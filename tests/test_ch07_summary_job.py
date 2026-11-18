import threading
import time
from types import SimpleNamespace
from langchain_core.messages import AIMessage
from app.config import Settings
from app.models import ConversationSummary
from app.services.history import append_message, create_conversation, get_conversation
from app.services.summary_job import (
    _render_row,
    build_summary_prompt,
    run_summary,
    schedule_summary,
    shutdown_summary_executor,
)


class FakeLLM:
    def __init__(self, summary="订单 1001，手机 13800000000，要求退款"):
        self.summary = summary
    def with_structured_output(self, schema, method=None):
        return self
    def invoke(self, messages):
        return type("R", (), {"summary": self.summary})()


def _settings(**kwargs):
    return Settings(_env_file=None, openai_api_key="x", **kwargs)


def _seed(session, summary_upto=None, layer1_from=None):
    cid = create_conversation(session, "u1").id
    for text in ["订单 1001 想退款", "好的，请提供手机号", "13800000000", "已登记"]:
        append_message(session, cid, "user" if text in ("订单 1001 想退款", "13800000000") else "assistant", text)
    session.commit()
    conv = get_conversation(session, cid)
    conv.summary_upto_msg_id = summary_upto
    conv.layer1_from_msg_id = layer1_from
    session.commit()
    return conv


def test_run_summary_appends_segment_and_advances_upto(session):
    conv = _seed(session, layer1_from=4)
    result = run_summary(conv.id, _settings(), session=session, llm=FakeLLM())
    assert result.status == "done" and result.seq == 1
    updated = get_conversation(session, conv.id)
    assert updated.summary_upto_msg_id == 4
    assert updated.summary.startswith("第1段:")
    assert "订单 1001" in updated.summary


def test_run_summary_does_not_rewrite_old_segments(session):
    conv = _seed(session, layer1_from=4)
    old = "旧段" + "旧" * 25  # >= 20 chars, passes summary validation
    new = "新段" + "新" * 25
    run_summary(conv.id, _settings(), session=session, llm=FakeLLM(old))
    append_message(session, conv.id, "user", "后来问物流")
    append_message(session, conv.id, "assistant", "在路上")
    session.commit()
    conv = get_conversation(session, conv.id)
    conv.layer1_from_msg_id = 6
    session.commit()
    run_summary(conv.id, _settings(), session=session, llm=FakeLLM(new))
    rows = session.query(ConversationSummary).order_by(ConversationSummary.seq).all()
    assert [row.content for row in rows] == [old, new]
    updated = get_conversation(session, conv.id)
    assert "第1段:" in updated.summary and "第2段:" in updated.summary
    assert f"第1段:{old}" in updated.summary
    assert f"第2段:{new}" in updated.summary


def test_run_summary_skips_empty_range(session):
    conv = _seed(session)
    result = run_summary(conv.id, _settings(), session=session, llm=FakeLLM())
    assert result.status == "skip" and result.reason == "no_range"


def test_run_summary_skips_short_nonempty_output(session):
    conv = _seed(session, layer1_from=4)
    result = run_summary(conv.id, _settings(), session=session, llm=FakeLLM("短"))
    assert result.status == "skip" and result.reason == "too_short"
    # A too-short non-empty summary must not consume the range.
    assert get_conversation(session, conv.id).summary_upto_msg_id is None


def test_run_summary_consumes_empty_summary_without_segment(session, caplog):
    conv = _seed(session, layer1_from=4)
    with caplog.at_level("INFO", logger="app.services.summary_job"):
        result = run_summary(conv.id, _settings(), session=session, llm=FakeLLM(""))

    assert result.status == "skip" and result.reason == "empty_consumed"
    assert result.from_msg_id == 1 and result.upto_msg_id == 4
    updated = get_conversation(session, conv.id)
    assert updated.summary_upto_msg_id == 4
    assert updated.summary is None  # projection untouched
    segments = (
        session.query(ConversationSummary)
        .filter_by(conversation_id=conv.id)
        .count()
    )
    assert segments == 0
    assert "reason=empty_consumed" in caplog.text


def test_schedule_summary_is_non_blocking_and_deduplicated(monkeypatch):
    started, release = threading.Event(), threading.Event()
    def slow(cid, settings):
        started.set(); release.wait(2)
    monkeypatch.setattr("app.services.summary_job.run_summary", slow)
    assert schedule_summary(1, _settings()) is True
    assert started.wait(1)
    assert schedule_summary(1, _settings()) is False
    release.set(); shutdown_summary_executor()


def test_schedule_summary_returns_false_when_submit_fails(monkeypatch):
    import app.services.summary_job as summary_job

    class BrokenExecutor:
        def submit(self, *args, **kwargs):
            raise RuntimeError("cannot schedule new futures after shutdown")

    monkeypatch.setattr(summary_job, "_get_executor", lambda: BrokenExecutor())
    try:
        assert schedule_summary(99, _settings()) is False
        assert 99 not in summary_job._inflight
    finally:
        shutdown_summary_executor()


def test_shutdown_is_idempotent():
    shutdown_summary_executor()
    shutdown_summary_executor()


def test_render_row_marks_order_audit_as_system_fact():
    line = _render_row(SimpleNamespace(role="user", content="[订单选择] 订单 1001"))
    assert line == "系统事实：用户选择了订单 1001"
    assert "用户：" not in line
    assert "[订单选择]" not in line


def test_build_summary_prompt_uses_configured_cap():
    messages = build_summary_prompt("旧摘要", "用户：你好", max_chars=80)
    system = messages[0].content
    assert "50–80" in system
    assert "400" not in system
    assert "不得少于 20" not in system


def test_run_summary_respects_configured_max_chars(session):
    conv = _seed(session, layer1_from=4)
    result = run_summary(
        conv.id,
        _settings(summary_max_chars=50),
        session=session,
        llm=FakeLLM("长" * 60),
    )
    assert result.status == "skip" and result.reason == "too_long"


def test_cancelled_queued_job_clears_inflight(monkeypatch):
    import app.services.summary_job as summary_job

    release, both_started = threading.Event(), threading.Event()
    running = []

    def slow(cid, settings):
        if cid <= 2:
            running.append(cid)
            if len(running) == 2:
                both_started.set()
            release.wait(2)

    monkeypatch.setattr("app.services.summary_job.run_summary", slow)
    try:
        assert schedule_summary(1, _settings()) is True
        assert schedule_summary(2, _settings()) is True
        assert both_started.wait(2)
        assert schedule_summary(3, _settings()) is True  # queued behind both workers
        assert 3 in summary_job._inflight
        shutdown_summary_executor()
        assert 3 not in summary_job._inflight
    finally:
        release.set()
        shutdown_summary_executor()
