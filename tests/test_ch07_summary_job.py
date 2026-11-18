import threading
import time
from langchain_core.messages import AIMessage
from app.config import Settings
from app.models import ConversationSummary
from app.services.history import append_message, create_conversation, get_conversation
from app.services.summary_job import run_summary, schedule_summary, shutdown_summary_executor


class FakeLLM:
    def __init__(self, summary="订单 1001，手机 13800000000，要求退款"):
        self.summary = summary
    def with_structured_output(self, schema, method=None):
        return self
    def invoke(self, messages):
        return type("R", (), {"summary": self.summary})()


def _settings():
    return Settings(_env_file=None, openai_api_key="x")


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


def test_run_summary_skips_empty_range(session):
    conv = _seed(session)
    result = run_summary(conv.id, _settings(), session=session, llm=FakeLLM())
    assert result.status == "skip" and result.reason == "no_range"


def test_run_summary_skips_invalid_output(session):
    conv = _seed(session, layer1_from=4)
    result = run_summary(conv.id, _settings(), session=session, llm=FakeLLM(""))
    assert result.status == "skip" and result.reason == "too_short"


def test_schedule_summary_is_non_blocking_and_deduplicated(monkeypatch):
    started, release = threading.Event(), threading.Event()
    def slow(cid, settings):
        started.set(); release.wait(2)
    monkeypatch.setattr("app.services.summary_job.run_summary", slow)
    assert schedule_summary(1, _settings()) is True
    assert started.wait(1)
    assert schedule_summary(1, _settings()) is False
    release.set(); shutdown_summary_executor()


def test_shutdown_is_idempotent():
    shutdown_summary_executor()
    shutdown_summary_executor()
