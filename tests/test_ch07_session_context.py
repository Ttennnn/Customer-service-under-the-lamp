import logging

from langchain_core.messages import AIMessage, HumanMessage
from app.config import Settings
from app.models import Conversation
from app.services.history import append_message, create_conversation, get_conversation
from app.services.session_context import prepare_context


def _settings(**kw):
    base = dict(_env_file=None, openai_api_key="x", model_context_window=18000,
                max_output_tokens=2000, max_user_input_tokens=2000,
                max_agent_steps=3, tool_result_max_tokens=1200, rerank_top_k=5)
    base.update(kw)
    return Settings(**base)


def _seed(session, turns=4, answer="答" * 100, question="问"):
    cid = create_conversation(session, "u1").id
    msgs, ids = [], []
    for i in range(turns):
        u = append_message(session, cid, "user", f"{question}{i}")
        a = append_message(session, cid, "assistant", answer)
        msgs += [HumanMessage(content=u.content, id=f"db-{u.id}"), AIMessage(content=a.content)]
        ids.append((u.id, a.id))
    session.commit()
    return get_conversation(session, cid), msgs, ids


def test_prepare_context_splits_and_logs_history_ctx(session, caplog):
    caplog.set_level(logging.INFO)
    conv, msgs, ids = _seed(session, turns=2)
    conv.layer1_from_msg_id = ids[0][1]  # 第一轮 assistant 之后是 L1 起点
    session.commit()
    plan = prepare_context(session, conv, msgs, len(msgs), _settings())
    assert plan.history_l2 and plan.history_l1
    assert "history_ctx" in caplog.text
    assert plan.prompt_history == plan.history_l2 + plan.history_l1


def test_prepare_context_degrades_l1_and_persists_anchor(session, caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr("app.services.session_context.schedule_summary", lambda cid, settings: True)
    conv, msgs, ids = _seed(session, turns=8, answer="答" * 200)
    plan = prepare_context(session, conv, msgs, len(msgs), _settings(model_context_window=4000, max_output_tokens=200))
    assert plan.degraded is True
    updated = get_conversation(session, conv.id)
    assert updated.layer1_from_msg_id is not None
    assert "层1 降级" in caplog.text


def test_prepare_context_merges_db_history_before_oldest_state_id(session):
    conv, msgs, ids = _seed(session, turns=3)
    # State 只剩最后一轮，前面两轮要从 MySQL 补回来
    plan = prepare_context(session, conv, msgs[4:], 0, _settings())
    assert any(m.content == "问0" for m in plan.prompt_history)
    assert any(m.content == "问1" for m in plan.prompt_history)


def test_prepare_context_triggers_summary_when_l2_over_budget(session, monkeypatch):
    conv, msgs, ids = _seed(session, turns=6, answer="答" * 400, question="问" + "细" * 500)
    conv.layer1_from_msg_id = ids[-2][1]  # 倒数第二轮 assistant 之后是最后一轮 L1
    session.commit()
    calls = []
    monkeypatch.setattr("app.services.session_context.schedule_summary", lambda cid, settings: calls.append(cid) or True)
    prepare_context(session, conv, msgs, len(msgs), _settings())
    assert calls == [conv.id]


def test_prepare_context_does_not_trigger_within_budget(session, monkeypatch):
    conv, msgs, _ = _seed(session, turns=2)
    calls = []
    monkeypatch.setattr("app.services.session_context.schedule_summary", lambda cid, settings: calls.append(cid) or True)
    prepare_context(session, conv, msgs, len(msgs), _settings())
    assert calls == []
