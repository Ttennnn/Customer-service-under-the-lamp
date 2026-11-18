import logging

from langchain_core.messages import AIMessage, HumanMessage
from app.config import Settings
from app.models import Conversation
from app.services.history import append_message, create_conversation, get_conversation
from app.services.session_context import _summary_upto_id, prepare_context


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


def test_prepare_context_prepends_missing_prefix_to_non_empty_state(session):
    conv, msgs, ids = _seed(session, turns=3)
    # State 从第二轮开始：第一轮必须补到前面，已在 State 的第二轮不能重复
    state = msgs[2:]
    plan = prepare_context(session, conv, state, len(state), _settings())
    contents = [m.content for m in plan.prompt_history]
    assert contents[0] == "问0"
    assert contents.count("问1") == 1


def test_prepare_context_rebuild_excludes_current_user(session):
    conv, msgs, ids = _seed(session, turns=3)
    plan = prepare_context(
        session, conv, [], 0, _settings(), current_msg_id=ids[-1][0]
    )
    contents = [m.content for m in plan.prompt_history]
    assert "问0" in contents and "问1" in contents
    assert "问2" not in contents  # 当前用户行不再补进历史


def test_prepare_context_triggers_summary_when_l2_over_budget(
    session, monkeypatch, caplog
):
    caplog.set_level(logging.INFO)
    conv, msgs, ids = _seed(session, turns=6, answer="答" * 400, question="问" + "细" * 500)
    conv.layer1_from_msg_id = ids[-2][1]  # 倒数第二轮 assistant 之后是最后一轮 L1
    session.commit()
    calls = []
    monkeypatch.setattr("app.services.session_context.schedule_summary", lambda cid, settings: calls.append(cid) or True)
    prepare_context(session, conv, msgs, len(msgs), _settings())
    assert calls == [conv.id]
    # 摘取范围必须止于 L1 起点之前，即倒数第二轮 assistant
    assert f"upto={ids[-2][1]}" in caplog.text


def test_summary_upto_uses_layer1_anchor_when_it_is_last_l2_id(session):
    conv, msgs, ids = _seed(session, turns=2)
    l2 = [
        HumanMessage(content="问0", id=f"db-{ids[0][0]}"),
        AIMessage(content="答0", id=f"db-{ids[0][1]}"),
    ]
    assert _summary_upto_id(session, conv.id, l2, [], ids[0][1]) == ids[0][1]


def test_prepare_context_does_not_trigger_within_budget(session, monkeypatch):
    conv, msgs, ids = _seed(session, turns=2)
    conv.layer1_from_msg_id = ids[0][1]  # L2 非空，但远低于层 2 预算
    session.commit()
    calls = []
    monkeypatch.setattr("app.services.session_context.schedule_summary", lambda cid, settings: calls.append(cid) or True)
    plan = prepare_context(session, conv, msgs, len(msgs), _settings())
    # 确认真的走到了 token 比较分支：L2 非空且未超预算
    assert plan.history_l2 and plan.l2_tokens <= plan.budget.layer2
    assert calls == []
