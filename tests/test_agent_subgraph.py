from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.subgraph import build_agent_subgraph
from app.config import Settings
from app.models import ToolAuditLog
from app.tools.registry import builtin_specs
from app.tools.spec import ToolRun
from tests.agent_fakes import ScriptedLLM, text_chunk, tool_chunk


def _settings():
    return Settings(_env_file=None, openai_api_key="test")


def _initial_state(**overrides):
    state = {
        "messages": [HumanMessage(content="你好")],
        "evidence": [],
        "route": "business",
        "step_count": 0,
        "token_estimate": 0,
        "token_budget": 12000,
        "max_steps": 6,
        "suggested_actions": [],
        "conversation_id": 1,
        "exhausted": False,
    }
    state.update(overrides)
    return state


def test_subgraph_converges_without_tool():
    llm = ScriptedLLM([text_chunk("你好")])
    result = build_agent_subgraph(llm).invoke(_initial_state())

    assert result["answer"] == "你好"
    assert result["step_count"] == 1
    assert isinstance(result["messages"][-1], AIMessage)


def test_subgraph_can_ask_follow_up():
    llm = ScriptedLLM([text_chunk("请提供订单号")])
    result = build_agent_subgraph(llm).invoke(_initial_state())

    assert result["answer"] == "请提供订单号"
    last = result["messages"][-1]
    assert isinstance(last, AIMessage)
    assert not last.tool_calls


def test_business_route_binds_registry_tools(monkeypatch):
    llm = ScriptedLLM([text_chunk("hi")])
    monkeypatch.setattr(
        "app.agent.subgraph.get_all_specs",
        lambda settings: {
            "query_order": builtin_specs()["query_order"],
            "create_ticket": builtin_specs()["create_ticket"],
            "query_faq": builtin_specs()["query_faq"],
            "suggest_handoff": builtin_specs()["suggest_handoff"],
        },
    )
    build_agent_subgraph(llm).invoke(_initial_state())
    assert llm.bound_tool_names == ["query_order", "create_ticket", "suggest_handoff"]


def test_tools_node_uses_engine(monkeypatch):
    seen = {}

    def fake_execute_tool_call(
        tool_call, conversation_id, *, confirmed=False, specs=None, settings=None
    ):
        seen["name"] = tool_call["name"]
        return ToolRun(status="ok", content="订单已签收")

    monkeypatch.setattr("app.agent.subgraph.execute_tool_call", fake_execute_tool_call)
    llm = ScriptedLLM([
        tool_chunk("query_order", '{"order_id":"1001"}'),
        text_chunk("已查"),
    ])
    result = build_agent_subgraph(llm).invoke(_initial_state())
    assert seen["name"] == "query_order"
    assert any(isinstance(m, ToolMessage) and m.content == "订单已签收"
               for m in result["messages"])


def test_subgraph_executes_tool_then_converges(monkeypatch):
    calls = []

    def fake_execute_tool_call(
        tool_call, conversation_id, *, confirmed=False, specs=None, settings=None
    ):
        calls.append((tool_call["name"], tool_call["args"], conversation_id))
        return ToolRun(status="ok", content="订单1001已发货")

    monkeypatch.setattr("app.agent.subgraph.execute_tool_call", fake_execute_tool_call)
    llm = ScriptedLLM(
        [
            tool_chunk("query_order", '{"order_id": "1001"}'),
            text_chunk("您的订单已发货"),
        ]
    )
    result = build_agent_subgraph(llm).invoke(_initial_state())

    assert result["answer"] == "您的订单已发货"
    assert result["step_count"] == 2
    assert calls == [("query_order", {"order_id": "1001"}, 1)]
    assert any(
        isinstance(m, ToolMessage) and m.content == "订单1001已发货"
        for m in result["messages"]
    )


def test_refund_route_binds_create_ticket_but_excludes_faq_and_order_tools():
    llm = ScriptedLLM([text_chunk("hi")])
    build_agent_subgraph(llm).invoke(
        _initial_state(route="refund_flow", order_data={"order_id": "1001"})
    )

    assert llm.bound_tool_names == ["submit_refund", "suggest_handoff", "create_ticket"]
    assert "query_faq" not in llm.bound_tool_names
    assert "query_order" not in llm.bound_tool_names


def test_business_route_binds_exposed_builtin_tools():
    llm = ScriptedLLM([text_chunk("hi")])
    build_agent_subgraph(llm).invoke(_initial_state())

    assert llm.bound_tool_names == [
        "query_order",
        "query_product",
        "query_promotions",
        "suggest_handoff",
        "create_ticket",
    ]
    assert "query_faq" not in llm.bound_tool_names


def test_business_route_hallucinated_query_faq_is_unknown_and_audited(
    session, monkeypatch
):
    calls = []

    def spy_retrieve(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("query_faq must not run on the business route")

    # The registry exposes query_faq to the parent graph only; if it leaks
    # into the business allowlist the engine would invoke the tool, which
    # calls retrieve_hybrid.
    monkeypatch.setattr("app.tools.builtin.faq.retrieve_hybrid", spy_retrieve)
    llm = ScriptedLLM(
        [
            tool_chunk("query_faq", '{"keyword": "邮费"}'),
            text_chunk("知识库暂未收录"),
        ]
    )

    result = build_agent_subgraph(llm, settings=_settings()).invoke(
        _initial_state()
    )

    assert calls == []
    session.expire_all()
    row = session.query(ToolAuditLog).filter_by(tool_name="query_faq").one()
    assert row.status == "失败"
    assert "未知工具" in (row.error_message or "")
    assert any(
        isinstance(message, ToolMessage) and "未知工具" in message.content
        for message in result["messages"]
    )


def test_business_route_hallucinated_submit_refund_is_unknown_and_audited(session):
    llm = ScriptedLLM(
        [
            tool_chunk("submit_refund", '{"order_id": "1001"}'),
            text_chunk("无法准备退款表单"),
        ]
    )

    result = build_agent_subgraph(llm, settings=_settings()).invoke(
        _initial_state(order_data={"order_id": "1001"})
    )

    assert result["suggested_actions"] == []
    session.expire_all()
    row = session.query(ToolAuditLog).filter_by(tool_name="submit_refund").one()
    assert row.status == "失败"
    assert "未知工具" in (row.error_message or "")
    assert any(
        isinstance(message, ToolMessage) and "未知工具" in message.content
        for message in result["messages"]
    )


def test_suggest_handoff_records_options_without_side_effect(monkeypatch):
    seen = {}

    def fake_execute(
        spec,
        args,
        *,
        conversation_id,
        tool_call_id=None,
        confirmed=False,
        settings=None,
        retries=None,
    ):
        seen["name"] = spec.name
        seen["args"] = args
        return ToolRun(status="ok", content="已记录转人工/建工单建议。")

    monkeypatch.setattr("app.agent.subgraph.execute", fake_execute)
    llm = ScriptedLLM(
        [
            tool_chunk(
                "suggest_handoff",
                '{"options": ["transfer", "ticket"], "ticket_type": "投诉"}',
            ),
            text_chunk("已为您记录转人工与建工单建议"),
        ]
    )
    result = build_agent_subgraph(llm).invoke(_initial_state())

    assert result["answer"] == "已为您记录转人工与建工单建议"
    assert seen["name"] == "suggest_handoff"
    assert seen["args"]["options"] == ["transfer", "ticket"]
    assert result["suggested_actions"] == [
        {"action": "transfer", "label": "转人工"},
        {"action": "ticket", "label": "建工单", "ticket_type": "投诉"},
    ]
    assert any(isinstance(m, ToolMessage) for m in result["messages"])


def test_invalid_handoff_denied_by_validation_without_execute(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("invalid handoff must not execute")

    monkeypatch.setattr("app.agent.subgraph.execute", boom)
    llm = ScriptedLLM(
        [
            tool_chunk("suggest_handoff", '{"options": ["bogus"]}'),
            text_chunk("已记录"),
        ]
    )
    result = build_agent_subgraph(llm).invoke(_initial_state())

    assert result["suggested_actions"] == []
    assert any(
        isinstance(m, ToolMessage) and "参数无效" in m.content
        for m in result["messages"]
    )


def test_submit_refund_runs_through_engine_and_emits_action(monkeypatch):
    seen = {}

    def fake_execute(
        spec,
        args,
        *,
        conversation_id,
        tool_call_id=None,
        confirmed=False,
        settings=None,
        retries=None,
    ):
        seen["name"] = spec.name
        return ToolRun(status="ok", content="已准备退款表单。")

    monkeypatch.setattr("app.agent.subgraph.execute", fake_execute)
    llm = ScriptedLLM([
        tool_chunk("submit_refund", '{"order_id": "1001"}'),
        text_chunk("可以退,请填写退款原因。"),
    ])
    graph = build_agent_subgraph(llm)
    result = graph.invoke({
        "messages": [HumanMessage(content="能退吗")],
        "route": "refund_flow",
        "order_data": {"order_id": "1001", "status": "已签收"},
        "suggested_actions": [],
        "token_estimate": 0,
        "conversation_id": 1,
    })
    assert seen["name"] == "submit_refund"
    assert result["suggested_actions"] == [
        {"action": "refund_form", "label": "提交退款工单", "order_id": "1001"}
    ]


def test_submit_refund_with_mismatched_order_emits_no_action(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("mismatched refund must not execute")

    monkeypatch.setattr("app.agent.subgraph.execute", boom)
    llm = ScriptedLLM([
        tool_chunk("submit_refund", '{"order_id": "1002"}'),
        text_chunk("当前订单暂不支持退款。"),
    ])
    result = build_agent_subgraph(llm).invoke(
        _initial_state(
            route="refund_flow",
            order_data={"order_id": "1001", "status": "已签收"},
        )
    )

    assert result["suggested_actions"] == []
    assert any(
        isinstance(m, ToolMessage) and "不一致" in m.content
        for m in result["messages"]
    )


def test_refund_route_binds_after_sales_tools_and_prompt():
    llm = ScriptedLLM([text_chunk("可以退")])
    graph = build_agent_subgraph(llm)
    graph.invoke({
        "messages": [HumanMessage(content="能退吗")],
        "route": "refund_flow",
        "order_data": {"order_id": "1001", "status": "已签收"},
        "evidence": [],
        "suggested_actions": [],
        "token_estimate": 0,
        "conversation_id": 1,
    })
    assert llm.bound_tool_names == ["submit_refund", "suggest_handoff", "create_ticket"]


def test_max_steps_runs_finalize(monkeypatch):
    monkeypatch.setattr(
        "app.agent.subgraph.execute_tool_call",
        lambda tool_call, conversation_id, **kwargs: ToolRun(
            status="ok", content="ok"
        ),
    )
    llm = ScriptedLLM(
        [tool_chunk("query_order", '{"order_id": "1"}') for _ in range(5)]
    )
    result = build_agent_subgraph(llm, max_steps=2).invoke(_initial_state())

    assert result["exhausted"] is True
    assert result["answer"]
    assert {"action": "transfer", "label": "转人工"} in result["suggested_actions"]


def test_legacy_token_budget_no_longer_runs_finalize():
    # ch07 retires AGENT_TOKEN_BUDGET as an Agent stop condition: the context
    # window and max_steps govern the turn instead.
    llm = ScriptedLLM([text_chunk("答")])
    result = build_agent_subgraph(llm).invoke(
        _initial_state(token_budget=1, token_estimate=999999)
    )

    assert result["exhausted"] is False
    assert result["answer"] == "答"


def test_custom_stream_emits_tool_then_tokens(monkeypatch):
    monkeypatch.setattr(
        "app.agent.subgraph.execute_tool_call",
        lambda tool_call, conversation_id, **kwargs: ToolRun(
            status="ok", content="ok"
        ),
    )
    llm = ScriptedLLM(
        [
            tool_chunk("query_order", '{"order_id": "1"}'),
            text_chunk("已发货"),
        ]
    )
    graph = build_agent_subgraph(llm)

    frames = list(graph.stream(_initial_state(), stream_mode="custom"))

    assert {"type": "tool", "tool": "query_order", "label": "查订单"} in frames
    assert {"type": "token", "text": "已发货"} in frames
    assert all(frame["type"] in {"token", "tool"} for frame in frames)
    first_tool = next(
        index for index, frame in enumerate(frames) if frame["type"] == "tool"
    )
    last_token = max(
        index for index, frame in enumerate(frames) if frame["type"] == "token"
    )
    assert first_tool < last_token
