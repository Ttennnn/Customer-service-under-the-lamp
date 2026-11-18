from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.subgraph import build_agent_subgraph
from app.agent.tools import AGENT_TOOLS
from tests.agent_fakes import ScriptedLLM, text_chunk, tool_chunk


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


def test_subgraph_executes_tool_then_converges(monkeypatch):
    calls = []

    def fake_execute_tool(tool, args, timeout_s=5):
        calls.append((tool.name, args, timeout_s))
        return "订单1001已发货"

    monkeypatch.setattr("app.agent.subgraph.execute_tool", fake_execute_tool)
    llm = ScriptedLLM(
        [
            tool_chunk("query_order", '{"order_id": "1001"}'),
            text_chunk("您的订单已发货"),
        ]
    )
    result = build_agent_subgraph(llm).invoke(_initial_state())

    assert result["answer"] == "您的订单已发货"
    assert result["step_count"] == 2
    assert calls == [("query_order", {"order_id": "1001"}, 5.0)]
    assert any(
        isinstance(m, ToolMessage) and m.content == "订单1001已发货"
        for m in result["messages"]
    )


def test_subgraph_does_not_bind_create_ticket_or_query_faq():
    llm = ScriptedLLM([text_chunk("hi")])
    build_agent_subgraph(llm).invoke(_initial_state())

    assert llm.bound_tool_names == [t.name for t in AGENT_TOOLS]
    assert "create_ticket" not in llm.bound_tool_names
    assert "query_faq" not in llm.bound_tool_names


def test_suggest_handoff_records_options_without_side_effect(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("suggest_handoff must not call execute_tool")

    monkeypatch.setattr("app.agent.subgraph.execute_tool", boom)
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
    assert result["suggested_actions"] == [
        {"action": "transfer", "label": "转人工"},
        {"action": "ticket", "label": "建工单", "ticket_type": "投诉"},
    ]
    assert any(isinstance(m, ToolMessage) for m in result["messages"])


def test_submit_refund_is_intercepted_without_execute_tool(monkeypatch):
    import app.agent.subgraph as sg

    monkeypatch.setattr(
        sg,
        "execute_tool",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not execute")),
    )
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
    assert result["suggested_actions"] == [
        {"action": "refund_form", "label": "提交退款工单", "order_id": "1001"}
    ]


def test_submit_refund_with_mismatched_order_emits_no_action(monkeypatch):
    import app.agent.subgraph as sg

    monkeypatch.setattr(
        sg,
        "execute_tool",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not execute")),
    )
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
    assert llm.bound_tool_names == ["submit_refund", "suggest_handoff"]


def test_max_steps_runs_finalize(monkeypatch):
    monkeypatch.setattr(
        "app.agent.subgraph.execute_tool",
        lambda tool, args, timeout_s=5: "ok",
    )
    llm = ScriptedLLM(
        [tool_chunk("query_order", '{"order_id": "1"}') for _ in range(5)]
    )
    result = build_agent_subgraph(llm, max_steps=2).invoke(_initial_state())

    assert result["exhausted"] is True
    assert result["answer"]
    assert {"action": "transfer", "label": "转人工"} in result["suggested_actions"]


def test_token_budget_runs_finalize(monkeypatch):
    monkeypatch.setattr(
        "app.agent.subgraph.execute_tool",
        lambda tool, args, timeout_s=5: "ok",
    )
    llm = ScriptedLLM([tool_chunk("query_order", '{"order_id": "1"}')])
    result = build_agent_subgraph(llm, token_budget=1).invoke(_initial_state())

    assert result["exhausted"] is True
    assert result["answer"]
    assert {"action": "transfer", "label": "转人工"} in result["suggested_actions"]


def test_custom_stream_emits_tool_then_tokens(monkeypatch):
    monkeypatch.setattr(
        "app.agent.subgraph.execute_tool",
        lambda tool, args, timeout_s=5: "ok",
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
