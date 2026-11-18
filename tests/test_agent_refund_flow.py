"""Refund sub-flow tests: fetch_order slot + interrupt/resume (Task 12)."""

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.agent.graph import (
    ORDER_FAILED_TEXT,
    ORDER_INVALID_TEXT,
    build_graph,
    extract_order_id,
)
from app.agent.intent import IntentResult
from app.tools.business import DEMO_ORDERS
from tests.agent_fakes import ScriptedLLM, ScriptedStructuredLLM

CONFIG = {"configurable": {"thread_id": "test-refund-thread"}}


class ExplodingLLM:
    """Agent LLM that fails the test if the Agent path is taken."""

    def bind_tools(self, tools, **kwargs):
        raise AssertionError("Agent LLM must not be called")

    def stream(self, messages):
        raise AssertionError("Agent LLM must not be called")


def _state(message="能退吗", resolved=None, **overrides):
    resolved = message if resolved is None else resolved
    state = {
        "messages": [HumanMessage(content=message)],
        "user_message": message,
        "resolved_query": resolved,
        "session_id": "test-session",
        "conversation_id": 1,
        "step_count": 0,
        "token_estimate": 0,
        "evidence": [],
        "sources": [],
        "suggested_actions": [],
        "gate_passed": False,
        "exhausted": False,
    }
    state.update(overrides)
    return state


def _build(intent="退款退货", resolved=None):
    intent_llm = ScriptedStructuredLLM(
        resolved or "能退吗",
        IntentResult(intent=intent, confidence=0.9),
    )
    return build_graph(ExplodingLLM(), intent_llm, checkpointer=InMemorySaver())


def _run_to_interrupt(graph, message="能退吗", resolved=None):
    list(
        graph.stream(
            _state(message, resolved=resolved),
            CONFIG,
            stream_mode="custom",
        )
    )
    return graph.get_state(CONFIG)


def _order_tool(monkeypatch, calls, payload=None):
    def fake_execute_tool(tool, args, **kwargs):
        calls.append((tool.name, dict(args)))
        data = payload
        if data is None:
            data = DEMO_ORDERS[args["order_id"]]
        return json.dumps(data, ensure_ascii=False)

    monkeypatch.setattr("app.agent.graph.execute_tool", fake_execute_tool)


def test_extract_order_id_patterns():
    assert extract_order_id("订单 1001 能退吗") == "1001"
    assert extract_order_id("订单号：1002") == "1002"
    assert extract_order_id("1003号订单") == "1003"
    assert extract_order_id("金额 299 元") is None
    assert extract_order_id("金额 1299 元") is None
    assert extract_order_id("1299 元") is None
    assert extract_order_id("1299块") is None
    assert extract_order_id("2026年能退吗") is None


def test_fetch_order_rejects_unknown_id_and_interrupts(monkeypatch):
    calls = []
    _order_tool(monkeypatch, calls)
    graph = _build(resolved="订单 1004 能退吗")

    snap = _run_to_interrupt(graph, resolved="订单 1004 能退吗")

    assert snap.interrupts
    # Prove the catalog-external ID actually reached the node and was rejected.
    assert snap.values["resolved_query"] == "订单 1004 能退吗"
    payload = snap.interrupts[0].value
    assert payload["kind"] == "select_order"
    assert payload["slot"] == "order_id"
    assert {order["order_id"] for order in payload["orders"]} == {
        "1001",
        "1002",
        "1003",
    }
    # Pre-interrupt code must stay read-only: no order lookup, no slot write.
    assert calls == []
    assert not any(
        isinstance(message, SystemMessage) for message in snap.values["messages"]
    )
    assert snap.values.get("order_id") is None


def test_fetch_order_resume_fills_slot_and_adds_system_message(monkeypatch):
    calls = []
    _order_tool(monkeypatch, calls)
    graph = _build()

    snap = _run_to_interrupt(graph)
    assert snap.interrupts
    assert calls == []

    list(graph.stream(Command(resume="1001"), CONFIG, stream_mode="custom"))
    values = graph.get_state(CONFIG).values

    assert values["order_id"] == "1001"
    assert values["order_data"]["order_id"] == "1001"
    assert calls == [("query_order", {"order_id": "1001"})]
    markers = [
        message
        for message in values["messages"]
        if isinstance(message, SystemMessage)
        and message.content == "[订单选择器] 用户选择了订单 1001"
    ]
    # The node restarts from the top on resume; the marker must appear once.
    assert len(markers) == 1


def test_fetch_order_invalid_resume_falls_back(monkeypatch):
    calls = []
    _order_tool(monkeypatch, calls)
    graph = _build()

    _run_to_interrupt(graph)
    list(graph.stream(Command(resume="9999"), CONFIG, stream_mode="custom"))
    values = graph.get_state(CONFIG).values

    assert values["answer"] == ORDER_INVALID_TEXT
    assert values.get("order_data") is None
    assert values["gate_reason"] == "order_slot_failed"
    assert values["gate_passed"] is False
    assert isinstance(values["messages"][-1], AIMessage)
    assert values["messages"][-1].content == ORDER_INVALID_TEXT
    assert calls == []


def test_fetch_order_valid_id_from_message_calls_query_once(monkeypatch):
    calls = []
    _order_tool(monkeypatch, calls)
    graph = _build(resolved="订单 1001 能退吗")

    values = graph.invoke(_state(resolved="订单 1001 能退吗"), CONFIG)

    assert values["order_id"] == "1001"
    assert values["order_data"]["order_id"] == "1001"
    assert calls == [("query_order", {"order_id": "1001"})]
    assert not graph.get_state(CONFIG).interrupts


def test_fetch_order_resume_tool_failure_keeps_selection(monkeypatch):
    calls = []

    def failing_execute_tool(tool, args, **kwargs):
        calls.append((tool.name, dict(args)))
        return "工具执行失败"

    monkeypatch.setattr("app.agent.graph.execute_tool", failing_execute_tool)
    graph = _build()

    _run_to_interrupt(graph)
    list(graph.stream(Command(resume="1001"), CONFIG, stream_mode="custom"))
    values = graph.get_state(CONFIG).values

    assert values["answer"] == ORDER_FAILED_TEXT
    assert values["gate_reason"] == "order_fetch_failed"
    assert values["order_id"] == "1001"
    assert values.get("order_data") is None
    markers = [
        message
        for message in values["messages"]
        if isinstance(message, SystemMessage)
        and message.content == "[订单选择器] 用户选择了订单 1001"
    ]
    assert len(markers) == 1
    assert isinstance(values["messages"][-1], AIMessage)
    assert values["messages"][-1].content == ORDER_FAILED_TEXT
    assert calls == [("query_order", {"order_id": "1001"})]


def test_fetch_order_tool_failure_falls_back(monkeypatch):
    calls = []

    def failing_execute_tool(tool, args, **kwargs):
        calls.append((tool.name, dict(args)))
        return "工具执行失败"

    monkeypatch.setattr("app.agent.graph.execute_tool", failing_execute_tool)
    graph = _build(resolved="订单 1001 能退吗")

    values = graph.invoke(_state(resolved="订单 1001 能退吗"), CONFIG)

    assert values["answer"] == ORDER_FAILED_TEXT
    assert values.get("order_data") is None
    assert values["gate_reason"] == "order_fetch_failed"
    assert values["gate_passed"] is False
    assert isinstance(values["messages"][-1], AIMessage)
    assert values["messages"][-1].content == ORDER_FAILED_TEXT
    assert calls == [("query_order", {"order_id": "1001"})]
