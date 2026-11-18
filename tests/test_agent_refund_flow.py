"""Refund sub-flow tests: fetch_order (Task 12) + retrieve_policy (Task 13)."""

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.agent.graph import (
    INFRA_TEXT,
    KNOWLEDGE_GAP_TEXT,
    ORDER_FAILED_TEXT,
    ORDER_INVALID_TEXT,
    RERANK_FAILED_TEXT,
    build_graph,
    extract_order_id,
)
from app.agent.intent import IntentResult
from app.config import Settings
from app.rag.retrieval import RetrievedChunk
from app.tools.business import DEMO_ORDERS
from tests.agent_fakes import (
    ScriptedLLM,
    ScriptedStructuredLLM,
    make_chunk,
    make_retrieval_result,
)

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


def _build(intent="退款退货", resolved=None, queries=None, **kwargs):
    intent_llm = ScriptedStructuredLLM(
        resolved or "能退吗",
        IntentResult(intent=intent, confidence=0.9),
        queries=queries,
    )
    return build_graph(
        ExplodingLLM(), intent_llm, checkpointer=InMemorySaver(), **kwargs
    )


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


def _chunk(chunk_id, score):
    return RetrievedChunk(
        chunk_id=chunk_id,
        score=score,
        questions=f"问题{chunk_id}",
        answer=f"答案{chunk_id}",
        section_path="售后/退款政策",
        category="售后",
        content_type="policy",
    )


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


def test_retrieve_policy_runs_three_queries_and_merges(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    tool_calls = []
    _order_tool(monkeypatch, tool_calls)
    retrieval_calls = []

    def fake_retrieve(query, **kwargs):
        retrieval_calls.append((query, kwargs.get("analysis")))
        return make_retrieval_result([make_chunk(0.9)])

    monkeypatch.setattr("app.agent.graph.retrieve_hybrid", fake_retrieve)
    graph = _build(resolved="订单 1001 能退吗", queries=["q1", "q2", "q3"])

    result = graph.invoke(_state(resolved="订单 1001 能退吗"), CONFIG)

    assert [call[0] for call in retrieval_calls] == ["q1", "q2", "q3"]
    # A prebuilt analysis skips retrieve_hybrid's second LLM rewrite.
    assert all(call[1] is not None for call in retrieval_calls)
    for query, analysis in retrieval_calls:
        assert analysis.original == query
        assert analysis.rewritten == query
        assert analysis.rewritten_by == "llm"
    assert result["expanded_queries"] == ["q1", "q2", "q3"]
    assert result["gate_passed"] is True
    assert result["gate_reason"] == "ok"
    assert result["confidence"] == 0.9
    # Task 13 keeps the refund pass edge on record_log; ExplodingLLM would
    # raise if main_agent were reached.
    assert any(
        "ch06 retrieve_policy order_id=1001" in record.getMessage()
        and "queries=['q1', 'q2', 'q3']" in record.getMessage()
        for record in caplog.records
    )


def test_retrieve_policy_dedupes_and_keeps_max_score(monkeypatch):
    tool_calls = []
    _order_tool(monkeypatch, tool_calls)
    by_query = {
        "q1": [_chunk(1, 0.4), _chunk(2, 0.3)],
        "q2": [_chunk(1, 0.8), _chunk(3, 0.7)],
        "q3": [],
    }

    def fake_retrieve(query, **kwargs):
        return make_retrieval_result(by_query[query])

    monkeypatch.setattr("app.agent.graph.retrieve_hybrid", fake_retrieve)
    settings = Settings(_env_file=None, openai_api_key="x", rerank_top_n=2)
    graph = _build(
        resolved="订单 1001 能退吗", queries=["q1", "q2", "q3"], settings=settings
    )

    result = graph.invoke(_state(resolved="订单 1001 能退吗"), CONFIG)

    scores = {source["chunk_id"]: source["score"] for source in result["sources"]}
    # chunk 1 appeared twice; the higher 0.8 wins.  Descending sort keeps
    # 0.8/0.7 and drops 0.3 at rerank_top_n=2.
    assert scores == {1: 0.8, 3: 0.7}
    assert len(result["sources"]) == 2
    assert result["confidence"] == 0.8


def test_retrieve_policy_infra_error(monkeypatch):
    tool_calls = []
    _order_tool(monkeypatch, tool_calls)
    retrieval_calls = []

    def fake_retrieve(query, **kwargs):
        retrieval_calls.append(query)
        if query == "q2":
            raise RuntimeError("milvus down")
        return make_retrieval_result([make_chunk(0.9)])

    monkeypatch.setattr("app.agent.graph.retrieve_hybrid", fake_retrieve)
    graph = _build(resolved="订单 1001 能退吗", queries=["q1", "q2", "q3"])

    result = graph.invoke(_state(resolved="订单 1001 能退吗"), CONFIG)

    assert retrieval_calls == ["q1", "q2"]
    assert result["expanded_queries"] == ["q1", "q2", "q3"]
    assert result["gate_reason"] == "infra_error"
    assert result["gate_passed"] is False
    assert result["answer"] == INFRA_TEXT
    assert result["evidence"] == []
    assert result["step_count"] == 0


def test_retrieve_policy_rerank_failure(monkeypatch):
    tool_calls = []
    _order_tool(monkeypatch, tool_calls)
    retrieval_calls = []

    def fake_retrieve(query, **kwargs):
        retrieval_calls.append(query)
        if query == "q2":
            return make_retrieval_result([_chunk(1, 0.9)], rerank_failed=True)
        return make_retrieval_result([_chunk(1, 0.9)])

    monkeypatch.setattr("app.agent.graph.retrieve_hybrid", fake_retrieve)
    graph = _build(resolved="订单 1001 能退吗", queries=["q1", "q2", "q3"])

    result = graph.invoke(_state(resolved="订单 1001 能退吗"), CONFIG)

    assert retrieval_calls == ["q1", "q2"]
    assert result["expanded_queries"] == ["q1", "q2", "q3"]
    assert result["gate_reason"] == "rerank_failed"
    assert result["gate_passed"] is False
    assert result["rerank_failed"] is True
    assert result["confidence"] is None
    assert result["evidence"] == []
    assert result["answer"] == RERANK_FAILED_TEXT
    assert result["step_count"] == 0


def test_retrieve_policy_low_score_fails_gate(monkeypatch):
    tool_calls = []
    _order_tool(monkeypatch, tool_calls)

    def fake_retrieve(query, **kwargs):
        return make_retrieval_result([_chunk(1, 0.02)])

    monkeypatch.setattr("app.agent.graph.retrieve_hybrid", fake_retrieve)
    graph = _build(resolved="订单 1001 能退吗", queries=["q1", "q2", "q3"])

    result = graph.invoke(_state(resolved="订单 1001 能退吗"), CONFIG)

    assert result["gate_passed"] is False
    assert result["gate_reason"] == "below_threshold"
    assert result["answer"] == KNOWLEDGE_GAP_TEXT
    assert result["step_count"] == 0
