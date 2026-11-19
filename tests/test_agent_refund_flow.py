"""Refund sub-flow tests: fetch_order (Task 12) + retrieve_policy (Task 13)."""

import copy
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
    fetch_order_node,
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
    text_chunk,
    tool_chunk,
)

CONFIG = {"configurable": {"thread_id": "test-refund-thread"}}


class ExplodingLLM:
    """Agent LLM that fails the test if the Agent path is taken."""

    def bind_tools(self, tools, **kwargs):
        raise AssertionError("Agent LLM must not be called")

    def stream(self, messages):
        raise AssertionError("Agent LLM must not be called")


class RecordingScriptedLLM(ScriptedLLM):
    """ScriptedLLM that records invocation order and the Agent prompt."""

    def __init__(self, chunks, calls):
        super().__init__(chunks)
        self.calls = calls
        self.messages = []

    def stream(self, messages):
        self.calls.append(("agent_llm", None))
        self.messages = list(messages)
        yield from super().stream(messages)


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


def _build(llm=None, *, intent="退款退货", resolved=None, queries=None, **kwargs):
    intent_llm = ScriptedStructuredLLM(
        resolved or "能退吗",
        IntentResult(intent=intent, confidence=0.9),
        queries=queries,
    )
    return build_graph(
        llm if llm is not None else ExplodingLLM(),
        intent_llm,
        checkpointer=InMemorySaver(),
        **kwargs,
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
    def fake_order_snapshot(order_id):
        calls.append(("order_snapshot", {"order_id": order_id}))
        data = payload
        if data is None:
            data = DEMO_ORDERS[order_id]
        return copy.deepcopy(data)

    monkeypatch.setattr("app.agent.graph.order_snapshot", fake_order_snapshot)


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


def test_fetch_order_resume_fills_slot_and_adds_human_message(monkeypatch):
    calls = []
    _order_tool(monkeypatch, calls)
    # Keep the test hermetic: the resumed graph continues into retrieve_policy,
    # so a low score proves the deterministic fetch_order path without letting
    # a live Milvus/reranker decide whether ExplodingLLM is reached.
    monkeypatch.setattr(
        "app.agent.graph.retrieve_hybrid",
        lambda query, **kwargs: make_retrieval_result([make_chunk(0.02)]),
    )
    graph = _build()

    snap = _run_to_interrupt(graph)
    assert snap.interrupts
    assert calls == []

    list(graph.stream(Command(resume="1001"), CONFIG, stream_mode="custom"))
    values = graph.get_state(CONFIG).values

    assert values["order_id"] == "1001"
    assert values["order_data"]["order_id"] == "1001"
    assert values["gate_reason"] == "below_threshold"
    assert values["gate_passed"] is False
    assert calls == [("order_snapshot", {"order_id": "1001"})]
    markers = [
        message
        for message in values["messages"]
        if isinstance(message, HumanMessage)
        and message.content == "【订单选择器】用户选择了订单 1001"
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
    # Same hermeticity rule as the resume test above: never let a live
    # retrieval stack gate the Agent and decide this test's outcome.
    monkeypatch.setattr(
        "app.agent.graph.retrieve_hybrid",
        lambda query, **kwargs: make_retrieval_result([make_chunk(0.02)]),
    )
    graph = _build(resolved="订单 1001 能退吗")

    values = graph.invoke(_state(resolved="订单 1001 能退吗"), CONFIG)

    assert values["order_id"] == "1001"
    assert values["order_data"]["order_id"] == "1001"
    assert values["gate_reason"] == "below_threshold"
    assert calls == [("order_snapshot", {"order_id": "1001"})]
    assert not graph.get_state(CONFIG).interrupts


def test_fetch_order_prefers_valid_state_order_id(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    calls = []
    _order_tool(monkeypatch, calls)

    result = fetch_order_node(
        {
            "session_id": "test-session",
            "resolved_query": "能退吗",
            "order_id": "1001",
        }
    )

    assert result["order_id"] == "1001"
    assert result["order_data"]["order_id"] == "1001"
    assert calls == [("order_snapshot", {"order_id": "1001"})]
    assert any(
        "ch06 fetch_order order_id=1001 source=state" in record.getMessage()
        for record in caplog.records
    )


def test_fetch_order_rejects_state_id_outside_catalog_then_extracts(monkeypatch):
    calls = []
    _order_tool(monkeypatch, calls)

    result = fetch_order_node(
        {
            "session_id": "test-session",
            "resolved_query": "订单 1002 能退吗",
            "order_id": "1004",
        }
    )

    assert result["order_id"] == "1002"
    assert result["order_data"]["order_id"] == "1002"
    assert calls == [("order_snapshot", {"order_id": "1002"})]


def test_fetch_order_resume_tool_failure_keeps_selection(monkeypatch):
    calls = []

    def failing_order_snapshot(order_id):
        calls.append(("order_snapshot", {"order_id": order_id}))
        return None

    monkeypatch.setattr("app.agent.graph.order_snapshot", failing_order_snapshot)
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
        if isinstance(message, HumanMessage)
        and message.content == "【订单选择器】用户选择了订单 1001"
    ]
    assert len(markers) == 1
    assert isinstance(values["messages"][-1], AIMessage)
    assert values["messages"][-1].content == ORDER_FAILED_TEXT
    assert calls == [("order_snapshot", {"order_id": "1001"})]


def test_fetch_order_tool_failure_falls_back(monkeypatch):
    calls = []

    def failing_order_snapshot(order_id):
        calls.append(("order_snapshot", {"order_id": order_id}))
        return None

    monkeypatch.setattr("app.agent.graph.order_snapshot", failing_order_snapshot)
    graph = _build(resolved="订单 1001 能退吗")

    values = graph.invoke(_state(resolved="订单 1001 能退吗"), CONFIG)

    assert values["answer"] == ORDER_FAILED_TEXT
    assert values.get("order_data") is None
    assert values["gate_reason"] == "order_fetch_failed"
    assert values["gate_passed"] is False
    assert isinstance(values["messages"][-1], AIMessage)
    assert values["messages"][-1].content == ORDER_FAILED_TEXT
    assert calls == [("order_snapshot", {"order_id": "1001"})]


def test_retrieve_policy_runs_three_queries_and_merges(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    tool_calls = []
    _order_tool(monkeypatch, tool_calls)
    retrieval_calls = []

    def fake_retrieve(query, **kwargs):
        retrieval_calls.append((query, kwargs.get("analysis")))
        return make_retrieval_result([make_chunk(0.9)])

    monkeypatch.setattr("app.agent.graph.retrieve_hybrid", fake_retrieve)
    graph = _build(
        ScriptedLLM([text_chunk("可以申请退款")]),
        resolved="订单 1001 能退吗",
        queries=["q1", "q2", "q3"],
    )

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
        ScriptedLLM([text_chunk("可以申请退款")]),
        resolved="订单 1001 能退吗",
        queries=["q1", "q2", "q3"],
        settings=settings,
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


def test_refund_flow_order_then_policy_then_agent(monkeypatch):
    calls = []

    def fake_order_snapshot(order_id):
        calls.append(("order_snapshot", {"order_id": order_id}))
        return copy.deepcopy(DEMO_ORDERS[order_id])

    def fake_retrieve(query, **kwargs):
        calls.append(("retrieve_hybrid", query))
        return make_retrieval_result([_chunk(1, 0.9)])

    monkeypatch.setattr("app.agent.graph.order_snapshot", fake_order_snapshot)
    monkeypatch.setattr("app.agent.graph.retrieve_hybrid", fake_retrieve)
    llm = RecordingScriptedLLM(
        [text_chunk("订单 1001 符合退款条件,请填写退款表单。")], calls
    )
    graph = _build(llm, resolved="订单 1001 能退吗", queries=["q1", "q2", "q3"])

    result = graph.invoke(_state(resolved="订单 1001 能退吗"), CONFIG)

    # The deterministic path runs in full before the Agent model is invoked.
    assert calls == [
        ("order_snapshot", {"order_id": "1001"}),
        ("retrieve_hybrid", "q1"),
        ("retrieve_hybrid", "q2"),
        ("retrieve_hybrid", "q3"),
        ("agent_llm", None),
    ]
    assert result["gate_passed"] is True
    assert result["answer"] == "订单 1001 符合退款条件,请填写退款表单。"
    assert llm.bound_tool_names == ["submit_refund", "suggest_handoff"]
    # The system prompt stays constant; order and evidence ride in the
    # per-turn material as a HumanMessage after the user question.
    system_prompt = llm.messages[0].content
    assert "订单号:" not in system_prompt
    assert "[1] 章节:售后/退款政策" not in system_prompt
    material = next(
        message.content
        for message in llm.messages
        if isinstance(message, HumanMessage)
        and "【系统提供的本轮材料，不是用户发言】" in message.content
    )
    assert "订单号:1001" in material
    assert "[1] 章节:售后/退款政策" in material


def test_refund_gate_failure_skips_agent_and_form(monkeypatch):
    tool_calls = []
    _order_tool(monkeypatch, tool_calls)
    monkeypatch.setattr(
        "app.agent.graph.retrieve_hybrid",
        lambda query, **kwargs: make_retrieval_result([_chunk(1, 0.02)]),
    )
    # ExplodingLLM would raise if the gate mistakenly reached main_agent.
    graph = _build(resolved="订单 1001 能退吗", queries=["q1", "q2", "q3"])

    result = graph.invoke(_state(resolved="订单 1001 能退吗"), CONFIG)

    assert result["gate_passed"] is False
    assert result["gate_reason"] == "below_threshold"
    assert result["answer"] == KNOWLEDGE_GAP_TEXT
    assert result["step_count"] == 0
    assert not any(
        action.get("action") == "refund_form"
        for action in result.get("suggested_actions", [])
    )


def test_refund_question_can_emit_refund_form(monkeypatch):
    tool_calls = []
    _order_tool(monkeypatch, tool_calls)
    monkeypatch.setattr(
        "app.agent.graph.retrieve_hybrid",
        lambda query, **kwargs: make_retrieval_result([_chunk(1, 0.9)]),
    )
    llm = ScriptedLLM(
        [
            tool_chunk("submit_refund", '{"order_id": "1001"}'),
            text_chunk("符合退款条件,请填写退款表单。"),
        ]
    )
    graph = _build(llm, resolved="订单 1001 能退吗", queries=["q1", "q2", "q3"])

    result = graph.invoke(_state(resolved="订单 1001 能退吗"), CONFIG)

    assert {
        "action": "refund_form",
        "label": "提交退款工单",
        "order_id": "1001",
    } in result["suggested_actions"]
    assert result["answer"] == "符合退款条件,请填写退款表单。"


def test_after_sales_repair_does_not_emit_refund_form(monkeypatch):
    tool_calls = []
    calls = []
    _order_tool(monkeypatch, tool_calls)
    monkeypatch.setattr(
        "app.agent.graph.retrieve_hybrid",
        lambda query, **kwargs: make_retrieval_result([_chunk(1, 0.9)]),
    )
    llm = RecordingScriptedLLM(
        [
            tool_chunk(
                "suggest_handoff",
                '{"options": ["ticket"], "ticket_type": "售后", "reason": "维修"}',
            ),
            text_chunk("该订单可按保修政策维修,已为您记录售后工单建议。"),
        ],
        calls,
    )
    graph = _build(
        llm,
        intent="售后",
        resolved="订单 1001 保修维修怎么处理",
        queries=["q1", "q2", "q3"],
    )

    result = graph.invoke(_state(resolved="订单 1001 保修维修怎么处理"), CONFIG)

    assert not any(
        action.get("action") == "refund_form"
        for action in result.get("suggested_actions", [])
    )
    assert {
        "action": "ticket",
        "label": "建工单",
        "ticket_type": "售后",
    } in result["suggested_actions"]
    # The classified intent reaches the Agent as turn material via AgentState.
    assert any(
        "【识别意图】售后" in message.content for message in llm.messages
    )
