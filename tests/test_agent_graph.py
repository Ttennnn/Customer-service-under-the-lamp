"""Parent workflow graph tests (Task 11, design §6.1-§6.6)."""

import json
import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.graph import (
    CHITCHAT_TEXT,
    COMPLAINT_TEXT,
    KNOWLEDGE_GAP_TEXT,
    OTHER_TEXT,
    RERANK_FAILED_TEXT,
    SELF_CHECK_FAILURE_TEXT,
    build_graph,
    sqlite_checkpointer,
)
from app.agent.intent import IntentResult
from app.config import Settings
from app.rag.self_check import EvidenceAssessment
from tests.agent_fakes import (
    ScriptedLLM,
    ScriptedStructuredLLM,
    make_chunk,
    make_retrieval_result,
    text_chunk,
    tool_chunk,
)


class ExplodingLLM:
    """Agent LLM that fails the test if the Agent path is taken."""

    def bind_tools(self, tools, **kwargs):
        raise AssertionError("Agent LLM must not be called")

    def stream(self, messages):
        raise AssertionError("Agent LLM must not be called")


CONFIG = {"configurable": {"thread_id": "test-thread"}}


def _state(message="退货政策", **overrides):
    state = {
        "messages": [HumanMessage(content=message)],
        "user_message": message,
        "resolved_query": message,
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


def _build(llm, intent, resolved=None, queries=None, **kwargs):
    intent_llm = ScriptedStructuredLLM(
        resolved or intent,
        IntentResult(intent=intent, confidence=0.9),
        queries=queries,
    )
    # The evidence score is the sole formal gate; pin an explicit threshold so
    # low-score tests do not depend on the old confidence_min_score floor.
    kwargs.setdefault(
        "settings",
        Settings(
            _env_file=None,
            openai_api_key="test",
            evidence_confidence_threshold=0.5,
        ),
    )
    return build_graph(llm, intent_llm, checkpointer=InMemorySaver(), **kwargs)


def test_resolve_reference_passthrough(caplog):
    caplog.set_level(logging.INFO)
    llm = ScriptedLLM([text_chunk("您好")])
    graph = _build(llm, "物流", resolved="你好")

    result = graph.invoke(_state("你好"), CONFIG)

    assert result["resolved_query"] == result["user_message"] == "你好"
    assert any(
        "ch06 resolve_reference" in record.message for record in caplog.records
    )


def test_business_route_writes_resolved_query_and_intent():
    llm = ScriptedLLM([text_chunk("您的包裹已发出")])
    graph = _build(llm, "物流", resolved="订单 1001 什么时候发货")

    result = graph.invoke(_state("这个订单什么时候发货"), CONFIG)

    assert result["resolved_query"] == "订单 1001 什么时候发货"
    assert result["intent"] == "物流"
    assert result["intent_confidence"] == 0.9
    assert result["route"] == "business"
    assert result["answer"] == "您的包裹已发出"


@pytest.mark.parametrize("intent,text", [("闲聊", CHITCHAT_TEXT), ("其他", OTHER_TEXT)])
def test_script_reply_is_fixed_and_does_not_call_llm(intent, text):
    graph = _build(ExplodingLLM(), intent)

    result = graph.invoke(_state("测试"), CONFIG)

    assert result["route"] == "fallback_script"
    assert result["answer"] == text
    assert result["suggested_actions"] == []


def test_complaint_route_uses_escalate_buttons():
    graph = _build(ExplodingLLM(), "投诉")

    result = graph.invoke(_state("我要投诉"), CONFIG)

    assert result["route"] == "escalate"
    assert result["answer"] == COMPLAINT_TEXT
    assert result["suggested_actions"] == [
        {"action": "transfer", "label": "转人工"},
        {"action": "ticket", "label": "建工单", "ticket_type": "投诉"},
    ]


def test_knowledge_route_passes_analysis_and_gates(monkeypatch):
    calls = []

    def fake_retrieve(query, **kwargs):
        calls.append((query, kwargs.get("analysis")))
        return make_retrieval_result([make_chunk(0.9)])

    monkeypatch.setattr("app.agent.graph.retrieve_hybrid", fake_retrieve)
    graph = _build(
        ScriptedLLM([text_chunk("支持降噪")]),
        "商品咨询",
        resolved="这款耳机支持降噪吗",
    )

    result = graph.invoke(_state("这款耳机支持降噪吗"), CONFIG)

    assert calls[0][0] == "这款耳机支持降噪吗"
    assert calls[0][1].rewritten == "这款耳机支持降噪吗"
    assert calls[0][1].original == "这款耳机支持降噪吗"
    assert calls[0][1].rewritten_by == "llm"
    assert result["route"] == "knowledge"
    assert result["gate_passed"] is True
    assert result["answer"] == "支持降噪"


def test_gate_failure_skips_agent(monkeypatch):
    monkeypatch.setattr(
        "app.agent.graph.retrieve_hybrid",
        lambda query, **kwargs: make_retrieval_result([]),
    )
    graph = _build(ScriptedLLM([]), "商品咨询")

    result = graph.invoke(_state("耳机降噪吗"), CONFIG)

    assert result["gate_passed"] is False
    assert result["fallback_source"] == "retrieval_low_conf"
    assert result["answer"] == KNOWLEDGE_GAP_TEXT
    assert result["step_count"] == 0


def test_confidence_failure_sets_retrieval_low_conf_with_snapshot(monkeypatch):
    monkeypatch.setattr(
        "app.agent.graph.retrieve_hybrid",
        lambda query, **kwargs: make_retrieval_result([make_chunk(0.02)]),
    )
    graph = _build(ScriptedLLM([]), "商品咨询")

    result = graph.invoke(_state("耳机降噪吗"), CONFIG)

    assert result["gate_passed"] is False
    assert result["fallback_source"] == "retrieval_low_conf"
    assert result["evidence_confidence"] is not None
    assert result["evidence_signals"]["top1"] == 0.02
    assert "evidence_confidence" in result["fallback_reason"]
    assert result["retrieved_snapshot"] == [
        {
            "question": "q",
            "answer": "a",
            "rerank_score": 0.02,
            "section_path": "s",
            "is_key_clause": False,
        }
    ]
    assert result["answer"] == KNOWLEDGE_GAP_TEXT
    assert result["step_count"] == 0


def test_selfcheck_failure_sets_self_check_and_skips_agent(monkeypatch):
    monkeypatch.setattr(
        "app.agent.graph.retrieve_hybrid",
        lambda query, **kwargs: make_retrieval_result([make_chunk(0.9)]),
    )
    intent_llm = ScriptedStructuredLLM(
        "",
        IntentResult(intent="商品咨询", confidence=0.9),
        evidence_assessment=EvidenceAssessment(useful=False, reason="只沾边"),
    )
    graph = build_graph(
        ExplodingLLM(), intent_llm, checkpointer=InMemorySaver()
    )

    result = graph.invoke(_state("耳机降噪吗"), CONFIG)

    assert result["gate_passed"] is False
    assert result["gate_reason"] == "self_check"
    assert result["fallback_source"] == "self_check"
    assert result["fallback_reason"] == "只沾边"
    assert result["answer"] == KNOWLEDGE_GAP_TEXT
    assert result["step_count"] == 0


def test_selfcheck_exception_uses_failure_text_without_pool_source(monkeypatch):
    monkeypatch.setattr(
        "app.agent.graph.retrieve_hybrid",
        lambda query, **kwargs: make_retrieval_result([make_chunk(0.9)]),
    )
    intent_llm = ScriptedStructuredLLM(
        "",
        IntentResult(intent="商品咨询", confidence=0.9),
        evidence_error=RuntimeError("self-check timeout"),
    )
    graph = build_graph(
        ExplodingLLM(), intent_llm, checkpointer=InMemorySaver()
    )

    result = graph.invoke(_state("耳机降噪吗"), CONFIG)

    assert result["gate_passed"] is False
    assert result["gate_reason"] == "self_check_failure"
    assert result["fallback_source"] == ""
    assert result["fallback_reason"] == ""
    assert result["answer"] == SELF_CHECK_FAILURE_TEXT
    assert result["step_count"] == 0


def test_threshold_only_failure_applies_even_when_top1_is_above_legacy_floor(
    monkeypatch,
):
    # Top1 0.2 clears the old 0.1 confidence_min_score, but the evidence score
    # (~0.37) is below the explicit 0.5 threshold, so the formal gate must fail.
    monkeypatch.setattr(
        "app.agent.graph.retrieve_hybrid",
        lambda query, **kwargs: make_retrieval_result([make_chunk(0.2)]),
    )
    settings = Settings(
        _env_file=None,
        openai_api_key="test",
        evidence_confidence_threshold=0.5,
    )
    graph = _build(ExplodingLLM(), "商品咨询", settings=settings)

    result = graph.invoke(_state("耳机降噪吗"), CONFIG)

    assert result["evidence_confidence"] < 0.5
    assert result["gate_passed"] is False
    assert result["gate_reason"] == "below_threshold"
    assert result["fallback_source"] == "retrieval_low_conf"
    assert result["answer"] == KNOWLEDGE_GAP_TEXT


def test_threshold_only_pass_ignores_legacy_floor(monkeypatch):
    # Top1 0.09 is below the old confidence_min_score floor, but the evidence
    # score (key clause + margin, ~0.235) clears the explicit 0.1 threshold.
    # This test fails if the deprecated legacy gate is still ANDed in.
    monkeypatch.setattr(
        "app.agent.graph.retrieve_hybrid",
        lambda query, **kwargs: make_retrieval_result(
            [make_chunk(0.09, is_key_clause=True)]
        ),
    )
    settings = Settings(
        _env_file=None,
        openai_api_key="test",
        evidence_confidence_threshold=0.1,
    )
    graph = _build(
        ScriptedLLM([text_chunk("支持降噪")]),
        "商品咨询",
        settings=settings,
    )

    result = graph.invoke(_state("耳机降噪吗"), CONFIG)

    assert result["evidence_confidence"] >= 0.1
    assert result["gate_passed"] is True
    assert result["gate_reason"] == "ok"
    assert result["fallback_source"] == ""
    assert result["answer"] == "支持降噪"


def test_empty_retrieval_fails_even_with_zero_threshold(monkeypatch):
    # score 0.0 >= threshold 0.0, so the explicit empty guard is what fails it.
    monkeypatch.setattr(
        "app.agent.graph.retrieve_hybrid",
        lambda query, **kwargs: make_retrieval_result([]),
    )
    settings = Settings(
        _env_file=None,
        openai_api_key="test",
        evidence_confidence_threshold=0.0,
    )
    graph = _build(ExplodingLLM(), "商品咨询", settings=settings)

    result = graph.invoke(_state("耳机降噪吗"), CONFIG)

    assert result["evidence_confidence"] == 0.0
    assert result["gate_passed"] is False
    assert result["gate_reason"] == "empty"
    assert result["fallback_source"] == "retrieval_low_conf"
    assert result["answer"] == KNOWLEDGE_GAP_TEXT


def test_retrieval_pass_stores_top3_snapshot(monkeypatch):
    chunks = [make_chunk(score) for score in (0.6, 0.9, 0.7, 0.8)]
    monkeypatch.setattr(
        "app.agent.graph.retrieve_hybrid",
        lambda query, **kwargs: make_retrieval_result(chunks),
    )
    graph = _build(ScriptedLLM([text_chunk("支持降噪")]), "商品咨询")

    result = graph.invoke(_state("耳机降噪吗"), CONFIG)

    assert result["gate_passed"] is True
    assert result["fallback_source"] == ""
    assert [row["rerank_score"] for row in result["retrieved_snapshot"]] == [
        0.9,
        0.8,
        0.7,
    ]
    assert set(result["retrieved_snapshot"][0]) == {
        "question",
        "answer",
        "rerank_score",
        "section_path",
        "is_key_clause",
    }


def test_rerank_failure_skips_agent_and_uses_infra_text(monkeypatch):
    monkeypatch.setattr(
        "app.agent.graph.retrieve_hybrid",
        lambda query, **kwargs: make_retrieval_result(
            [make_chunk(0.02)], rerank_failed=True
        ),
    )
    graph = _build(ScriptedLLM([]), "商品咨询")

    result = graph.invoke(_state("耳机降噪吗"), CONFIG)

    assert result["gate_reason"] == "rerank_failed"
    assert result["gate_passed"] is False
    assert result["answer"] == RERANK_FAILED_TEXT
    assert result["step_count"] == 0


def test_infra_error_skips_agent_and_does_not_become_empty_gap(monkeypatch):
    def boom(query, **kwargs):
        raise RuntimeError("milvus down")

    monkeypatch.setattr("app.agent.graph.retrieve_hybrid", boom)
    graph = _build(ScriptedLLM([]), "商品咨询")

    result = graph.invoke(_state("耳机降噪吗"), CONFIG)

    assert result["gate_reason"] == "infra_error"
    assert "知识库暂时不可用" in result["answer"]
    assert result["answer"] != KNOWLEDGE_GAP_TEXT
    assert result["step_count"] == 0


def test_business_route_skips_retrieval(monkeypatch):
    def unexpected_retrieval(query, **kwargs):
        raise AssertionError("business route must not retrieve")

    monkeypatch.setattr("app.agent.graph.retrieve_hybrid", unexpected_retrieval)
    llm = ScriptedLLM([text_chunk("您的包裹已发出")])
    graph = _build(llm, "物流")

    result = graph.invoke(_state("我的快递到哪了"), CONFIG)

    assert result["route"] == "business"
    assert result["answer"] == "您的包裹已发出"


@pytest.mark.parametrize("case", ["complaint", "script", "gate_fail"])
def test_fixed_paths_append_ai_message(monkeypatch, case):
    intent = {"complaint": "投诉", "script": "其他", "gate_fail": "商品咨询"}[case]
    if case == "gate_fail":
        monkeypatch.setattr(
            "app.agent.graph.retrieve_hybrid",
            lambda query, **kwargs: make_retrieval_result([]),
        )
    graph = _build(ExplodingLLM(), intent)

    result = graph.invoke(_state("测试"), CONFIG)

    last = result["messages"][-1]
    assert isinstance(last, AIMessage)
    assert last.content == result["answer"]


def test_record_log_has_ch06_fields(caplog):
    caplog.set_level(logging.INFO)
    graph = _build(
        ScriptedLLM([text_chunk("答")]),
        "物流",
        resolved="订单 1001 什么时候发货",
    )

    graph.invoke(_state("订单 1001 什么时候发货"), CONFIG)

    payloads = [
        json.loads(record.getMessage().split("turn_log ", 1)[1])
        for record in caplog.records
        if "turn_log " in record.getMessage()
    ]
    payload = payloads[-1]
    assert payload["resolved_query"] == "订单 1001 什么时候发货"
    assert payload["intent_confidence"] == 0.9
    assert "order_id" in payload
    assert payload["expanded_queries"] == []
    assert payload["refund_form"] is False


def test_build_graph_warns_cost_mode_falls_back_to_accuracy(caplog):
    caplog.set_level(logging.WARNING)
    settings = Settings(
        _env_file=None, openai_api_key="test", intent_mode="cost"
    )

    _build(ExplodingLLM(), "物流", settings=settings)

    assert any(
        "intent_mode='cost'" in record.getMessage()
        and "using accuracy mode" in record.getMessage()
        for record in caplog.records
    )


def test_checkpointer_keeps_history_across_turns():
    llm = ScriptedLLM([text_chunk("第一轮回答"), text_chunk("第二轮回答")])
    graph = _build(llm, "物流")

    graph.invoke(_state("第一句"), CONFIG)
    graph.invoke(_state("第二句"), CONFIG)

    messages = graph.get_state(CONFIG).values["messages"]
    first_answer = next(
        index
        for index, message in enumerate(messages)
        if isinstance(message, AIMessage) and message.content == "第一轮回答"
    )
    second_question = next(
        index
        for index, message in enumerate(messages)
        if isinstance(message, HumanMessage) and message.content == "第二句"
    )
    assert first_answer < second_question
    assert isinstance(messages[-1], AIMessage)
    assert messages[-1].content == "第二轮回答"


def test_parent_stream_receives_subgraph_custom_frames(monkeypatch):
    from app.tools.spec import ToolRun

    monkeypatch.setattr(
        "app.agent.subgraph.execute_tool_call",
        lambda tool_call, conversation_id, **kwargs: ToolRun(
            status="ok", content="订单1001已发货"
        ),
    )
    llm = ScriptedLLM(
        [
            tool_chunk("query_order", '{"order_id": "1001"}'),
            text_chunk("已发货"),
        ]
    )
    graph = _build(llm, "物流")

    raw = list(
        graph.stream(
            _state("查一下订单"),
            CONFIG,
            stream_mode="custom",
            subgraphs=True,
            version="v2",
        )
    )

    # langgraph 1.2.20 with subgraphs=True yields
    # ``{"type": "custom", "ns": (...), "data": <frame>}`` items.
    assert all(item["type"] == "custom" for item in raw)
    frames = [item["data"] for item in raw]
    namespaces = [item["ns"] for item in raw]
    assert {"type": "tool", "tool": "query_order", "label": "查订单"} in frames
    assert {"type": "token", "text": "已发货"} in frames
    # Both custom frames must come from inside the nested Agent subgraph.
    for index, frame in enumerate(frames):
        if frame in (
            {"type": "tool", "tool": "query_order", "label": "查订单"},
            {"type": "token", "text": "已发货"},
        ):
            assert namespaces[index]


def test_sqlite_checkpointer_creates_file(tmp_path):
    path = tmp_path / "checkpoints.sqlite"

    with sqlite_checkpointer(str(path)) as saver:
        assert path.exists()
        # setup() is idempotent; a second call must not raise either.
        saver.setup()


def test_classify_intent_reports_to_langfuse(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "app.agent.graph.update_intent_trace",
        lambda settings, intent, confidence: calls.append(
            (settings, intent, confidence)
        ),
    )
    graph = _build(ScriptedLLM([]), "投诉", resolved="我要投诉")

    result = graph.invoke(_state("我要投诉"), CONFIG)

    assert result["intent"] == "投诉"
    assert len(calls) == 1
    _, intent, confidence = calls[0]
    assert intent == "投诉"
    assert confidence == 0.9


def test_classify_intent_trace_failure_does_not_break_graph(monkeypatch):
    def exploding_trace(*args, **kwargs):
        raise RuntimeError("langfuse down")

    monkeypatch.setattr(
        "app.agent.graph.update_intent_trace", exploding_trace
    )
    graph = _build(ScriptedLLM([]), "投诉", resolved="我要投诉")

    result = graph.invoke(_state("我要投诉"), CONFIG)

    assert result["intent"] == "投诉"
    assert result["route"] == "escalate"


def test_compiled_graph_with_config_keeps_streaming_and_state_reads():
    from app.observability import _state_graph

    graph = _build(ScriptedLLM([]), "投诉", resolved="我要投诉")
    binding = graph.with_config({"callbacks": []})

    result = binding.invoke(_state("我要投诉"), CONFIG)

    assert result["intent"] == "投诉"
    assert _state_graph(binding).get_state(CONFIG).values["intent"] == "投诉"
