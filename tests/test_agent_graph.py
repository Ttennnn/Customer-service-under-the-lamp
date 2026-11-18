"""Parent workflow graph tests (Task 9, design §6.1-§6.6)."""

import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.graph import (
    KNOWLEDGE_GAP_TEXT,
    RERANK_FAILED_TEXT,
    build_graph,
    sqlite_checkpointer,
)
from tests.agent_fakes import (
    ScriptedLLM,
    make_chunk,
    make_retrieval_result,
    text_chunk,
    tool_chunk,
)


class FakeIntentLLM:
    def __init__(self, intent):
        self.intent = intent

    def with_structured_output(self, schema, method=None):
        return self

    def invoke(self, messages):
        from app.agent.intent import IntentResult

        return IntentResult(intent=self.intent, reason="test")


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
        "resolved_message": message,
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


def _build(llm, intent, **kwargs):
    return build_graph(llm, FakeIntentLLM(intent), checkpointer=InMemorySaver(), **kwargs)


def test_resolve_reference_passthrough(caplog):
    caplog.set_level(logging.INFO)
    llm = ScriptedLLM([text_chunk("您好")])
    graph = _build(llm, "物流")

    result = graph.invoke(_state("你好"), CONFIG)

    assert result["resolved_message"] == result["user_message"] == "你好"
    assert any(
        "ch05 resolve_reference passthrough" in record.message
        for record in caplog.records
    )


def test_knowledge_route_calls_retrieval_and_gate(monkeypatch):
    calls = []

    def fake_retrieve(query):
        calls.append(query)
        return make_retrieval_result([make_chunk(0.9)])

    monkeypatch.setattr("app.agent.graph.retrieve_hybrid", fake_retrieve)
    llm = ScriptedLLM([text_chunk("政策回答")])
    graph = _build(llm, "退款退货")

    result = graph.invoke(_state("退货政策"), CONFIG)

    assert calls == ["退货政策"]
    assert result["route"] == "knowledge"
    assert result["gate_passed"] is True
    assert result["answer"] == "政策回答"


def test_gate_failure_skips_agent(monkeypatch):
    monkeypatch.setattr(
        "app.agent.graph.retrieve_hybrid",
        lambda query: make_retrieval_result([]),
    )
    graph = _build(ScriptedLLM([]), "退款退货")

    result = graph.invoke(_state("退货政策"), CONFIG)

    assert result["gate_passed"] is False
    assert result["answer"] == KNOWLEDGE_GAP_TEXT
    assert result["step_count"] == 0


def test_rerank_failure_skips_agent_and_uses_infra_text(monkeypatch):
    monkeypatch.setattr(
        "app.agent.graph.retrieve_hybrid",
        lambda query: make_retrieval_result([make_chunk(0.02)], rerank_failed=True),
    )
    graph = _build(ScriptedLLM([]), "退款退货")

    result = graph.invoke(_state("退货政策"), CONFIG)

    assert result["gate_reason"] == "rerank_failed"
    assert result["gate_passed"] is False
    assert "可靠性校验" in result["answer"]
    assert result["step_count"] == 0


def test_infra_error_skips_agent_and_does_not_become_empty_gap(monkeypatch):
    def boom(query):
        raise RuntimeError("milvus down")

    monkeypatch.setattr("app.agent.graph.retrieve_hybrid", boom)
    graph = _build(ScriptedLLM([]), "退款退货")

    result = graph.invoke(_state("退货政策"), CONFIG)

    assert result["gate_reason"] == "infra_error"
    assert "知识库暂时不可用" in result["answer"]
    assert result["answer"] != KNOWLEDGE_GAP_TEXT
    assert result["step_count"] == 0


def test_business_route_skips_retrieval(monkeypatch):
    def unexpected_retrieval(query):
        raise AssertionError("business route must not retrieve")

    monkeypatch.setattr("app.agent.graph.retrieve_hybrid", unexpected_retrieval)
    llm = ScriptedLLM([text_chunk("您的包裹已发出")])
    graph = _build(llm, "物流")

    result = graph.invoke(_state("我的快递到哪了"), CONFIG)

    assert result["route"] == "business"
    assert result["answer"] == "您的包裹已发出"


@pytest.mark.parametrize("intent", ["投诉", "闲聊"])
def test_complaint_and_chitchat_do_not_call_llm(intent):
    graph = _build(ExplodingLLM(), intent)

    result = graph.invoke(_state("测试"), CONFIG)

    assert result["answer"]
    if intent == "投诉":
        assert result["suggested_actions"] == [
            {"action": "transfer", "label": "转人工"},
            {"action": "ticket", "label": "建工单", "ticket_type": "投诉"},
        ]
    else:
        assert result["suggested_actions"] == []


@pytest.mark.parametrize("case", ["complaint", "chitchat", "gate_fail"])
def test_fixed_paths_append_ai_message(monkeypatch, case):
    intent = {"complaint": "投诉", "chitchat": "闲聊", "gate_fail": "退款退货"}[case]
    if case == "gate_fail":
        monkeypatch.setattr(
            "app.agent.graph.retrieve_hybrid",
            lambda query: make_retrieval_result([]),
        )
    graph = _build(ExplodingLLM(), intent)

    result = graph.invoke(_state("测试"), CONFIG)

    last = result["messages"][-1]
    assert isinstance(last, AIMessage)
    assert last.content == result["answer"]


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
    monkeypatch.setattr(
        "app.agent.subgraph.execute_tool",
        lambda tool, args, timeout_s=5: "订单1001已发货",
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
