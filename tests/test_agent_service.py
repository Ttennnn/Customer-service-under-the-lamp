"""Chat service tests: ch05 streaming baseline plus ch06 interrupt/resume."""

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.agent.service import (
    NoPendingInterrupt,
    has_pending_interrupt,
    resume_chat_agent,
    stream_chat_agent,
)
from app.config import Settings
from app.models import LowConfidenceQuestion, Message
from app.services.history import get_or_create_conversation
from tests.agent_fakes import FakeGraph

TOOL_FRAME = {"type": "tool", "tool": "query_order", "label": "查订单"}
SOURCES_FRAME = {
    "type": "sources",
    "sources": [
        {
            "n": 1,
            "chunk_id": 7,
            "section_path": "售后/退款",
            "question": "退款多久到账?",
            "answer": "1-3 个工作日",
            "category": "售后",
            "content_type": "faq",
            "score": 0.91,
        }
    ],
}
TRANSFER_ACTION = {"action": "transfer", "label": "转人工"}
ORDER_INTERRUPT = {
    "kind": "select_order",
    "slot": "order_id",
    "orders": [{"order_id": "1001", "title": "无线蓝牙耳机 ×1"}],
}


def _settings():
    return Settings(_env_file=None, openai_api_key="test")


def _snapshot(values=None, interrupts=()):
    return SimpleNamespace(values=values or {}, interrupts=interrupts)


def _pending_snapshot(payload=None):
    payload = ORDER_INTERRUPT if payload is None else payload
    return _snapshot(interrupts=(SimpleNamespace(value=payload),))


def _run(session, frames=(), final=None, message="你好", user_id="u1", **kwargs):
    graph = FakeGraph(list(frames), final or {})
    events = list(
        stream_chat_agent(
            graph, session, user_id, message, settings=_settings(), **kwargs
        )
    )
    # The leading ``conversation`` frame is covered by test_ch07_service; the
    # payload assertions below only care about the frames after it.
    return graph, [e for e in events if '"type":"conversation"' not in e]


def _dumps(payload):
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def test_service_forwards_tokens_tools_sources_and_options(session):
    graph, events = _run(
        session,
        frames=[TOOL_FRAME, SOURCES_FRAME, {"type": "token", "text": "答"}],
        final={"answer": "答", "suggested_actions": [TRANSFER_ACTION]},
    )

    assert events == [
        _dumps(TOOL_FRAME),
        _dumps(SOURCES_FRAME),
        "答",
        _dumps({"type": "options", "options": [TRANSFER_ACTION]}),
    ]


def test_service_resets_transient_state(session):
    graph, _ = _run(
        session,
        frames=[],
        final={"answer": "答"},
        message="退款政策是什么?",
    )

    inputs = graph.inputs[-1]
    assert inputs["step_count"] == 0
    assert inputs["token_estimate"] == 0
    assert inputs["evidence"] == []
    assert inputs["sources"] == []
    assert inputs["gate_passed"] is False
    assert inputs["gate_reason"] == ""
    assert inputs["rerank_failed"] is False
    assert inputs["resolved_query"] == ""
    assert inputs["intent_confidence"] is None
    assert inputs["session_id"] == "u1"
    assert inputs["order_id"] == ""
    assert inputs["order_data"] is None
    assert inputs["expanded_queries"] == []
    assert inputs["answer"] == ""
    assert inputs["suggested_actions"] == []
    assert inputs["exhausted"] is False
    assert inputs["conversation_id"] is not None
    assert inputs["token_budget"] == _settings().agent_token_budget
    assert inputs["max_steps"] == _settings().agent_max_steps
    assert inputs["user_message"] == "退款政策是什么?"
    assert inputs["resolved_query"] == ""
    assert inputs["summary"] == ""
    assert inputs["prompt_history"] == []
    assert inputs["turn_start_index"] == 0
    assert inputs["context_budget"]["layer1"] > 0

    cid = inputs["conversation_id"]
    assert graph.config == {"configurable": {"thread_id": f"conv:{cid}"}}
    assert len(inputs["messages"]) == 1
    assert isinstance(inputs["messages"][0], HumanMessage)
    assert inputs["messages"][0].content == "退款政策是什么?"
    assert inputs["messages"][0].id.startswith("db-")


def test_service_persists_assistant_answer(session):
    _run(session, frames=[{"type": "token", "text": "已"}], final={"answer": "已答复"})

    rows = session.query(Message).order_by(Message.id.asc()).all()
    assert [(row.role, row.content) for row in rows] == [
        ("user", "你好"),
        ("assistant", "已答复"),
    ]


def test_service_records_low_confidence_for_gate_gap(session):
    cid = get_or_create_conversation(session, "u1").id
    _run(
        session,
        frames=[],
        final={
            "route": "knowledge",
            "gate_passed": False,
            "gate_reason": "below_threshold",
            "answer": "抱歉,知识库暂未收录足够的信息。",
        },
    )

    rows = session.query(LowConfidenceQuestion).all()
    assert len(rows) == 1
    row = rows[0]
    assert row.source == "confidence_gate"
    assert row.conversation_id == cid
    assert row.raw_question == "你好"
    assert row.reason == "below_threshold"


@pytest.mark.parametrize("gate_reason", ["infra_error", "rerank_failed"])
def test_service_does_not_pool_infra_or_rerank_failure(session, gate_reason):
    _run(
        session,
        frames=[],
        final={
            "route": "knowledge",
            "gate_passed": False,
            "gate_reason": gate_reason,
            "answer": "知识库暂时不可用。",
        },
    )

    assert session.query(LowConfidenceQuestion).count() == 0


def test_service_ignores_unknown_frames(session, caplog):
    graph, events = _run(
        session,
        frames=[TOOL_FRAME, {"type": "debug", "detail": "internal"}, {"type": "token", "text": "答"}],
        final={"answer": "答"},
    )

    assert events == [_dumps(TOOL_FRAME), "答"]
    assert any("unknown" in record.message.lower() for record in caplog.records)


def test_stream_chat_reemits_pending_interrupt_without_new_turn(session):
    graph = FakeGraph(
        frames=[], final={}, snapshots=[_pending_snapshot()]
    )
    events = list(
        stream_chat_agent(graph, session, "s1", "我要退款", settings=_settings())
    )

    assert '"type":"conversation"' in events[0]
    assert events[1:] == [_dumps({"type": "interrupt", **ORDER_INTERRUPT})]
    assert graph.stream_calls == []
    assert session.query(Message).count() == 0


def test_stream_chat_emits_interrupt_and_skips_answer(session):
    graph = FakeGraph(
        frames=[{"type": "token", "text": "部分"}],
        final={"answer": "不应写入"},
        snapshots=[_snapshot(), _pending_snapshot()],
    )
    events = list(
        stream_chat_agent(graph, session, "s1", "我要退款", settings=_settings())
    )

    assert '"type":"conversation"' in events[0]
    assert events[1:] == [
        "部分",
        _dumps({"type": "interrupt", **ORDER_INTERRUPT}),
    ]
    assert len(graph.stream_calls) == 1
    rows = session.query(Message).order_by(Message.id.asc()).all()
    assert [(row.role, row.content) for row in rows] == [("user", "我要退款")]


def test_resume_chat_uses_command_and_writes_answer(session):
    final = {
        "answer": "可以退,已为您准备退款表单。",
        "suggested_actions": [TRANSFER_ACTION],
        "user_message": "我要退款",
    }
    graph = FakeGraph(
        frames=[{"type": "token", "text": "可以退。"}],
        final=final,
        snapshots=[_pending_snapshot(), _snapshot(values=final)],
    )
    events = list(
        resume_chat_agent(graph, session, "s1", "1001", settings=_settings())
    )

    assert isinstance(graph.inputs[-1], Command)
    assert graph.inputs[-1].resume == "1001"
    assert '"type":"conversation"' in events[0]
    assert events[1:] == [
        "可以退。",
        _dumps({"type": "options", "options": [TRANSFER_ACTION]}),
    ]
    rows = session.query(Message).order_by(Message.id.asc()).all()
    assert [(row.role, row.content) for row in rows] == [
        ("user", "[订单选择] 订单 1001"),
        ("assistant", "可以退,已为您准备退款表单。"),
    ]


def test_resume_chat_agent_passes_dict_to_command(session):
    graph = FakeGraph(
        [],
        {},
        snapshots=[_pending_snapshot({"kind": "confirm_ticket"})],
    )

    list(
        resume_chat_agent(
            graph, session, "u1", {"confirmed": True}, settings=_settings()
        )
    )

    assert isinstance(graph.inputs[-1], Command)
    assert graph.inputs[-1].resume == {"confirmed": True}
    rows = (
        session.query(Message)
        .filter(Message.role == "user")
        .order_by(Message.id.asc())
        .all()
    )
    assert [row.content for row in rows] == ["[工单确认] 确认提交"]


def test_resume_chat_agent_audits_ticket_cancel(session):
    graph = FakeGraph(
        [],
        {},
        snapshots=[_pending_snapshot({"kind": "confirm_ticket"})],
    )

    list(
        resume_chat_agent(
            graph, session, "u1", {"confirmed": False}, settings=_settings()
        )
    )

    assert graph.inputs[-1].resume == {"confirmed": False}
    rows = (
        session.query(Message)
        .filter(Message.role == "user")
        .order_by(Message.id.asc())
        .all()
    )
    assert [row.content for row in rows] == ["[工单确认] 取消"]


def test_resume_chat_raises_without_pending(session):
    graph = FakeGraph(frames=[], final={}, snapshots=[_snapshot()])

    with pytest.raises(NoPendingInterrupt):
        list(resume_chat_agent(graph, session, "s1", "1001", settings=_settings()))

    assert graph.stream_calls == []


def test_has_pending_interrupt_reads_thread_snapshot(session):
    graph = FakeGraph(frames=[], final={}, snapshots=[_pending_snapshot()])
    assert has_pending_interrupt(graph, 7) is True
    assert graph.config == {"configurable": {"thread_id": "conv:7"}}

    assert has_pending_interrupt(FakeGraph(frames=[], final={}), 7) is False


def test_service_graph_error_yields_error_frame(session):
    class RaisingGraph:
        def __init__(self):
            self.state_calls = 0

        def stream(self, *args, **kwargs):
            raise RuntimeError("graph exploded")

        def get_state(self, config):
            # Only the pre-turn pending check may run; the stream error must
            # short-circuit before the post-stream snapshot.
            self.state_calls += 1
            return _snapshot()

    graph = RaisingGraph()
    events = list(
        stream_chat_agent(graph, session, "u1", "你好", settings=_settings())
    )

    assert graph.state_calls == 1

    assert '"type":"conversation"' in events[0]
    assert len(events) == 2
    assert json.loads(events[-1]) == {"error": "客服服务暂时不可用,请稍后重试"}
    assert session.query(Message).filter(Message.role == "assistant").count() == 0
