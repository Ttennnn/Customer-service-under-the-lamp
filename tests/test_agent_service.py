"""Ch05 chat service tests (Task 10, design §6.6-§6.7)."""

import json

import pytest
from langchain_core.messages import HumanMessage

from app.agent.service import stream_chat_agent
from app.config import Settings
from app.models import LowConfidenceQuestion, Message
from app.services.history import get_or_create_conversation
from tests.agent_fakes import FakeGraph

TOOL_FRAME = {"type": "tool", "tool": "query_logistics", "label": "查物流"}
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


def _settings():
    return Settings(_env_file=None, openai_api_key="test")


def _run(session, frames=(), final=None, message="你好", user_id="u1"):
    graph = FakeGraph(list(frames), final or {})
    events = list(
        stream_chat_agent(
            graph, session, user_id, message, settings=_settings()
        )
    )
    return graph, events


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

    inputs = graph.inputs
    assert inputs is not None
    assert inputs["step_count"] == 0
    assert inputs["token_estimate"] == 0
    assert inputs["evidence"] == []
    assert inputs["sources"] == []
    assert inputs["gate_passed"] is False
    assert inputs["gate_reason"] == ""
    assert inputs["rerank_failed"] is False
    assert inputs["answer"] == ""
    assert inputs["suggested_actions"] == []
    assert inputs["exhausted"] is False
    assert inputs["conversation_id"] is not None
    assert inputs["token_budget"] == _settings().agent_token_budget
    assert inputs["max_steps"] == _settings().agent_max_steps
    assert inputs["user_message"] == inputs["resolved_message"] == "退款政策是什么?"

    assert graph.config == {"configurable": {"thread_id": "user:u1"}}
    assert len(inputs["messages"]) == 1
    assert isinstance(inputs["messages"][0], HumanMessage)
    assert inputs["messages"][0].content == "退款政策是什么?"


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


def test_service_graph_error_yields_error_frame(session):
    class RaisingGraph:
        def stream(self, *args, **kwargs):
            raise RuntimeError("graph exploded")

        def get_state(self, config):
            raise AssertionError("get_state must not run after a graph error")

    events = list(
        stream_chat_agent(
            RaisingGraph(), session, "u1", "你好", settings=_settings()
        )
    )

    assert len(events) == 1
    assert json.loads(events[0]) == {"error": "客服服务暂时不可用,请稍后重试"}
    assert session.query(Message).filter(Message.role == "assistant").count() == 0
