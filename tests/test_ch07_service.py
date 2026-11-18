"""Ch07 conversation-aware runtime tests.

The runtime must pin every turn to ``conv:{conversation_id}``, emit the
``conversation`` frame the frontend uses to track the active session, and
write audit rows into the resolved conversation.
"""

import json
from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from app.agent.service import stream_chat_agent
from app.config import Settings
from app.models import Message
from tests.agent_fakes import FakeGraph


def _settings():
    return Settings(_env_file=None, openai_api_key="x")


def _run(session, frames=(), final=None, message="你好", user_id="u1", **kw):
    graph = FakeGraph(
        list(frames),
        final or {"answer": "答"},
        snapshots=[SimpleNamespace(values={}, interrupts=())],
    )
    events = list(
        stream_chat_agent(graph, session, user_id, message, settings=_settings(), **kw)
    )
    return graph, events


def test_stream_agent_thread_is_conversation_scoped(session):
    graph, events = _run(session, new_conversation=True)
    assert graph.config["configurable"]["thread_id"].startswith("conv:")
    assert any('"type":"conversation"' in e for e in events)


def test_stream_agent_uses_existing_conversation(session):
    from app.services.history import create_conversation

    cid = create_conversation(session, "u1").id
    graph, _ = _run(session, conversation_id=cid)
    assert graph.config["configurable"]["thread_id"] == f"conv:{cid}"
    rows = session.query(Message).filter_by(conversation_id=cid).all()
    assert any(row.content == "你好" for row in rows)


def test_stream_agent_rejects_foreign_conversation(session):
    from app.services.history import create_conversation

    cid = create_conversation(session, "u2").id
    graph, events = _run(session, conversation_id=cid)
    assert any('"error"' in e for e in events)
    assert graph.inputs == []


def test_stream_agent_injects_plan_fields_and_db_message_id(session):
    graph, _ = _run(session, message="一")
    inputs = graph.inputs[-1]
    assert inputs["messages"][0].id.startswith("db-")
    assert isinstance(inputs["messages"][0], HumanMessage)
    assert inputs["summary"] == ""
    assert inputs["prompt_history"] == []
    assert inputs["history_l2"] == []
    assert inputs["history_l1"] == []
    assert inputs["turn_start_index"] == 0
    assert inputs["l2_tokens"] == 0
    assert inputs["l1_tokens"] == 0
    assert inputs["context_budget"]["layer1"] > 0


def test_resume_uses_conversation_thread_and_audit_row(session):
    from langgraph.types import Command

    from app.agent.service import resume_chat_agent
    from app.services.history import create_conversation

    cid = create_conversation(session, "u1").id
    graph = FakeGraph(
        [],
        {"answer": "答"},
        snapshots=[
            SimpleNamespace(
                values={},
                interrupts=(SimpleNamespace(value={"kind": "select_order"}),),
            )
        ],
    )
    events = list(
        resume_chat_agent(
            graph,
            session,
            "u1",
            "1001",
            settings=_settings(),
            conversation_id=cid,
        )
    )
    assert graph.config["configurable"]["thread_id"] == f"conv:{cid}"
    assert any(isinstance(call, Command) for call in graph.inputs)
    assert events[0] == json.dumps(
        {"type": "conversation", "conversation_id": cid},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    rows = session.query(Message).filter_by(conversation_id=cid, role="user").all()
    assert rows[-1].content == "[订单选择] 订单 1001"
