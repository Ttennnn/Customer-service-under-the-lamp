"""Ch07 API aliases and conversation-aware refund/ticket ownership."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.ch06 import get_ch06_graph_dep
from app.config import Settings, get_settings
from app.main import app
from tests.agent_fakes import FakeGraph


@pytest.fixture
def client(session):
    app.dependency_overrides[get_settings] = lambda: Settings(openai_api_key="test")
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_ch07_chat_alias_streams_conversation_frame(client):
    app.dependency_overrides[get_ch06_graph_dep] = lambda: FakeGraph(
        [], {"answer": "答"}, snapshots=[SimpleNamespace(values={}, interrupts=())]
    )
    with client.stream(
        "POST", "/ch07/chat", json={"session_id": "s1", "message": "你好"}
    ) as r:
        body = "".join(r.iter_text())
    assert "conversation" in body and "data: [DONE]" in body


def test_new_conversation_creates_a_row(client, session):
    graph = FakeGraph(
        [], {"answer": "答"}, snapshots=[SimpleNamespace(values={}, interrupts=())]
    )
    app.dependency_overrides[get_ch06_graph_dep] = lambda: graph
    with client.stream(
        "POST",
        "/ch07/chat",
        json={"session_id": "s1", "message": "一", "new_conversation": True},
    ) as r:
        body = "".join(r.iter_text())
    assert '"conversation_id"' in body


def test_refund_and_ticket_use_request_conversation(client, session):
    from app.models import Ticket
    from app.services.history import create_conversation

    c1 = create_conversation(session, "s1")
    c2 = create_conversation(session, "s1")
    r = client.post(
        "/ch06/refunds",
        json={
            "session_id": "s1",
            "order_id": "1001",
            "reason": "质量问题",
            "conversation_id": c1.id,
        },
    )
    assert r.status_code == 200
    assert session.query(Ticket).filter_by(conversation_id=c1.id).count() == 1
    r2 = client.post(
        "/ch05/tickets",
        json={
            "session_id": "s1",
            "description": "售后",
            "ticket_type": "售后",
            "conversation_id": c2.id,
        },
    )
    assert r2.status_code == 200
    assert session.query(Ticket).filter_by(conversation_id=c2.id).count() == 1


def test_refund_rejects_foreign_conversation(client, session):
    from app.models import Ticket
    from app.services.history import create_conversation

    foreign = create_conversation(session, "other")
    r = client.post(
        "/ch06/refunds",
        json={
            "session_id": "s1",
            "order_id": "1001",
            "reason": "质量问题",
            "conversation_id": foreign.id,
        },
    )

    assert r.status_code == 404
    assert session.query(Ticket).count() == 0


def test_ticket_rejects_foreign_conversation(client, session):
    from app.models import Ticket
    from app.services.history import create_conversation

    foreign = create_conversation(session, "other")
    r = client.post(
        "/ch05/tickets",
        json={
            "session_id": "s1",
            "description": "售后",
            "ticket_type": "售后",
            "conversation_id": foreign.id,
        },
    )

    assert r.status_code == 404
    assert session.query(Ticket).count() == 0


def test_ch07_resume_alias_answers_409_without_pending(client):
    app.dependency_overrides[get_ch06_graph_dep] = lambda: FakeGraph([], {})
    r = client.post(
        "/ch07/chat/resume", json={"session_id": "s1", "order_id": "1001"}
    )
    assert r.status_code == 409
