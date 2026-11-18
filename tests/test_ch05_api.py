"""Ch05 compatibility tests: redirects plus the retained ticket endpoint.

The ch05 workflow API moved to ``/ch06`` in Task 16. ``/ch05`` and
``/ch05/chat`` are 307 compatibility redirects; ``/ch05/tickets`` is kept
because the chat page still creates complaint/after-sale tickets through it.
"""

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app
from app.models import Conversation, Ticket
from app.services.history import create_conversation, get_or_create_conversation
from app.tools.ticket import TICKET_NO_PATTERN


@pytest.fixture
def client(session):
    app.dependency_overrides[get_settings] = lambda: Settings(openai_api_key="test")
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_ch05_page_serves_latest_ch06_page(client):
    r = client.get("/ch05", follow_redirects=False)

    assert r.status_code == 200
    assert "lamp-cs-ch06-session-id" in r.text


def test_ch05_page_redirect_lands_on_ch06(client):
    r = client.get("/ch05")

    assert r.status_code == 200
    assert "lamp-cs-ch06-session-id" in r.text


def test_ch05_chat_redirects_to_ch06(client):
    r = client.post(
        "/ch05/chat",
        json={"session_id": "s", "message": "x"},
        follow_redirects=False,
    )

    assert r.status_code == 307
    assert r.headers["location"] == "/ch06/chat"


def test_ch05_chat_redirect_does_not_validate_body(client):
    # The compatibility layer must redirect legacy clients even if their body
    # does not match the new request schema (e.g. an old client sending extra
    # fields); it never reads the payload itself.
    r = client.post(
        "/ch05/chat",
        json={"unexpected": True},
        follow_redirects=False,
    )

    assert r.status_code == 307


def test_ch05_ticket_writes_ticket_without_transfer(client, session):
    conversation = get_or_create_conversation(session, "s1")

    r = client.post(
        "/ch05/tickets",
        json={
            "session_id": "s1",
            "description": " 我要投诉 ",
            "ticket_type": "投诉",
        },
    )

    assert r.status_code == 200
    payload = r.json()
    assert payload["message"] == "工单已创建"
    assert TICKET_NO_PATTERN.match(payload["ticket_no"])

    session.expire_all()
    ticket = session.query(Ticket).one()
    assert ticket.ticket_no == payload["ticket_no"]
    assert ticket.conversation_id == conversation.id
    assert ticket.description == "我要投诉"
    assert ticket.ticket_type == "投诉"

    stored = session.query(Conversation).filter_by(id=conversation.id).one()
    assert stored.status == "进行中"


def test_ch05_ticket_uses_request_conversation_not_latest(client, session):
    requested = create_conversation(session, "s1")
    create_conversation(session, "s1")  # newer in-progress conversation

    r = client.post(
        "/ch05/tickets",
        json={
            "session_id": "s1",
            "description": "售后",
            "ticket_type": "售后",
            "conversation_id": requested.id,
        },
    )

    assert r.status_code == 200
    session.expire_all()
    assert session.query(Ticket).one().conversation_id == requested.id


def test_ch05_ticket_rejects_foreign_conversation(client, session):
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


def test_ch05_ticket_defaults_to_consultation(client, session):
    r = client.post(
        "/ch05/tickets", json={"session_id": "s2", "description": "咨询一下"}
    )

    assert r.status_code == 200
    session.expire_all()
    assert session.query(Ticket).one().ticket_type == "咨询"


def test_ch05_ticket_accepts_refund_type(client, session):
    r = client.post(
        "/ch05/tickets",
        json={"session_id": "s1", "description": "退款", "ticket_type": "退款"},
    )

    assert r.status_code == 200
    session.expire_all()
    assert session.query(Ticket).one().ticket_type == "退款"


def test_ch05_ticket_invalid_type_422(client):
    r = client.post(
        "/ch05/tickets",
        json={"session_id": "s1", "description": "x", "ticket_type": "违规"},
    )
    assert r.status_code == 422


def test_ch05_ticket_failure_502(client, session, monkeypatch):
    calls = []

    def fake_execute_tool(*args, **kwargs):
        calls.append(kwargs)
        return "工具执行失败"

    monkeypatch.setattr("app.api.ch05.execute_tool", fake_execute_tool)

    r = client.post(
        "/ch05/tickets", json={"session_id": "s1", "description": "描述"}
    )

    assert r.status_code == 502
    assert session.query(Ticket).count() == 0
    # 工单写入非幂等:必须禁用自动重试,超时宁可 502 让用户手动确认重试。
    assert calls == [{"timeout_s": 5, "retries": 0}]
