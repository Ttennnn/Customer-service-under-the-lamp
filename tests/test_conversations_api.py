"""Conversation list and read-only history API tests."""

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app
from app.services.history import (
    append_message,
    create_conversation,
    update_conversation_summary,
)


@pytest.fixture
def client(session):
    app.dependency_overrides[get_settings] = lambda: Settings(openai_api_key="test")
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_list_conversations_newest_first(client, session):
    c1 = create_conversation(session, "u1")
    append_message(session, c1.id, "user", "订单 1001 退款")
    update_conversation_summary(session, c1.id, "第1段：订单 1001", 1)
    c2 = create_conversation(session, "u1")
    append_message(session, c2.id, "user", "物流到哪")
    r = client.get("/api/conversations", params={"user_id": "u1"})
    assert r.status_code == 200
    rows = r.json()["conversations"]
    assert [row["id"] for row in rows] == [c2.id, c1.id]
    assert rows[1]["preview"].startswith("订单 1001") and rows[1]["has_summary"] is True


def test_messages_404_and_user_scope(client, session):
    cid = create_conversation(session, "u1").id
    assert client.get(f"/api/conversations/{cid}/messages").status_code == 200
    assert client.get(f"/api/conversations/{cid}/messages", params={"user_id": "u2"}).status_code == 404
    assert client.get("/api/conversations/9999/messages").status_code == 404


def test_messages_filter_audit_rows(client, session):
    cid = create_conversation(session, "u1").id
    append_message(session, cid, "user", "问")
    append_message(session, cid, "user", "[订单选择] 订单 1001")
    append_message(session, cid, "assistant", "答")
    session.commit()
    rows = client.get(f"/api/conversations/{cid}/messages").json()["messages"]
    assert [row["content"] for row in rows] == ["问", "答"]
