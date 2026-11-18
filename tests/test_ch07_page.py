"""Ch07 page tests: multi-conversation sidebar, root page and ch06 frames."""

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app


@pytest.fixture
def client(session):
    app.dependency_overrides[get_settings] = lambda: Settings(openai_api_key="test")
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_ch07_page_has_sidebar_and_apis(client):
    r = client.get("/ch07")
    assert r.status_code == 200
    text = r.text
    assert "lamp-cs-ch07-user-id" in text
    assert "新对话" in text
    assert "/api/conversations" in text
    assert "conversation_id" in text
    assert "@media (max-width: 780px)" in text
    assert "streaming || pickerPending" in text


def test_root_serves_ch07_and_ch06_still_works(client):
    assert "lamp-cs-ch07-user-id" in client.get("/").text
    assert "lamp-cs-ch06-session-id" in client.get("/ch06").text


def test_ch06_page_ignores_conversation_frame():
    text = open("app/static/ch06.html", encoding="utf-8").read()
    assert 'frame.type === "conversation"' in text
