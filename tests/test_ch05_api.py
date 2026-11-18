"""Ch05 API tests (Task 11, design §6.7-§6.8)."""

import threading
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api import ch05 as ch05_api
from app.api.ch05 import get_ch05_graph_dep
from app.config import Settings, get_settings
from app.main import app
from app.models import Conversation, Ticket
from app.services.history import get_or_create_conversation
from app.tools.ticket import TICKET_NO_PATTERN
from tests.agent_fakes import FakeGraph

TRANSFER_ACTION = {"action": "transfer", "label": "转人工"}


@pytest.fixture
def client(session):
    app.dependency_overrides[get_settings] = lambda: Settings(openai_api_key="test")
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_ch05_chat_streams_frames_and_done(client):
    graph = FakeGraph(
        frames=[{"type": "token", "text": "您好"}],
        final={"answer": "您好", "suggested_actions": [TRANSFER_ACTION]},
    )
    app.dependency_overrides[get_ch05_graph_dep] = lambda: graph

    with client.stream(
        "POST", "/ch05/chat", json={"session_id": "s1", "message": "你好"}
    ) as r:
        body = "".join(r.iter_text())

    assert r.status_code == 200
    assert "data: 您好" in body
    assert "options" in body
    assert "data: [DONE]" in body


def test_ch05_chat_blank_message_422(client):
    app.dependency_overrides[get_ch05_graph_dep] = lambda: FakeGraph([], {})
    r = client.post("/ch05/chat", json={"session_id": "s1", "message": "  "})
    assert r.status_code == 422


def test_ch05_chat_graph_error_yields_error_and_done(client):
    class RaisingGraph:
        def stream(self, *args, **kwargs):
            raise RuntimeError("graph exploded")

        def get_state(self, config):
            raise AssertionError("get_state must not run after a graph error")

    app.dependency_overrides[get_ch05_graph_dep] = lambda: RaisingGraph()

    with client.stream(
        "POST", "/ch05/chat", json={"session_id": "s1", "message": "你好"}
    ) as r:
        body = "".join(r.iter_text())

    assert r.status_code == 200
    assert '"error"' in body
    assert "data: [DONE]" in body


def test_ch05_chat_graph_build_failure_503(client):
    def failing_dep():
        raise HTTPException(status_code=503, detail="客服服务暂时不可用,请稍后重试")

    app.dependency_overrides[get_ch05_graph_dep] = failing_dep
    r = client.post("/ch05/chat", json={"session_id": "s1", "message": "你好"})
    assert r.status_code == 503


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


def test_ch05_ticket_defaults_to_consultation(client, session):
    r = client.post(
        "/ch05/tickets", json={"session_id": "s2", "description": "咨询一下"}
    )

    assert r.status_code == 200
    session.expire_all()
    assert session.query(Ticket).one().ticket_type == "咨询"


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


def _fake_request_state():
    state = SimpleNamespace(
        ch05_graph=None,
        ch05_checkpoint_handle=None,
        ch05_graph_lock=threading.Lock(),
    )
    return SimpleNamespace(app=SimpleNamespace(state=state))


def test_ch05_lazy_graph_builds_once_and_closes(monkeypatch):
    built = []
    closed = []

    class Handle:
        saver = object()

        def close(self):
            closed.append(True)

    monkeypatch.setattr(
        ch05_api, "open_sqlite_checkpointer", lambda path: Handle()
    )
    monkeypatch.setattr(
        ch05_api, "build_llm", lambda settings, temperature=0.7: object()
    )

    def fake_build_graph(llm, intent_llm, *, checkpointer=None, settings=None):
        built.append(checkpointer)
        return object()

    monkeypatch.setattr(ch05_api, "build_graph", fake_build_graph)
    request = _fake_request_state()
    settings = Settings(_env_file=None, openai_api_key="test")

    first = ch05_api.get_ch05_graph_dep(request, settings)
    second = ch05_api.get_ch05_graph_dep(request, settings)

    assert first is second
    assert len(built) == 1
    assert request.app.state.ch05_checkpoint_handle is not None

    ch05_api.close_ch05_graph(request.app)
    assert closed == [True]
    assert request.app.state.ch05_graph is None
    assert request.app.state.ch05_checkpoint_handle is None


def test_ch05_lazy_graph_failure_raises_503_and_closes_handle(monkeypatch):
    closed = []

    class Handle:
        saver = object()

        def close(self):
            closed.append(True)

    monkeypatch.setattr(
        ch05_api, "open_sqlite_checkpointer", lambda path: Handle()
    )
    monkeypatch.setattr(
        ch05_api, "build_llm", lambda settings, temperature=0.7: object()
    )

    def failing_build_graph(*args, **kwargs):
        raise RuntimeError("no model")

    monkeypatch.setattr(ch05_api, "build_graph", failing_build_graph)
    request = _fake_request_state()

    with pytest.raises(HTTPException) as excinfo:
        ch05_api.get_ch05_graph_dep(
            request, Settings(_env_file=None, openai_api_key="test")
        )

    assert excinfo.value.status_code == 503
    assert closed == [True]
    assert request.app.state.ch05_graph is None


def test_ch05_app_does_not_build_graph_at_startup(client):
    # ``client`` has already entered ``with TestClient(app)``, i.e. lifespan
    # startup ran; neither the graph nor the checkpointer may exist yet.
    assert app.state.ch05_graph is None
    assert app.state.ch05_checkpoint_handle is None
