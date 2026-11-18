"""Ch06 API tests: page routes, chat/resume SSE, refunds and lazy graph."""

import threading
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api import ch06 as ch06_api
from app.api.ch06 import get_ch06_graph_dep
from app.config import Settings, get_settings
from app.main import app
from app.models import Conversation, Message, Ticket
from app.tools.ticket import TICKET_NO_PATTERN
from tests.agent_fakes import FakeGraph

ORDER_INTERRUPT = {
    "kind": "select_order",
    "slot": "order_id",
    "orders": [{"order_id": "1001", "title": "无线蓝牙耳机 ×1"}],
}


@pytest.fixture
def client(session):
    app.dependency_overrides[get_settings] = lambda: Settings(openai_api_key="test")
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _pending_snapshot():
    return SimpleNamespace(
        values={}, interrupts=(SimpleNamespace(value=ORDER_INTERRUPT),)
    )


def _idle_snapshot(values=None):
    return SimpleNamespace(values=values or {}, interrupts=())


def test_ch06_page_and_root(client):
    assert client.get("/ch06").status_code == 200
    assert client.get("/").status_code == 200
    assert "lamp-cs-ch06-session-id" in client.get("/ch06").text


def test_ch05_redirects(client):
    assert client.get("/ch05", follow_redirects=False).status_code == 307
    assert (
        client.post(
            "/ch05/chat",
            json={"session_id": "s", "message": "x"},
            follow_redirects=False,
        ).status_code
        == 307
    )


def test_ch05_redirects_point_at_ch06(client):
    page = client.get("/ch05", follow_redirects=False)
    chat = client.post(
        "/ch05/chat",
        json={"session_id": "s", "message": "x"},
        follow_redirects=False,
    )

    assert page.headers["location"] == "/ch06"
    assert chat.headers["location"] == "/ch06/chat"


def test_ch06_chat_streams_frames_and_done(client):
    graph = FakeGraph(
        frames=[{"type": "token", "text": "您好"}],
        final={
            "answer": "您好",
            "suggested_actions": [{"action": "transfer", "label": "转人工"}],
        },
    )
    app.dependency_overrides[get_ch06_graph_dep] = lambda: graph

    with client.stream(
        "POST", "/ch06/chat", json={"session_id": "s1", "message": "你好"}
    ) as r:
        body = "".join(r.iter_text())

    assert r.status_code == 200
    assert "data: 您好" in body
    assert "options" in body
    assert "data: [DONE]" in body


def test_ch06_chat_blank_message_422(client):
    app.dependency_overrides[get_ch06_graph_dep] = lambda: FakeGraph([], {})
    r = client.post("/ch06/chat", json={"session_id": "s1", "message": "  "})
    assert r.status_code == 422


def test_ch06_chat_graph_error_yields_error_and_done(client):
    class RaisingGraph:
        def stream(self, *args, **kwargs):
            raise RuntimeError("graph exploded")

        def get_state(self, config):
            # The service's pre-check runs first; only the stream may fail.
            return SimpleNamespace(values={}, interrupts=())

    app.dependency_overrides[get_ch06_graph_dep] = lambda: RaisingGraph()

    with client.stream(
        "POST", "/ch06/chat", json={"session_id": "s1", "message": "你好"}
    ) as r:
        body = "".join(r.iter_text())

    assert r.status_code == 200
    assert '"error"' in body
    assert "data: [DONE]" in body


def test_resume_without_pending_409(client, session):
    graph = FakeGraph([], {})
    app.dependency_overrides[get_ch06_graph_dep] = lambda: graph

    r = client.post("/ch06/chat/resume", json={"session_id": "s", "order_id": "1001"})

    assert r.status_code == 409
    assert r.json()["detail"] == "当前没有待选择的订单"
    # 409 must not start a stream or write an audit row.
    assert graph.stream_calls == []
    assert session.query(Message).count() == 0


def test_resume_with_pending_streams_and_writes_audit(client, session):
    graph = FakeGraph(
        frames=[{"type": "token", "text": "这一单可以退"}],
        final={"answer": "这一单可以退"},
        snapshots=[_pending_snapshot(), _pending_snapshot()],
    )
    app.dependency_overrides[get_ch06_graph_dep] = lambda: graph

    with client.stream(
        "POST",
        "/ch06/chat/resume",
        json={"session_id": "s1", "order_id": "1001"},
    ) as r:
        body = "".join(r.iter_text())

    assert r.status_code == 200
    assert "data: 这一单可以退" in body
    assert "data: [DONE]" in body
    assert graph.stream_calls[0].resume == "1001"

    session.expire_all()
    contents = [
        row.content
        for row in session.query(Message).filter(Message.role == "user").all()
    ]
    assert "[订单选择] 订单 1001" in contents


def test_ch06_refund_writes_ticket(client, session):
    r = client.post(
        "/ch06/refunds",
        json={"session_id": "s", "order_id": "1001", "reason": "质量问题"},
    )

    assert r.status_code == 200
    payload = r.json()
    assert TICKET_NO_PATTERN.match(payload["ticket_no"])
    assert payload["message"] == "退款单已创建"

    session.expire_all()
    ticket = session.query(Ticket).one()
    assert ticket.ticket_type == "退款"
    assert "订单 1001" in ticket.description
    assert "质量问题" in ticket.description


def test_ch06_refund_invalid_reason_422(client, session):
    r = client.post(
        "/ch06/refunds",
        json={"session_id": "s", "order_id": "1001", "reason": "违规"},
    )

    assert r.status_code == 422
    assert session.query(Ticket).count() == 0


def test_ch06_refund_unknown_order_422(client, session):
    r = client.post(
        "/ch06/refunds",
        json={"session_id": "s", "order_id": "9999", "reason": "质量问题"},
    )

    assert r.status_code == 422
    # Validation happens before any conversation/ticket write.
    assert session.query(Ticket).count() == 0
    assert session.query(Conversation).count() == 0


def test_ch06_refund_failure_502(client, session, monkeypatch):
    calls = []

    def fake_execute_tool(*args, **kwargs):
        calls.append(kwargs)
        return "工具执行失败"

    monkeypatch.setattr("app.api.ch06.execute_tool", fake_execute_tool)

    r = client.post(
        "/ch06/refunds",
        json={"session_id": "s", "order_id": "1001", "reason": "质量问题"},
    )

    assert r.status_code == 502
    assert session.query(Ticket).count() == 0
    # 退款单写入非幂等:必须禁用自动重试,超时宁可 502 让用户手动确认重试。
    assert calls == [{"timeout_s": 5, "retries": 0}]


def _fake_request_state():
    state = SimpleNamespace(
        ch06_graph=None,
        ch06_checkpoint_handle=None,
        ch06_graph_lock=threading.Lock(),
    )
    return SimpleNamespace(app=SimpleNamespace(state=state))


def test_ch06_lazy_graph_builds_once_and_closes(monkeypatch):
    built = []
    closed = []
    paths = []
    llm_calls = []

    class Handle:
        saver = object()

        def close(self):
            closed.append(True)

    monkeypatch.setattr(
        ch06_api,
        "open_sqlite_checkpointer",
        lambda path: (paths.append(path), Handle())[1],
    )

    def fake_build_llm(settings, temperature=0.7, model=None):
        llm_calls.append((temperature, model))
        return object()

    monkeypatch.setattr(ch06_api, "build_llm", fake_build_llm)

    def fake_build_graph(llm, intent_llm, *, checkpointer=None, settings=None):
        built.append(checkpointer)
        return object()

    monkeypatch.setattr(ch06_api, "build_graph", fake_build_graph)
    request = _fake_request_state()
    settings = Settings(
        _env_file=None, openai_api_key="test", intent_model="intent-x"
    )

    first = ch06_api.get_ch06_graph_dep(request, settings)
    second = ch06_api.get_ch06_graph_dep(request, settings)

    assert first is second
    assert len(built) == 1
    assert paths == [settings.ch06_checkpoint_db]
    assert llm_calls == [(0.7, None), (0.0, "intent-x")]
    assert request.app.state.ch06_checkpoint_handle is not None

    ch06_api.close_ch06_graph(request.app)
    assert closed == [True]
    assert request.app.state.ch06_graph is None
    assert request.app.state.ch06_checkpoint_handle is None


def test_ch06_lazy_graph_failure_raises_503_and_closes_handle(monkeypatch):
    closed = []

    class Handle:
        saver = object()

        def close(self):
            closed.append(True)

    monkeypatch.setattr(
        ch06_api, "open_sqlite_checkpointer", lambda path: Handle()
    )
    monkeypatch.setattr(
        ch06_api,
        "build_llm",
        lambda settings, temperature=0.7, model=None: object(),
    )

    def failing_build_graph(*args, **kwargs):
        raise RuntimeError("no model")

    monkeypatch.setattr(ch06_api, "build_graph", failing_build_graph)
    request = _fake_request_state()

    with pytest.raises(HTTPException) as excinfo:
        ch06_api.get_ch06_graph_dep(
            request, Settings(_env_file=None, openai_api_key="test")
        )

    assert excinfo.value.status_code == 503
    assert closed == [True]
    assert request.app.state.ch06_graph is None


def test_ch06_app_does_not_build_graph_at_startup(client):
    # ``client`` has already entered ``with TestClient(app)``, i.e. lifespan
    # startup ran; neither the graph nor the checkpointer may exist yet.
    assert app.state.ch06_graph is None
    assert app.state.ch06_checkpoint_handle is None
