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
from app.tools.spec import ToolRun
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


def test_ch05_page_serves_latest_and_chat_redirects(client):
    page = client.get("/ch05", follow_redirects=False)
    assert page.status_code == 200
    assert "lamp-cs-ch06-session-id" in page.text
    assert (
        client.post(
            "/ch05/chat",
            json={"session_id": "s", "message": "x"},
            follow_redirects=False,
        ).status_code
        == 307
    )


def test_ch05_chat_redirects_point_at_ch06_chat(client):
    chat = client.post(
        "/ch05/chat",
        json={"session_id": "s", "message": "x"},
        follow_redirects=False,
    )

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


def test_chat_with_pending_interrupt_reemits_frame_without_stream(client, session):
    graph = FakeGraph(
        frames=[{"type": "token", "text": "不应出现"}],
        final={},
        snapshots=[_pending_snapshot()],
    )
    app.dependency_overrides[get_ch06_graph_dep] = lambda: graph

    with client.stream(
        "POST", "/ch06/chat", json={"session_id": "s1", "message": "你好"}
    ) as r:
        body = "".join(r.iter_text())

    assert r.status_code == 200
    assert '"type":"interrupt"' in body
    assert "select_order" in body
    assert "不应出现" not in body
    # The new message is dropped and the graph must not run while paused.
    assert graph.stream_calls == []
    assert session.query(Message).count() == 0


def test_resume_pending_check_failure_503(client, session):
    class RaisingStateGraph:
        def get_state(self, config):
            raise RuntimeError("checkpoint unavailable")

        def stream(self, *args, **kwargs):
            raise AssertionError("stream must not start")

    app.dependency_overrides[get_ch06_graph_dep] = lambda: RaisingStateGraph()

    r = client.post(
        "/ch06/chat/resume", json={"session_id": "s", "order_id": "1001"}
    )

    assert r.status_code == 503
    assert r.json()["detail"] == ch06_api.ERROR_TEXT
    assert session.query(Message).count() == 0


def test_resume_stream_error_yields_error_and_done(client, session):
    class RaisingStreamGraph:
        def get_state(self, config):
            return _pending_snapshot()

        def stream(self, *args, **kwargs):
            raise RuntimeError("graph exploded")

    app.dependency_overrides[get_ch06_graph_dep] = lambda: RaisingStreamGraph()

    with client.stream(
        "POST",
        "/ch06/chat/resume",
        json={"session_id": "s1", "order_id": "1001"},
    ) as r:
        body = "".join(r.iter_text())

    assert r.status_code == 200
    assert '"error"' in body
    assert "data: [DONE]" in body


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


def test_ch06_refund_uses_confirmed_engine_retries_zero(
    client, session, monkeypatch
):
    seen = {}

    def fake_execute(spec, args, **kwargs):
        seen["spec"] = spec.name
        seen["args"] = args
        seen.update(kwargs)
        return ToolRun(
            status="ok",
            content='{"ticket_no":"T20261119001"}',
            data="T20261119001",
        )

    monkeypatch.setattr("app.api.ch06.execute", fake_execute)

    r = client.post(
        "/ch06/refunds",
        json={
            "session_id": "s",
            "order_id": "1001",
            "reason": "质量问题",
            # The refund form submission is the confirmation; a body field
            # must not override it.
            "confirmed": False,
        },
    )

    assert r.status_code == 200
    assert r.json() == {"ticket_no": "T20261119001", "message": "退款单已创建"}
    assert seen["spec"] == "create_ticket"
    assert seen["args"]["ticket_type"] == "退款"
    assert "订单 1001" in seen["args"]["description"]
    assert "质量问题" in seen["args"]["description"]
    assert seen["confirmed"] is True
    # Refund tickets are non-idempotent: the write must never retry.
    assert seen["retries"] == 0
    assert seen["conversation_id"] is not None


def test_ch06_refund_engine_failure_502(client, session, monkeypatch):
    def fake_execute(spec, args, **kwargs):
        return ToolRun(
            status="error",
            content="工具暂时不可用：create_ticket（TimeoutError）",
        )

    monkeypatch.setattr("app.api.ch06.execute", fake_execute)

    r = client.post(
        "/ch06/refunds",
        json={"session_id": "s", "order_id": "1001", "reason": "质量问题"},
    )

    assert r.status_code == 502
    assert r.json()["detail"] == "退款单创建失败,请稍后重试"
    assert session.query(Ticket).count() == 0


def test_ch06_refund_missing_number_502(client, session, monkeypatch):
    def fake_execute(spec, args, **kwargs):
        return ToolRun(status="ok", content="{}", data=None)

    monkeypatch.setattr("app.api.ch06.execute", fake_execute)

    r = client.post(
        "/ch06/refunds",
        json={"session_id": "s", "order_id": "1001", "reason": "质量问题"},
    )

    assert r.status_code == 502
    assert session.query(Ticket).count() == 0


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
    assert paths == [settings.ch07_checkpoint_db]
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


def test_ch06_lazy_graph_attaches_tracing_when_enabled(monkeypatch):
    class Handle:
        saver = object()

        def close(self):
            pass

    monkeypatch.setattr(
        ch06_api, "open_sqlite_checkpointer", lambda path: Handle()
    )
    monkeypatch.setattr(
        ch06_api,
        "build_llm",
        lambda settings, temperature=0.7, model=None: object(),
    )
    built = object()
    monkeypatch.setattr(ch06_api, "build_graph", lambda *a, **k: built)

    bound = object()
    seen = []

    def fake_attach(graph, settings):
        seen.append((graph, settings))
        return bound

    monkeypatch.setattr(ch06_api, "attach_tracing", fake_attach)
    request = _fake_request_state()
    settings = Settings(
        _env_file=None,
        openai_api_key="test",
        langfuse_public_key="pk-lf-test",
        langfuse_secret_key="sk-lf-test",
    )

    result = ch06_api.get_ch06_graph_dep(request, settings)

    assert result is bound
    assert request.app.state.ch06_graph is bound
    assert seen and seen[0][0] is built and seen[0][1] is settings

    ch06_api.close_ch06_graph(request.app)


def test_lifespan_flushes_tracing_on_shutdown(monkeypatch):
    calls = []
    monkeypatch.setattr("app.main.flush_tracing", lambda: calls.append(True))

    with TestClient(app) as c:
        assert c.get("/ch06").status_code == 200

    assert calls == [True]
