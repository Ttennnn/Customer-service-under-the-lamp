import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app, get_llm_dep, get_store_dep
from app.schemas import AfterSaleExtraction, AppealType
from app.services.session import SessionStore


class FakeChunk:
    def __init__(self, content):
        self.content = content


class FakeStructured:
    def invoke(self, messages):
        return AfterSaleExtraction(
            order_id="123",
            appeal_type=AppealType.REFUND,
            expected_solution="退款",
        )


class FakeLLM:
    def stream(self, messages):
        yield FakeChunk("你")

    def with_structured_output(self, schema, **kwargs):
        return FakeStructured()


class RaisingLLM:
    def stream(self, messages):
        raise RuntimeError("upstream boom")

    def with_structured_output(self, schema, **kwargs):
        raise NotImplementedError


@pytest.fixture
def client():
    app.dependency_overrides[get_settings] = lambda: Settings(openai_api_key="test")
    app.dependency_overrides[get_llm_dep] = lambda: FakeLLM()
    app.dependency_overrides[get_store_dep] = lambda: SessionStore()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_chat_streams_sse_and_done(client):
    with client.stream("POST", "/chat", json={"session_id": "s1", "message": "在吗"}) as r:
        body = "".join(r.iter_text())
    assert "data: 你" in body and "data: [DONE]" in body


def test_chat_blank_message_422(client):
    assert client.post("/chat", json={"session_id": "s1", "message": "  "}).status_code == 422


def test_extract_returns_three_fields(client):
    r = client.post("/extract", json={"text": "订单123 退款"}).json()
    assert set(r.keys()) == {"order_id", "appeal_type", "expected_solution"}


def test_extract_upstream_error_502(client):
    app.dependency_overrides[get_llm_dep] = lambda: RaisingLLM()
    r = client.post("/extract", json={"text": "订单123 退款"})
    assert r.status_code == 502
    assert r.json()["detail"] != ""


def test_chat_upstream_error_not_hang(client):
    app.dependency_overrides[get_llm_dep] = lambda: RaisingLLM()
    with client.stream("POST", "/chat", json={"session_id": "s1", "message": "hi"}) as r:
        body = "".join(r.iter_text())
    assert "error" in body and "data: [DONE]" in body
