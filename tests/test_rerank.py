import json

import httpx
import pytest

from app.config import Settings
from app.rag import rerank as rerank_mod
from app.rag.rerank import RerankError, RerankHit, SiliconFlowReranker, get_reranker


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = {} if payload is None else payload
        self.text = json.dumps(self._payload, ensure_ascii=False)

    def json(self):
        return self._payload


class FakeClient:
    """Minimal stand-in for ``httpx.Client`` used by the reranker tests."""

    def __init__(self, *, json=None, statuses=None, errors=None):
        self._json = {} if json is None else json
        self._statuses = [200] if statuses is None else list(statuses)
        self._errors = list(errors) if errors else []
        self._calls = 0
        self.calls = []
        self.last = None

    def post(self, url, *, headers=None, json=None):
        call = {"url": url, "headers": headers, "json": json}
        self.calls.append(call)
        self.last = call
        index = self._calls
        self._calls += 1
        if index < len(self._errors) and self._errors[index] is not None:
            raise self._errors[index]
        status = self._statuses[min(index, len(self._statuses) - 1)]
        return FakeResponse(status, self._json)

    def close(self):
        pass


def test_rerank_posts_expected_payload_and_sorts():
    client = FakeClient(json={"results": [
        {"index": 1, "relevance_score": 0.9},
        {"index": 0, "relevance_score": 0.2},
    ]})
    r = SiliconFlowReranker("https://api.siliconflow.cn/v1", "k", "BAAI/bge-reranker-v2-m3", client=client)
    hits = r.rerank("q", ["a", "b"], top_n=2)
    assert client.last["json"] == {"model": "BAAI/bge-reranker-v2-m3", "query": "q", "documents": ["a", "b"], "top_n": 2, "return_documents": False}
    assert [(h.index, h.score) for h in hits] == [(1, 0.9), (0, 0.2)]


def test_rerank_retries_429_then_raises():
    client = FakeClient(statuses=[429, 200], json={"results": [{"index": 0, "relevance_score": 0.8}]})
    sleeps = []
    r = SiliconFlowReranker("u", "k", "m", client=client, sleep=sleeps.append)
    assert r.rerank("q", ["a"])[0].score == 0.8
    assert sleeps == [0.5]


def test_rerank_rejects_out_of_range_index():
    client = FakeClient(json={"results": [{"index": 9, "relevance_score": 0.5}]})
    with pytest.raises(RerankError):
        SiliconFlowReranker("u", "k", "m", client=client).rerank("q", ["a"])


def test_rerank_omits_top_n_when_not_given_and_sorts_all():
    client = FakeClient(json={"results": [
        {"index": 2, "relevance_score": 0.1},
        {"index": 0, "relevance_score": 0.7},
        {"index": 1, "relevance_score": 0.4},
    ]})
    r = SiliconFlowReranker("https://x/v1", "k", "m", client=client)
    hits = r.rerank("q", ["a", "b", "c"])

    assert client.last["url"] == "https://x/v1/rerank"
    assert client.last["headers"]["Authorization"] == "Bearer k"
    assert "top_n" not in client.last["json"]
    assert client.last["json"]["return_documents"] is False
    assert hits == [RerankHit(index=0, score=0.7), RerankHit(index=1, score=0.4), RerankHit(index=2, score=0.1)]


def test_rerank_top_n_truncates_sorted_results():
    client = FakeClient(json={"results": [
        {"index": 0, "relevance_score": 0.2},
        {"index": 1, "relevance_score": 0.9},
        {"index": 2, "relevance_score": 0.5},
    ]})
    hits = SiliconFlowReranker("u", "k", "m", client=client).rerank("q", ["a", "b", "c"], top_n=2)
    assert [(h.index, h.score) for h in hits] == [(1, 0.9), (2, 0.5)]


def test_rerank_401_raises_without_retry():
    client = FakeClient(statuses=[401], json={"message": "bad key"})
    with pytest.raises(RerankError) as excinfo:
        SiliconFlowReranker("u", "k", "m", client=client, sleep=lambda _: None).rerank("q", ["a"])
    assert "401" in str(excinfo.value)
    assert len(client.calls) == 1


def test_rerank_retries_transport_error_then_succeeds():
    client = FakeClient(
        errors=[httpx.ConnectError("boom"), None],
        json={"results": [{"index": 0, "relevance_score": 0.6}]},
    )
    sleeps = []
    hits = SiliconFlowReranker("u", "k", "m", client=client, sleep=sleeps.append).rerank("q", ["a"])
    assert hits == [RerankHit(index=0, score=0.6)]
    assert sleeps == [0.5]


def test_rerank_gives_up_after_max_retries():
    client = FakeClient(statuses=[503], json={"message": "busy"})
    sleeps = []
    with pytest.raises(RerankError) as excinfo:
        SiliconFlowReranker("u", "k", "m", client=client, max_retries=2, sleep=sleeps.append).rerank("q", ["a"])
    assert "503" in str(excinfo.value)
    assert len(client.calls) == 3
    assert sleeps == [0.5, 1.0]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"results": "nope"},
        {"results": [{"index": 0}]},
        {"results": [{"relevance_score": 1.0}]},
        {"results": [{"index": "0", "relevance_score": 1.0}]},
        {"results": [{"index": 0, "relevance_score": "high"}]},
    ],
)
def test_rerank_rejects_invalid_payload(payload):
    client = FakeClient(json=payload)
    with pytest.raises(RerankError):
        SiliconFlowReranker("u", "k", "m", client=client).rerank("q", ["a"])


def test_rerank_empty_documents_returns_empty_without_request():
    client = FakeClient()

    def fail_post(*args, **kwargs):
        raise AssertionError("no request expected")

    client.post = fail_post
    assert SiliconFlowReranker("u", "k", "m", client=client).rerank("q", []) == []


def _settings(**overrides):
    values = {
        "openai_api_key": "x",
        "embedding_base_url": "https://embed.example/v1",
        "embedding_api_key": "",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_get_reranker_requires_api_key(monkeypatch):
    settings = _settings(rerank_api_key="")
    monkeypatch.setattr(rerank_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(rerank_mod, "_reranker", None)

    with pytest.raises(RerankError, match="rerank_api_key"):
        get_reranker()


def test_get_reranker_falls_back_to_embedding_settings(monkeypatch):
    settings = _settings(
        embedding_api_key="embed-key",
        embedding_base_url="https://embed.example/v1",
        rerank_api_key="",
        rerank_base_url="",
    )
    monkeypatch.setattr(rerank_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(rerank_mod, "_reranker", None)

    reranker = get_reranker()
    assert reranker.base_url == "https://embed.example/v1"
    assert reranker.api_key == "embed-key"
    assert reranker.model == "BAAI/bge-reranker-v2-m3"
    assert reranker.timeout_s == 10.0
    reranker.close()


def test_get_reranker_prefers_rerank_settings_and_caches(monkeypatch):
    settings = _settings(
        embedding_api_key="embed-key",
        embedding_base_url="https://embed.example/v1",
        rerank_api_key="rerank-key",
        rerank_base_url="https://rerank.example/v1/",
        rerank_model="custom/model",
        rerank_timeout_s=3.5,
    )
    monkeypatch.setattr(rerank_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(rerank_mod, "_reranker", None)

    first = get_reranker()
    second = get_reranker()
    assert first is second
    assert first.base_url == "https://rerank.example/v1"
    assert first.api_key == "rerank-key"
    assert first.model == "custom/model"
    assert first.timeout_s == 3.5
    first.close()
