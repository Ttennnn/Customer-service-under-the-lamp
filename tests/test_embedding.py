import json

import httpx
import pytest

from app.config import Settings
from app.rag import embedding as embedding_mod
from app.rag.embedding import EmbeddingError, SiliconFlowEmbedder, get_embedder


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def _ok(data):
    return httpx.Response(200, json={"data": data})


def test_batches_and_orders_by_index():
    calls = []

    def handler(req):
        payload = json.loads(req.content)
        calls.append(payload)
        assert req.headers["authorization"] == "Bearer k"
        offset = sum(len(call["input"]) for call in calls[:-1])
        data = [
            {"index": i, "embedding": [float(offset + i)] * 1024}
            for i in range(len(payload["input"]))
        ]
        return _ok(list(reversed(data)))

    e = SiliconFlowEmbedder("https://x/v1", "k", batch_size=2, client=_client(handler))
    out = e.embed(["a", "b", "c"])

    assert len(calls) == 2 and calls[0]["input"] == ["a", "b"]
    assert calls[0]["model"] == "BAAI/bge-m3"
    assert calls[0]["encoding_format"] == "float"
    assert [v[0] for v in out] == [0.0, 1.0, 2.0]
    assert all(len(v) == 1024 for v in out)


def test_dimension_mismatch_raises():
    def handler(req):
        return _ok([{"index": 0, "embedding": [0.0] * 3}])

    e = SiliconFlowEmbedder("https://x/v1", "k", client=_client(handler))
    with pytest.raises(EmbeddingError) as excinfo:
        e.embed_one("a")
    assert "1024" in str(excinfo.value)


def test_retries_503_then_succeeds():
    n = {"i": 0}
    delays = []

    def handler(req):
        n["i"] += 1
        if n["i"] == 1:
            return httpx.Response(503, json={"message": "busy"})
        return _ok([{"index": 0, "embedding": [0.0] * 1024}])

    e = SiliconFlowEmbedder(
        "https://x/v1", "k", client=_client(handler), sleep=delays.append
    )
    assert len(e.embed_one("a")) == 1024 and n["i"] == 2
    assert delays == [0.5]


def test_401_raises_without_retry():
    n = {"i": 0}

    def handler(req):
        n["i"] += 1
        return httpx.Response(401, json={"message": "bad key"})

    e = SiliconFlowEmbedder(
        "https://x/v1", "k", client=_client(handler), sleep=lambda _: None
    )
    with pytest.raises(EmbeddingError) as excinfo:
        e.embed_one("a")
    assert "401" in str(excinfo.value) and n["i"] == 1


def test_transport_error_retries_then_succeeds():
    n = {"i": 0}
    delays = []

    def handler(req):
        n["i"] += 1
        if n["i"] == 1:
            raise httpx.ConnectError("boom")
        return _ok([{"index": 0, "embedding": [0.0] * 1024}])

    e = SiliconFlowEmbedder(
        "https://x/v1", "k", client=_client(handler), sleep=delays.append
    )
    assert len(e.embed_one("a")) == 1024 and n["i"] == 2
    assert delays == [0.5]


def test_gives_up_after_max_retries():
    n = {"i": 0}
    delays = []

    def handler(req):
        n["i"] += 1
        return httpx.Response(503, json={"message": "busy"})

    e = SiliconFlowEmbedder(
        "https://x/v1",
        "k",
        client=_client(handler),
        max_retries=2,
        sleep=delays.append,
    )
    with pytest.raises(EmbeddingError) as excinfo:
        e.embed_one("a")
    assert "503" in str(excinfo.value)
    assert n["i"] == 3
    assert delays == [0.5, 1.0]


def test_empty_input_returns_empty_without_request():
    def handler(req):
        raise AssertionError("no request expected")

    e = SiliconFlowEmbedder("https://x/v1", "k", client=_client(handler))
    assert e.embed([]) == []


def test_get_embedder_requires_api_key(monkeypatch):
    settings = Settings(_env_file=None, openai_api_key="x", embedding_api_key="")
    monkeypatch.setattr(embedding_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(embedding_mod, "_embedder", None)

    with pytest.raises(EmbeddingError, match="embedding_api_key"):
        get_embedder()


def test_get_embedder_caches_and_uses_settings(monkeypatch):
    settings = Settings(
        _env_file=None,
        openai_api_key="x",
        embedding_api_key="k",
        embedding_base_url="https://api.example.test/v1",
        embedding_batch_size=8,
    )
    monkeypatch.setattr(embedding_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(embedding_mod, "_embedder", None)

    first = get_embedder()
    second = get_embedder()
    assert first is second
    assert first.base_url == "https://api.example.test/v1"
    assert first.model == "BAAI/bge-m3"
    assert first.dim == 1024
    assert first.batch_size == 8
    first.close()
