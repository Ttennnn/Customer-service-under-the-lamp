"""Ch09 Langfuse observability core tests (no network, no real SDK client)."""

from types import SimpleNamespace

import pytest

import app.observability as obs


def _settings(
    *,
    public: str = "pk-lf-test",
    secret: str = "sk-lf-test",
    base_url: str = "http://localhost:3000",
    environment: str = "development",
    trace_name: str = "cs-chat",
):
    return SimpleNamespace(
        langfuse_public_key=public,
        langfuse_secret_key=secret,
        langfuse_base_url=base_url,
        langfuse_environment=environment,
        langfuse_trace_name=trace_name,
    )


def _disabled_settings():
    return _settings(public="", secret="", base_url="")


@pytest.fixture(autouse=True)
def _reset_singletons(monkeypatch):
    """Keep each test from inheriting a process-level client/handler."""
    monkeypatch.setattr(obs, "_client", None)
    monkeypatch.setattr(obs, "_handler", None)


class FakeClient:
    def __init__(self):
        self.trace_updates = []
        self.flush_calls = 0

    def update_current_trace(self, **kwargs):
        self.trace_updates.append(kwargs)

    def flush(self):
        self.flush_calls += 1


class FakeGraph:
    def __init__(self, bound="compiled-graph"):
        self.bound = bound
        self.configs = []
        self.with_config_error = None

    def with_config(self, config):
        if self.with_config_error is not None:
            raise self.with_config_error
        self.configs.append(config)
        return SimpleNamespace(bound=self.bound, config=config)


# --- tracing_enabled -------------------------------------------------------


def test_tracing_disabled_without_keys():
    assert obs.tracing_enabled(_disabled_settings()) is False


def test_tracing_disabled_when_any_required_value_missing():
    assert obs.tracing_enabled(_settings(public="")) is False
    assert obs.tracing_enabled(_settings(secret="")) is False
    assert obs.tracing_enabled(_settings(base_url="")) is False


def test_tracing_enabled_with_all_three_values():
    assert obs.tracing_enabled(_settings()) is True


# --- client / handler singletons ------------------------------------------


def test_client_is_a_process_level_singleton(monkeypatch):
    created = []

    def fake_new_client(settings):
        client = FakeClient()
        created.append((settings, client))
        return client

    monkeypatch.setattr(obs, "_new_client", fake_new_client)
    settings = _settings()

    first = obs.get_langfuse_client(settings)
    second = obs.get_langfuse_client(settings)

    assert first is second
    assert len(created) == 1


def test_callback_handler_is_shared_across_calls(monkeypatch):
    client = FakeClient()
    handler = object()
    calls = []

    monkeypatch.setattr(obs, "_new_client", lambda settings: client)
    monkeypatch.setattr(
        obs, "_new_handler", lambda: (calls.append(True), handler)[1]
    )

    first = obs.get_callback_handler(_settings())
    second = obs.get_callback_handler(_settings())

    assert first is second is handler
    assert calls == [True]


# --- attach_tracing --------------------------------------------------------


def test_attach_tracing_noop_when_disabled():
    graph = object()
    assert obs.attach_tracing(graph, _disabled_settings()) is graph


def test_attach_tracing_binds_shared_handler(monkeypatch):
    handler = object()
    monkeypatch.setattr(obs, "get_callback_handler", lambda settings: handler)
    graph = FakeGraph()

    bound = obs.attach_tracing(graph, _settings())

    assert bound.bound is graph.bound
    assert graph.configs == [{"callbacks": [handler]}]


def test_two_attach_tracing_calls_return_bindings(monkeypatch):
    handler = object()
    monkeypatch.setattr(obs, "get_callback_handler", lambda settings: handler)
    graph = FakeGraph()

    first = obs.attach_tracing(graph, _settings())
    second = obs.attach_tracing(graph, _settings())

    assert first.bound == second.bound == graph.bound
    assert len(graph.configs) == 2
    assert all(cfg == {"callbacks": [handler]} for cfg in graph.configs)


def test_attach_tracing_returns_original_graph_when_handler_fails(monkeypatch):
    def failing_handler(settings):
        raise ImportError("langfuse not installed")

    graph = FakeGraph()
    monkeypatch.setattr(obs, "get_callback_handler", failing_handler)

    assert obs.attach_tracing(graph, _settings()) is graph


def test_attach_tracing_returns_original_graph_when_with_config_fails(
    monkeypatch,
):
    handler = object()
    monkeypatch.setattr(obs, "get_callback_handler", lambda settings: handler)
    graph = FakeGraph()
    graph.with_config_error = RuntimeError("binding failed")

    assert obs.attach_tracing(graph, _settings()) is graph


# --- _state_graph ----------------------------------------------------------


def test_state_graph_unwraps_binding():
    class Binding:
        bound = "graph"

    assert obs._state_graph(Binding()) == "graph"
    assert obs._state_graph("graph") == "graph"


# --- update_intent_trace ---------------------------------------------------


def test_update_intent_trace_writes_metadata_and_tag(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(obs, "_new_client", lambda settings: client)

    obs.update_intent_trace(_settings(), "退款", 0.87)

    assert client.trace_updates == [
        {
            "metadata": {"intent": "退款", "intent_confidence": 0.87},
            "tags": ["intent:退款"],
        }
    ]


def test_update_intent_trace_noop_when_disabled(monkeypatch):
    def unexpected(settings):
        raise AssertionError("client must not be created")

    monkeypatch.setattr(obs, "_new_client", unexpected)

    obs.update_intent_trace(_disabled_settings(), "退款", 0.87)


def test_update_intent_trace_never_raises(monkeypatch):
    class ExplodingClient(FakeClient):
        def update_current_trace(self, **kwargs):
            raise RuntimeError("langfuse down")

    monkeypatch.setattr(obs, "_new_client", lambda settings: ExplodingClient())

    # Must not raise into the graph.
    obs.update_intent_trace(_settings(), "退款", 0.87)


# --- flush_tracing ---------------------------------------------------------


def test_flush_tracing_flushes_existing_client():
    client = FakeClient()
    obs._client = client

    obs.flush_tracing()

    assert client.flush_calls == 1


def test_flush_tracing_noop_without_client():
    obs.flush_tracing()


def test_flush_tracing_swallows_errors():
    class ExplodingClient(FakeClient):
        def flush(self):
            raise RuntimeError("flush failed")

    obs._client = ExplodingClient()

    obs.flush_tracing()
