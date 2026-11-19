"""异步桥与 MCP Client 发现/合并测试。

全部使用 fake client,不启动 MCP server、不访问网络。
"""

import asyncio
import json

import pytest

from app.config import Settings
from app.tools.async_bridge import run_async, shutdown_async_bridge
from app.tools.builtin.orders import query_order
from app.tools.mcp_client import _build_client, fetch_mcp_specs, normalize_mcp_result
from app.tools.registry import get_all_specs, load_builtin_tools, reset_registry
from app.tools.spec import ToolSpec


@pytest.fixture(autouse=True)
def _stop_bridge():
    yield
    shutdown_async_bridge()


def _settings(**overrides) -> Settings:
    values = {"_env_file": None, "openai_api_key": "test"}
    values.update(overrides)
    return Settings(**values)


# --- 异步桥 ---


def test_run_async_returns_result():
    assert run_async(asyncio.sleep(0, result="ok")) == "ok"


def test_run_async_timeout_raises_builtin_timeout():
    async def slow():
        await asyncio.sleep(5)
        return "never"

    with pytest.raises(TimeoutError):
        run_async(slow(), timeout=0.05)


def test_run_async_rejects_call_from_loop_thread():
    async def inner():
        with pytest.raises(RuntimeError, match="event loop thread"):
            run_async(asyncio.sleep(0))
        return "done"

    assert run_async(inner()) == "done"


# --- 结果归一化 ---


def test_normalize_mcp_result_unwraps_text_blocks():
    payload = {"tracking_no": "SF1", "status_code": "TRANSIT"}
    raw = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
    assert normalize_mcp_result(raw) == payload


def test_normalize_mcp_result_keeps_non_text_blocks():
    raw = [{"type": "image", "base64": "abc", "mime_type": "image/png"}]
    assert normalize_mcp_result(raw) is raw


# --- 发现与合并 ---


FULL_INPUT_SCHEMA = {
    "type": "object",
    "properties": {"tracking_no": {"type": "string"}},
    "required": ["tracking_no"],
}


class FakeTool:
    """Adapter-shaped fake: dict ``args_schema`` holds the full inputSchema."""

    def __init__(self, name: str):
        self.name = name
        self.description = f"{name} mock"
        self.args_schema = dict(FULL_INPUT_SCHEMA)


class FakeClient:
    def __init__(self, tools_by_server, delays=None):
        self.tools_by_server = tools_by_server
        self.delays = delays or {}

    async def get_tools(self, *, server_name=None):
        delay = self.delays.get(server_name, 0)
        if delay:
            await asyncio.sleep(delay)
        result = self.tools_by_server[server_name]
        if isinstance(result, Exception):
            raise result
        return list(result)


def test_disabled_mcp_returns_no_specs(monkeypatch):
    settings = _settings(mcp_enabled=False)
    monkeypatch.setattr(
        "app.tools.mcp_client._fetch_all",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no network")),
    )
    assert fetch_mcp_specs(settings) == []


def test_build_client_uses_streamable_http_transport():
    settings = _settings(mcp_enabled=True, mcp_logistics_url="http://x:8101/mcp")
    client = _build_client(settings)
    assert client.connections["logistics"] == {
        "transport": "streamable_http",
        "url": "http://x:8101/mcp",
    }
    assert client.connections["aftersales"]["transport"] == "streamable_http"
    assert client.handle_tool_errors is False


def test_fetch_builds_mcp_specs(monkeypatch):
    client = FakeClient(
        {
            "logistics": [FakeTool("query_logistics")],
            "aftersales": [
                FakeTool("query_warranty"),
                FakeTool("query_return_status"),
            ],
        }
    )
    monkeypatch.setattr("app.tools.mcp_client._build_client", lambda settings: client)
    settings = _settings(mcp_enabled=True, mcp_tool_timeout_s=10)

    specs = {spec.name: spec for spec in fetch_mcp_specs(settings)}

    assert set(specs) == {"query_logistics", "query_warranty", "query_return_status"}
    logistics = specs["query_logistics"]
    assert logistics.source == "mcp"
    assert logistics.mcp_server == "logistics"
    assert logistics.permission == "read"
    assert logistics.timeout == 10
    assert logistics.classify is None
    assert logistics.format_result is None
    # The full adapter inputSchema survives, not just its ``properties`` map.
    assert logistics.json_schema["type"] == "object"
    assert logistics.json_schema["required"] == ["tracking_no"]
    assert specs["query_warranty"].mcp_server == "aftersales"


def test_server_failure_is_skipped(monkeypatch):
    client = FakeClient(
        {
            "logistics": RuntimeError("down"),
            "aftersales": [FakeTool("query_warranty")],
        }
    )
    monkeypatch.setattr("app.tools.mcp_client._build_client", lambda settings: client)

    specs = fetch_mcp_specs(_settings(mcp_enabled=True))

    assert [spec.name for spec in specs] == ["query_warranty"]


def test_discovery_timeout_skips_server(monkeypatch):
    client = FakeClient(
        {
            "logistics": [FakeTool("query_logistics")],
            "aftersales": [FakeTool("query_warranty")],
        },
        delays={"logistics": 0.2},
    )
    monkeypatch.setattr("app.tools.mcp_client._build_client", lambda settings: client)

    specs = fetch_mcp_specs(
        _settings(mcp_enabled=True, mcp_discovery_timeout_s=0.05)
    )

    assert [spec.name for spec in specs] == ["query_warranty"]


def test_merge_drops_mcp_duplicate(monkeypatch):
    reset_registry()
    load_builtin_tools()
    fake = ToolSpec(
        name="query_order",
        description="",
        json_schema={},
        tool=query_order,
        permission="read",
        source="mcp",
        mcp_server="logistics",
    )
    monkeypatch.setattr("app.tools.registry.fetch_mcp_specs", lambda settings: [fake])
    specs = get_all_specs(_settings(mcp_enabled=True))
    assert specs["query_order"].source == "builtin"
