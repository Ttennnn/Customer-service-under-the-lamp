"""ToolSpec 注册中心测试。

覆盖:内置插件扫描、重复注册、MCP 关闭时不触网、MCP 合并去重
(内置优先)、以及模块已缓存时 ``reset_registry`` + 重新加载仍可用。
"""

import pytest
from langchain.tools import tool

from app.config import Settings
from app.tools.registry import (
    builtin_specs,
    get_all_specs,
    load_builtin_tools,
    register,
    reset_registry,
)
from app.tools.spec import ToolSpec


@tool
def hello(name: str) -> str:
    """Say hello."""
    return f"hi {name}"


def _spec(name="hello", routes=None, source="builtin", mcp_server=None):
    return ToolSpec(
        name=name,
        description="Say hello.",
        json_schema=hello.args_schema.model_json_schema(),
        tool=hello,
        permission="read",
        source=source,
        mcp_server=mcp_server,
        routes=routes,
    )


def test_builtin_scan_registers_promotions_tool():
    reset_registry()
    load_builtin_tools()
    specs = builtin_specs()
    assert "query_promotions" in specs
    spec = specs["query_promotions"]
    assert spec.source == "builtin"
    assert spec.permission == "read"
    assert spec.description
    assert spec.json_schema["type"] == "object"


def test_duplicate_builtin_registration_raises():
    reset_registry()
    register(_spec())
    with pytest.raises(ValueError):
        register(_spec())


def test_get_all_specs_without_mcp_has_only_builtins(monkeypatch):
    reset_registry()
    load_builtin_tools()
    settings = Settings(_env_file=None, openai_api_key="test", mcp_enabled=False)
    monkeypatch.setattr(
        "app.tools.registry.fetch_mcp_specs",
        lambda settings: (_ for _ in ()).throw(AssertionError("no network")),
    )
    specs = get_all_specs(settings)
    assert "query_promotions" in specs
    assert all(spec.source == "builtin" for spec in specs.values())


def test_get_all_specs_merges_mcp_and_builtin_wins(monkeypatch):
    reset_registry()
    load_builtin_tools()
    duplicate = _spec(name="query_promotions", source="mcp", mcp_server="logistics")
    mcp_only = _spec(name="mcp_query_stock", source="mcp", mcp_server="logistics")
    monkeypatch.setattr(
        "app.tools.registry.fetch_mcp_specs",
        lambda settings: [duplicate, mcp_only],
    )
    settings = Settings(_env_file=None, openai_api_key="test", mcp_enabled=True)
    specs = get_all_specs(settings)
    assert specs["query_promotions"].source == "builtin"
    assert specs["mcp_query_stock"].source == "mcp"
    assert specs["mcp_query_stock"].mcp_server == "logistics"


def test_reset_registry_reloads_when_modules_already_cached():
    reset_registry()
    load_builtin_tools()
    reset_registry()
    load_builtin_tools()
    assert "query_promotions" in builtin_specs()
