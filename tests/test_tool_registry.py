"""ToolSpec 注册中心测试。

覆盖:内置插件扫描、重复注册、MCP 关闭时不触网、MCP 合并去重
(内置优先)、以及模块已缓存时 ``reset_registry`` + 重新加载仍可用。
"""

import importlib
import pkgutil
import threading

import pytest
from langchain.tools import tool

import app.tools.builtin as builtin_pkg

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


def _expected_builtin_names() -> set[str]:
    names: set[str] = set()
    for module_info in pkgutil.iter_modules(builtin_pkg.__path__):
        module = importlib.import_module(
            f"{builtin_pkg.__name__}.{module_info.name}"
        )
        names.update(spec.name for spec in getattr(module, "SPECS", ()))
    return names


def test_concurrent_first_load_registers_one_spec_set():
    reset_registry()
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def load() -> None:
        try:
            barrier.wait(timeout=5)
            load_builtin_tools()
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=load) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []
    assert set(builtin_specs()) == _expected_builtin_names()


def test_failed_scan_leaves_registry_untouched_and_retryable(monkeypatch):
    import app.tools.registry as registry

    reset_registry()
    calls = {"n": 0}

    class _FlakyImportlib:
        """Proxy for the registry's module-level ``importlib`` reference."""

        @staticmethod
        def import_module(name):
            calls["n"] += 1
            if calls["n"] == 2:
                raise ImportError("simulated plugin import failure")
            return importlib.import_module(name)

    monkeypatch.setattr(registry, "importlib", _FlakyImportlib)
    with pytest.raises(ImportError):
        load_builtin_tools()

    # The failed scan must not leave a partial registry behind, and a later
    # call must be able to retry from a clean slate instead of hitting a
    # duplicate-tool error.
    assert registry._BUILTIN == {}
    assert registry._LOADED is False

    monkeypatch.setattr(registry, "importlib", importlib)
    load_builtin_tools()
    assert set(builtin_specs()) == _expected_builtin_names()
