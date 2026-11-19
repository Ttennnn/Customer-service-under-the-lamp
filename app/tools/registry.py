"""工具注册表。

集中登记客服机器人可用的业务工具,供后续 Agent / 路由统一查找。

文件包含两代 API,迁移期并存:

- 旧 API(``ALL_TOOLS`` / ``TOOLS_BY_NAME`` / ``TOOL_LABELS`` /
  ``TOOL_TIMEOUTS`` / ``tool_timeout``):仍被 ``app.main``、
  ``app.services.chat``、``app.agent.subgraph`` 消费,待后续任务迁移。
- 新 API(``ToolSpec`` + ``register`` / ``load_builtin_tools`` /
  ``get_all_specs`` 等):内置工具通过 ``app.tools.builtin`` 包自动扫描,
  MCP 工具由 ``fetch_mcp_specs`` 合并。
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

from app.tools.builtin.orders import query_order, query_product
from app.tools.faq import query_faq
from app.tools.spec import ToolSpec
from app.tools.ticket import create_ticket

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)

# 已实现的业务工具,按最终注册顺序排列。
ALL_TOOLS: list[BaseTool] = [
    query_order,
    query_product,
    query_faq,
    create_ticket,
]

TOOLS_BY_NAME: dict[str, BaseTool] = {t.name: t for t in ALL_TOOLS}

# 各工具的执行超时(秒);未列出的工具使用默认值。
# query_faq 覆盖改写 + 召回 + 重排的完整链路,给到 25 秒(低于前端 30 秒
# 无数据超时);配合 stream_chat 的 retries=0,超时只发生一次。
TOOL_TIMEOUTS: dict[str, float] = {"query_faq": 25.0}
DEFAULT_TOOL_TIMEOUT = 5.0


def tool_timeout(name: str) -> float:
    """Return the execution timeout in seconds for tool *name*."""
    return TOOL_TIMEOUTS.get(name, DEFAULT_TOOL_TIMEOUT)

# 四个工具的展示名(仅为中文标签字符串,不包含工具对象)。
TOOL_LABELS: dict[str, str] = {
    "query_order": "查订单",
    "query_product": "查商品",
    "query_faq": "查常见问题",
    "create_ticket": "创建工单",
}


# --- 新注册中心(ToolSpec) ---

_BUILTIN: dict[str, ToolSpec] = {}
_LOADED = False


def fetch_mcp_specs(settings: Settings) -> list[ToolSpec]:
    """Return specs discovered from MCP servers.

    Temporary stub: the real MCP client arrives in Task 7, which replaces this
    name with ``app.tools.mcp_client.fetch_mcp_specs``. Tests monkeypatch it.
    """
    return []


def register(spec: ToolSpec) -> None:
    """Register *spec*, rejecting duplicate builtin names."""
    if spec.name in _BUILTIN:
        raise ValueError(f"duplicate tool: {spec.name}")
    _BUILTIN[spec.name] = spec


def load_builtin_tools() -> None:
    """Import every module in ``app.tools.builtin`` and register its ``SPECS``.

    Builtin modules expose a module-level ``SPECS: list[ToolSpec]`` instead of
    registering as an import side effect, so reset + reload still works even
    when the modules are already cached in ``sys.modules``.
    """
    global _LOADED
    if _LOADED:
        return

    import app.tools.builtin as builtin_pkg

    for module_info in pkgutil.iter_modules(builtin_pkg.__path__):
        module = importlib.import_module(f"{builtin_pkg.__name__}.{module_info.name}")
        for spec in getattr(module, "SPECS", ()):
            register(spec)
    _LOADED = True


def builtin_specs() -> dict[str, ToolSpec]:
    """Return a copy of the registered builtin specs."""
    load_builtin_tools()
    return dict(_BUILTIN)


def get_all_specs(settings: Settings | None = None) -> dict[str, ToolSpec]:
    """Return builtin specs, plus MCP specs when MCP is enabled.

    On a name conflict the builtin tool wins and a warning is logged.
    """
    load_builtin_tools()
    merged = dict(_BUILTIN)
    if settings is not None and settings.mcp_enabled:
        for spec in fetch_mcp_specs(settings):
            if spec.name in merged:
                logger.warning(
                    "duplicate tool %r from MCP server %r; keeping builtin",
                    spec.name,
                    spec.mcp_server,
                )
                continue
            merged[spec.name] = spec
    return merged


def reset_registry() -> None:
    """Clear the registry so tests can start from a clean slate."""
    global _LOADED
    _BUILTIN.clear()
    _LOADED = False
