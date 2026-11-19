"""MCP 工具发现客户端。

按 settings 构造 ``MultiServerMCPClient``,逐个 server 拉取工具清单并转换成
:class:`ToolSpec`。MCP 关闭时不触网;单台 server 发现失败/超时只记录日志并
跳过,不影响另一台与内置工具。

所有 MCP 工具在本项目的策略里恒为只读,``permission`` 固定为 ``"read"``;
``classify`` / ``format_result`` 留空,交给引擎默认分类器/格式化器按工具名
处理 ``found:false`` 与内部枚举码翻译。
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from langchain_mcp_adapters.client import MultiServerMCPClient

from app.tools.async_bridge import run_async
from app.tools.spec import ToolSpec

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from app.config import Settings

logger = logging.getLogger(__name__)

# server 名称固定,对应两套连接配置与 ``mcp_server`` 审计字段。
SERVER_NAMES: tuple[str, ...] = ("logistics", "aftersales")


def _build_client(settings: Settings) -> MultiServerMCPClient:
    """Build the multi-server client from settings (no I/O yet)."""
    return MultiServerMCPClient(
        {
            "logistics": {
                "transport": "streamable_http",
                "url": settings.mcp_logistics_url,
            },
            "aftersales": {
                "transport": "streamable_http",
                "url": settings.mcp_aftersales_url,
            },
        }
    )


def _spec_from_mcp_tool(tool: BaseTool, server_name: str, settings: Settings) -> ToolSpec:
    schema = getattr(tool, "args", None)
    if not schema:
        args_schema = getattr(tool, "args_schema", None)
        if args_schema is not None:
            schema = args_schema.model_json_schema()
    return ToolSpec(
        name=tool.name,
        description=getattr(tool, "description", "") or "",
        json_schema=schema or {"type": "object", "properties": {}},
        tool=tool,
        permission="read",
        source="mcp",
        mcp_server=server_name,
        timeout=settings.mcp_tool_timeout_s,
        classify=None,
        format_result=None,
    )


async def _fetch_all(
    settings: Settings, client: MultiServerMCPClient | None = None
) -> list[ToolSpec]:
    """Discover tools from every configured server, skipping failures."""
    client = client or _build_client(settings)
    specs: list[ToolSpec] = []
    for server_name in SERVER_NAMES:
        try:
            tools = await asyncio.wait_for(
                client.get_tools(server_name=server_name),
                settings.mcp_discovery_timeout_s,
            )
        except Exception:  # noqa: BLE001 - 单台失败不拖垮整体发现
            logger.warning("MCP server %r 工具发现失败,已跳过", server_name, exc_info=True)
            continue
        specs.extend(
            _spec_from_mcp_tool(tool, server_name, settings) for tool in tools
        )
    return specs


def fetch_mcp_specs(settings: Settings) -> list[ToolSpec]:
    """Return MCP tool specs; empty when disabled or discovery fails."""
    if not settings.mcp_enabled:
        return []

    try:
        # 每台 server 有独立 wait_for;总超时取 2 倍单台上限兜底。
        return run_async(_fetch_all(settings), timeout=settings.mcp_discovery_timeout_s * 2)
    except Exception:  # noqa: BLE001 - 发现失败降级为无 MCP 工具
        logger.warning("MCP 工具发现整体失败,已降级为内置工具", exc_info=True)
        return []
