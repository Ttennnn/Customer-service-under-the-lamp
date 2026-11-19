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
import json
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
        },
        # MCP ``isError=True`` 结果改为抛异常,由引擎按失败处理而不是伪装成功。
        handle_tool_errors=False,
    )


def _coerce_json_schema(schema: object) -> dict | None:
    """Return *schema* as a full JSON Schema dict, or ``None``."""
    if isinstance(schema, dict):
        return schema
    model_json_schema = getattr(schema, "model_json_schema", None)
    if callable(model_json_schema):
        return model_json_schema()
    return None


def _json_schema_from_tool(tool: BaseTool) -> dict:
    """Derive the tool's **full** JSON Schema.

    ``langchain-mcp-adapters`` builds ``StructuredTool(args_schema=tool.inputSchema)``
    with a dict, so ``tool.tool_call_schema`` (and ``args_schema``) is already the
    complete ``{"type": "object", "properties": ..., "required": ...}`` schema.

    ``BaseTool.args`` must NOT be used: in ``langchain-core`` it returns only the
    ``properties`` map, which would silently disable required/type validation.
    """
    for attr in ("tool_call_schema", "args_schema"):
        schema = _coerce_json_schema(getattr(tool, attr, None))
        if schema is not None:
            return schema

    # 最后兜底:把 ``args`` 的 properties 映射包成合法 object schema。
    args = getattr(tool, "args", None)
    if isinstance(args, dict):
        return {"type": "object", "properties": args}
    return {"type": "object", "properties": {}}


def _spec_from_mcp_tool(tool: BaseTool, server_name: str, settings: Settings) -> ToolSpec:
    return ToolSpec(
        name=tool.name,
        description=getattr(tool, "description", "") or "",
        json_schema=_json_schema_from_tool(tool),
        tool=tool,
        permission="read",
        source="mcp",
        mcp_server=server_name,
        timeout=settings.mcp_tool_timeout_s,
        classify=None,
        format_result=None,
    )


def normalize_mcp_result(raw: object) -> object:
    """Unwrap MCP content blocks to the inner text/JSON payload.

    ``langchain-mcp-adapters`` returns ``[{"type": "text", "text": "<json>"}]``
    from ``_convert_call_tool_result``; passing that wrapper to the default
    classifier/formatter hides ``found:false`` and internal enum codes. We
    concatenate text blocks, parse JSON when possible, and otherwise return
    the text. Non-text blocks are left untouched for the generic formatter.
    """
    if not isinstance(raw, list):
        return raw
    if not raw:
        return ""
    if not all(
        isinstance(block, dict) and block.get("type") == "text" for block in raw
    ):
        return raw

    text = "".join(str(block.get("text") or "") for block in raw)
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


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
