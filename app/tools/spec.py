"""工具规格定义。

``ToolSpec`` 是工具的统一定义:把 LangChain 工具对象与其权限、来源、
超时、展示名等元数据放在一起,供注册中心、执行器和 Agent 统一消费。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from langchain_core.tools import BaseTool

Permission = Literal["read", "write"]
ToolSource = Literal["builtin", "mcp"]


@dataclass(frozen=True)
class ToolSpec:
    """A registered tool plus the metadata the rest of the system needs."""

    name: str
    description: str
    json_schema: dict
    tool: BaseTool
    permission: Permission
    source: ToolSource
    mcp_server: str | None = None
    timeout: float | None = None
    inject_conversation: bool = False
    format_result: Callable | None = None
    classify: Callable | None = None
    exposed_to_agent: bool = True
    routes: frozenset[str] | None = None
    label: str | None = None


def spec_from_tool(
    tool: BaseTool,
    *,
    permission: Permission = "read",
    source: ToolSource = "builtin",
    mcp_server: str | None = None,
    timeout: float | None = None,
    inject_conversation: bool = False,
    exposed_to_agent: bool = True,
    routes: frozenset[str] | None = None,
    label: str | None = None,
    classify: Callable | None = None,
    format_result: Callable | None = None,
) -> ToolSpec:
    """Build a :class:`ToolSpec` from a LangChain tool.

    Name, description and JSON schema come from the tool itself; everything
    else is explicit metadata.
    """
    return ToolSpec(
        name=tool.name,
        description=tool.description,
        json_schema=tool.args_schema.model_json_schema(),
        tool=tool,
        permission=permission,
        source=source,
        mcp_server=mcp_server,
        timeout=timeout,
        inject_conversation=inject_conversation,
        format_result=format_result,
        classify=classify,
        exposed_to_agent=exposed_to_agent,
        routes=routes,
        label=label,
    )
