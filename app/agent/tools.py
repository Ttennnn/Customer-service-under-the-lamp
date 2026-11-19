"""Agent 子图可用的工具集合与控制工具。

工具绑定由注册中心驱动:``tools_for_route`` 根据 ``ToolSpec`` 的
``exposed_to_agent`` / ``routes`` / ``mcp_server`` 元数据选出本轮可用的
工具。通用路由绑定所有显式暴露且适用于该路由的内置工具(``query_faq``
由父图负责检索,不绑定);退款路由只绑定退款/转人工/建工单控制工具以及
售后 MCP 工具。``legacy_tools`` 保留旧调用方需要的三个只读查询工具。

``suggest_handoff`` 与 ``submit_refund`` 都是纯记录的控制工具:它们只
校验参数并返回确认,不查库、不写库、不调用其他工具;真正的状态写入由
子图的 ``tools`` 节点拦截完成。``submit_refund`` 只表示「准备退款表单」,
不会自动提交退款。
"""

from collections.abc import Sequence

from langchain_core.tools import BaseTool

from app.tools.builtin.refunds import suggest_handoff, submit_refund
from app.tools.spec import ToolSpec

HANDOFF_LABELS: dict[str, str] = {"transfer": "转人工", "ticket": "建工单"}

_AFTER_SALES_SERVER = "aftersales"
_REFUND_ROUTE_TOOLS: tuple[str, ...] = (
    "submit_refund",
    "suggest_handoff",
    "create_ticket",
)
_LEGACY_TOOL_NAMES: tuple[str, ...] = (
    "query_order",
    "query_product",
    "query_faq",
)

__all__ = [
    "HANDOFF_LABELS",
    "legacy_tools",
    "submit_refund",
    "suggest_handoff",
    "suggested_actions_from_handoff",
    "suggested_actions_from_refund",
    "tools_for_route",
]


def tools_for_route(route: str, specs: dict[str, ToolSpec]) -> list[BaseTool]:
    """Return the tools bound for ``route`` from the registry *specs*.

    ``refund_flow`` gets exactly the refund/handoff/ticket control tools plus
    every aftersales MCP tool. Other routes get every spec that is exposed to
    the agent and either route-agnostic or scoped to the route; aftersales MCP
    tools are reserved for the refund flow. Registry insertion order is kept
    so the bound tool list is deterministic. A fresh list is returned so
    callers cannot mutate the registry.
    """
    if route == "refund_flow":
        selected: list[BaseTool] = []
        for name in _REFUND_ROUTE_TOOLS:
            spec = specs.get(name)
            if spec is not None:
                selected.append(spec.tool)
        selected.extend(
            spec.tool
            for spec in specs.values()
            if spec.mcp_server == _AFTER_SALES_SERVER
        )
        return selected

    return [
        spec.tool
        for spec in specs.values()
        if spec.exposed_to_agent
        and (spec.routes is None or route in spec.routes)
        and spec.mcp_server != _AFTER_SALES_SERVER
    ]


def legacy_tools(specs: dict[str, ToolSpec]) -> list[BaseTool]:
    """Return the three legacy read tools in their historical order."""
    return [specs[name].tool for name in _LEGACY_TOOL_NAMES]


def suggested_actions_from_refund(order_id: str) -> list[dict]:
    """Map a validated refund request to the front-end refund form action."""
    return [
        {
            "action": "refund_form",
            "label": "提交退款工单",
            "order_id": order_id,
        }
    ]


def suggested_actions_from_handoff(
    options: Sequence[str],
    ticket_type: str = "咨询",
) -> list[dict]:
    """Map handoff options to front-end actions, deduped and order-preserving.

    Unknown options are dropped. ``ticket_type`` is only attached to the
    ``ticket`` action; ``transfer`` and ``ticket`` are independent so both
    can be suggested in one turn.
    """
    actions: list[dict] = []
    seen: set[str] = set()
    for option in options:
        if option in seen or option not in HANDOFF_LABELS:
            continue
        seen.add(option)
        if option == "transfer":
            actions.append({"action": "transfer", "label": HANDOFF_LABELS[option]})
        else:
            actions.append(
                {
                    "action": "ticket",
                    "label": HANDOFF_LABELS[option],
                    "ticket_type": ticket_type,
                }
            )
    return actions
