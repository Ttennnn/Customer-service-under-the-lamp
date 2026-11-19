"""Agent 子图可用的工具集合与控制工具。

通用路由只绑定三个业务查询工具和 ``suggest_handoff``;退款路由只绑定
``submit_refund`` 和 ``suggest_handoff``。``query_faq`` 由父图负责知识
检索后以证据形式注入,``create_ticket`` 永不绑定给 Agent。

``suggest_handoff`` 与 ``submit_refund`` 都是纯记录的控制工具:它们只
校验参数并返回确认,不查库、不写库、不调用其他工具;真正的状态写入由
子图的 ``tools`` 节点拦截完成。``submit_refund`` 只表示「准备退款表单」,
不会自动提交退款。
"""

from collections.abc import Sequence

from langchain_core.tools import BaseTool

from app.tools.builtin.orders import query_order, query_product
from app.tools.builtin.refunds import suggest_handoff, submit_refund

HANDOFF_LABELS: dict[str, str] = {"transfer": "转人工", "ticket": "建工单"}

__all__ = [
    "AGENT_TOOLS",
    "AGENT_TOOLS_BY_NAME",
    "BUSINESS_TOOL_NAMES",
    "GENERAL_AGENT_TOOLS",
    "HANDOFF_LABELS",
    "REFUND_AGENT_TOOLS",
    "submit_refund",
    "suggest_handoff",
    "suggested_actions_from_handoff",
    "suggested_actions_from_refund",
    "tools_for_route",
]


GENERAL_AGENT_TOOLS: list[BaseTool] = [
    query_order,
    query_product,
    suggest_handoff,
]

REFUND_AGENT_TOOLS: list[BaseTool] = [
    submit_refund,
    suggest_handoff,
]

# 兼容 ch05 的旧名字:默认仍指通用工具集(独立列表,避免别名被改写)。
AGENT_TOOLS: list[BaseTool] = list(GENERAL_AGENT_TOOLS)

AGENT_TOOLS_BY_NAME: dict[str, BaseTool] = {
    t.name: t for t in [*GENERAL_AGENT_TOOLS, *REFUND_AGENT_TOOLS]
}

BUSINESS_TOOL_NAMES: frozenset[str] = frozenset(
    {"query_order", "query_product"}
)


def tools_for_route(route: str) -> list[BaseTool]:
    """Return the tool subset bound for ``route``.

    ``refund_flow`` gets only the refund control tool plus handoff; every
    other route keeps the business tools plus handoff. A fresh list is
    returned so callers cannot mutate the shared registries.
    """
    if route == "refund_flow":
        return list(REFUND_AGENT_TOOLS)
    return list(GENERAL_AGENT_TOOLS)


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
