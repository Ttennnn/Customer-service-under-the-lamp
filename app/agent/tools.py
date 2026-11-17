"""Agent 子图可用的工具集合与控制工具。

Agent 只绑定三个业务查询工具和 ``suggest_handoff`` 控制工具:
``query_faq`` 由父图负责知识检索后以证据形式注入,``create_ticket`` 永不
绑定给 Agent。``suggest_handoff`` 是纯记录工具:它只校验参数并返回确认,
不查库、不调用其他工具;真正的状态写入由子图的 ``tools`` 节点拦截完成。
"""

import json
from collections.abc import Sequence
from typing import Literal

from langchain.tools import tool
from langchain_core.tools import BaseTool

from app.tools.business import query_logistics, query_order, query_product

HANDOFF_LABELS: dict[str, str] = {"transfer": "转人工", "ticket": "建工单"}


@tool
def suggest_handoff(
    options: list[Literal["transfer", "ticket"]],
    ticket_type: Literal["售后", "投诉", "咨询"] = "咨询",
    reason: str = "",
) -> str:
    """建议转人工和/或建工单(仅记录建议,不立即执行)。

    当问题超出知识库范围、需要人工介入,或用户明确要求转人工/投诉时调用。
    本工具不会真正转接或创建工单;子图会把建议收集到 ``suggested_actions``,
    由前端展示「转人工」「建工单」按钮。

    Args:
        options: 建议的动作,``"transfer"`` 表示转人工,``"ticket"`` 表示建工单
        ticket_type: 工单类型,取值「售后」「投诉」「咨询」之一
        reason: 一句话说明为什么需要人工介入或建单
    """
    return json.dumps(
        {
            "acknowledged": True,
            "options": list(options),
            "ticket_type": ticket_type,
            "reason": reason,
        },
        ensure_ascii=False,
    )


AGENT_TOOLS: list[BaseTool] = [
    query_order,
    query_product,
    query_logistics,
    suggest_handoff,
]

AGENT_TOOLS_BY_NAME: dict[str, BaseTool] = {t.name: t for t in AGENT_TOOLS}

BUSINESS_TOOL_NAMES: frozenset[str] = frozenset(
    {"query_order", "query_product", "query_logistics"}
)


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
