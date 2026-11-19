"""退款与转人工控制工具(内置插件)。

``suggest_handoff`` 与 ``submit_refund`` 都是纯记录的控制工具:它们只
校验参数并返回确认,不查库、不写库、不调用其他工具;真正的状态写入由
子图的 ``tools`` 节点拦截完成。``submit_refund`` 只表示「准备退款表单」,
不会自动提交退款。
"""

import json
from typing import Literal

from langchain.tools import tool

from app.tools.spec import spec_from_tool


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


@tool
def submit_refund(order_id: str) -> str:
    """准备退款表单(仅记录建议,不提交退款)。

    仅当用户诉求是退款或退货时调用。本工具不会提交退款,也不会写库;
    子图会校验 ``order_id`` 与当前订单一致后,把「提交退款工单」按钮
    收集到 ``suggested_actions``,由用户在前端表单中确认提交。

    Args:
        order_id: 当前申请退款的订单号
    """
    return json.dumps(
        {"acknowledged": True, "order_id": order_id},
        ensure_ascii=False,
    )


SPECS = [
    spec_from_tool(
        submit_refund,
        permission="read",
        routes=frozenset({"refund_flow"}),
    ),
    spec_from_tool(suggest_handoff, permission="read"),
]
