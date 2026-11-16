"""Deterministic router for pure knowledge questions.

The router only decides whether to force ``query_faq`` on the **initial**
model call. It is intentionally lexical and conservative:

* any order/logistics/handoff action term wins (the turn may need another
  tool), so mixed questions keep automatic tool choice and the model can
  combine tools;
* otherwise a knowledge term marks the turn as knowledge-only.

No model call is made here, and the router never guesses categories or
filters.
"""

from __future__ import annotations

KNOWLEDGE_TERMS: tuple[str, ...] = (
    "退货",
    "退款",
    "运费",
    "邮费",
    "快递费",
    "发票",
    "保修",
    "换货",
    "维修",
    "上门取件",
    "发货",
    "配送",
    "型号",
    "色温",
    "功率",
    "尺码",
    "政策",
    "流程",
    "多久",
    "能不能",
    "支持",
)

ORDER_ACTION_TERMS: tuple[str, ...] = (
    "订单号",
    "订单",
    "物流单号",
    "快递单号",
    "到哪",
    "进度",
    "催单",
    "运单",
    # Handoff/ticket and logistics actions must always win over generic
    # knowledge words such as 能不能/支持, otherwise forced query_faq would
    # lock out create_ticket/query_logistics for the turn.
    "人工",
    "转人工",
    "人工客服",
    "投诉",
    "工单",
    "催",
    "物流",
    "包裹",
    "派送",
    "签收",
    "退回件",
)


def is_pure_knowledge_question(message: str) -> bool:
    """Return True when *message* should force a ``query_faq`` first call.

    Order/logistics action terms always win, so ``"我的订单到哪了"`` and
    mixed questions are routed to automatic tool choice.
    """
    if any(term in message for term in ORDER_ACTION_TERMS):
        return False
    return any(term in message for term in KNOWLEDGE_TERMS)
