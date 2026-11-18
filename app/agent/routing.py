"""硬编码的意图 → 出口映射(不调模型决定路由)。"""

import logging

from app.agent.state import Intent, Route

logger = logging.getLogger(__name__)

INTENT_ROUTES: dict[Intent, Route] = {
    "商品咨询": "knowledge",
    "退款退货": "refund_flow",
    "售后": "refund_flow",
    "物流": "business",
    "订单": "business",
    "投诉": "escalate",
    "闲聊": "fallback_script",
    "其他": "fallback_script",
}


def route_for_intent(intent: str) -> Route:
    """返回意图对应的出口;未知意图兜底 ``business`` 并记 warning。"""
    route = INTENT_ROUTES.get(intent)  # type: ignore[arg-type]
    if route is None:
        logger.warning("ch06 unknown intent %r, falling back to business", intent)
        return "business"
    return route
