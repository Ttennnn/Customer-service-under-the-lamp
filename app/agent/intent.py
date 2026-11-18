"""简单 JSON 意图分类器(第 5 章)。

七类意图:物流、订单、商品咨询、退款退货、售后、投诉、闲聊。分类失败或
JSON 解析异常一律兜底 ``售后``,绝不猜 ``投诉`` / ``闲聊``。
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.agent.state import Intent

logger = logging.getLogger(__name__)


class IntentResult(BaseModel):
    """意图分类结果;``reason`` 只用于日志与排查。"""

    intent: Intent
    reason: str = ""


INTENT_SYSTEM_PROMPT = """你是电商客服系统的意图分类器。请把用户消息归入以下七类之一。

类别定义与样例:
- 物流:查询包裹位置、配送进度、快递单号。例:「我的快递到哪了」「怎么还没发货」
- 订单:查询订单状态、订单内容、取消订单。例:「订单 1001 到哪一步了」「帮我看看我的订单」
- 商品咨询:询问商品参数、库存、价格、型号推荐。例:「这款耳机支持降噪吗」「有什么颜色的」
- 退款退货:询问退款进度、退货流程、退款政策。例:「我要退款」「退货怎么寄回去」
- 售后:商品质量、使用、维修、安装等售后问题。例:「收到的东西坏了」「怎么保修」
- 投诉:明确表达不满、要求追责或投诉。例:「我要投诉你们」「服务太差了,给我个说法」
- 闲聊:与购物无关的寒暄或闲聊。例:「你好」「今天天气不错」

判定规则:
- 优先按用户的核心诉求归类,同一个消息只选一个类别。
- 不确定时归「售后」,不要猜「投诉」或「闲聊」。
- 只输出 JSON,不要输出任何解释或 Markdown。

输出格式(JSON):
{"intent": "<七类之一>", "reason": "<一句话理由>"}
"""


def classify_intent(llm, message: str) -> IntentResult:
    """调用 LLM 分类意图;任何异常都兜底 ``售后``。"""
    try:
        structured = llm.with_structured_output(IntentResult, method="json_mode")
        return structured.invoke(
            [
                SystemMessage(content=INTENT_SYSTEM_PROMPT),
                HumanMessage(content=message),
            ]
        )
    except Exception as exc:
        logger.warning("ch05 classify_intent fallback to 售后: %s", exc)
        return IntentResult(intent="售后", reason=f"fallback: {exc}")
