"""九类意图分类器(第 6 章)。

四件套:九类枚举成选择题、强制 JSON ``{intent, confidence}``、边界
few-shot、``其他`` 兜底。分类失败、越界或空输出一律回退
``IntentResult(intent="其他", confidence=0.0)``,绝不猜业务类别;评估脚本
传 ``raise_on_error=True`` 时改为抛出,以便把「真实其他」和网络/解析失败
区分开。
"""

import logging
import math
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from pydantic import BaseModel

from app.agent.state import Intent
from app.services.context import render_recent_dialogue

logger = logging.getLogger(__name__)

#: 九类意图,顺序与 ``app.agent.state.Intent`` 一致。
INTENT_LABELS: tuple[str, ...] = (
    "物流",
    "订单",
    "商品咨询",
    "退款退货",
    "售后",
    "投诉",
    "人工",
    "闲聊",
    "其他",
)


class IntentResult(BaseModel):
    """意图分类结果:单标签加 0–1 置信度。"""

    intent: Intent
    confidence: float = 0.0


INTENT_SYSTEM_PROMPT = """你是电商客服系统的意图分类器。请把用户消息归入以下九类之一,并给出 0 到 1 之间的 confidence。

类别定义(选择题,每类都写了该归与不该归):
- 物流:包裹位置、配送进度、快递单号;不包括订单内容、退款政策。
- 订单:订单状态、订单内容、取消订单;不包括物流轨迹、商品参数。
- 商品咨询:商品参数、库存、价格、型号推荐;不包括已购买后的质量问题。
- 退款退货:退款进度、退货流程、退款/退货政策、明确要求退钱或退货;不包括单纯的质量维修。
- 售后:商品质量、使用、维修、安装、补发等售后处理;不包括退款/退货流程本身。
- 投诉:明确表达不满、要求追责、找主管、曝光;带情绪的退款诉求,如果核心是退款,仍归退款退货。
- 人工:明确要求创建工单、转人工客服、人工跟进;不包括仅表达不满的投诉。
- 闲聊:寒暄、无关闲聊、问机器人身份。
- 其他:与购物/售后无关、语义不明、无法归入以上任何一类;拿不准就选它。

边界样例(注意为什么不是另一类):
- 「这个还能退吗」→ 退款退货(不是商品咨询)。
- 「刚收到就坏了，怎么保修」→ 售后(不是退款退货)。
- 「我要投诉，顺便把钱退了」→ 投诉(核心是投诉;若只是抱怨且核心诉求是退款,则归退款退货)。
- 「你好，在吗」→ 闲聊(不是订单)。
- 「帮我写一首诗」→ 其他(不是闲聊/商品咨询)。
- 「麻烦帮我开张工单，让售后的同事继续处理」→ 人工(不是售后)。
- 「我不想跟机器人聊，帮我安排真人客服」→ 人工(不是投诉)。

兜底规则:
- 语义不明或无法归入业务类时选「其他」,不要硬塞。
- 同一条消息只选一个类别。

输出要求:
- JSON 字段只有 intent 和 confidence;confidence 是 0 到 1 之间的小数。
- 不要理由字段,不要解释,不要 Markdown。
- 只输出 JSON。

输出格式(JSON):
{"intent": "<九类之一>", "confidence": 0.0}
"""

_ROLE_MESSAGES = {
    "user": HumanMessage,
    "human": HumanMessage,
    "assistant": AIMessage,
    "ai": AIMessage,
    "system": SystemMessage,
}


def _history_messages(history: Sequence[Any]) -> list[BaseMessage]:
    """把历史条目统一成 LangChain 消息。

    同时接受 ``BaseMessage``、``{"role", "content"}`` 字典(评估集)和
    ``(role, content)`` 二元组(便捷调用);未知角色或空内容跳过。
    """
    messages: list[BaseMessage] = []
    for entry in history:
        if isinstance(entry, BaseMessage):
            messages.append(entry)
            continue
        if isinstance(entry, dict):
            role, content = entry.get("role"), entry.get("content")
        elif isinstance(entry, (tuple, list)) and len(entry) == 2:
            role, content = entry
        else:
            continue
        message_type = _ROLE_MESSAGES.get(str(role).lower())
        if message_type is None or not isinstance(content, str):
            continue
        messages.append(message_type(content=content))
    return messages


def _build_human_message(
    message: str,
    history: Sequence[Any],
    summary: str = "",
) -> HumanMessage:
    """把最近对话与早前摘要渲染进 HumanMessage,让单次调用看到完整上下文。"""
    recent = render_recent_dialogue(_history_messages(history))
    prefix = f"(早前对话摘要: {summary})\n" if summary else ""
    if not recent:
        if not prefix:
            return HumanMessage(content=message)
        return HumanMessage(content=f"{prefix}\n当前用户消息：{message}")
    return HumanMessage(
        content=f"{prefix}最近对话：\n{recent}\n\n当前用户消息：{message}"
    )


def _clamp_confidence(value: Any) -> float:
    """把 confidence 夹到 [0,1];缺失、非数字或 NaN 按 0.0。"""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(confidence):
        return 0.0
    return min(1.0, max(0.0, confidence))


def classify_intent(
    llm,
    message: str,
    history: Sequence[Any] = (),
    *,
    raise_on_error: bool = False,
    summary: str = "",
) -> IntentResult:
    """调用 LLM 分类意图;异常、越界或空输出回退 ``其他``。

    ``raise_on_error=True`` 时把异常原样抛给调用方,供评估脚本区分
    「真实其他」和网络/解析失败;``summary`` 是更早对话的摘要,作为上下文
    前缀传给模型。
    """
    try:
        structured = llm.with_structured_output(IntentResult, method="json_mode")
        result = structured.invoke(
            [
                SystemMessage(content=INTENT_SYSTEM_PROMPT),
                _build_human_message(message, history, summary),
            ]
        )
        intent = getattr(result, "intent", None)
        if intent not in INTENT_LABELS:
            raise ValueError(f"intent out of range: {intent!r}")
        confidence = _clamp_confidence(getattr(result, "confidence", 0.0))
    except Exception as exc:
        if raise_on_error:
            logger.warning("ch06 classify_intent raised for eval: %s", exc)
            raise
        logger.warning("ch06 classify_intent fallback to 其他: %s", exc)
        return IntentResult(intent="其他", confidence=0.0)

    logger.info("ch06 classify_intent intent=%s confidence=%.2f", intent, confidence)
    return IntentResult(intent=intent, confidence=confidence)
