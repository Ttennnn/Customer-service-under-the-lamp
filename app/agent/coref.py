"""指代消解与 Query 改写(第 6 章)。

一次 LLM 调用把「它能退吗」这类依赖上下文的问法补全成不依赖上下文也能
看懂的完整问题;问题已完整时原样透传。失败或空输出一律回退原文。
"""

import logging
from collections.abc import Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from app.services.context import render_recent_dialogue

logger = logging.getLogger(__name__)


class CorefRewrite(BaseModel):
    """指代消解结果;只保留一个能独立看懂的问题。"""

    resolved_query: str


COREF_SYSTEM_PROMPT = """你是电商客服系统的指代消解与 Query 改写器。请把用户当前这句话改写成一句不依赖上下文也能看懂的完整问题。

规则:
1. 把指代词(它、这个、那个、该订单、这单、刚才那单、上面的)补成最近对话里明确的对象,例如订单号、商品名、型号。
2. 保留所有实体(订单号、商品名、型号、金额)原样,不要编造或改写。
3. 不要引入最近对话里没有的条件或事实,例如不要凭空补日期、地址、退款原因。
4. 如果问题已经完整、指代已经明确,原样返回,不要为了「更标准」而改写。
5. 口语归一:邮费→运费、退钱→退款、灯不亮→故障/维修、换个新的→换货。
6. 如果无法从最近对话确定指代对象,保留原句,不要猜测。
7. 只输出 JSON:{"resolved_query": "<改写后的完整问题>"},不要解释,不要 Markdown。
"""


def _build_human_message(message: str, history: Sequence[BaseMessage]) -> HumanMessage:
    """把最近对话渲染进 HumanMessage,让单次调用同时看到上下文和当前问题。"""
    recent = render_recent_dialogue(history)
    if not recent:
        return HumanMessage(content=message)
    return HumanMessage(content=f"最近对话：\n{recent}\n\n当前用户消息：{message}")


def resolve_query(
    llm,
    message: str,
    history: Sequence[BaseMessage] = (),
) -> str:
    """把 *message* 结合 *history* 补全成独立问题;异常/空输出回退原文。"""
    try:
        structured = llm.with_structured_output(CorefRewrite, method="json_mode")
        result = structured.invoke(
            [
                SystemMessage(content=COREF_SYSTEM_PROMPT),
                _build_human_message(message, history),
            ]
        )
        resolved = getattr(result, "resolved_query", "") or ""
        resolved = resolved.strip() if isinstance(resolved, str) else ""
    except Exception as exc:
        logger.warning("ch06 resolve_reference fallback to original: %s", exc)
        return message
    if not resolved:
        logger.warning("ch06 resolve_reference blank output, fallback to original")
        return message
    logger.info(
        "ch06 resolve_reference original=%r resolved=%r changed=%s",
        message,
        resolved,
        resolved != message,
    )
    return resolved
