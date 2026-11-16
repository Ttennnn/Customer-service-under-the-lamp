"""FAQ hybrid-retrieval tool emitting cited evidence.

The tool runs :func:`retrieve_hybrid` (default ``hybrid_rerank``), reorders the
chunks so the strongest evidence sits at both ends of the prompt, registers
them with the turn's :class:`CitationCollector`, and formats numbered blocks::

    [1] 章节:退货政策 > 退款说明
    问:退款说明
    答:退货验收通过后,退款将在 1-3 个工作日内原路退回……

The collector lives in a contextvar as a **mutable** object, so writes from
the ``execute_tool`` worker thread are visible to the chat orchestrator.
Infrastructure failures are caught and returned as ``ERROR_TEXT`` so a
retrieval problem never breaks the chat round; a real miss returns
``MISS_TEXT``. Both are public constants because ``stream_chat`` (Task 8)
distinguishes "knowledge base unavailable" from "no matching knowledge".
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from langchain.tools import tool

from app.rag.retrieval import RetrievedChunk, retrieve_hybrid
from app.tools.context import CitationCollector, current_citation_collector

logger = logging.getLogger(__name__)

MISS_TEXT = "未找到相关问题,请换个关键词再试试。"
ERROR_TEXT = "知识库暂时不可用,请稍后重试或联系人工客服。"


def order_for_prompt(
    items: Sequence[RetrievedChunk],
) -> list[RetrievedChunk]:
    """Reorder score-descending *items* so the strongest sit first and last.

    Positions are filled alternately from the front with the even ranks
    (1st, 3rd, 5th, ...) and from the back with the odd ranks (2nd, 4th, ...),
    i.e. ``[1, 3, 5, ..., 6, 4, 2]``.  For ``n=5`` this yields ranks
    ``[1, 3, 5, 4, 2]`` and for ``n=10`` ``[1, 3, 5, 7, 9, 10, 8, 6, 4, 2]``
    (1-based).  Citation numbers and the ``sources`` payload follow this
    order; Recall/MRR evaluation keeps the score order.
    """
    ordered = list(items)
    return ordered[0::2] + ordered[1::2][::-1]


@tool
def query_faq(
    keyword: str,
    category: str | None = None,
    content_type: str | None = None,
) -> str:
    """在知识库中语义检索常见问题(FAQ),返回带引用编号的证据。

    结果按「最强在首尾」顺序返回,每条形如 ``[n] 章节:.../问:.../答:...``。
    ``[n]`` 是引用编号,回答时必须保留并引用对应证据;调用方据此把编号映射
    回来源 chunk 的章节路径与原文。

    ``category`` 与 ``content_type`` 是精确匹配过滤,只应传用户明确指定且
    确信取值正确的值(例如用户说「商品」或「手册」);不确定就不要过滤,
    省略该参数,不要自行猜测品类/类型。

    Args:
        keyword: 用于语义检索的问题或关键词
        category: 精确匹配的知识品类,不确定时省略
        content_type: 精确匹配的内容类型,不确定时省略
    """
    collector = current_citation_collector.get()
    if collector is None:
        # Direct call (tests/scripts) without an orchestrator context.
        collector = CitationCollector()
    collector.faq_called = True

    try:
        result = retrieve_hybrid(
            keyword, category=category, content_type=content_type
        )
    except Exception:  # noqa: BLE001 - 工具必须返回字符串,不能打断对话
        logger.exception("query_faq retrieval failed")
        return ERROR_TEXT

    ordered = order_for_prompt(result.chunks)
    if not ordered:
        return MISS_TEXT

    numbers = collector.add(ordered)
    return "\n\n".join(
        f"[{number}] 章节:{chunk.section_path or '未知'}\n"
        f"问:{chunk.questions}\n"
        f"答:{chunk.answer}"
        for number, chunk in zip(numbers, ordered)
    )
