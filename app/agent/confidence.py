"""置信度闸:对已检索的 chunk 做纯判定。

这是「检索之后、进 Agent 之前」的纯判断,不调用重排器、Milvus 或 LLM。
它只区分两类失败:

* 知识缺口(``empty`` / ``below_threshold``)——证据不足,后续可进低置信度池;
* 基础设施故障(``rerank_failed``)——重排器不可用时返回的是 RRF 分数,
  与重排分不可比,因此必须先于分数阈值判断,且不得当作知识缺口落池。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.rag.retrieval import RetrievedChunk


@dataclass(frozen=True)
class ConfidenceVerdict:
    passed: bool
    top_score: float | None
    reason: str  # "ok" | "empty" | "below_threshold" | "rerank_failed"


def assess_confidence(
    chunks: Sequence[RetrievedChunk],
    min_score: float,
    *,
    rerank_failed: bool = False,
) -> ConfidenceVerdict:
    """判断检索证据是否足够进入主力 Agent。

    ``rerank_failed`` 优先于一切阈值判断:此时 ``chunks`` 的分数是 RRF
    融合分,不能与重排阈值比较。``top_score`` 仍会报告以便日志排查。
    """
    top_score = max((c.score for c in chunks), default=None)

    if rerank_failed:
        return ConfidenceVerdict(passed=False, top_score=top_score, reason="rerank_failed")

    if top_score is None:
        return ConfidenceVerdict(passed=False, top_score=None, reason="empty")

    if top_score >= min_score:
        return ConfidenceVerdict(passed=True, top_score=top_score, reason="ok")

    return ConfidenceVerdict(passed=False, top_score=top_score, reason="below_threshold")
