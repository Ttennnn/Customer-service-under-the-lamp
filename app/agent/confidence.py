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

# Fixed weights for the ch09 evidence-confidence score.  They are deliberately
# module constants rather than settings: calibration changes the threshold,
# not the shape of the signal.
EVIDENCE_WEIGHTS = {
    "top1": 0.50,
    "valid_count": 0.20,
    "margin": 0.20,
    "key_clause": 0.10,
}


@dataclass(frozen=True)
class EvidenceConfidence:
    """Calibratable confidence score plus the raw signals behind it."""

    score: float
    signals: dict


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


def _hit_score(hit: RetrievedChunk | dict) -> float:
    if isinstance(hit, dict):
        return float(hit.get("score", 0.0) or 0.0)
    return float(hit.score)


def _hit_is_key_clause(hit: RetrievedChunk | dict) -> bool:
    if isinstance(hit, dict):
        return bool(hit.get("is_key_clause", False))
    return bool(hit.is_key_clause)


def compute_evidence_confidence(
    hits: Sequence[RetrievedChunk | dict],
    *,
    valid_floor: float = 0.1,
    valid_cap: int = 3,
    margin_scale: float = 0.2,
) -> EvidenceConfidence:
    """Score retrieved evidence on a calibratable 0-1 scale.

    Supported signals, each normalised to 0-1 before weighting:

    * ``top1``        - best rerank score, clamped to 0-1 (weight 0.50);
    * ``valid_count`` - hits with ``score >= valid_floor``, capped at
      ``valid_cap`` (weight 0.20);
    * ``margin``      - ``top1 - top2``, or ``top1`` when there is only one
      hit, divided by ``margin_scale`` and capped at 1 (weight 0.20);
    * ``key_clause``  - any hit carrying ``is_key_clause=True`` (weight 0.10).

    ``signals`` carries both the raw values and every constant needed to
    interpret the score.  Pure function: no retrieval, DB or LLM calls.
    """
    scores = sorted(
        (_hit_score(hit) for hit in hits), reverse=True
    )
    key_clause_hit = any(_hit_is_key_clause(hit) for hit in hits)

    top1 = scores[0] if scores else 0.0
    top2 = scores[1] if len(scores) > 1 else None
    margin = top1 - top2 if top2 is not None else top1
    valid_count = sum(1 for score in scores if score >= valid_floor)

    top1_norm = min(max(top1, 0.0), 1.0)
    valid_norm = (
        min(valid_count / valid_cap, 1.0) if valid_cap > 0 else 0.0
    )
    margin_denom = margin_scale if margin_scale > 0 else 1.0
    margin_norm = min(max(margin, 0.0) / margin_denom, 1.0)

    score = (
        EVIDENCE_WEIGHTS["top1"] * top1_norm
        + EVIDENCE_WEIGHTS["valid_count"] * valid_norm
        + EVIDENCE_WEIGHTS["margin"] * margin_norm
        + EVIDENCE_WEIGHTS["key_clause"] * (1.0 if key_clause_hit else 0.0)
    )
    # Guard against floating-point drift at the boundaries.
    score = min(max(score, 0.0), 1.0)

    signals = {
        "top1": top1,
        "top1_score": top1_norm,
        "top2": top2,
        "top2_score": top2,
        "valid_count": valid_count,
        "valid_count_norm": valid_norm,
        "margin": margin,
        "margin_norm": margin_norm,
        "key_clause_hit": key_clause_hit,
        "valid_floor": valid_floor,
        "valid_cap": valid_cap,
        "margin_scale": margin_scale,
        "weights": dict(EVIDENCE_WEIGHTS),
    }
    return EvidenceConfidence(score=score, signals=signals)
