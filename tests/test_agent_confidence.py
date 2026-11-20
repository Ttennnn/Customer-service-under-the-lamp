import pytest

from app.agent.confidence import (
    EVIDENCE_WEIGHTS,
    assess_confidence,
    compute_evidence_confidence,
)
from app.rag.retrieval import RetrievedChunk


def chunk(score, key_clause=False):
    return RetrievedChunk(
        chunk_id=1, score=score, questions="q", answer="a",
        section_path="s", category="商品", content_type="faq",
        is_key_clause=key_clause,
    )


def hit(score, key=False):
    return {"score": score, "is_key_clause": key}


def test_empty_is_a_knowledge_gap():
    out = assess_confidence([], 0.3)
    assert (out.passed, out.reason, out.top_score) == (False, "empty", None)


def test_below_threshold_fails():
    out = assess_confidence([chunk(0.29)], 0.3)
    assert (out.passed, out.reason) == (False, "below_threshold")
    assert out.top_score == 0.29


def test_equal_threshold_passes():
    out = assess_confidence([chunk(0.3)], 0.3)
    assert (out.passed, out.reason) == (True, "ok")


def test_rerank_failure_is_not_a_knowledge_gap():
    out = assess_confidence([chunk(0.02)], 0.3, rerank_failed=True)
    assert (out.passed, out.reason) == (False, "rerank_failed")


# --- evidence confidence pure function ---------------------------------------


def test_evidence_empty_is_zero():
    out = compute_evidence_confidence([])
    assert out.score == 0.0
    assert out.signals["valid_count"] == 0
    assert out.signals["key_clause_hit"] is False


def test_evidence_key_clause_adds_weight():
    plain = compute_evidence_confidence([hit(0.8), hit(0.5)])
    keyed = compute_evidence_confidence([hit(0.8, True), hit(0.5)])
    assert keyed.score > plain.score
    assert keyed.signals["key_clause_hit"] is True
    assert keyed.score - plain.score == pytest.approx(
        EVIDENCE_WEIGHTS["key_clause"]
    )


def test_evidence_single_hit_margin_is_top1():
    out = compute_evidence_confidence([hit(0.7)])
    assert out.signals["margin"] == 0.7
    assert out.signals["margin_norm"] == pytest.approx(1.0)


def test_evidence_accepts_retrieved_chunks():
    out = compute_evidence_confidence([chunk(0.9, True), chunk(0.4)])
    assert out.signals["top1_score"] == 0.9
    assert out.signals["valid_count"] == 2
    assert out.signals["key_clause_hit"] is True


def test_evidence_valid_count_caps_and_margin_normalizes():
    # Five valid hits: raw valid_count stays 5, but the signal is capped at 3.
    out = compute_evidence_confidence([hit(0.5)] * 5)
    assert out.signals["valid_count"] == 5
    assert out.signals["valid_count_norm"] == pytest.approx(1.0)
    # Identical scores mean zero margin.
    assert out.signals["margin"] == pytest.approx(0.0)
    assert out.signals["margin_norm"] == pytest.approx(0.0)


def test_evidence_score_uses_fixed_weights_and_clamps_to_one():
    out = compute_evidence_confidence(
        [hit(1.4, True), hit(0.1), hit(0.05)]
    )
    expected = (
        EVIDENCE_WEIGHTS["top1"] * 1.0
        + EVIDENCE_WEIGHTS["valid_count"] * (2 / 3)
        + EVIDENCE_WEIGHTS["margin"] * 1.0
        + EVIDENCE_WEIGHTS["key_clause"]
    )
    assert out.signals["top1_score"] == 1.0
    assert out.score == pytest.approx(expected)
    assert 0.0 <= out.score <= 1.0


def test_evidence_signals_include_threshold_constants():
    out = compute_evidence_confidence(
        [hit(0.8)], valid_floor=0.2, valid_cap=4, margin_scale=0.5
    )
    assert out.signals["valid_floor"] == 0.2
    assert out.signals["valid_cap"] == 4
    assert out.signals["margin_scale"] == 0.5
    assert out.signals["weights"] == EVIDENCE_WEIGHTS
