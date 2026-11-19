"""Offline tests for the ch09 evidence-confidence calibration job.

``main`` is exercised with fake ``analyze``/``retrieve_hybrid`` functions and
fake settings so the tests never touch Milvus, MySQL or an LLM.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.jobs import calibrate_confidence as job
from app.rag.query_understanding import QueryAnalysis
from app.rag.retrieval import RetrievalStrategy, RetrievedChunk


# --- threshold selection -----------------------------------------------------


def test_youden_picks_separating_threshold():
    scored = [(0.9, True), (0.8, True), (0.2, False), (0.1, False)]
    out = job.select_threshold(scored)
    assert 0.2 < out["threshold"] <= 0.8
    assert out["youden_j"] == 1.0
    assert out["tpr"] == 1.0
    assert out["fpr"] == 0.0


def test_tie_break_prefers_higher_precision():
    # Two thresholds share J=0.375: TP=2/FP=1 (precision 2/3) and
    # TP=3/FP=3 (precision 1/2).  The higher-precision point wins.
    scored = [
        (0.9, True),
        (0.8, False),
        (0.7, True),
        (0.6, False),
        (0.55, False),
        (0.5, True),
        (0.4, False),
        (0.3, False),
        (0.2, False),
        (0.1, False),
        (0.05, True),
        (0.04, False),
    ]
    out = job.select_threshold(scored)
    assert out["youden_j"] == pytest.approx(0.375)
    assert out["precision"] == pytest.approx(2 / 3)
    assert out["threshold"] > 0.6


def test_tie_break_then_prefers_closer_to_half():
    scored = [(0.9, True), (0.8, True), (0.2, False), (0.1, False)]
    out = job.select_threshold(scored)
    assert out["youden_j"] == 1.0
    assert out["threshold"] == pytest.approx(0.5)


def test_select_threshold_requires_both_labels():
    with pytest.raises(ValueError, match="positive and negative"):
        job.select_threshold([(0.9, True)])
    with pytest.raises(ValueError, match="no scored"):
        job.select_threshold([])


# --- main job ----------------------------------------------------------------


def _chunk(score: float, key: bool = False) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=int(score * 100),
        score=score,
        questions="q",
        answer="a",
        section_path="s",
        category="商品",
        content_type="faq",
        is_key_clause=key,
    )


def _write_cases(path):
    cases = [
        {"id": "P1", "bucket": "A_policy", "query": "正1", "answerable": True},
        {"id": "P2", "bucket": "E_multi", "query": "正2", "answerable": True},
        {"id": "D1", "bucket": "D_absent", "query": "负1", "answerable": False},
        {"id": "X1", "bucket": "A_policy", "query": "跳过", "answerable": True, "excluded": True},
        {"id": "R1", "bucket": "B_model", "query": "失败", "answerable": True},
        {"id": "R2", "bucket": "C_colloquial", "query": "报错", "answerable": True},
    ]
    path.write_text(json.dumps({"cases": cases}, ensure_ascii=False), encoding="utf-8")


def _fake_settings(tmp_path):
    return SimpleNamespace(
        ch09_report_dir=str(tmp_path / "reports"),
        evidence_valid_floor=0.1,
        evidence_valid_cap=3,
        evidence_margin_scale=0.2,
    )


def test_main_writes_reports_and_excludes_bad_runs(monkeypatch, tmp_path):
    cases_path = tmp_path / "cases.json"
    _write_cases(cases_path)
    analyze_calls: list[str] = []

    def fake_analyze(query):
        analyze_calls.append(query)
        return QueryAnalysis(query, query, [], "llm")

    def fake_retrieve(query, **kwargs):
        assert kwargs["strategy"] is RetrievalStrategy.HYBRID_RERANK
        assert kwargs["top_k"] == 10
        assert kwargs["analysis"].original == query
        if query == "报错":
            raise RuntimeError("milvus down")
        if query == "失败":
            return SimpleNamespace(chunks=[_chunk(0.95)], rerank_failed=True)
        score = {"正1": 0.9, "正2": 0.8, "负1": 0.2}[query]
        return SimpleNamespace(chunks=[_chunk(score)], rerank_failed=False)

    monkeypatch.setattr(job, "get_settings", lambda: _fake_settings(tmp_path))
    monkeypatch.setattr(job, "analyze", fake_analyze)
    monkeypatch.setattr(job, "retrieve_hybrid", fake_retrieve)

    code = job.main(
        ["--cases", str(cases_path), "--out-dir", str(tmp_path / "reports")]
    )

    assert code == 0
    # One analyze per attempted (non-excluded) case, reused for retrieval.
    assert analyze_calls == ["正1", "正2", "负1", "失败", "报错"]

    out_dir = tmp_path / "reports"
    txt = (out_dir / "confidence_calibration.txt").read_text(encoding="utf-8")
    report = json.loads(
        (out_dir / "confidence_calibration.json").read_text(encoding="utf-8")
    )
    assert "confidence_calibration" in txt
    assert report["samples"]["dataset_size"] == 6
    assert report["samples"]["dataset_excluded"] == 1
    assert report["samples"]["evaluable"] == 3
    assert report["samples"]["positives"] == 2
    assert report["samples"]["negatives"] == 1
    assert report["samples"]["excluded"] == {
        "rerank_failed": 1,
        "retrieval_error": 1,
    }
    assert len(report["curve"]) == 101
    assert report["recommended"]["threshold"] == pytest.approx(0.5)
    assert report["recommended"]["youden_j"] == 1.0
    assert report["formula"]["weights"]["top1"] == 0.5
    assert report["formula"]["valid_floor"] == 0.1
    assert report["formula"]["valid_cap"] == 3
    assert report["formula"]["margin_scale"] == 0.2
    assert report["formula"]["top_k"] == 10
    assert len(report["scored"]) == 3
    assert "0.50" in txt
