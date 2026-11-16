import json
from pathlib import Path

from app.jobs.eval_prompts import (
    score_rewrite_sample,
    score_self_check_sample,
)
from app.rag.query_understanding import QueryAnalysis
from app.rag.self_check import EvidenceAssessment

REWRITE_SAMPLES = (
    Path(__file__).parent / "eval" / "query_rewrite_samples.json"
)
SELF_CHECK_SAMPLES = (
    Path(__file__).parent / "eval" / "self_check_samples.json"
)


def test_eval_prompt_samples_are_valid_json():
    for path in (REWRITE_SAMPLES, SELF_CHECK_SAMPLES):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["samples"]


def test_rewrite_samples_schema_and_count():
    data = json.loads(REWRITE_SAMPLES.read_text(encoding="utf-8"))
    samples = data["samples"]
    assert len(samples) == 10
    for sample in samples:
        assert isinstance(sample["query"], str) and sample["query"]
        assert isinstance(sample["expect_rewritten_any"], list)
        assert isinstance(sample["expect_keywords_any"], list)
        assert sample["expect_rewritten_any"] or sample["expect_keywords_any"]


def test_self_check_samples_balance_and_required_queries():
    data = json.loads(SELF_CHECK_SAMPLES.read_text(encoding="utf-8"))
    samples = data["samples"]
    assert len(samples) == 10
    assert sum(1 for s in samples if s["expect_useful"]) == 5
    assert sum(1 for s in samples if not s["expect_useful"]) == 5
    queries = " ".join(s["query"] for s in samples)
    assert "支持以旧换新吗" in queries
    assert "MH-Z900" in queries and "猫砂容量" in queries
    for sample in samples:
        assert isinstance(sample["sources"], list)
        assert isinstance(sample["expect_useful"], bool)
        for source in sample["sources"]:
            assert "n" in source
            assert "section_path" in source
            assert "answer" in source


def test_score_rewrite_sample_accepts_matching_output():
    sample = {
        "query": "邮费多少",
        "expect_rewritten_any": ["运费"],
        "expect_keywords_any": ["运费", "配送费"],
    }
    analysis = QueryAnalysis(
        original="邮费多少",
        rewritten="运费怎么计算",
        keywords=["运费", "配送费"],
        rewritten_by="llm",
    )
    assert score_rewrite_sample(sample, analysis)


def test_score_rewrite_sample_rejects_missing_rewrite_or_keywords():
    analysis = QueryAnalysis(
        original="邮费多少",
        rewritten="邮费多少",
        keywords=["运费"],
        rewritten_by="fallback",
    )
    assert not score_rewrite_sample(
        {"query": "邮费多少", "expect_rewritten_any": ["运费"], "expect_keywords_any": []},
        analysis,
    )
    assert not score_rewrite_sample(
        {"query": "邮费多少", "expect_rewritten_any": [], "expect_keywords_any": ["运费"]},
        QueryAnalysis(
            original="邮费多少",
            rewritten="运费怎么计算",
            keywords=["邮费"],
            rewritten_by="llm",
        ),
    )


def test_score_self_check_sample_matches_expectation():
    sample = {"query": "支持以旧换新吗", "sources": [], "expect_useful": False}
    assert score_self_check_sample(
        sample, EvidenceAssessment(useful=False, reason="证据未提以旧换新")
    )
    assert not score_self_check_sample(
        sample, EvidenceAssessment(useful=True, reason="")
    )


def test_main_returns_2_for_unreadable_samples(tmp_path):
    from app.jobs.eval_prompts import main

    missing = tmp_path / "missing.json"
    assert (
        main(["--suite", "rewrite", "--samples", str(missing)]) == 2
    )
