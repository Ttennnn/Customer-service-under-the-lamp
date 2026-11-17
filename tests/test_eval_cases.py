"""Validate the hand-curated retrieval eval baseline against the real corpus.

The eval set is data, not code.  Every non-excluded ``ground_truth.section_paths``
value must come from ``split_document(load_document(path))`` so Recall/MRR can
be computed by section path.  Cases whose expected section or expected points
are not present in the corpus are marked ``excluded`` and are not scored by the
evaluation job.  No Milvus/MySQL/LLM is involved.
"""

import json
import re
from collections import Counter
from pathlib import Path

from app.rag.docs import load_documents
from app.rag.splitter import split_document

EVAL_PATH = Path("tests/eval/rag_cases.json")
KNOWLEDGE_DIR = Path("knowledge")
BUCKETS = {"A_policy", "B_model", "C_colloquial", "D_absent", "E_multi"}
EXCLUSION_REASONS = {"section_missing", "points_missing", "corpus_conflict"}

# A few expected points are paraphrases of two facts that appear separately in
# the corpus.  Keep the alias list tiny and explicit; anything else must match
# the corpus text directly.
POINT_ALIASES = {
    "7天无理由退货": ("7天内", "无理由退货"),
    "不参与包邮": ("偏远地区运费另计",),
}


def _load_cases() -> list[dict]:
    data = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    return data["cases"]


def _section_texts() -> dict[str, str]:
    texts: dict[str, list[str]] = {}
    for doc in load_documents(KNOWLEDGE_DIR):
        for chunk in split_document(doc):
            if chunk.section_path:
                texts.setdefault(chunk.section_path, []).append(chunk.answer)
    return {path: "\n".join(parts) for path, parts in texts.items()}


def _normalized(text: str) -> str:
    return (
        re.sub(r"\s+", "", text)
        .replace("（", "(")
        .replace("）", ")")
    )


def _point_terms(point: str) -> list[str]:
    return [part.strip() for part in re.split(r"[,，、]", point) if part.strip()]


def _point_present(point: str, corpus: str) -> bool:
    if all(_normalized(term) in corpus for term in _point_terms(point)):
        return True
    normalized_point = _normalized(point)
    return any(
        needle in normalized_point
        and all(_normalized(alt) in corpus for alt in alternatives)
        for needle, alternatives in POINT_ALIASES.items()
    )


def test_eval_cases_cover_all_buckets_and_refusal_flags():
    cases = _load_cases()
    assert len(cases) == 300
    assert len({case["id"] for case in cases}) == 300
    assert Counter(case["bucket"] for case in cases) == {
        bucket: 60 for bucket in BUCKETS
    }
    for case in cases:
        assert case["answerable"] is (not case["should_refuse"]), case["id"]
        if case["bucket"] == "D_absent":
            assert case["should_refuse"] is True
            assert case["expect_section"] == ""
            assert case["expect_points"] == []
            assert case["ground_truth"]["section_paths"] == []
        else:
            assert case["should_refuse"] is False
            assert case["expect_section"]
            assert case["expect_points"]
            assert case["ground_truth"]["must_contain"] == case["expect_points"]
        assert isinstance(case["excluded"], bool)
        if case["excluded"]:
            assert case["coverage"] in EXCLUSION_REASONS, case["id"]
            assert case["exclusion_reason"] == case["coverage"]
        else:
            assert case["coverage"] == "covered", case["id"]
            assert case["exclusion_reason"] is None


def test_eval_cases_reference_real_paths_and_covered_points():
    sections = _section_texts()
    corpus = _normalized("\n".join(sections.values()))
    for case in _load_cases():
        for path in case["ground_truth"]["section_paths"]:
            assert path in sections, (case["id"], path)
        if case["excluded"]:
            if case["coverage"] == "section_missing":
                assert case["ground_truth"]["section_paths"] == []
            elif case["coverage"] == "points_missing":
                assert case["missing_points"], case["id"]
            elif case["coverage"] == "corpus_conflict":
                assert case["bucket"] == "D_absent"
                assert case["conflict_section_paths"]
            continue
        if case["bucket"] == "D_absent":
            continue
        assert case["ground_truth"]["section_paths"], case["id"]
        for point in case["expect_points"]:
            assert _point_present(point, corpus), (case["id"], point)
