"""Validate the hand-curated retrieval eval set against the real corpus.

The eval set is data, not code: every ``ground_truth.section_paths`` value
must come from ``split_document(load_document(path))`` so Recall/MRR can be
computed by section path later.  No Milvus/MySQL/LLM is involved.
"""

import json
from collections import Counter
from pathlib import Path

from app.rag.docs import load_documents
from app.rag.splitter import split_document

EVAL_PATH = Path("tests/eval/rag_cases.json")
KNOWLEDGE_DIR = Path("knowledge")
BUCKETS = {"A_policy", "B_model", "C_colloquial", "D_unanswerable", "E_multi"}
DIFFICULTIES = {"easy", "medium", "hard"}
ALLOWED_FILTERS = ({"category": "商品"}, {"content_type": "manual"})


def _load_cases() -> list[dict]:
    data = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    return data["cases"]


def _corpus_section_paths() -> set[str]:
    docs = load_documents(KNOWLEDGE_DIR)
    return {chunk.section_path for doc in docs for chunk in split_document(doc)}


def test_eval_cases_reference_real_section_paths():
    paths = _corpus_section_paths()
    cases = _load_cases()
    assert len(cases) == 35
    assert len({c["id"] for c in cases}) == 35
    assert {c["bucket"] for c in cases} == BUCKETS
    for c in cases:
        assert c["difficulty"] in DIFFICULTIES
        assert c["answerable"] is (c["bucket"] != "D_unanswerable")
        for p in c.get("ground_truth", {}).get("section_paths", []):
            assert p in paths, (c["id"], p)


def test_eval_buckets_are_balanced_and_filters_are_valid():
    cases = _load_cases()
    assert Counter(c["bucket"] for c in cases) == {bucket: 7 for bucket in BUCKETS}
    for bucket in BUCKETS:
        difficulties = {c["difficulty"] for c in cases if c["bucket"] == bucket}
        assert difficulties == DIFFICULTIES, (bucket, difficulties)
    for c in cases:
        if c["bucket"] == "D_unanswerable":
            assert c["ground_truth"].get("section_paths", []) == []
    filtered = [c for c in cases if c.get("filters")]
    assert len(filtered) >= 3
    for c in filtered:
        assert c["filters"] in ALLOWED_FILTERS, c["id"]
