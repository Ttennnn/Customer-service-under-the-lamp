from app.agent.confidence import assess_confidence
from app.rag.retrieval import RetrievedChunk


def chunk(score):
    return RetrievedChunk(
        chunk_id=1, score=score, questions="q", answer="a",
        section_path="s", category="商品", content_type="faq",
    )


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
