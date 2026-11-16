"""query_faq citation/format/filter tests (Task 6)."""

import pytest

from app.rag.query_understanding import QueryAnalysis
from app.rag.retrieval import RetrievalResult, RetrievalStrategy, RetrievedChunk
from app.tools.context import CitationCollector, current_citation_collector
from app.tools.faq import ERROR_TEXT, MISS_TEXT, query_faq

FIXED_ANALYSIS = QueryAnalysis("q", "标准问法", ["关键词"], "llm")


def make_chunk(chunk_id: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        score=1.0 - chunk_id / 100,
        questions=f"问题{chunk_id}",
        answer=f"答案{chunk_id}",
        section_path=f"退货政策 > 小节{chunk_id}",
        category="商品",
        content_type="faq",
    )


def _result(chunks: list[RetrievedChunk]) -> RetrievalResult:
    return RetrievalResult(
        chunks, RetrievalStrategy.HYBRID_RERANK, FIXED_ANALYSIS
    )


@pytest.fixture(autouse=True)
def _reset_collector():
    """Keep a collector set by one test from leaking into the next one."""
    yield
    current_citation_collector.set(None)


def test_query_faq_returns_cited_evidence(monkeypatch):
    monkeypatch.setattr(
        "app.tools.faq.retrieve_hybrid",
        lambda *a, **k: _result([make_chunk(1), make_chunk(2)]),
    )
    out = query_faq.func("完全不相关的问题")
    assert "[1] 章节:" in out and "[2] 章节:" in out
    assert "完全不相关的问题" not in out  # evidence, not the raw query echo


def test_query_faq_passes_filters(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "app.tools.faq.retrieve_hybrid",
        lambda q, **k: calls.append(k) or _result([]),
    )
    query_faq.func("色温", category="商品", content_type="manual")
    assert calls[0] == {"category": "商品", "content_type": "manual"}


def test_query_faq_miss_sets_faq_called(monkeypatch):
    collector = CitationCollector()
    current_citation_collector.set(collector)
    monkeypatch.setattr(
        "app.tools.faq.retrieve_hybrid", lambda *a, **k: _result([])
    )
    assert query_faq.func("不存在") == MISS_TEXT
    assert collector.faq_called is True and collector.payload() == []


def test_query_faq_infrastructure_error_returns_error_text(monkeypatch):
    collector = CitationCollector()
    current_citation_collector.set(collector)

    def boom(*a, **k):
        raise RuntimeError("milvus down")

    monkeypatch.setattr("app.tools.faq.retrieve_hybrid", boom)
    assert query_faq.func("邮费") == ERROR_TEXT
    assert collector.faq_called is True and collector.payload() == []


def test_query_faq_does_not_reset_numbering_per_call(monkeypatch):
    collector = CitationCollector()
    current_citation_collector.set(collector)
    results = iter([_result([make_chunk(1)]), _result([make_chunk(2)])])
    monkeypatch.setattr(
        "app.tools.faq.retrieve_hybrid", lambda *a, **k: next(results)
    )

    assert "[1] 章节:" in query_faq.func("第一次")
    assert "[2] 章节:" in query_faq.func("第二次")
    assert [s["chunk_id"] for s in collector.payload()] == [1, 2]


def test_query_faq_orders_and_numbers_through_collector(monkeypatch):
    collector = CitationCollector()
    current_citation_collector.set(collector)
    chunks = [make_chunk(i) for i in range(5)]
    monkeypatch.setattr(
        "app.tools.faq.retrieve_hybrid", lambda *a, **k: _result(chunks)
    )

    out = query_faq.func("灯")

    # order_for_prompt -> [0, 2, 4, 3, 1], numbered 1..5 in that order.
    assert [s["chunk_id"] for s in collector.payload()] == [0, 2, 4, 3, 1]
    assert [s["n"] for s in collector.payload()] == [1, 2, 3, 4, 5]
    assert out.index("问题0") < out.index("问题2") < out.index("问题4")
    assert out.index("问题4") < out.index("问题3") < out.index("问题1")
