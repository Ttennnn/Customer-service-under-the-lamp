from types import SimpleNamespace

import pytest
from sqlalchemy.orm import sessionmaker

from app.models import KnowledgeChunk
from app.rag.milvus_store import SearchHit
from app.rag.query_understanding import QueryAnalysis
from app.rag.rerank import RerankError, RerankHit
from app.rag.retrieval import (
    RetrievalStrategy,
    RetrievedChunk,
    build_filter_expr,
    retrieve,
    retrieve_hybrid,
)

FIXED_ANALYSIS = QueryAnalysis("q", "标准问法", ["关键词"], "llm")


class FakeEmbedder:
    def __init__(self):
        self.texts = []

    def embed_one(self, text):
        self.texts.append(text)
        return [0.1] * 1024


class FakeStore:
    """Dense-only store mirroring the legacy ch03 interface."""

    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def dense_search(self, vector, top_k, expr=None):
        self.calls.append((vector, top_k))
        return self.hits


class FakeHybridStore:
    def __init__(self, dense=None, bm25=None, hybrid=None):
        self.dense_hits = list(dense or [])
        self.bm25_hits = list(bm25 or [])
        self.hybrid_hits = list(hybrid or [])
        self.dense_calls = []
        self.bm25_calls = []
        self.hybrid_calls = []

    def dense_search(self, vector, top_k, expr=None):
        self.dense_calls.append((vector, top_k, expr))
        return self.dense_hits

    def bm25_search(self, query_text, top_k, expr=None):
        self.bm25_calls.append((query_text, top_k, expr))
        return self.bm25_hits

    def hybrid_search(self, vector, query_text, top_k, expr=None, rrf_k=60):
        # expr is at index 1 so tests can assert the filter contract directly.
        self.hybrid_calls.append((query_text, expr, top_k, rrf_k, vector))
        return self.hybrid_hits


class FakeReranker:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def rerank(self, query, documents, top_n=None):
        self.calls.append((query, documents, top_n))
        return self.hits[:top_n] if top_n is not None else self.hits


class BoomReranker:
    def rerank(self, query, documents, top_n=None):
        raise RerankError("boom")


def _row(session, q, a, section_path=None):
    c = KnowledgeChunk(
        category="售后",
        questions=q,
        answer=a,
        section_path=section_path,
        content_type="faq",
    )
    session.add(c)
    session.commit()
    session.refresh(c)
    return c


class _SpySession:
    def __init__(self, inner):
        self._inner = inner
        self.closed = False

    def query(self, *args, **kwargs):
        return self._inner.query(*args, **kwargs)

    def close(self):
        self.closed = True
        self._inner.close()


# --- legacy retrieve() compatibility wrapper ---------------------------------


def test_retrieve_orders_by_score_and_filters(session):
    a = _row(session, "问A", "答A")
    b = _row(session, "问B", "答B")
    hits = [SearchHit(b.id, 0.9), SearchHit(a.id, 0.2)]
    out = retrieve(
        "邮费",
        session=session,
        embedder=FakeEmbedder(),
        store=FakeStore(hits),
        threshold=0.5,
    )
    assert [(r.chunk_id, r.answer) for r in out] == [(b.id, "答B")]
    assert out[0] == RetrievedChunk(
        b.id, 0.9, "问B", "答B", None, "售后", "faq"
    )


def test_retrieve_empty(session):
    assert (
        retrieve(
            "x",
            session=session,
            embedder=FakeEmbedder(),
            store=FakeStore([]),
            threshold=0.5,
        )
        == []
    )


def test_retrieve_skips_ids_missing_from_mysql(session):
    a = _row(session, "问A", "答A")
    hits = [SearchHit(a.id, 0.9), SearchHit(99999, 0.8)]
    out = retrieve(
        "x",
        session=session,
        embedder=FakeEmbedder(),
        store=FakeStore(hits),
        threshold=0.1,
    )
    assert [(r.chunk_id, r.answer) for r in out] == [(a.id, "答A")]


def test_retrieve_uses_settings_defaults(session):
    row = _row(session, "问A", "答A")
    store = FakeStore([SearchHit(row.id, 0.5), SearchHit(row.id, 0.34)])
    retrieve(
        "x",
        session=session,
        embedder=FakeEmbedder(),
        store=store,
        threshold=0.5,
    )
    assert store.calls[0][1] == 3


def test_retrieve_opens_and_closes_its_own_session(session, monkeypatch):
    row = _row(session, "问A", "答A")
    factory = sessionmaker(
        bind=session.get_bind(), autoflush=False, expire_on_commit=False
    )
    spies = []

    def make():
        spy = _SpySession(factory())
        spies.append(spy)
        return spy

    monkeypatch.setattr(
        "app.rag.retrieval.db_session.get_session_factory", lambda: make
    )
    out = retrieve(
        "x",
        embedder=FakeEmbedder(),
        store=FakeStore([SearchHit(row.id, 0.9)]),
        threshold=0.1,
    )
    assert [r.answer for r in out] == ["答A"]
    assert len(spies) == 1 and spies[0].closed is True


def test_retrieve_does_not_close_injected_session(session):
    row = _row(session, "问A", "答A")
    spy = _SpySession(session)
    out = retrieve(
        "x",
        session=spy,
        embedder=FakeEmbedder(),
        store=FakeStore([SearchHit(row.id, 0.9)]),
        threshold=0.1,
    )
    assert [r.answer for r in out] == ["答A"]
    assert spy.closed is False


def test_retrieve_resolves_embedder_and_store_from_module(monkeypatch, session):
    row = _row(session, "问A", "答A")
    calls = []

    def fake_embedder():
        calls.append("embedder")
        return FakeEmbedder()

    def fake_store():
        calls.append("store")
        return FakeStore([SearchHit(row.id, 0.9)])

    monkeypatch.setattr("app.rag.retrieval.get_embedder", fake_embedder)
    monkeypatch.setattr("app.rag.retrieval.get_vector_store", fake_store)
    monkeypatch.setattr(
        "app.rag.retrieval.get_settings",
        lambda: SimpleNamespace(faq_top_k=3, faq_score_threshold=0.5),
    )
    out = retrieve("x", session=session, threshold=0.5)
    assert [r.answer for r in out] == ["答A"]
    assert calls == ["embedder", "store"]


def test_retrieve_without_threshold_uses_hybrid_rerank(session, monkeypatch):
    row = _row(session, "问A", "答A")
    store = FakeHybridStore(hybrid=[SearchHit(row.id, 0.9)])
    reranker = FakeReranker([RerankHit(0, 0.7)])
    monkeypatch.setattr(
        "app.rag.retrieval.analyze", lambda query: FIXED_ANALYSIS
    )
    monkeypatch.setattr("app.rag.retrieval.get_reranker", lambda: reranker)

    out = retrieve("x", session=session, embedder=FakeEmbedder(), store=store)

    assert [r.chunk_id for r in out] == [row.id]
    assert store.hybrid_calls and not store.dense_calls


# --- four strategies ---------------------------------------------------------


def test_dense_strategy_only_calls_dense_search(session):
    row = _row(session, "问", "答")
    store = FakeHybridStore(dense=[SearchHit(row.id, 0.8)])
    result = retrieve_hybrid(
        "q",
        strategy=RetrievalStrategy.DENSE,
        session=session,
        embedder=FakeEmbedder(),
        store=store,
        analysis=FIXED_ANALYSIS,
    )
    assert [c.chunk_id for c in result.chunks] == [row.id]
    assert store.dense_calls and not store.bm25_calls and not store.hybrid_calls
    assert result.strategy is RetrievalStrategy.DENSE


def test_bm25_strategy_uses_expanded_query(session):
    row = _row(session, "问", "答")
    store = FakeHybridStore(bm25=[SearchHit(row.id, 1.2)])
    retrieve_hybrid(
        "邮费",
        strategy=RetrievalStrategy.BM25,
        session=session,
        embedder=FakeEmbedder(),
        store=store,
        analysis=QueryAnalysis(
            "邮费", "运费怎么算", ["运费", "配送费"], "llm"
        ),
    )
    assert "运费" in store.bm25_calls[0][0]
    assert store.bm25_calls[0][0] == "运费怎么算 运费 配送费"


def test_bm25_merges_static_expansions_even_when_llm_returns_one(session):
    row = _row(session, "问", "答")
    store = FakeHybridStore(bm25=[SearchHit(row.id, 1.2)])
    retrieve_hybrid(
        "邮费咋算",
        strategy=RetrievalStrategy.BM25,
        session=session,
        store=store,
        analysis=QueryAnalysis("邮费咋算", "运费怎么计算", ["运费"], "llm"),
    )
    # LLM supplied 运费 but missed 配送费; the static expansion still lands
    # in the BM25 query, and duplicates are removed order-preservingly.
    assert store.bm25_calls[0][0] == "运费怎么计算 运费 配送费"


def test_hybrid_strategy_calls_hybrid_search_with_rrf(session):
    row = _row(session, "问", "答")
    store = FakeHybridStore(hybrid=[SearchHit(row.id, 0.03)])
    result = retrieve_hybrid(
        "q",
        strategy=RetrievalStrategy.HYBRID,
        session=session,
        embedder=FakeEmbedder(),
        store=store,
        analysis=FIXED_ANALYSIS,
    )
    assert [c.chunk_id for c in result.chunks] == [row.id]
    assert result.rerank_failed is False
    assert store.hybrid_calls[0][3] == 60


def test_hybrid_rerank_sorts_by_rerank_score_and_flags_failure(session):
    a, b = _row(session, "q1", "a1"), _row(session, "q2", "a2")
    store = FakeHybridStore(hybrid=[SearchHit(a.id, 0.9), SearchHit(b.id, 0.8)])
    reranker = FakeReranker([RerankHit(1, 0.99), RerankHit(0, 0.1)])
    result = retrieve_hybrid(
        "q",
        strategy=RetrievalStrategy.HYBRID_RERANK,
        session=session,
        embedder=FakeEmbedder(),
        store=store,
        reranker=reranker,
        analysis=FIXED_ANALYSIS,
    )
    assert [c.chunk_id for c in result.chunks] == [b.id, a.id]
    assert [c.score for c in result.chunks] == [0.99, 0.1]
    assert result.rerank_failed is False


def test_hybrid_rerank_failure_falls_back_and_sets_flag(session):
    row = _row(session, "q", "a")
    store = FakeHybridStore(hybrid=[SearchHit(row.id, 0.9)])
    result = retrieve_hybrid(
        "q",
        strategy=RetrievalStrategy.HYBRID_RERANK,
        session=session,
        embedder=FakeEmbedder(),
        store=store,
        reranker=BoomReranker(),
        analysis=FIXED_ANALYSIS,
    )
    assert result.rerank_failed is True
    assert [c.chunk_id for c in result.chunks] == [row.id]


def test_hybrid_rerank_receives_documents_and_top_n(session):
    row = _row(session, "问", "答", section_path="第1节")
    store = FakeHybridStore(hybrid=[SearchHit(row.id, 0.9)])
    reranker = FakeReranker([RerankHit(0, 0.5)])
    result = retrieve_hybrid(
        "q",
        strategy=RetrievalStrategy.HYBRID_RERANK,
        session=session,
        embedder=FakeEmbedder(),
        store=store,
        reranker=reranker,
        analysis=FIXED_ANALYSIS,
        top_k=3,
    )
    query, documents, top_n = reranker.calls[0]
    assert query == FIXED_ANALYSIS.rewritten
    assert documents == ["第1节\n问\n答"]
    assert top_n == 3
    assert result.chunks[0].score == 0.5


def test_hybrid_rerank_skips_ids_missing_from_mysql(session):
    row = _row(session, "q", "a")
    store = FakeHybridStore(
        hybrid=[SearchHit(99999, 0.9), SearchHit(row.id, 0.8)]
    )
    reranker = FakeReranker([RerankHit(0, 0.7)])

    result = retrieve_hybrid(
        "q",
        strategy=RetrievalStrategy.HYBRID_RERANK,
        session=session,
        embedder=FakeEmbedder(),
        store=store,
        reranker=reranker,
        analysis=FIXED_ANALYSIS,
    )

    assert [c.chunk_id for c in result.chunks] == [row.id]
    assert len(reranker.calls[0][1]) == 1


def test_empty_candidates_do_not_resolve_reranker(session, monkeypatch):
    def boom():
        raise AssertionError("reranker must not be resolved")

    monkeypatch.setattr("app.rag.retrieval.get_reranker", boom)
    result = retrieve_hybrid(
        "q",
        session=session,
        embedder=FakeEmbedder(),
        store=FakeHybridStore(),
        analysis=FIXED_ANALYSIS,
    )
    assert result.chunks == [] and result.rerank_failed is False


def test_recall_top_k_is_passed_and_top_k_cuts(session):
    rows = [_row(session, f"q{i}", f"a{i}") for i in range(3)]
    store = FakeHybridStore(
        hybrid=[SearchHit(row.id, 1.0 - index) for index, row in enumerate(rows)]
    )
    result = retrieve_hybrid(
        "q",
        strategy=RetrievalStrategy.HYBRID,
        session=session,
        embedder=FakeEmbedder(),
        store=store,
        analysis=FIXED_ANALYSIS,
        recall_top_k=7,
        top_k=2,
    )
    assert store.hybrid_calls[0][2] == 7
    assert len(result.chunks) == 2


# --- filters and analysis ----------------------------------------------------


def test_filter_expr_is_passed_to_both_branches(session):
    row = _row(session, "q", "a")
    store = FakeHybridStore(hybrid=[SearchHit(row.id, 0.9)])
    retrieve_hybrid(
        "q",
        strategy=RetrievalStrategy.HYBRID,
        session=session,
        embedder=FakeEmbedder(),
        store=store,
        category="商品",
        content_type="manual",
        analysis=FIXED_ANALYSIS,
    )
    expr = store.hybrid_calls[0][1]
    assert 'category == "商品"' in expr and 'content_type == "manual"' in expr


def test_filter_expr_reaches_dense_and_bm25(session):
    row = _row(session, "q", "a")
    store = FakeHybridStore(
        dense=[SearchHit(row.id, 0.8)], bm25=[SearchHit(row.id, 1.1)]
    )
    retrieve_hybrid(
        "q",
        strategy=RetrievalStrategy.DENSE,
        session=session,
        embedder=FakeEmbedder(),
        store=store,
        category="商品",
        content_type="manual",
        analysis=FIXED_ANALYSIS,
    )
    retrieve_hybrid(
        "q",
        strategy=RetrievalStrategy.BM25,
        session=session,
        store=store,
        category="商品",
        content_type="manual",
        analysis=FIXED_ANALYSIS,
    )
    assert 'category == "商品"' in store.dense_calls[0][2]
    assert 'content_type == "manual"' in store.bm25_calls[0][2]


def test_no_filters_means_no_expr(session):
    assert build_filter_expr(None, None) is None
    assert build_filter_expr("", None) == 'category == ""'


def test_filter_values_are_escaped():
    assert build_filter_expr('a"b\\c', None) == 'category == "a\\"b\\\\c"'


def test_injected_analysis_skips_analyzer(session):
    row = _row(session, "q", "a")

    def boom(_):
        raise AssertionError("analyzer must not be called")

    retrieve_hybrid(
        "q",
        session=session,
        embedder=FakeEmbedder(),
        store=FakeHybridStore(dense=[SearchHit(row.id, 0.9)]),
        analyzer=boom,
        analysis=FIXED_ANALYSIS,
    )


def test_analyzer_is_used_when_analysis_missing(session):
    row = _row(session, "q", "a")
    store = FakeHybridStore(bm25=[SearchHit(row.id, 1.2)])
    calls = []

    def analyzer(query):
        calls.append(query)
        return QueryAnalysis(query, "改写后", ["词"], "llm")

    result = retrieve_hybrid(
        "原问",
        strategy=RetrievalStrategy.BM25,
        session=session,
        store=store,
        analyzer=analyzer,
    )
    assert calls == ["原问"]
    assert result.analysis.rewritten == "改写后"
    assert store.bm25_calls[0][0] == "改写后 词"


def test_disabled_query_rewrite_skips_analyzer_and_uses_fallback(
    session, monkeypatch
):
    row = _row(session, "q", "a")
    store = FakeHybridStore(bm25=[SearchHit(row.id, 1.2)])
    calls = []

    def boom(query):
        calls.append(query)
        raise AssertionError("analyzer must not be called when rewrite is off")

    monkeypatch.setattr(
        "app.rag.retrieval.get_settings",
        lambda: SimpleNamespace(
            query_rewrite_enabled=False,
            hybrid_recall_top_k=50,
            rerank_top_n=10,
            hybrid_rrf_k=60,
        ),
    )

    result = retrieve_hybrid(
        "邮费咋算",
        strategy=RetrievalStrategy.BM25,
        session=session,
        store=store,
        analyzer=boom,
    )

    assert calls == []
    assert result.analysis.original == "邮费咋算"
    assert result.analysis.rewritten == "邮费咋算"
    assert result.analysis.keywords == ["运费", "配送费"]
    assert result.analysis.rewritten_by == "fallback"
    assert store.bm25_calls[0][0] == "邮费咋算 运费 配送费"


def test_injected_analysis_wins_even_when_rewrite_disabled(session, monkeypatch):
    row = _row(session, "q", "a")
    store = FakeHybridStore(bm25=[SearchHit(row.id, 1.2)])
    monkeypatch.setattr(
        "app.rag.retrieval.get_settings",
        lambda: SimpleNamespace(
            query_rewrite_enabled=False,
            hybrid_recall_top_k=50,
            rerank_top_n=10,
            hybrid_rrf_k=60,
        ),
    )

    result = retrieve_hybrid(
        "q",
        strategy=RetrievalStrategy.BM25,
        session=session,
        store=store,
        analysis=FIXED_ANALYSIS,
    )

    assert result.analysis is FIXED_ANALYSIS


def test_default_strategy_is_hybrid_rerank(session):
    row = _row(session, "q", "a")
    store = FakeHybridStore(hybrid=[SearchHit(row.id, 0.9)])
    reranker = FakeReranker([RerankHit(0, 0.4)])
    result = retrieve_hybrid(
        "q",
        session=session,
        embedder=FakeEmbedder(),
        store=store,
        reranker=reranker,
        analysis=FIXED_ANALYSIS,
    )
    assert result.strategy is RetrievalStrategy.HYBRID_RERANK


def test_infrastructure_error_is_not_swallowed(session):
    class DownStore:
        def dense_search(self, *args, **kwargs):
            raise RuntimeError("milvus down")

    with pytest.raises(RuntimeError, match="milvus down"):
        retrieve_hybrid(
            "q",
            strategy=RetrievalStrategy.DENSE,
            session=session,
            embedder=FakeEmbedder(),
            store=DownStore(),
            analysis=FIXED_ANALYSIS,
        )
