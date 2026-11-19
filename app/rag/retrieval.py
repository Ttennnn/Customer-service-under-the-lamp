"""Multi-strategy retrieval over the hybrid knowledge base.

Four strategies share one entry point, :func:`retrieve_hybrid`:

* ``dense``   - cosine similarity over the BGE-M3 dense vector.
* ``bm25``    - Milvus native BM25 over the ``text`` field.
* ``hybrid``  - dense + BM25 Top-N fused with Reciprocal Rank Fusion (RRF).
* ``hybrid_rerank`` - hybrid candidates re-scored by the BGE reranker.

Scores are **not comparable across strategies**: dense reports a cosine
similarity, BM25 an unbounded lexical score, ``hybrid`` an RRF fused score
built from ranks, and ``hybrid_rerank`` the reranker's relevance score.
Within one strategy the chunks are returned best-first.

Filtering is exact-match on ``category`` / ``content_type`` and comes only
from the explicit arguments; query understanding never guesses filter values.

An id present in Milvus can be missing from MySQL: a knowledge rebuild deletes
rows first and only afterwards cleans up vectors, so a search can race that
window.  Such ids are skipped rather than raising.  Infrastructure failures
(Milvus/embedding) propagate to the caller; only reranker failures are
degraded, because the hybrid RRF order is still a usable result.

:func:`retrieve` is the ch03 compatibility wrapper.  With an explicit
``threshold`` it keeps the old dense-only Top-K + score cutoff contract; with
no threshold it delegates to ``retrieve_hybrid(strategy=hybrid_rerank)``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from enum import Enum

from app.config import get_settings
from app.db import session as db_session
from app.models import KnowledgeChunk
from app.rag.embedding import get_embedder
from app.rag.milvus_store import get_vector_store
from app.rag.query_understanding import QueryAnalysis, analyze, expand_synonyms
from app.rag.rerank import RerankError, get_reranker

logger = logging.getLogger(__name__)


class RetrievalStrategy(str, Enum):
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    score: float
    questions: str
    answer: str
    section_path: str | None
    category: str
    content_type: str | None
    # Placed last so existing positional constructors keep working.
    is_key_clause: bool = False


@dataclass(frozen=True)
class RetrievalResult:
    """Chunks plus the strategy/analysis that produced them.

    ``rerank_failed`` is only ever true for ``hybrid_rerank``: it means the
    reranker was unreachable and ``chunks`` follow the hybrid RRF order
    instead.  Callers must surface this flag (logs/eval reports) rather than
    treating the fallback as a successful rerank.
    """

    chunks: list[RetrievedChunk]
    strategy: RetrievalStrategy
    analysis: QueryAnalysis
    rerank_failed: bool = False


def _quote(value: str) -> str:
    """Return *value* as a safely quoted Milvus string literal."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def build_filter_expr(
    category: str | None = None, content_type: str | None = None
) -> str | None:
    """Build an exact-match Milvus filter expression.

    Only explicit arguments are used.  ``None`` means "no filter"; an empty
    string is a real value and yields an expression that matches nothing.
    """
    clauses: list[str] = []
    if category is not None:
        clauses.append(f"category == {_quote(category)}")
    if content_type is not None:
        clauses.append(f"content_type == {_quote(content_type)}")
    return " and ".join(clauses) or None


def _dedupe(tokens: list[str]) -> list[str]:
    """Drop empty/duplicate tokens while preserving first-seen order."""
    unique: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token and token not in seen:
            seen.add(token)
            unique.append(token)
    return unique


def _as_chunk(row: KnowledgeChunk, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=row.id,
        score=score,
        questions=row.questions,
        answer=row.answer,
        section_path=row.section_path,
        category=row.category,
        content_type=row.content_type,
        is_key_clause=bool(row.is_key_clause),
    )


def _fetch_chunks(hits, session) -> list[RetrievedChunk]:
    """Load MySQL rows for *hits*, preserving hit order and skipping misses."""
    if not hits:
        return []

    own_session = session is None
    if own_session:
        session = db_session.get_session_factory()()
    try:
        ids = [hit.chunk_id for hit in hits]
        rows = (
            session.query(KnowledgeChunk)
            .filter(KnowledgeChunk.id.in_(ids))
            .all()
        )
        by_id = {row.id: row for row in rows}
        return [
            _as_chunk(by_id[hit.chunk_id], hit.score)
            for hit in hits
            if hit.chunk_id in by_id
        ]
    finally:
        if own_session:
            session.close()


def _rerank_chunks(
    analysis: QueryAnalysis,
    candidates: list[RetrievedChunk],
    reranker,
    top_k: int,
) -> tuple[list[RetrievedChunk], bool]:
    """Rerank *candidates*; on any reranker failure keep their order."""
    if not candidates:
        return [], False

    documents = [
        f"{chunk.section_path or ''}\n{chunk.questions}\n{chunk.answer}"
        for chunk in candidates
    ]
    try:
        if reranker is None:
            reranker = get_reranker()
        reranked = reranker.rerank(analysis.rewritten, documents, top_n=top_k)
    except RerankError:
        logger.exception(
            "rerank failed; falling back to hybrid RRF order for query %r",
            analysis.original,
        )
        return candidates[:top_k], True

    chunks: list[RetrievedChunk] = []
    seen: set[int] = set()
    for hit in reranked:
        if not 0 <= hit.index < len(candidates):
            logger.warning(
                "reranker returned out-of-range index %s for %s candidates",
                hit.index,
                len(candidates),
            )
            continue
        chunk = candidates[hit.index]
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        chunks.append(replace(chunk, score=hit.score))
    return chunks[:top_k], False


def retrieve_hybrid(
    query: str,
    *,
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_RERANK,
    session=None,
    embedder=None,
    store=None,
    reranker=None,
    analyzer=None,
    analysis: QueryAnalysis | None = None,
    category: str | None = None,
    content_type: str | None = None,
    recall_top_k: int | None = None,
    top_k: int | None = None,
) -> RetrievalResult:
    """Run one of the four retrieval strategies and return ranked chunks.

    ``analysis`` takes precedence over ``analyzer``; when neither is supplied
    :func:`analyze` runs the LLM rewrite (with its deterministic fallback),
    unless ``settings.query_rewrite_enabled`` is false, in which case the
    deterministic analysis is built directly without the LLM.  Recall defaults
    to ``settings.hybrid_recall_top_k`` and the final cut to
    ``settings.effective_rerank_top_k``.

    Scores carry the strategy's own semantics and must not be compared across
    strategies (see the module docstring).  ``rerank_failed=True`` means the
    reranker raised :class:`RerankError` and the chunks are the hybrid RRF
    fallback, not reranked results.
    """
    strategy = RetrievalStrategy(strategy)
    settings = get_settings()
    if recall_top_k is None:
        recall_top_k = settings.hybrid_recall_top_k
    if top_k is None:
        top_k = settings.effective_rerank_top_k

    if analysis is None:
        if settings.query_rewrite_enabled:
            analysis = (analyzer or analyze)(query)
        else:
            # Rewrite disabled: skip the LLM entirely and use the same
            # deterministic analysis :func:`analyze` falls back to.
            analysis = QueryAnalysis(
                original=query,
                rewritten=query,
                keywords=expand_synonyms(query),
                rewritten_by="fallback",
            )

    if store is None:
        store = get_vector_store()

    expr = build_filter_expr(category, content_type)
    # Spec §4/§6: BM25 query = rewritten + LLM keywords + static expansions,
    # even when the LLM already returned some of the synonyms.
    bm25_text = " ".join(
        _dedupe(
            [
                analysis.rewritten,
                *analysis.keywords,
                *expand_synonyms(analysis.original),
            ]
        )
    )

    if strategy is RetrievalStrategy.DENSE:
        if embedder is None:
            embedder = get_embedder()
        hits = store.dense_search(
            embedder.embed_one(analysis.rewritten), recall_top_k, expr
        )
        return RetrievalResult(
            _fetch_chunks(hits[:top_k], session), strategy, analysis
        )

    if strategy is RetrievalStrategy.BM25:
        hits = store.bm25_search(bm25_text, recall_top_k, expr)
        return RetrievalResult(
            _fetch_chunks(hits[:top_k], session), strategy, analysis
        )

    if embedder is None:
        embedder = get_embedder()
    hits = store.hybrid_search(
        embedder.embed_one(analysis.rewritten),
        bm25_text,
        recall_top_k,
        expr,
        settings.hybrid_rrf_k,
    )

    if strategy is RetrievalStrategy.HYBRID:
        return RetrievalResult(
            _fetch_chunks(hits[:top_k], session), strategy, analysis
        )

    # hybrid_rerank: Milvus only stores ids and metadata, so the MySQL texts
    # must be fetched before the reranker can read them.
    candidates = _fetch_chunks(hits, session)
    chunks, rerank_failed = _rerank_chunks(
        analysis, candidates, reranker, top_k
    )
    return RetrievalResult(chunks, strategy, analysis, rerank_failed)


def retrieve(
    keyword: str,
    *,
    session=None,
    embedder=None,
    store=None,
    top_k: int | None = None,
    threshold: float | None = None,
) -> list[RetrievedChunk]:
    """Compatibility wrapper for ch03 callers.

    With an explicit ``threshold`` this keeps the legacy dense-only contract:
    embed *keyword*, take ``top_k`` dense hits, drop ``score < threshold``,
    then backfill MySQL preserving Milvus order.  Without a threshold it
    delegates to the chapter-4 default, ``hybrid_rerank``, and returns its
    chunks.  The returned scores are therefore only comparable between calls
    that used the same path/strategy.
    """
    if threshold is None:
        return retrieve_hybrid(
            keyword,
            strategy=RetrievalStrategy.HYBRID_RERANK,
            session=session,
            embedder=embedder,
            store=store,
            top_k=top_k,
        ).chunks

    if top_k is None:
        top_k = get_settings().faq_top_k
    if embedder is None:
        embedder = get_embedder()
    if store is None:
        store = get_vector_store()

    hits = [
        hit
        for hit in store.dense_search(
            embedder.embed_one(keyword), top_k, None
        )
        if hit.score >= threshold
    ]
    return _fetch_chunks(hits, session)
