"""Semantic retrieval over the knowledge base.

``retrieve`` embeds one query with BGE-M3, searches the Milvus ``knowledge``
collection, drops hits below the score threshold, and then loads the matching
``knowledge_chunks`` rows from MySQL while preserving the Milvus hit order.

An id present in Milvus can be missing from MySQL: a knowledge rebuild deletes
rows first (``reset_*``) and only afterwards cleans up vectors, so a search can
race that window.  Such ids are skipped rather than raising.

The caller may inject ``session`` / ``embedder`` / ``store`` (tests do).  Only
an internally opened session is closed; an injected one belongs to the caller.
``embedder`` and ``store`` resolve lazily from the settings-backed singletons.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import get_settings
from app.db import session as db_session
from app.models import KnowledgeChunk
from app.rag.embedding import get_embedder
from app.rag.milvus_store import get_vector_store


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    score: float
    questions: str
    answer: str


def retrieve(
    keyword: str,
    *,
    session=None,
    embedder=None,
    store=None,
    top_k: int | None = None,
    threshold: float | None = None,
) -> list[RetrievedChunk]:
    """Return knowledge chunks semantically matching *keyword*.

    Hits with ``score < threshold`` are dropped.  The returned chunks follow
    the Milvus hit order (highest score first) and skip ids absent from MySQL.
    """
    if top_k is None or threshold is None:
        settings = get_settings()
        if top_k is None:
            top_k = settings.faq_top_k
        if threshold is None:
            threshold = settings.faq_score_threshold

    if embedder is None:
        embedder = get_embedder()
    if store is None:
        store = get_vector_store()

    hits = [
        hit for hit in store.search(embedder.embed_one(keyword), top_k)
        if hit.score >= threshold
    ]
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
            RetrievedChunk(
                chunk_id=hit.chunk_id,
                score=hit.score,
                questions=by_id[hit.chunk_id].questions,
                answer=by_id[hit.chunk_id].answer,
            )
            for hit in hits
            if hit.chunk_id in by_id
        ]
    finally:
        if own_session:
            session.close()
