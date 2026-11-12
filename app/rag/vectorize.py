"""Vectorize pending knowledge chunks into Milvus, idempotently.

The flow is a resumable two-phase write: MySQL rows are created ``pending``
first, and only after the matching Milvus upsert succeeds are they marked
``done`` with their ``vector_id`` backfilled.  A failed or interrupted run
therefore leaves unprocessed rows ``pending``; rerunning the function picks
them up, and re-upserting the same ``chunk_{id}`` primary key is idempotent.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.models import KnowledgeChunk
from app.rag.milvus_store import VectorItem

EMBEDDING_DIM = 1024


@dataclass(frozen=True)
class VectorizeReport:
    processed: int
    failed: int


def vectorization_text(chunk: KnowledgeChunk) -> str:
    """Compose the exact text that is embedded for *chunk*."""
    return f"{chunk.category}\n{chunk.questions}\n{chunk.answer}"


def vector_id_for(chunk_id: int) -> str:
    """Milvus primary key for a chunk; stable, so upserts are idempotent."""
    return f"chunk_{chunk_id}"


def vectorize_pending(
    session,
    embedder,
    store,
    batch_size: int = 16,
    max_chunks: int | None = None,
) -> VectorizeReport:
    """Embed and upsert every ``pending`` chunk, committing once per batch.

    Rows are selected in primary-key order.  On any exception the current
    batch is rolled back (leaving its rows ``pending``) and the exception is
    re-raised; already committed batches stay ``done``.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if max_chunks is not None and max_chunks < 0:
        raise ValueError("max_chunks must not be negative")

    query = (
        session.query(KnowledgeChunk)
        .filter(KnowledgeChunk.vectorize_status == "pending")
        .order_by(KnowledgeChunk.id)
    )
    if max_chunks is not None:
        query = query.limit(max_chunks)
    rows = query.all()

    processed = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        try:
            texts = [vectorization_text(row) for row in batch]
            vectors = embedder.embed(texts)
            if len(vectors) != len(batch):
                raise ValueError(
                    f"embedder returned {len(vectors)} vectors for "
                    f"{len(batch)} texts"
                )
            for vector in vectors:
                if len(vector) != EMBEDDING_DIM:
                    raise ValueError(
                        f"embedding dimension mismatch: expected {EMBEDDING_DIM}, "
                        f"got {len(vector)}"
                    )

            # Milvus first: a row is only marked done after this returns.
            store.upsert(
                [
                    VectorItem(vector_id_for(row.id), row.id, vector)
                    for row, vector in zip(batch, vectors)
                ]
            )
            for row in batch:
                row.vector_id = vector_id_for(row.id)
                row.vectorize_status = "done"
            session.commit()
        except Exception:
            session.rollback()
            raise
        processed += len(batch)

    # Any failure re-raises above, so a returned report means zero failures.
    return VectorizeReport(processed=processed, failed=0)
