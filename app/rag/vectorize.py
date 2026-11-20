"""Vectorize knowledge chunks into Milvus, idempotently.

The flow is a resumable two-phase write: MySQL rows are created ``pending``
first, and only after the matching Milvus upsert succeeds are they marked
``done`` with their ``vector_id`` backfilled.  A failed or interrupted run
therefore leaves unprocessed rows ``pending``; rerunning the function picks
them up, and re-upserting the same ``chunk_{id}`` primary key is idempotent.

``vectorize_pending`` handles the incremental path; ``vectorize_rows`` is the
scoped variant that publishes only caller-supplied rows; ``vectorize_all``
re-embeds every row (including ``done`` ones) to backfill the hybrid
collection's BM25 ``text`` and metadata from the legacy dense-only setup.
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
    _validate_batch_args(batch_size, max_chunks)
    query = (
        session.query(KnowledgeChunk)
        .filter(KnowledgeChunk.vectorize_status == "pending")
        .order_by(KnowledgeChunk.id)
    )
    if max_chunks is not None:
        query = query.limit(max_chunks)
    rows = query.all()
    return _vectorize_rows(rows, session, embedder, store, batch_size)


def vectorize_rows(
    session,
    rows,
    embedder,
    store,
    batch_size: int = 16,
) -> VectorizeReport:
    """Embed and upsert only *rows*, committing once per batch.

    Unlike :func:`vectorize_pending`, the caller decides exactly which rows are
    published.  The approval writeback uses this so vectorizing an approved
    answer can never publish unrelated pending chunks (for example a stale
    answer left behind by a failed earlier approval).  The transaction
    semantics are identical to :func:`vectorize_pending`: each batch is
    committed only after its Milvus upsert succeeds, and any exception rolls
    the current batch back and is re-raised.

    The rows are used as given (no status filter); callers that need the
    incremental path should keep using :func:`vectorize_pending`.
    """
    _validate_batch_args(batch_size, None)
    return _vectorize_rows(list(rows), session, embedder, store, batch_size)


def vectorize_all(
    session,
    embedder,
    store,
    batch_size: int = 16,
    max_chunks: int | None = None,
) -> VectorizeReport:
    """Re-embed and upsert *every* chunk, including rows already ``done``.

    This is the full-migration path onto the hybrid collection: the old
    ``knowledge`` vectors were dense-only, so each row is embedded again to
    populate the BM25 ``text`` and metadata fields.  There is deliberately no
    ``vectorize_status`` filter; idempotence comes from the stable
    ``chunk_{id}`` primary key, not from the status column.  Rows are
    selected in primary-key order and batched exactly like
    :func:`vectorize_pending`.
    """
    _validate_batch_args(batch_size, max_chunks)
    query = session.query(KnowledgeChunk).order_by(KnowledgeChunk.id)
    if max_chunks is not None:
        query = query.limit(max_chunks)
    rows = query.all()
    return _vectorize_rows(rows, session, embedder, store, batch_size)


def _validate_batch_args(batch_size: int, max_chunks: int | None) -> None:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if max_chunks is not None and max_chunks < 0:
        raise ValueError("max_chunks must not be negative")


def _vectorize_rows(
    rows,
    session,
    embedder,
    store,
    batch_size: int,
) -> VectorizeReport:
    """Embed, upsert, and mark *rows* done one batch at a time.

    The transaction boundary lives here: each batch is committed only after
    its Milvus upsert succeeds, so a failure rolls back just the batch that
    failed and is re-raised.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

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
                    VectorItem(
                        vector_id_for(row.id),
                        row.id,
                        vector,
                        text=text,
                        category=row.category,
                        content_type=row.content_type,
                        section_path=row.section_path,
                    )
                    for row, vector, text in zip(batch, vectors, texts)
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
