"""Ingest document chunks and mined QA into ``knowledge_chunks``, safely.

Document chunks are identified by a content fingerprint, so rebuilding the
same document reuses existing rows instead of duplicating them.  Rows are
inserted ``pending``; vectorization is a separate, resumable step.

The two reset helpers enforce the data-safety boundary for rebuilds: Milvus
vectors are deleted *before* any MySQL row, so a failed vector deletion never
leaves orphaned vectors behind.  ``reset_document_domain`` deliberately keeps
mined QA rows; only ``reset_all`` clears them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from sqlalchemy import or_

from app.models import KnowledgeChunk
from app.rag.splitter import Chunk

SECTION_PATH_MAX = 512
CATEGORY_MAX = 255
DELETE_BATCH_SIZE = 500
MINED_CONTENT_TYPE = "mined_qa"
_PATH_SEP = " > "


@dataclass(frozen=True)
class IngestReport:
    inserted: int
    reused: int


@dataclass(frozen=True)
class MinedQA:
    question: str
    answer: str
    category: str


def chunk_fingerprint(chunk: Chunk) -> tuple[str, str, str, str, str]:
    """Return the identity tuple that decides whether a chunk already exists."""
    return (
        chunk.content_type or "",
        chunk.section_path or "",
        chunk.category,
        chunk.questions,
        chunk.answer,
    )


def _row_fingerprint(row: KnowledgeChunk) -> tuple[str, str, str, str, str]:
    return (
        row.content_type or "",
        row.section_path or "",
        row.category,
        row.questions,
        row.answer,
    )


def _truncate_section_path(path: str | None) -> str | None:
    """Fit *path* into 512 chars while keeping the final heading segment.

    Leading segments are dropped whole; if even the leaf alone is too long it
    is hard-cut.  This preserves the most specific heading that identifies the
    chunk.
    """
    if path is None or len(path) <= SECTION_PATH_MAX:
        return path
    parts = path.split(_PATH_SEP)
    while len(parts) > 1 and len(_PATH_SEP.join(parts)) > SECTION_PATH_MAX:
        parts.pop(0)
    return _PATH_SEP.join(parts)[:SECTION_PATH_MAX]


def _truncate_category(category: str | None) -> str:
    return (category or "")[:CATEGORY_MAX]


def _normalized(chunk: Chunk) -> Chunk:
    """Return *chunk* with its stored fields fitted to the DDL lengths."""
    return replace(
        chunk,
        category=_truncate_category(chunk.category),
        section_path=_truncate_section_path(chunk.section_path),
    )


def _document_domain_filter():
    return or_(
        KnowledgeChunk.content_type != MINED_CONTENT_TYPE,
        KnowledgeChunk.content_type.is_(None),
    )


def ingest_document_chunks(
    session, chunks: list[Chunk]
) -> tuple[list[KnowledgeChunk], IngestReport]:
    """Insert or reuse one document's chunks, then link neighbours in order.

    Existing document-domain rows are matched by fingerprint; matched rows are
    reused as-is (including their vectorization status), new rows are written
    ``pending``.  ``prev_chunk_id`` / ``next_chunk_id`` follow the input order.
    """
    existing = (
        session.query(KnowledgeChunk).filter(_document_domain_filter()).all()
    )
    by_fingerprint = {_row_fingerprint(row): row for row in existing}

    rows: list[KnowledgeChunk] = []
    inserted = 0
    reused = 0
    for chunk in chunks:
        normalized = _normalized(chunk)
        fingerprint = chunk_fingerprint(normalized)
        row = by_fingerprint.get(fingerprint)
        if row is None:
            row = KnowledgeChunk(
                category=normalized.category,
                questions=normalized.questions,
                answer=normalized.answer,
                section_path=normalized.section_path,
                content_type=normalized.content_type,
                is_key_clause=normalized.is_key_clause,
                vectorize_status="pending",
            )
            session.add(row)
            by_fingerprint[fingerprint] = row
            inserted += 1
        else:
            reused += 1
        rows.append(row)

    session.flush()
    for index, row in enumerate(rows):
        row.prev_chunk_id = rows[index - 1].id if index > 0 else None
        row.next_chunk_id = rows[index + 1].id if index + 1 < len(rows) else None
    session.commit()

    return rows, IngestReport(inserted=inserted, reused=reused)


def insert_mined_qa(session, items: list[MinedQA]) -> list[KnowledgeChunk]:
    """Insert mined QA pairs, reusing rows with the same fingerprint.

    Mined rows carry ``content_type='mined_qa'``, no section path, and never
    participate in neighbour links.
    """
    existing = (
        session.query(KnowledgeChunk)
        .filter(KnowledgeChunk.content_type == MINED_CONTENT_TYPE)
        .all()
    )
    by_fingerprint = {_row_fingerprint(row): row for row in existing}

    rows: list[KnowledgeChunk] = []
    for item in items:
        category = _truncate_category(item.category)
        fingerprint = (MINED_CONTENT_TYPE, "", category, item.question, item.answer)
        row = by_fingerprint.get(fingerprint)
        if row is None:
            row = KnowledgeChunk(
                category=category,
                questions=item.question,
                answer=item.answer,
                section_path=None,
                content_type=MINED_CONTENT_TYPE,
                is_key_clause=False,
                vectorize_status="pending",
                prev_chunk_id=None,
                next_chunk_id=None,
            )
            session.add(row)
            by_fingerprint[fingerprint] = row
        rows.append(row)

    session.commit()
    return rows


def _delete_vectors(store, rows: list[KnowledgeChunk]) -> None:
    """Delete every non-null vector id in batches of 500, before any SQL delete."""
    vector_ids = [row.vector_id for row in rows if row.vector_id is not None]
    for start in range(0, len(vector_ids), DELETE_BATCH_SIZE):
        store.delete_ids(vector_ids[start : start + DELETE_BATCH_SIZE])


def _purge(session, query, store) -> int:
    rows = query.order_by(KnowledgeChunk.id).all()
    count = len(rows)
    # Milvus first: if this raises, MySQL rows are still intact.
    _delete_vectors(store, rows)
    query.delete(synchronize_session=False)
    session.commit()
    return count


def reset_document_domain(session, store) -> int:
    """Delete document chunks and their vectors, keeping ``mined_qa`` rows."""
    query = session.query(KnowledgeChunk).filter(_document_domain_filter())
    return _purge(session, query, store)


def reset_all(session, store) -> int:
    """Delete every knowledge chunk and its vector, including ``mined_qa``."""
    return _purge(session, session.query(KnowledgeChunk), store)
