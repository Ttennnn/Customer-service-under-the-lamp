"""Build the knowledge base from the markdown corpus, then vectorize it.

The job is deliberately resumable and idempotent: document chunks are matched
by fingerprint, existing rows are reused as-is, and only ``pending`` rows are
sent to the embedder/Milvus.  A run interrupted during vectorization therefore
leaves the remaining rows ``pending`` for the next run.

Documents are processed one at a time so ``prev_chunk_id`` / ``next_chunk_id``
never link across document boundaries.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings
from app.db.session import get_session_factory
from app.rag.docs import load_documents
from app.rag.embedding import get_embedder
from app.rag.ingest import (
    ingest_document_chunks,
    reset_all as reset_all_chunks,
    reset_document_domain,
)
from app.rag.milvus_store import get_vector_store
from app.rag.splitter import split_document
from app.rag.vectorize import vectorize_pending

DEFAULT_DOCS_DIR = "knowledge"
DEFAULT_MAX_CHARS = 500
DEFAULT_OVERLAP_CHARS = 80


@dataclass(frozen=True)
class BuildReport:
    documents: int
    chunks: int
    inserted: int
    reused: int
    vectorized: int


def build_knowledge(
    session,
    docs_dir,
    embedder,
    store,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    reset: bool = False,
    reset_all: bool = False,
    do_vectorize: bool = True,
    max_chunks: int | None = None,
) -> BuildReport:
    """Rebuild ``knowledge_chunks`` from *docs_dir* and vectorize new rows.

    ``reset_all`` wins over ``reset`` when both are true.  Each document's
    chunks are ingested as one batch so neighbor links stay document-local.
    """
    if reset_all:
        reset_all_chunks(session, store)
    elif reset:
        reset_document_domain(session, store)

    documents = load_documents(Path(docs_dir))
    chunk_count = 0
    inserted = 0
    reused = 0
    for document in documents:
        chunks = split_document(
            document, max_chars=max_chars, overlap_chars=overlap_chars
        )
        chunk_count += len(chunks)
        _, ingest_report = ingest_document_chunks(session, chunks)
        inserted += ingest_report.inserted
        reused += ingest_report.reused

    vectorized = 0
    if do_vectorize:
        vectorized = vectorize_pending(
            session, embedder, store, max_chunks=max_chunks
        ).processed

    return BuildReport(
        documents=len(documents),
        chunks=chunk_count,
        inserted=inserted,
        reused=reused,
        vectorized=vectorized,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild knowledge chunks from markdown and vectorize them."
    )
    parser.add_argument(
        "--docs-dir",
        help=(
            "directory of *.md knowledge documents "
            f"(default: KNOWLEDGE_DOCS_DIR, currently {DEFAULT_DOCS_DIR})"
        ),
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="delete document-domain chunks and vectors first; keep mined_qa",
    )
    parser.add_argument(
        "--reset-all",
        action="store_true",
        help="delete every chunk and vector first, including mined_qa (needs --yes)",
    )
    parser.add_argument(
        "--yes", action="store_true", help="confirm the --reset-all blast radius"
    )
    parser.add_argument(
        "--no-vectorize",
        action="store_true",
        help="ingest only; leave new rows pending for a later vectorize run",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="vectorize at most this many pending chunks (controlled interruption)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="chunk size limit (default: CHUNK_MAX_CHARS)",
    )
    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=None,
        help="chunk overlap limit (default: CHUNK_OVERLAP_CHARS)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 2 when --reset-all lacks --yes."""
    args = _build_parser().parse_args(argv)

    if args.reset_all and not args.yes:
        print(
            "--reset-all would delete EVERY row in knowledge_chunks "
            "(including mined_qa) and every vector in the Milvus knowledge "
            "collection."
        )
        print("Refusing to run without --yes; nothing was deleted.")
        return 2

    settings = get_settings()
    docs_dir = Path(args.docs_dir or settings.knowledge_docs_dir)
    max_chars = (
        args.max_chars if args.max_chars is not None else settings.chunk_max_chars
    )
    overlap_chars = (
        args.overlap_chars
        if args.overlap_chars is not None
        else settings.chunk_overlap_chars
    )
    do_vectorize = not args.no_vectorize

    # A plain ingest (no reset, no vectorize) must not require the embedding
    # API key or a reachable Milvus instance.
    needs_store = args.reset or args.reset_all or do_vectorize
    store = get_vector_store() if needs_store else None
    embedder = get_embedder() if do_vectorize else None

    session = get_session_factory()()
    try:
        report = build_knowledge(
            session,
            docs_dir,
            embedder,
            store,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            reset=args.reset,
            reset_all=args.reset_all,
            do_vectorize=do_vectorize,
            max_chunks=args.max_chunks,
        )
    finally:
        session.close()

    print(
        "build_knowledge: "
        f"documents={report.documents} chunks={report.chunks} "
        f"inserted={report.inserted} reused={report.reused} "
        f"vectorized={report.vectorized}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
