"""Vectorize pending knowledge chunks as a standalone, resumable job.

Only rows still marked ``pending`` are embedded and upserted, so rerunning
after a failure or a ``--max-chunks`` interruption converges without
duplicating MySQL rows or Milvus entities.
"""

from __future__ import annotations

import argparse

from app.config import get_settings
from app.db.session import get_session_factory
from app.rag.embedding import get_embedder
from app.rag.milvus_store import get_vector_store
from app.rag.vectorize import vectorize_pending


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed pending knowledge chunks and upsert them into Milvus."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="chunks embedded and committed per batch (default: EMBEDDING_BATCH_SIZE)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="process at most this many pending chunks (controlled interruption)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: vectorize pending chunks and print a one-line summary."""
    args = _build_parser().parse_args(argv)
    settings = get_settings()
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else settings.embedding_batch_size
    )

    session = get_session_factory()()
    try:
        report = vectorize_pending(
            session,
            get_embedder(),
            get_vector_store(),
            batch_size=batch_size,
            max_chunks=args.max_chunks,
        )
    finally:
        session.close()

    print(f"vectorize: processed={report.processed} failed={report.failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
