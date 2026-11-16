"""Full reindex of MySQL knowledge chunks into the hybrid collection.

Unlike the resumable ``app.jobs.vectorize`` job, this migrates every chunk,
including rows already marked ``done`` in the legacy dense flow: the hybrid
collection needs the BM25 ``text`` and filter metadata that the old vectors
never carried.  Re-embedding (rather than copying old vectors) is deliberate;
upserts use the stable ``chunk_{id}`` primary key, so reruns are idempotent.

``--recreate`` drops and recreates only ``knowledge_hybrid`` and therefore
requires ``--yes``.  The legacy ``knowledge`` collection is never touched.
"""

from __future__ import annotations

import argparse
import json

from app.db.session import get_session_factory
from app.rag.embedding import get_embedder
from app.rag.milvus_store import get_vector_store
from app.rag.vectorize import vectorize_all


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-embed every knowledge chunk into the hybrid Milvus collection."
        )
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="drop and recreate the hybrid collection before reindexing",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="confirm the destructive --recreate step",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process at most this many chunks (smoke runs); 0 processes none",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; prints a JSON summary and returns an exit code."""
    args = _build_parser().parse_args(argv)

    # Destructive-flag guard runs before any settings/service resolution so a
    # refused invocation never opens MySQL, Milvus, or the embedding client.
    if args.recreate and not args.yes:
        print("reindex_hybrid: refusing to recreate without --yes")
        return 2
    if args.limit is not None and args.limit < 0:
        print("reindex_hybrid: --limit must not be negative")
        return 2

    store = get_vector_store()
    if args.recreate:
        store.recreate_collection()

    if args.limit == 0:
        # Smoke guard: no rows, so skip building a session and embedder.
        print(json.dumps({"chunks": 0, "embedded": 0, "upserted": 0}))
        return 0

    session = get_session_factory()()
    try:
        report = vectorize_all(
            session,
            get_embedder(),
            store,
            max_chunks=args.limit,
        )
    finally:
        session.close()

    print(
        json.dumps(
            {
                "chunks": report.processed,
                "embedded": report.processed,
                "upserted": report.processed,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
