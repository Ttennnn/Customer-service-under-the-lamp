"""CLI: classify the low-confidence pool through the side-channel service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.db.session import get_session_factory
from scripts.ch10.classify_lib import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify the ch10 pool")
    parser.add_argument("--base-url", default="http://127.0.0.1:8110")
    parser.add_argument("--min-batch", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reclassify", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--out", default="data/ch10/reports/classify_run.json"
    )
    args = parser.parse_args(argv)

    if args.reclassify and not args.yes:
        print("classify-pool: --reclassify also requires --yes")
        return 1

    session = get_session_factory()()
    try:
        result = run(
            session,
            args.base_url,
            min_batch=args.min_batch,
            force=args.force,
            limit=args.limit,
            reclassify=args.reclassify,
        )
    finally:
        session.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "classify-pool: "
        f"pending={result.get('pending')} "
        f"classified={result.get('classified')} "
        f"skipped={result.get('skipped')} "
        f"reason={result.get('reason', '')}"
    )
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
