"""Retrieval evaluation harness.

Runs every annotated query in ``tests/eval/faq_cases.json`` through the real
``retrieve`` path (cloud BGE-M3 embedding + Milvus + MySQL) and reports the
Top-K hits, their scores, and PASS/FAIL per case.  Exits non-zero when any
case fails so it can gate a release run.

This is a manual evaluation job: importing the module never touches the
network.  Only ``main`` resolves settings and runs retrieval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import get_settings
from app.rag.retrieval import retrieve

_DEFAULT_CASES = (
    Path(__file__).resolve().parents[2] / "tests" / "eval" / "faq_cases.json"
)


def _value(obj, field: str) -> str:
    """Read *field* from a mapping or an object (evaluation data may use both)."""
    if isinstance(obj, dict):
        return obj.get(field) or ""
    return getattr(obj, field, "") or ""


def _snippet(text: str, limit: int = 120) -> str:
    """Collapse whitespace and truncate long answers for one-line display."""
    collapsed = " ".join(str(text or "").split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit] + "..."


def score_faq_case(case: dict, hits: list) -> bool:
    """Return whether retrieval *hits* satisfy one annotated FAQ *case*.

    ``expect_hit`` false passes only when no chunk survived the score
    threshold.  For positive cases any hit whose answer contains any of the
    ``expect_any`` keywords passes; a positive case without ``expect_any``
    only requires at least one hit.
    """
    if not case.get("expect_hit", True):
        return not hits
    if not hits:
        return False
    keywords = case.get("expect_any") or []
    if not keywords:
        return True
    return any(
        any(keyword in _value(hit, "answer") for keyword in keywords)
        for hit in hits
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the annotated FAQ retrieval evaluation set."
    )
    parser.add_argument(
        "--cases",
        default=None,
        help=f"path to the cases JSON (default: {_DEFAULT_CASES})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the retrieval evaluation and return a process exit code."""
    args = _build_parser().parse_args(argv)
    path = Path(args.cases) if args.cases else _DEFAULT_CASES
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"eval_retrieval: cannot read {path}: {exc}")
        return 2

    cases = data.get("cases") or []
    if not cases:
        print(f"eval_retrieval: no cases found in {path}")
        return 2

    settings = get_settings()
    print(
        f"eval_retrieval: cases={len(cases)} "
        f"top_k={settings.faq_top_k} threshold={settings.faq_score_threshold}"
    )

    passed = 0
    failed = 0
    for number, case in enumerate(cases, 1):
        query = case.get("query", "")
        expect_hit = case.get("expect_hit", True)
        print(f"\n[{number}/{len(cases)}] query={query!r} expect_hit={expect_hit}")
        try:
            hits = retrieve(query)
        except Exception as exc:  # noqa: BLE001 - an eval run reports, never crashes
            failed += 1
            print(f"  ERROR: {exc}")
            print("  FAIL")
            continue

        for rank, hit in enumerate(hits, 1):
            print(
                f"  {rank}. score={hit.score:.4f} "
                f"questions={_snippet(hit.questions)} "
                f"answer={_snippet(hit.answer)}"
            )
        if not hits:
            print("  (no hits above threshold)")

        ok = score_faq_case(case, hits)
        if ok:
            passed += 1
            print("  PASS")
        else:
            failed += 1
            print("  FAIL")

    print(f"\neval_retrieval: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
