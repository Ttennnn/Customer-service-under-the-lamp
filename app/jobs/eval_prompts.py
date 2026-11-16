"""Prompt-quality sample validation harness.

Runs the real query rewrite (:func:`analyze`) or evidence self-check
(:func:`assess_evidence`) over annotated samples and prints PASS/FAIL per
sample, exiting non-zero when any sample fails.

Sample schemas (top-level object with a ``samples`` list)::

    rewrite:    {"samples": [{"query": ...,
                              "expect_rewritten_any": [...],
                              "expect_keywords_any": [...]}]}
    self_check: {"samples": [{"query": ...,
                              "sources": [...],
                              "expect_useful": true|false}]}

The suite is a manual validation job: importing this module never touches the
network. Only ``main`` resolves settings and builds the LLM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import build_llm, get_settings
from app.rag.query_understanding import QueryAnalysis, analyze
from app.rag.self_check import EvidenceAssessment, assess_evidence

_EVAL_DIR = Path(__file__).resolve().parents[2] / "tests" / "eval"

_DEFAULT_SAMPLES = {
    "rewrite": _EVAL_DIR / "query_rewrite_samples.json",
    "self_check": _EVAL_DIR / "self_check_samples.json",
}


def score_rewrite_sample(sample: dict, analysis: QueryAnalysis) -> bool:
    """Return whether a real *analysis* satisfies one rewrite *sample*.

    ``expect_rewritten_any`` is a substring allow-list for the rewritten
    query; ``expect_keywords_any`` matches extracted keywords in either
    direction (``运费`` accepts ``运费标准`` and vice versa). An empty list is
    a wildcard. Every non-empty list must be satisfied by the same analysis.
    """
    rewritten_tokens = sample.get("expect_rewritten_any") or []
    if rewritten_tokens and not any(
        token in analysis.rewritten for token in rewritten_tokens
    ):
        return False

    keyword_tokens = sample.get("expect_keywords_any") or []
    if keyword_tokens:
        keywords = analysis.keywords or []
        if not any(
            token in keyword or keyword in token
            for token in keyword_tokens
            for keyword in keywords
        ):
            return False
    return True


def score_self_check_sample(
    sample: dict, assessment: EvidenceAssessment
) -> bool:
    """Return whether the self-check verdict matches the annotated label."""
    return assessment.useful == sample.get("expect_useful")


def _snippet(text: str, limit: int = 160) -> str:
    """Collapse whitespace and truncate long text for one-line display."""
    collapsed = " ".join(str(text or "").split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit] + "..."


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate prompt behaviour on annotated rewrite/self-check samples."
    )
    parser.add_argument(
        "--suite",
        required=True,
        choices=sorted(_DEFAULT_SAMPLES),
        help="which sample suite to run",
    )
    parser.add_argument(
        "--samples",
        default=None,
        help="path to the samples JSON (default: the suite's annotated file)",
    )
    return parser


def _load_samples(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return list(data)
    return list(data.get("samples") or [])


def _run_rewrite(samples: list[dict], llm) -> tuple[int, int, int]:
    """Run the rewrite suite; return ``(passed, failed, fallbacks)``."""
    passed = failed = fallbacks = 0
    for number, sample in enumerate(samples, 1):
        query = sample.get("query", "")
        print(f"\n[{number}/{len(samples)}] query={query!r}")
        analysis = analyze(query, llm=llm)
        if analysis.rewritten_by == "fallback":
            fallbacks += 1
        print(f"  rewritten={_snippet(analysis.rewritten)!r}")
        print(f"  keywords={analysis.keywords} via={analysis.rewritten_by}")
        if score_rewrite_sample(sample, analysis):
            passed += 1
            print("  PASS")
        else:
            failed += 1
            print(
                "  FAIL: expect_rewritten_any="
                f"{sample.get('expect_rewritten_any')} "
                f"expect_keywords_any={sample.get('expect_keywords_any')}"
            )
    return passed, failed, fallbacks


def _run_self_check(samples: list[dict], llm) -> tuple[int, int]:
    """Run the self-check suite; return ``(passed, failed)``."""
    passed = failed = 0
    for number, sample in enumerate(samples, 1):
        query = sample.get("query", "")
        sources = list(sample.get("sources") or [])
        expect = sample.get("expect_useful")
        print(
            f"\n[{number}/{len(samples)}] query={query!r} "
            f"expect_useful={expect} sources={len(sources)}"
        )
        try:
            assessment = assess_evidence(llm, query, sources)
        except Exception as exc:  # noqa: BLE001 - an eval run reports, never crashes
            failed += 1
            print(f"  ERROR: {exc}")
            print("  FAIL")
            continue

        print(
            f"  useful={assessment.useful} "
            f"reason={_snippet(assessment.reason)}"
        )
        if score_self_check_sample(sample, assessment):
            passed += 1
            print("  PASS")
        else:
            failed += 1
            print("  FAIL")
    return passed, failed


def main(argv: list[str] | None = None) -> int:
    """Run one prompt sample suite and return a process exit code."""
    args = _build_parser().parse_args(argv)
    path = Path(args.samples) if args.samples else _DEFAULT_SAMPLES[args.suite]
    try:
        samples = _load_samples(path)
    except (OSError, json.JSONDecodeError, TypeError, AttributeError) as exc:
        print(f"eval_prompts: cannot read {path}: {exc}")
        return 2
    if not samples:
        print(f"eval_prompts: no samples found in {path}")
        return 2

    settings = get_settings()
    llm = build_llm(settings, temperature=0.0)
    print(
        f"eval_prompts: suite={args.suite} samples={len(samples)} "
        f"model={settings.model_name}"
    )

    if args.suite == "rewrite":
        passed, failed, fallbacks = _run_rewrite(samples, llm)
        if fallbacks:
            print(
                f"\neval_prompts: {fallbacks}/{len(samples)} rewrites fell back "
                "to the deterministic path (LLM may be unavailable)"
            )
    else:
        passed, failed = _run_self_check(samples, llm)

    print(f"\neval_prompts: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
