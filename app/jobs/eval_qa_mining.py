"""QA-mining evaluation harness.

Runs the real configured LLM (``build_llm`` + ``extract_qa_batch``) over the
annotated transcripts in ``tests/eval/qa_mining_samples.json`` and checks that
each expected question/answer pair is present in the extracted candidates.

LLM output is non-deterministic, so scoring is keyword-based rather than an
exact diff: an expected item passes when one extracted candidate's question
contains a question keyword *and* that same candidate's answer contains an
answer keyword.  The job is manual; importing the module never calls the LLM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import build_llm, get_settings
from app.rag.embedding import get_embedder
from app.services.qa_mining import (
    ConversationTranscript,
    dedup_questions,
    extract_qa_batch,
)

_DEFAULT_SAMPLES = (
    Path(__file__).resolve().parents[2] / "tests" / "eval" / "qa_mining_samples.json"
)


def _value(obj, field: str) -> str:
    """Read *field* from a mapping or an object (evaluation data may use both)."""
    if isinstance(obj, dict):
        return obj.get(field) or ""
    return getattr(obj, field, "") or ""


def _snippet(text: str, limit: int = 120) -> str:
    """Collapse whitespace and truncate long text for one-line display."""
    collapsed = " ".join(str(text or "").split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit] + "..."


def _expected_items(case: dict) -> list[dict]:
    """Return the expectation list, accepting a single-item case too."""
    if "expected" in case:
        return list(case.get("expected") or [])
    if "question_any" in case or "answer_any" in case:
        return [case]
    return []


def score_mining_case(case: dict, got: list) -> bool:
    """Return whether extracted *got* items satisfy every expectation in *case*.

    Each expectation is ``{"question_any": [...], "answer_any": [...]}`` and
    must be matched by one single candidate: keywords may not be combined
    across different candidates.  Empty keyword lists act as wildcards.
    """
    for expected in _expected_items(case):
        question_keywords = expected.get("question_any") or []
        answer_keywords = expected.get("answer_any") or []
        matched = False
        for item in got:
            question = str(_value(item, "question"))
            answer = str(_value(item, "answer"))
            question_ok = (
                any(keyword in question for keyword in question_keywords)
                if question_keywords
                else True
            )
            answer_ok = (
                any(keyword in answer for keyword in answer_keywords)
                if answer_keywords
                else True
            )
            if question_ok and answer_ok:
                matched = True
                break
        if not matched:
            return False
    return True


def summarize_dedup(
    candidates: list[str], embedder, threshold: float
) -> tuple[int, int, list[bool]]:
    """Run real dedup over *candidates* and return ``(kept, discarded, flags)``.

    ``existing`` is empty because the eval measures within-sample duplicate
    collapse, not collisions against production knowledge.
    """
    flags = dedup_questions(candidates, [], embedder, threshold=threshold)
    kept = sum(flags)
    return kept, len(flags) - kept, flags


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the annotated QA-mining evaluation set."
    )
    parser.add_argument(
        "--samples",
        default=None,
        help=f"path to the samples JSON (default: {_DEFAULT_SAMPLES})",
    )
    return parser


def _transcript(sample: dict) -> ConversationTranscript:
    messages = [
        (message.get("role", ""), message.get("content", ""))
        for message in sample.get("messages") or []
    ]
    return ConversationTranscript(
        source_ref=str(sample.get("source_ref", "")),
        messages=messages,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the QA-mining evaluation and return a process exit code."""
    args = _build_parser().parse_args(argv)
    path = Path(args.samples) if args.samples else _DEFAULT_SAMPLES
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"eval_qa_mining: cannot read {path}: {exc}")
        return 2

    samples = data.get("transcripts") or []
    if not samples:
        print(f"eval_qa_mining: no transcripts found in {path}")
        return 2

    settings = get_settings()
    llm = build_llm(settings)
    print(f"eval_qa_mining: transcripts={len(samples)} model={settings.model_name}")

    passed = 0
    failed = 0
    candidates: list[str] = []
    for number, sample in enumerate(samples, 1):
        source_ref = sample.get("source_ref", "")
        print(f"\n[{number}/{len(samples)}] source_ref={source_ref!r}")
        try:
            got = extract_qa_batch(llm, [_transcript(sample)])
        except Exception as exc:  # noqa: BLE001 - an eval run reports, never crashes
            failed += 1
            print(f"  ERROR: {exc}")
            print("  FAIL")
            continue

        for item in got:
            print(
                f"  - [{_value(item, 'category')}] "
                f"Q: {_snippet(_value(item, 'question'))} | "
                f"A: {_snippet(_value(item, 'answer'))}"
            )
            candidates.append(str(_value(item, "question")))
        if not got:
            print("  (no QA pairs extracted)")

        ok = score_mining_case(sample, got)
        if ok:
            passed += 1
            print("  PASS")
        else:
            failed += 1
            print("  FAIL")
            for expected in _expected_items(sample):
                print(
                    "    missing: question_any="
                    f"{expected.get('question_any')} "
                    f"answer_any={expected.get('answer_any')}"
                )

    print(f"\neval_qa_mining: extraction {passed} passed, {failed} failed")

    # Run the real semantic dedup over every extracted candidate.  The
    # annotated samples repeat several user questions, so dedup must collapse
    # at least one candidate (kept < total); this exercises the 0.92 cosine
    # path against the real BGE-M3 embedder, which extraction-only scoring
    # never does.
    threshold = settings.qa_dedup_threshold
    total = len(candidates)
    print(f"eval_qa_mining: dedup candidates={total} threshold={threshold}")
    if total < 2:
        print("  FAIL: expected at least two candidates to exercise dedup")
        failed += 1
    else:
        try:
            kept, discarded, flags = summarize_dedup(
                candidates, get_embedder(), threshold
            )
        except Exception as exc:  # noqa: BLE001 - report embedder outages
            print(f"  ERROR: {exc}")
            print("  FAIL")
            failed += 1
        else:
            for question, keep in zip(candidates, flags):
                print(f"  {'KEEP' if keep else 'DROP'} Q: {_snippet(question)}")
            print(
                f"  kept={kept} discarded={discarded} of {total}"
            )
            if kept < total:
                print("  PASS")
            else:
                print("  FAIL: no duplicate candidate was collapsed")
                failed += 1

    print(f"\neval_qa_mining: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
