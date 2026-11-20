"""Fetch matched low-confidence questions and clean them with the LLM."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from app.core.taxonomy import TOPIC_CLASSES  # noqa: F401 - import check
from scripts.ch10.corpus_lib import (
    Sample,
    dedupe_exact,
    read_jsonl,
    redact_pii,
    stable_id,
    write_jsonl,
)
from scripts.ch10.llm_io import CallBudget, invoke_structured
from scripts.ch10.prompts import CLEAN_SYSTEM
from scripts.ch10.schemas import CleanBatch

DEFAULT_RAW = "data/ch10/corpus_raw.jsonl"
DEFAULT_CLEAN = "data/ch10/corpus_clean.jsonl"
DEFAULT_CHANGES = "data/ch10/reports/clean_changes.json"


def fetch_real_questions(session) -> list[dict]:
    """Return only questions already merged into the flywheel review queue."""
    from app.models import LowConfidenceQuestion, ReviewQueue

    rows = (
        session.query(LowConfidenceQuestion, ReviewQueue)
        .join(
            ReviewQueue,
            LowConfidenceQuestion.matched_review_id == ReviewQueue.id,
        )
        .filter(LowConfidenceQuestion.matched_review_id.isnot(None))
        .order_by(LowConfidenceQuestion.id.asc())
        .all()
    )
    return [
        {
            "question_id": question.id,
            "review_id": review.id,
            "raw_question": question.raw_question,
            "normalized_question": review.normalized_question,
            "source": question.source,
            "created_at": question.created_at,
        }
        for question, review in rows
    ]


def choose_text(row: dict) -> tuple[str, bool]:
    normalized = row.get("normalized_question")
    if isinstance(normalized, str) and normalized.strip():
        return normalized.strip(), True
    raw = row.get("raw_question")
    return (raw.strip() if isinstance(raw, str) else ""), False


def build_real_samples(rows: list[dict]) -> list[Sample]:
    samples: list[Sample] = []
    for row in rows:
        text, normalized = choose_text(row)
        if not text:
            continue
        cleaned, flags = redact_pii(text)
        created_at = row.get("created_at")
        samples.append(
            Sample(
                id=stable_id("real", cleaned),
                text=cleaned,
                origin="real",
                source_refs=(
                    {
                        "question_id": row["question_id"],
                        "review_id": row["review_id"],
                    },
                ),
                meta={
                    "source": row.get("source"),
                    "normalized": normalized,
                    "created_at": (
                        created_at.isoformat()
                        if hasattr(created_at, "isoformat")
                        else created_at
                    ),
                    "pii_flags": flags,
                },
            )
        )
    return dedupe_exact(samples)


def _batches(items: list, size: int):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def clean_with_llm(
    llm,
    samples: list[Sample],
    budget: CallBudget,
    batch_size: int = 10,
) -> tuple[list[Sample], list[dict]]:
    """Clean typos/format; a failed batch is left unchanged and logged."""
    cleaned: list[Sample] = []
    changes: list[dict] = []
    for chunk in _batches(samples, batch_size):
        expected = {sample.id for sample in chunk}
        human = "\n".join(f"{sample.id}\t{sample.text}" for sample in chunk)
        try:
            data = invoke_structured(
                llm, CLEAN_SYSTEM, human, CleanBatch, budget
            )
            parsed = CleanBatch.model_validate(data)
            got = {item.id for item in parsed.items}
            if got != expected:
                raise ValueError(
                    "clean response id mismatch: "
                    f"missing={sorted(expected - got)} "
                    f"extra={sorted(got - expected)}"
                )
            by_id = {item.id: item.text.strip() for item in parsed.items}
            if any(not text for text in by_id.values()):
                raise ValueError("clean response contains empty text")
        except Exception as exc:  # noqa: BLE001 - keep the batch unchanged
            for sample in chunk:
                changes.append({"id": sample.id, "error": str(exc)})
            cleaned.extend(chunk)
            continue

        for sample in chunk:
            after = by_id[sample.id]
            changes.append(
                {"id": sample.id, "before": sample.text, "after": after}
            )
            cleaned.append(replace(sample, text=after))
    return cleaned, changes


def _read_samples(path: Path) -> list[Sample]:
    return [
        Sample(
            id=row["id"],
            text=row["text"],
            labels=tuple(row.get("labels", ())),
            origin=row.get("origin", "real"),
            source_refs=tuple(row.get("source_refs", ())),
            meta=dict(row.get("meta", {})),
        )
        for row in read_jsonl(path)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and clean ch10 corpus")
    parser.add_argument(
        "--stage", choices=("fetch", "clean", "all"), default="all"
    )
    parser.add_argument("--raw", default=DEFAULT_RAW)
    parser.add_argument("--clean", default=DEFAULT_CLEAN)
    parser.add_argument("--changes", default=DEFAULT_CHANGES)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-llm-calls", type=int, default=None)
    args = parser.parse_args(argv)

    raw_path = Path(args.raw)
    if args.stage in ("fetch", "all"):
        from app.db.session import get_session_factory

        session = get_session_factory()()
        try:
            rows = fetch_real_questions(session)
        finally:
            session.close()
        samples = build_real_samples(rows)
        if not samples:
            print("build_corpus: no matched low-confidence questions")
            return 1
        write_jsonl(raw_path, [asdict(sample) for sample in samples])
        print(f"build_corpus: wrote {len(samples)} real samples to {raw_path}")

    if args.stage in ("clean", "all"):
        if not raw_path.exists():
            print(f"build_corpus: missing {raw_path}; run --stage fetch first")
            return 1
        from app.config import build_llm, get_settings

        samples = _read_samples(raw_path)
        llm = build_llm(get_settings(), temperature=0.0)
        budget = CallBudget(args.max_llm_calls)
        try:
            cleaned, changes = clean_with_llm(
                llm, samples, budget, batch_size=args.batch_size
            )
        except Exception as exc:  # noqa: BLE001 - CLI reports and exits
            print(f"build_corpus: clean failed: {exc}")
            return 1
        clean_path = Path(args.clean)
        write_jsonl(clean_path, [asdict(sample) for sample in cleaned])
        changes_path = Path(args.changes)
        changes_path.parent.mkdir(parents=True, exist_ok=True)
        changes_path.write_text(
            json.dumps(changes, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"build_corpus: wrote {len(cleaned)} cleaned samples to "
            f"{clean_path} ({budget.calls} LLM calls)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
