"""Prelabel real questions against the 17-class taxonomy."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

from scripts.ch10.corpus_lib import Sample, read_jsonl, write_jsonl
from scripts.ch10.llm_io import CallBudget, invoke_structured
from scripts.ch10.prompts import PRELABEL_SYSTEM
from scripts.ch10.schemas import PrelabelBatch

DEFAULT_INPUT = "data/ch10/corpus_clean.jsonl"
DEFAULT_OUTPUT = "data/ch10/corpus_real_labeled.jsonl"


def _batches(items: list, size: int):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def prelabel_samples(
    llm,
    samples: list[Sample],
    budget: CallBudget,
    batch_size: int = 10,
    system: str = PRELABEL_SYSTEM,
) -> list[Sample]:
    """Prelabel every sample, preserving input order and rejecting bad ids."""
    labels_by_id: dict[str, tuple[str, ...]] = {}
    for chunk in _batches(samples, batch_size):
        expected = {sample.id for sample in chunk}
        human = "\n".join(f"{sample.id}\t{sample.text}" for sample in chunk)
        data = invoke_structured(llm, system, human, PrelabelBatch, budget)
        parsed = PrelabelBatch.model_validate(data)
        got = {item.id for item in parsed.items}
        if got != expected:
            missing = sorted(expected - got)
            extra = sorted(got - expected)
            raise ValueError(
                f"prelabel response id mismatch: missing={missing} extra={extra}"
            )
        for item in parsed.items:
            labels_by_id[item.id] = tuple(item.labels)

    return [
        replace(sample, labels=labels_by_id[sample.id])
        for sample in samples
    ]


def _load_samples(path: Path) -> list[Sample]:
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
    parser = argparse.ArgumentParser(description="Prelabel ch10 real questions")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-llm-calls", type=int, default=None)
    args = parser.parse_args(argv)

    from app.config import build_llm, get_settings

    samples = _load_samples(Path(args.input))
    if not samples:
        print("prelabel: no input samples")
        return 1

    llm = build_llm(get_settings(), temperature=0.0)
    budget = CallBudget(args.max_llm_calls)
    try:
        labeled = prelabel_samples(
            llm, samples, budget, batch_size=args.batch_size
        )
    except Exception as exc:  # noqa: BLE001 - CLI reports and exits non-zero
        print(f"prelabel: failed: {exc}")
        return 1

    write_jsonl(Path(args.out), [asdict(sample) for sample in labeled])
    print(
        f"prelabel: wrote {len(labeled)} samples to {args.out} "
        f"({budget.calls} LLM calls)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
