"""Augment only the training split, accepting a variant only if labels survive."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from app.core.taxonomy import normalize_labels
from scripts.ch10.corpus_lib import (
    Sample,
    read_jsonl,
    stable_id,
    write_jsonl,
)
from scripts.ch10.llm_io import CallBudget, invoke_structured
from scripts.ch10.prelabel import prelabel_samples
from scripts.ch10.prompts import AUGMENT_SYSTEM
from scripts.ch10.schemas import AugmentBatch

MAX_DROP_RATE = 0.20


def accept_variant(
    original: Sample,
    variant_text: str,
    predicted_labels: list[str],
) -> bool:
    if not variant_text.strip():
        return False
    expected = set(normalize_labels(original.labels))
    got = set(normalize_labels(predicted_labels))
    return expected == got


def _batches(items: list, size: int):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def generate_variants(
    llm,
    train: list[Sample],
    budget: CallBudget,
    batch_size: int = 10,
) -> list[tuple[Sample, str]]:
    pairs: list[tuple[Sample, str]] = []
    for chunk in _batches(train, batch_size):
        expected = {sample.id for sample in chunk}
        human = "\n".join(f"{sample.id}\t{sample.text}" for sample in chunk)
        try:
            data = invoke_structured(
                llm, AUGMENT_SYSTEM, human, AugmentBatch, budget
            )
            parsed = AugmentBatch.model_validate(data)
            by_id = {
                item.id: item.text.strip()
                for item in parsed.items
                if item.id in expected
            }
        except Exception:  # noqa: BLE001 - failures are reported and skipped
            by_id = {}
        for sample in chunk:
            pairs.append((sample, by_id.get(sample.id, "")))
    return pairs


def prelabel_variants(
    prelabel_llm,
    pairs: list[tuple[Sample, str]],
    budget: CallBudget,
    batch_size: int = 10,
) -> dict[str, list[str]]:
    variants: dict[str, Sample] = {}
    for original, text in pairs:
        if not text or text in variants:
            continue
        variants[text] = Sample(
            id=stable_id("variant", text),
            text=text,
            labels=original.labels,
            origin="augmented",
            source_refs=original.source_refs,
        )
    if not variants:
        return {}
    labeled = prelabel_samples(
        prelabel_llm,
        list(variants.values()),
        budget,
        batch_size=batch_size,
    )
    return {sample.text: list(sample.labels) for sample in labeled}


def assemble(
    train: list[Sample],
    pairs: list[tuple[Sample, str]],
    predicted: dict[str, list[str]],
) -> tuple[list[Sample], dict]:
    out: list[Sample] = []
    kept = 0
    dropped = 0
    failures: list[dict] = []
    dropped_details: list[dict] = []

    for original, variant_text in pairs:
        text = variant_text.strip()
        if not text or text == original.text:
            failures.append(
                {"id": original.id, "reason": "empty_or_unchanged"}
            )
            out.append(original)
            continue
        labels = predicted.get(text, [])
        if not accept_variant(original, text, labels):
            dropped += 1
            dropped_details.append(
                {
                    "id": original.id,
                    "variant": text,
                    "expected": list(original.labels),
                    "predicted": labels,
                }
            )
            out.append(original)
            continue
        kept += 1
        out.append(original)
        out.append(
            replace(
                original,
                id=stable_id("augmented", text),
                text=text,
                origin="augmented",
                meta={
                    **original.meta,
                    "augmented_from": original.id,
                },
            )
        )

    total = len(train)
    report = {
        "origins": total,
        "kept": kept,
        "dropped": dropped,
        "failures": failures,
        "dropped_details": dropped_details,
        "drop_rate": (dropped / total) if total else 0.0,
    }
    return out, report


def augment_train(
    llm,
    train: list[Sample],
    budget: CallBudget | None,
    prelabel_fn=None,
    batch_size: int = 10,
) -> tuple[list[Sample], dict]:
    if prelabel_fn is None:
        prelabel_fn = getattr(llm, "prelabel_fn", None)
    if prelabel_fn is None:
        raise TypeError("augment_train requires prelabel_fn")
    budget = budget or CallBudget(None)
    pairs = generate_variants(llm, train, budget, batch_size=batch_size)
    predicted = {
        text: list(prelabel_fn(text))
        for _, text in pairs
        if text
    }
    return assemble(train, pairs, predicted)


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
    parser = argparse.ArgumentParser(description="Augment ch10 train split")
    parser.add_argument("--train", default="data/ch10/dataset/train.jsonl")
    parser.add_argument(
        "--out", default="data/ch10/dataset/train_augmented.jsonl"
    )
    parser.add_argument(
        "--report", default="data/ch10/dataset/augment_report.json"
    )
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-llm-calls", type=int, default=None)
    args = parser.parse_args(argv)

    from app.config import build_llm, get_settings

    train_path = Path(args.train)
    if not train_path.exists():
        print(f"augment: missing {train_path}; run the split first")
        return 1
    train = _load_samples(train_path)
    if not train:
        print("augment: empty train split")
        return 1

    settings = get_settings()
    budget = CallBudget(args.max_llm_calls)
    try:
        pairs = generate_variants(
            build_llm(settings, temperature=0.7),
            train,
            budget,
            batch_size=args.batch_size,
        )
        predicted = prelabel_variants(
            build_llm(settings, temperature=0.0),
            pairs,
            budget,
            batch_size=args.batch_size,
        )
    except Exception as exc:  # noqa: BLE001 - CLI reports and exits
        print(f"augment: LLM failure: {exc}")
        return 1

    out, report = assemble(train, pairs, predicted)
    write_jsonl(Path(args.out), [asdict(sample) for sample in out])
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"augment: origins={report['origins']} kept={report['kept']} "
        f"dropped={report['dropped']} drop_rate={report['drop_rate']:.3f} "
        f"-> {args.out}"
    )
    if report["drop_rate"] > MAX_DROP_RATE:
        print(
            f"augment: drop_rate above {MAX_DROP_RATE:.0%}; inspect "
            f"{args.report} before training"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
