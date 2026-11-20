"""Augment only the training split, accepting a variant only if labels survive."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from app.core.taxonomy import normalize_labels
from scripts.ch10.corpus_lib import (
    Sample,
    jaccard_3gram,
    normalize_text,
    read_jsonl,
    stable_id,
    write_jsonl,
)
from scripts.ch10.llm_io import BudgetExceeded, CallBudget, invoke_structured
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
        except BudgetExceeded:
            raise
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


def _is_leak(
    text: str,
    leak_norm: set[str],
    leak_texts: list[str],
    threshold: float,
) -> bool:
    if normalize_text(text) in leak_norm:
        return True
    return any(
        jaccard_3gram(text, leak) >= threshold for leak in leak_texts
    )


def assemble(
    train: list[Sample],
    pairs: list[tuple[Sample, str]],
    predicted: dict[str, list[str]],
    leak_texts: list[str] | None = None,
    leak_threshold: float = 0.85,
) -> tuple[list[Sample], dict]:
    out: list[Sample] = []
    kept = 0
    dropped = 0
    leak_dropped = 0
    duplicates = 0
    failures: list[dict] = []
    dropped_details: list[dict] = []

    leak_texts = list(leak_texts or [])
    leak_norm = {normalize_text(text) for text in leak_texts}
    seen_texts = {normalize_text(sample.text) for sample in train}

    for original, variant_text in pairs:
        text = variant_text.strip()
        if not text or text == original.text:
            failures.append(
                {"id": original.id, "reason": "empty_or_unchanged"}
            )
            out.append(original)
            continue
        if _is_leak(text, leak_norm, leak_texts, leak_threshold):
            leak_dropped += 1
            out.append(original)
            continue
        normalized = normalize_text(text)
        if normalized in seen_texts:
            duplicates += 1
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
        seen_texts.add(normalized)
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
        "leak_dropped": leak_dropped,
        "duplicates": duplicates,
        "leak_threshold": leak_threshold,
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
    leak_texts: list[str] | None = None,
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
    return assemble(train, pairs, predicted, leak_texts=leak_texts)


def check_augmented_leak(
    augmented: list[Sample],
    val_texts: list[str],
    test_texts: list[str],
    threshold: float = 0.85,
) -> dict:
    leak_texts = list(val_texts) + list(test_texts)
    leak_norm = {normalize_text(text) for text in leak_texts}
    exact = 0
    near = 0
    for sample in augmented:
        if sample.origin != "augmented":
            continue
        if normalize_text(sample.text) in leak_norm:
            exact += 1
        elif any(
            jaccard_3gram(sample.text, leak) >= threshold
            for leak in leak_texts
        ):
            near += 1
    return {
        "exact": exact,
        "near": near,
        "total": exact + near,
        "threshold": threshold,
    }


def filter_augmented_rows(
    rows: list[dict],
    leak_texts: list[str],
    threshold: float = 0.85,
) -> tuple[list[dict], list[dict]]:
    """Drop augmented rows that duplicate train or leak into val/test."""
    leak_norm = {normalize_text(text) for text in leak_texts}
    kept: list[dict] = []
    dropped: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        text = row.get("text", "")
        normalized = normalize_text(text)
        if row.get("origin") != "augmented":
            kept.append(row)
            seen.add(normalized)
            continue
        if normalized in seen or _is_leak(
            text, leak_norm, leak_texts, threshold
        ):
            dropped.append(row)
            continue
        seen.add(normalized)
        kept.append(row)
    return kept, dropped


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


def _row_sample(row: dict) -> Sample:
    return Sample(
        id=row["id"],
        text=row["text"],
        labels=tuple(row.get("labels", ())),
        origin=row.get("origin", "real"),
        source_refs=tuple(row.get("source_refs", ())),
        meta=dict(row.get("meta", {})),
    )


def _update_manifest(train_path: Path, updates: dict) -> None:
    path = Path(train_path).parent / "manifest.json"
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(data, dict):
        return
    data.update(updates)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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
    parser.add_argument("--val", default="data/ch10/dataset/val.jsonl")
    parser.add_argument("--test", default="data/ch10/dataset/test.jsonl")
    parser.add_argument("--filter-existing", action="store_true")
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

    val_path = Path(args.val)
    test_path = Path(args.test)
    val_texts = (
        [sample.text for sample in _load_samples(val_path)]
        if val_path.exists()
        else []
    )
    test_texts = (
        [sample.text for sample in _load_samples(test_path)]
        if test_path.exists()
        else []
    )
    leak_texts = val_texts + test_texts
    out_path = Path(args.out)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if args.filter_existing:
        if not out_path.exists():
            print(f"augment: missing {out_path}")
            return 1
        rows = read_jsonl(out_path)
        kept, filtered = filter_augmented_rows(rows, leak_texts)
        write_jsonl(out_path, kept)
        before = [_row_sample(row) for row in rows]
        after = [_row_sample(row) for row in kept]
        report = {
            "origins": len(train),
            "kept": sum(
                1 for row in kept if row.get("origin") == "augmented"
            ),
            "filtered_out": len(filtered),
            "leak_before": check_augmented_leak(
                before, val_texts, test_texts
            ),
            "augment_leak": check_augmented_leak(
                after, val_texts, test_texts
            ),
            "complete": True,
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _update_manifest(
            train_path,
            {
                "augment_leak": report["augment_leak"],
                "train_augmented_count": len(kept),
            },
        )
        print(
            f"augment: filtered={report['filtered_out']} "
            f"leak_before={report['leak_before']['total']} "
            f"leak_after={report['augment_leak']['total']} -> {out_path}"
        )
        return 0 if report["augment_leak"]["total"] == 0 else 1

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
    except BudgetExceeded as exc:
        print(f"augment: LLM budget exhausted: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 - CLI reports and exits
        print(f"augment: LLM failure: {exc}")
        return 1

    out, report = assemble(
        train, pairs, predicted, leak_texts=leak_texts
    )
    write_jsonl(out_path, [asdict(sample) for sample in out])
    leaked = [sample for sample in out if sample.origin == "augmented"]
    report["augment_leak"] = check_augmented_leak(
        leaked, val_texts, test_texts
    )
    report["complete"] = True
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _update_manifest(
        train_path,
        {
            "augment_leak": report["augment_leak"],
            "train_augmented_count": len(out),
        },
    )
    print(
        f"augment: origins={report['origins']} kept={report['kept']} "
        f"dropped={report['dropped']} leak_dropped={report['leak_dropped']} "
        f"drop_rate={report['drop_rate']:.3f} "
        f"leak={report['augment_leak']['total']} -> {args.out}"
    )
    if report["drop_rate"] > MAX_DROP_RATE:
        print(
            f"augment: drop_rate above {MAX_DROP_RATE:.0%}; inspect "
            f"{args.report} before training"
        )
        return 1
    if report["augment_leak"]["total"] > 0:
        print("augment: augmented leak remains; refusing to ship")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
