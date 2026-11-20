"""Stratified 80/10/10 split with exact-text leak gate and manifest."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from app.core.taxonomy import TOPIC_CLASSES, normalize_labels
from scripts.ch10.corpus_lib import (
    Sample,
    dedupe_near,
    normalize_text,
    read_jsonl,
    write_jsonl,
)

SPLITS = ("train", "val", "test")
MIN_COMBO_SIZE = 10


def _combo_key(sample: Sample) -> str:
    return "|".join(normalize_labels(sample.labels)) or "其他"


def _primary_label(sample: Sample) -> str:
    normalized = normalize_labels(sample.labels)
    return normalized[0] if normalized else "其他"


def _fill_coverage(splits: dict[str, list[Sample]]) -> None:
    """Move one train sample per missing class, never emptying train."""
    for label in TOPIC_CLASSES:
        for split_name in ("val", "test"):
            if any(label in sample.labels for sample in splits[split_name]):
                continue
            if len(splits["train"]) <= 1:
                continue
            candidates = [
                sample for sample in splits["train"] if label in sample.labels
            ]
            if not candidates:
                continue

            def score(sample: Sample) -> int:
                others = [item for item in sample.labels if item != label]
                return sum(
                    1
                    for item in others
                    if any(item in other.labels for other in splits[split_name])
                )

            chosen = max(candidates, key=score)
            splits["train"].remove(chosen)
            splits[split_name].append(chosen)


def split_samples(
    samples: list[Sample],
    seed: int = 42,
    *,
    dedup: bool = True,
) -> dict[str, list[Sample]]:
    if dedup:
        samples, _ = dedupe_near(samples)

    base_keys = [_combo_key(sample) for sample in samples]
    combo_counts = Counter(base_keys)
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample, key in zip(samples, base_keys):
        if combo_counts[key] < MIN_COMBO_SIZE:
            key = _primary_label(sample)
        groups[key].append(sample)

    splits: dict[str, list[Sample]] = {name: [] for name in SPLITS}
    rng = random.Random(seed)
    for key in sorted(groups):
        group = list(groups[key])
        rng.shuffle(group)
        size = len(group)
        if size < 3:
            splits["train"].extend(group)
            continue
        n_train = int(size * 0.8)
        n_val = int(size * 0.1)
        n_test = size - n_train - n_val
        if n_val == 0:
            n_train -= 1
            n_val = 1
        if n_test == 0:
            n_train -= 1
            n_test = 1
        splits["train"].extend(group[:n_train])
        splits["val"].extend(group[n_train : n_train + n_val])
        splits["test"].extend(
            group[n_train + n_val : n_train + n_val + n_test]
        )

    _fill_coverage(splits)
    return splits


def check_no_leak(splits: dict[str, list[Sample]]) -> dict[str, int]:
    texts = {
        name: {normalize_text(sample.text) for sample in splits[name]}
        for name in SPLITS
    }
    return {
        "train_val": len(texts["train"] & texts["val"]),
        "train_test": len(texts["train"] & texts["test"]),
        "val_test": len(texts["val"] & texts["test"]),
    }


def class_coverage(splits: dict[str, list[Sample]]) -> dict[str, list[bool]]:
    return {
        label: [
            any(label in sample.labels for sample in splits[name])
            for name in SPLITS
        ]
        for label in TOPIC_CLASSES
    }


def write_dataset(
    splits: dict[str, list[Sample]],
    out_dir: Path,
    manifest_extra: dict | None = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        write_jsonl(
            out_dir / f"{name}.jsonl",
            [asdict(sample) for sample in splits[name]],
        )

    counts = {name: len(splits[name]) for name in SPLITS}
    origin_counts = {
        name: dict(Counter(sample.origin for sample in splits[name]))
        for name in SPLITS
    }
    per_class: dict[str, dict] = {}
    for label in TOPIC_CLASSES:
        per_class[label] = {}
        for name in SPLITS:
            count = sum(
                1 for sample in splits[name] if label in sample.labels
            )
            total = counts[name]
            per_class[label][name] = {
                "count": count,
                "ratio": (count / total) if total else 0.0,
            }

    coverage = class_coverage(splits)
    coverage_gaps = [
        {"label": label, "split": name}
        for label, flags in coverage.items()
        for name, present in zip(SPLITS, flags)
        if not present
    ]
    manifest = {
        "counts": counts,
        "origin_counts": origin_counts,
        "per_class": per_class,
        "coverage": coverage,
        "coverage_gaps": coverage_gaps,
        "leak": check_no_leak(splits),
        **(manifest_extra or {}),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


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
    parser = argparse.ArgumentParser(description="Split ch10 dataset")
    parser.add_argument("--labeled", default="data/ch10/corpus_labeled.jsonl")
    parser.add_argument("--out", default="data/ch10/dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    labeled_path = Path(args.labeled)
    if not labeled_path.exists():
        print(f"build_dataset: missing {labeled_path}")
        return 1
    samples = _load_samples(labeled_path)
    unique, merged = dedupe_near(samples)
    splits = split_samples(unique, seed=args.seed, dedup=False)
    leak = check_no_leak(splits)
    if any(leak.values()):
        print(f"build_dataset: leak detected {leak}; refusing to write")
        return 1

    manifest = write_dataset(
        splits,
        Path(args.out),
        {
            "seed": args.seed,
            "near_duplicates_merged": merged,
        },
    )
    print(
        f"build_dataset: train={manifest['counts']['train']} "
        f"val={manifest['counts']['val']} test={manifest['counts']['test']} "
        f"leak={manifest['leak']} gaps={len(manifest['coverage_gaps'])} "
        f"near_merged={merged}"
    )
    return 0 if not manifest["coverage_gaps"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
