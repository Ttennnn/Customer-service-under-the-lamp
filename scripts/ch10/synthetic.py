"""Fill per-class gaps with LLM-generated questions, frozen by content id."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from app.core.taxonomy import (
    LABEL2ID,
    TAXONOMY_VERSION,
    TOPIC_CLASSES,
    normalize_labels,
)
from scripts.ch10.corpus_lib import (
    Sample,
    apply_corrections,
    dedupe_exact,
    label_counts,
    normalize_text,
    read_jsonl,
    stable_id,
    write_jsonl,
)
from scripts.ch10.llm_io import CallBudget, invoke_structured
from scripts.ch10.prelabel import prelabel_samples
from scripts.ch10.prompts import SYNTHETIC_SYSTEM
from scripts.ch10.schemas import SyntheticBatch

DEFAULT_RATIOS = {"dialect": 0.15, "typo": 0.10, "multi": 0.15}


def target_gap(label_counts: dict[str, int], target: int = 100) -> dict[str, int]:
    return {
        label: max(0, target - int(label_counts.get(label, 0)))
        for label in TOPIC_CLASSES
    }


def run_id_for(params: dict, *, force: bool = False) -> str:
    payload = {"taxonomy": TAXONOMY_VERSION, "params": params}
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:8]
    if force:
        digest += "-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return digest


def validate_synthetic(samples: list[Sample]) -> list[Sample]:
    valid: list[Sample] = []
    for sample in samples:
        text = sample.text.strip()
        if not text or not sample.labels:
            continue
        try:
            labels = tuple(normalize_labels(sample.labels))
        except ValueError:
            continue
        valid.append(replace(sample, text=text, labels=labels))
    return dedupe_exact(valid)


def _load_samples(path: Path) -> list[Sample]:
    return [
        Sample(
            id=row["id"],
            text=row["text"],
            labels=tuple(row.get("labels", ())),
            origin=row.get("origin", "synthetic"),
            source_refs=tuple(row.get("source_refs", ())),
            meta=dict(row.get("meta", {})),
        )
        for row in read_jsonl(path)
    ]


def _write_generation(
    path: Path,
    samples: list[Sample],
    run_id: str,
    gap: dict[str, int],
    ratios: dict,
    calls: int,
) -> None:
    write_jsonl(path, [asdict(sample) for sample in samples])
    counts = label_counts(samples)
    multi = (
        sum(1 for sample in samples if len(sample.labels) > 1) / len(samples)
        if samples
        else 0.0
    )
    manifest = {
        "run_id": run_id,
        "gap": gap,
        "counts": counts,
        "samples": len(samples),
        "multi_ratio": multi,
        "requested_ratios": ratios,
        "llm_calls": calls,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (path.parent / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_or_generate(
    llm,
    gap: dict[str, int],
    out_dir: Path,
    *,
    ratios: dict | None = None,
    budget: CallBudget,
    force: bool = False,
    batch_size: int = 10,
    max_rounds: int = 400,
) -> list[Sample]:
    """Reuse a frozen run when present; otherwise generate until gaps close."""
    ratios = dict(ratios or DEFAULT_RATIOS)
    out_dir = Path(out_dir)
    params = {
        "gap": {label: int(value) for label, value in gap.items() if value > 0},
        "ratios": ratios,
        "batch_size": batch_size,
    }
    run_id = run_id_for(params, force=force)
    path = out_dir / run_id / "synthetic.jsonl"
    if path.exists():
        return validate_synthetic(_load_samples(path))

    remaining = {
        label: int(gap.get(label, 0)) for label in TOPIC_CLASSES
    }
    seen: set[str] = set()
    samples: list[Sample] = []
    rounds = 0
    stale_rounds = 0
    try:
        while any(value > 0 for value in remaining.values()) and rounds < max_rounds:
            rounds += 1
            needed = "\n".join(
                f"- {label}: 还需 {remaining[label]} 条"
                for label in TOPIC_CLASSES
                if remaining[label] > 0
            )
            human = (
                f"还需补足以下类目的正例数量:\n{needed}\n\n"
                f"请生成 {batch_size} 条问题，优先覆盖缺口大的类目。\n"
                f"风格配比: 方言 {ratios.get('dialect', 0.15):.0%}, "
                f"错别字 {ratios.get('typo', 0.10):.0%}, "
                f"多诉求 {ratios.get('multi', 0.15):.0%}。"
            )
            data = invoke_structured(
                llm, SYNTHETIC_SYSTEM, human, SyntheticBatch, budget
            )
            parsed = SyntheticBatch.model_validate(data)
            before = len(samples)
            for item in parsed.items:
                text = normalize_text(item.text)
                if not text or text in seen:
                    continue
                labels = tuple(item.labels)
                if not labels:
                    continue
                seen.add(text)
                samples.append(
                    Sample(
                        id=stable_id("synthetic", text),
                        text=text,
                        labels=labels,
                        origin="synthetic",
                    )
                )
                for label in labels:
                    if remaining.get(label, 0) > 0:
                        remaining[label] -= 1
            stale_rounds = stale_rounds + 1 if len(samples) == before else 0
            if stale_rounds >= 5:
                break
    except Exception:
        _write_generation(path, samples, run_id, gap, ratios, budget.calls)
        raise

    _write_generation(path, samples, run_id, gap, ratios, budget.calls)
    return samples


def cross_check_labels(samples: list[Sample], prelabel_fn) -> dict:
    disagreements: list[dict] = []
    agreed = 0
    for sample in samples:
        predicted = list(prelabel_fn(sample.text))
        if set(predicted) == set(sample.labels):
            agreed += 1
            continue
        disagreements.append(
            {
                "id": sample.id,
                "text": sample.text,
                "generated": list(sample.labels),
                "predicted": predicted,
            }
        )
    total = len(samples)
    return {
        "total": total,
        "agreed": agreed,
        "agreement_rate": (agreed / total) if total else 0.0,
        "disagreements": disagreements,
    }


def cross_check_predictions(
    samples: list[Sample], predicted: dict[str, list[str]]
) -> dict:
    disagreements: list[dict] = []
    agreed = 0
    for sample in samples:
        got = list(predicted.get(sample.id, []))
        if set(got) == set(sample.labels):
            agreed += 1
            continue
        disagreements.append(
            {
                "id": sample.id,
                "text": sample.text,
                "generated": list(sample.labels),
                "predicted": got,
            }
        )
    total = len(samples)
    return {
        "total": total,
        "agreed": agreed,
        "agreement_rate": (agreed / total) if total else 0.0,
        "disagreements": disagreements,
    }


def merge_labeled(
    real_labeled: list[Sample],
    synthetic: list[Sample],
    corrections: dict[str, list[str]] | None = None,
) -> list[Sample]:
    combined = list(real_labeled) + list(synthetic)
    if corrections:
        combined = apply_corrections(combined, corrections)
    return dedupe_exact(combined)


def _archive_corrections(path: Path, run_id: str) -> int:
    if not path.exists():
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    archived = path.with_name(f"audit_corrections.{run_id}.json")
    shutil.copyfile(path, archived)
    return len(data) if isinstance(data, dict) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate ch10 synthetic corpus")
    parser.add_argument(
        "--real-labeled", default="data/ch10/corpus_real_labeled.jsonl"
    )
    parser.add_argument("--out-dir", default="data/ch10/generated")
    parser.add_argument("--labeled-out", default="data/ch10/corpus_labeled.jsonl")
    parser.add_argument("--corrections", default="data/ch10/audit_corrections.json")
    parser.add_argument(
        "--cross-check-out", default="data/ch10/reports/cross_check.json"
    )
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-llm-calls", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    from app.config import build_llm, get_settings

    real_path = Path(args.real_labeled)
    if not real_path.exists():
        print(f"synthetic: missing {real_path}; run prelabel first")
        return 1
    real = validate_synthetic(_load_samples(real_path))
    gap = target_gap(label_counts(real), target=args.target)
    corrections_path = Path(args.corrections)
    if args.force:
        invalidated = _archive_corrections(corrections_path, run_id_for({}, force=True))
        if invalidated:
            print(
                f"synthetic: --force archived {invalidated} corrections; "
                "review them before merging again"
            )

    settings = get_settings()
    llm = build_llm(settings, temperature=0.8)
    check_llm = build_llm(settings, temperature=0.0)
    budget = CallBudget(args.max_llm_calls)
    try:
        synthetic = load_or_generate(
            llm,
            gap,
            Path(args.out_dir),
            ratios=DEFAULT_RATIOS,
            budget=budget,
            force=args.force,
            batch_size=args.batch_size,
        )
    except Exception as exc:  # noqa: BLE001 - CLI reports and exits
        print(f"synthetic: generation failed: {exc}")
        return 1

    cross_path = Path(args.cross_check_out)
    fingerprint = hashlib.sha1(
        "|".join(sorted(sample.id for sample in synthetic)).encode("utf-8")
    ).hexdigest()[:12]
    report = None
    if cross_path.exists() and not args.force:
        cached = json.loads(cross_path.read_text(encoding="utf-8"))
        if cached.get("fingerprint") == fingerprint:
            report = cached
            print("synthetic: reusing cached cross-check for this corpus")

    if report is None:
        try:
            labeled = prelabel_samples(
                check_llm, synthetic, budget, batch_size=args.batch_size
            )
        except Exception as exc:  # noqa: BLE001 - CLI reports and exits
            print(f"synthetic: cross-check prelabel failed: {exc}")
            return 1
        predicted = {sample.id: list(sample.labels) for sample in labeled}
        report = cross_check_predictions(synthetic, predicted)
        report["fingerprint"] = fingerprint
        cross_path.parent.mkdir(parents=True, exist_ok=True)
        cross_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    corrections: dict[str, list[str]] = {}
    if corrections_path.exists():
        corrections = json.loads(corrections_path.read_text(encoding="utf-8"))
    merged = merge_labeled(real, synthetic, corrections)
    write_jsonl(Path(args.labeled_out), [asdict(sample) for sample in merged])

    manifest = {
        "real_samples": len(real),
        "synthetic_samples": len(synthetic),
        "merged_samples": len(merged),
        "label_counts": label_counts(merged),
        "gap": gap,
        "cross_check_agreement": report["agreement_rate"],
        "llm_calls": budget.calls,
    }
    manifest_path = cross_path.parent / "corpus_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"synthetic: real={len(real)} synthetic={len(synthetic)} "
        f"merged={len(merged)} agreement={report['agreement_rate']:.3f} "
        f"calls={budget.calls} -> {args.labeled_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
