"""Golden-sample gate for the ch10 prelabel prompt (>= 80% exact sets)."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from app.core.taxonomy import normalize_labels
from scripts.ch10.corpus_lib import Sample, read_jsonl
from scripts.ch10.llm_io import CallBudget
from scripts.ch10.prelabel import prelabel_samples

DEFAULT_GOLDEN = "data/ch10/golden_samples.jsonl"
DEFAULT_OUT = "data/ch10/reports/golden_report.json"
DEFAULT_MIN_PASS_RATE = 0.8


def compare_labels(golden: list[dict], predicted: dict[str, list[str]]) -> dict:
    """Compare predicted label sets to gold, order-insensitively."""
    passed = 0
    mismatches: list[dict] = []
    for row in golden:
        gold = normalize_labels(row.get("labels", []))
        got = list(predicted.get(row["id"], []))
        if set(gold) == set(got):
            passed += 1
            continue
        mismatches.append(
            {"id": row["id"], "gold": gold, "predicted": got}
        )
    total = len(golden)
    return {
        "total": total,
        "passed": passed,
        "pass_rate": (passed / total) if total else 0.0,
        "mismatches": mismatches,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the prelabel prompt")
    parser.add_argument("--golden", default=DEFAULT_GOLDEN)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--min-pass-rate", type=float, default=DEFAULT_MIN_PASS_RATE)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-llm-calls", type=int, default=None)
    args = parser.parse_args(argv)

    from app.config import build_llm, get_settings

    rows = read_jsonl(Path(args.golden))
    if not rows:
        print("golden: no golden samples")
        return 1

    samples = [
        Sample(id=row["id"], text=row["text"], origin="golden")
        for row in rows
    ]
    try:
        llm = build_llm(get_settings(), temperature=0.0)
        budget = CallBudget(args.max_llm_calls)
        labeled = prelabel_samples(
            llm, samples, budget, batch_size=args.batch_size
        )
    except Exception as exc:  # noqa: BLE001 - CLI reports and exits non-zero
        print(f"golden: prelabel failed: {exc}")
        return 1

    predicted = {sample.id: list(sample.labels) for sample in labeled}
    report = compare_labels(rows, predicted)
    report.update(
        {
            "min_pass_rate": args.min_pass_rate,
            "model": get_settings().model_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"golden: pass_rate={report['pass_rate']:.3f} "
        f"({report['passed']}/{report['total']}) -> {out}"
    )
    return 0 if report["pass_rate"] >= args.min_pass_rate else 1


if __name__ == "__main__":
    raise SystemExit(main())
