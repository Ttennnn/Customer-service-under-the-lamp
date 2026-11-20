"""Re-score validation through :8110 once, then replay the threshold scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

from app.core.taxonomy import TOPIC_CLASSES
from scripts.ch10.corpus_lib import read_jsonl
from scripts.ch10.eval_lib import check_service_threshold, replay_threshold
from scripts.ch10.train_lib import DEFAULT_CANDIDATES


def _scores(base_url: str, texts: list[str]) -> list[dict]:
    response = httpx.post(
        f"{base_url.rstrip('/')}/classify",
        json={"texts": texts},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["items"]


def replay(
    val_path: Path,
    base_url: str,
    threshold_path: Path,
    out_dir: Path,
    batch_size: int = 32,
) -> dict:
    rows = read_jsonl(Path(val_path))
    if not rows:
        raise ValueError("validation split is empty")
    items: list[dict] = []
    for index in range(0, len(rows), batch_size):
        chunk = rows[index : index + batch_size]
        batch = _scores(base_url, [row["text"] for row in chunk])
        if len(batch) != len(chunk):
            raise ValueError(
                f"service returned {len(batch)} items for {len(chunk)} texts"
            )
        items.extend(batch)

    scores = []
    for row, item in zip(rows, items):
        probs = item.get("scores") or {}
        scores.append(
            {
                "id": row.get("id"),
                "probs": [float(probs.get(label, 0.0)) for label in TOPIC_CLASSES],
                "gold": sorted(set(row.get("labels", []))),
            }
        )

    threshold_data = json.loads(
        Path(threshold_path).read_text(encoding="utf-8")
    )
    health = httpx.get(f"{base_url.rstrip('/')}/healthz", timeout=10)
    health.raise_for_status()
    problem = check_service_threshold(health.json(), threshold_data)
    if problem:
        raise ValueError(problem)
    expected = float(threshold_data["threshold"])
    result = replay_threshold(scores, DEFAULT_CANDIDATES, expected)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "val_scores.json").write_text(
        json.dumps(
            {
                "thresholds": list(DEFAULT_CANDIDATES),
                "rows": scores,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "threshold_scan.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay the ch10 threshold scan through the service"
    )
    parser.add_argument("--val", default="data/ch10/dataset/val.jsonl")
    parser.add_argument("--base-url", default="http://127.0.0.1:8110")
    parser.add_argument(
        "--threshold", default="data/ch10/model/threshold.json"
    )
    parser.add_argument("--out", default="data/ch10/reports")
    args = parser.parse_args(argv)

    try:
        result = replay(
            Path(args.val),
            args.base_url,
            Path(args.threshold),
            Path(args.out),
        )
    except httpx.ConnectError:
        print(
            "threshold-scan: classifier service is not up; "
            "run make classifier-up"
        )
        return 1
    except Exception as exc:  # noqa: BLE001 - CLI reports and exits
        print(f"threshold-scan: failed: {exc}")
        return 1

    print(
        f"threshold-scan: expected={result['expected']:.2f} "
        f"best={result['best_threshold']:.2f} "
        f"micro_f1={result['best_micro_f1']:.4f} "
        f"matches={result['matches']}"
    )
    return 0 if result["matches"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
