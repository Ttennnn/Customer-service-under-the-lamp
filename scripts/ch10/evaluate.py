"""Test-set evaluation: per-class red lines, confusion matrices, error review."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx

from app.core.taxonomy import TOPIC_CLASSES
from scripts.ch10.corpus_lib import read_jsonl
from scripts.ch10.eval_lib import (
    check_service_threshold,
    error_directions,
    reconcile_errors,
    red_line_for,
    tier_for,
)
from scripts.ch10.train_lib import compute_metrics
from scripts.ch10.validate_golden import compare_labels


def _call_service(base_url: str, texts: list[str]) -> list[dict]:
    response = httpx.post(
        f"{base_url.rstrip('/')}/classify",
        json={"texts": texts},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["items"]


def _predict(base_url: str, rows: list[dict], batch_size: int = 32) -> list[dict]:
    items: list[dict] = []
    for index in range(0, len(rows), batch_size):
        chunk = rows[index : index + batch_size]
        batch = _call_service(base_url, [row["text"] for row in chunk])
        if len(batch) != len(chunk):
            raise ValueError(
                f"service returned {len(batch)} items for {len(chunk)} texts"
            )
        items.extend(batch)
    return items


def _matrix_dict(matrix) -> dict:
    tn, fp, fn, tp = [int(value) for value in matrix.ravel()]
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def _class_rows(matrices: dict) -> list[dict]:
    rows = []
    for label in TOPIC_CLASSES:
        item = matrices[label]
        tp, fp, fn = item["tp"], item["fp"], item["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall)
            else 0.0
        )
        line = red_line_for(label)
        rows.append(
            {
                "label": label,
                "tier": tier_for(label),
                "red_line": line,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": tp + fn,
                "pass": (f1 >= line) if line is not None else None,
            }
        )
    return rows


def _render_markdown(report: dict) -> str:
    lines = [
        "# ch10 主题分类器评测报告",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 测试集：{report['dataset_size']} 条",
        f"- 阈值：{report['threshold']['threshold']:.2f}"
        f"（source={report['threshold']['source']}）",
        f"- 语料构成：{report['corpus_composition']}",
        "",
        "## 汇总",
        "",
        "| 指标 | 值 |",
        "|---|---|",
    ]
    for key in (
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    ):
        lines.append(f"| {key} | {report['overall'][key]:.4f} |")

    lines += [
        "",
        "## 每类指标",
        "",
        "| 类目 | 档位 | 红线 | P | R | F1 | support | 判定 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in report["per_class"]:
        line = "—" if row["red_line"] is None else f"{row['red_line']:.2f}"
        verdict = (
            "—"
            if row["pass"] is None
            else ("✅" if row["pass"] else "❌ 先回头搞数据")
        )
        lines.append(
            f"| {row['label']} | {row['tier']} | {line} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['support']} | {verdict} |"
        )

    lines += [
        "",
        "## 混淆矩阵（2×2）",
        "",
        "| 类目 | TP | FP | FN | TN |",
        "|---|---|---|---|---|",
    ]
    for label, item in report["confusion_matrices"].items():
        lines.append(
            f"| {label} | {item['tp']} | {item['fp']} | "
            f"{item['fn']} | {item['tn']} |"
        )

    counts = report["error_counts"]
    lines += [
        "",
        "## 错误方向记账",
        "",
        f"- 漏打：{counts['miss_total']}",
        f"- 多打：{counts['extra_total']}",
        f"- 错位：{counts['misplaced_total']}",
        f"- 对账：{'通过' if counts['reconciled'] else '不通过'}",
    ]

    golden = report.get("golden_model")
    if golden:
        lines += [
            "",
            "## 模型黄金样例",
            "",
            f"- 通过：{golden['passed']}/{golden['total']}"
            f"（{golden['pass_rate']:.1%}）",
            f"- 多诉求：{golden['multi_claim']['passed']}"
            f"/{golden['multi_claim']['total']}",
        ]
        for mismatch in golden.get("mismatches", []):
            lines.append(
                f"- {mismatch['id']}: gold={mismatch['gold']} "
                f"pred={mismatch['predicted']}"
            )
    return "\n".join(lines) + "\n"


def evaluate(
    test_path: Path,
    threshold_path: Path,
    base_url: str,
    out_dir: Path,
    golden_path: Path | None = None,
) -> dict:
    rows = read_jsonl(Path(test_path))
    if not rows:
        raise ValueError("test split is empty")
    threshold_data = json.loads(
        Path(threshold_path).read_text(encoding="utf-8")
    )
    health = httpx.get(f"{base_url.rstrip('/')}/healthz", timeout=10)
    health.raise_for_status()
    problem = check_service_threshold(health.json(), threshold_data)
    if problem:
        raise ValueError(problem)
    items = _predict(base_url, rows)

    gold_sets = [
        {label for label in row.get("labels", [])} for row in rows
    ]
    pred_sets = [set(item.get("labels", [])) for item in items]
    overall = compute_metrics(gold_sets, pred_sets)

    golden_model = None
    if golden_path is not None and Path(golden_path).exists():
        golden_rows = read_jsonl(Path(golden_path))
        golden_items = _predict(base_url, golden_rows)
        golden_pred = {
            row["id"]: list(item.get("labels", []))
            for row, item in zip(golden_rows, golden_items)
        }
        golden_model = compare_labels(golden_rows, golden_pred)
        multi_rows = [
            row
            for row in golden_rows
            if len(set(row.get("labels", []))) > 1
        ]
        golden_model["multi_claim"] = {
            "total": len(multi_rows),
            "passed": sum(
                1
                for row in multi_rows
                if set(golden_pred.get(row["id"], []))
                == set(row["labels"])
            ),
        }

    try:
        from sklearn.metrics import multilabel_confusion_matrix
    except ImportError as exc:  # pragma: no cover - .venv-ml has sklearn
        raise RuntimeError("scikit-learn is required for evaluation") from exc

    y_true = [
        [1 if label in gold else 0 for label in TOPIC_CLASSES]
        for gold in gold_sets
    ]
    y_pred = [
        [1 if label in pred else 0 for label in TOPIC_CLASSES]
        for pred in pred_sets
    ]
    matrices = multilabel_confusion_matrix(y_true, y_pred)
    matrix_map = {
        label: _matrix_dict(matrix)
        for label, matrix in zip(TOPIC_CLASSES, matrices)
    }
    per_class = _class_rows(matrix_map)
    directions = error_directions(gold_sets, pred_sets)

    errors = []
    for row, gold, pred in zip(rows, gold_sets, pred_sets):
        if gold == pred:
            continue
        errors.append(
            {
                "id": row.get("id"),
                "text": row["text"],
                "gold": sorted(gold),
                "pred": sorted(pred),
            }
        )

    error_counts = {
        **directions,
        "misplaced_total": len(directions["misplaced_pairs"]),
        "reconciled": False,
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_size": len(rows),
        "threshold": {
            "threshold": float(threshold_data["threshold"]),
            "source": threshold_data.get("source", "unknown"),
        },
        "corpus_composition": dict(
            Counter(row.get("origin", "unknown") for row in rows)
        ),
        "overall": overall,
        "per_class": per_class,
        "confusion_matrices": matrix_map,
        "error_counts": error_counts,
        "errors": errors,
        "golden_model": golden_model,
    }
    error_counts["reconciled"] = reconcile_errors(report)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    markdown = _render_markdown(report)
    (out_dir / "eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "eval_report.md").write_text(markdown, encoding="utf-8")

    error_lines = [
        "# ch10 判错样本",
        "",
        "| id | 句子 | 标准标签 | 预测标签 |",
        "|---|---|---|---|",
    ]
    for item in errors:
        error_lines.append(
            "| {id} | {text} | {gold} | {pred} |".format(
                id=item["id"],
                text=item["text"].replace("|", "\\|"),
                gold="、".join(item["gold"]),
                pred="、".join(item["pred"]),
            )
        )
    (out_dir / "error_samples.md").write_text(
        "\n".join(error_lines) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate ch10 classifier")
    parser.add_argument("--test", default="data/ch10/dataset/test.jsonl")
    parser.add_argument(
        "--threshold", default="data/ch10/model/threshold.json"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8110")
    parser.add_argument("--out", default="data/ch10/reports")
    parser.add_argument(
        "--golden", default="data/ch10/golden_samples.jsonl"
    )
    args = parser.parse_args(argv)

    try:
        report = evaluate(
            Path(args.test),
            Path(args.threshold),
            args.base_url,
            Path(args.out),
            Path(args.golden),
        )
    except httpx.ConnectError:
        print("evaluate: classifier service is not up; run make classifier-up")
        return 1
    except Exception as exc:  # noqa: BLE001 - CLI reports and exits
        print(f"evaluate: failed: {exc}")
        return 1

    red = [
        row["label"]
        for row in report["per_class"]
        if row["pass"] is False
    ]
    print(
        f"evaluate: micro_f1={report['overall']['micro_f1']:.4f} "
        f"macro_f1={report['overall']['macro_f1']:.4f} "
        f"errors={len(report['errors'])} red={red}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
