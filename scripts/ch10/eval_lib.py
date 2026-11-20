"""Evaluation tiers, error-direction accounting, and threshold replay."""

from __future__ import annotations

from app.core.taxonomy import TOPIC_CLASSES, labels_from_scores
from scripts.ch10.train_lib import compute_metrics

STRICT = ("退换货", "物流", "尺码", "发票", "质量问题")
MEDIUM = (
    "运费",
    "优惠活动",
    "价保",
    "支付",
    "订单修改",
    "库存补货",
    "商品信息",
    "保修维修",
)
BROAD = ("账号", "会员积分", "评价", "其他")

_TIERS = {**{label: "严" for label in STRICT}, **{label: "中" for label in MEDIUM}, **{label: "宽" for label in BROAD}}
_RED_LINES = {**{label: 0.9 for label in STRICT}, **{label: 0.8 for label in MEDIUM}}


def tier_for(label: str) -> str:
    return _TIERS.get(label, "宽")


def red_line_for(label: str) -> float | None:
    return _RED_LINES.get(label)


def error_directions(
    gold: list[set[str]], pred: list[set[str]]
) -> dict:
    if len(gold) != len(pred):
        raise ValueError("gold and pred must have the same length")
    miss: dict[str, int] = {}
    extra: dict[str, int] = {}
    misplaced: list[dict] = []
    for gold_row, pred_row in zip(gold, pred):
        missed = sorted(gold_row - pred_row)
        extras = sorted(pred_row - gold_row)
        for label in missed:
            miss[label] = miss.get(label, 0) + 1
        for label in extras:
            extra[label] = extra.get(label, 0) + 1
        for gold_label, pred_label in zip(missed, extras):
            misplaced.append({"gold": gold_label, "pred": pred_label})
    return {
        "miss": miss,
        "extra": extra,
        "misplaced_pairs": misplaced,
        "miss_total": sum(miss.values()),
        "extra_total": sum(extra.values()),
    }


def reconcile_errors(report: dict) -> bool:
    matrices = report.get("confusion_matrices") or {}
    total_fp = sum(int(item.get("fp", 0)) for item in matrices.values())
    total_fn = sum(int(item.get("fn", 0)) for item in matrices.values())
    counts = report.get("error_counts") or {}
    return (
        int(counts.get("miss_total", 0)) + int(counts.get("extra_total", 0))
        == total_fp + total_fn
    )


def check_service_threshold(health: dict, threshold_data: dict) -> str | None:
    """Return an error string when the live service is not using this file."""
    source = threshold_data.get("source")
    if source != "onnx":
        return f"threshold source is {source!r}, expected 'onnx'"
    try:
        service_threshold = float(health.get("threshold"))
    except (TypeError, ValueError):
        return "service health has no usable threshold"
    expected = float(threshold_data.get("threshold", 0.0))
    if abs(service_threshold - expected) > 1e-9:
        return (
            f"service threshold {service_threshold} != file {expected}; "
            "restart the classifier"
        )
    return None


def replay_threshold(
    val_scores: list[dict],
    candidates,
    expected: float,
) -> dict:
    scan: list[dict] = []
    for threshold in candidates:
        pred = []
        gold = []
        for row in val_scores:
            probs = row["probs"]
            scores = {
                label: float(probs[index])
                for index, label in enumerate(TOPIC_CLASSES)
                if index < len(probs)
            }
            pred.append(set(labels_from_scores(scores, threshold)))
            gold.append(set(row.get("gold", [])))
        metrics = compute_metrics(gold, pred)
        scan.append(
            {
                "threshold": threshold,
                "micro_f1": metrics["micro_f1"],
                "macro_f1": metrics["macro_f1"],
            }
        )
    best = max(scan, key=lambda row: (row["micro_f1"], row["threshold"]))
    return {
        "best_threshold": best["threshold"],
        "best_micro_f1": best["micro_f1"],
        "expected": expected,
        "matches": abs(float(best["threshold"]) - float(expected)) < 1e-9,
        "scan": scan,
    }
