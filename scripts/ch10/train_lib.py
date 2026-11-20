"""Pure multi-label helpers shared by training and evaluation (no torch)."""

from __future__ import annotations

from app.core.taxonomy import LABEL2ID, TOPIC_CLASSES, labels_from_scores, normalize_labels

DEFAULT_CANDIDATES = tuple(round(0.30 + 0.05 * i, 2) for i in range(9))


def multi_hot(labels, label2id: dict[str, int] | None = None) -> list[float]:
    label2id = label2id or LABEL2ID
    vector = [0.0] * len(label2id)
    for label in normalize_labels(labels):
        vector[label2id[label]] = 1.0
    return vector


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall)
        else 0.0
    )
    return precision, recall, f1


def compute_metrics(
    gold: list[set[str]],
    pred: list[set[str]],
) -> dict:
    if len(gold) != len(pred):
        raise ValueError("gold and pred must have the same length")

    labels = sorted(
        {label for item in gold for label in item}
        | {label for item in pred for label in item}
    )
    tp_micro = fp_micro = fn_micro = 0
    per_label: dict[str, tuple[float, float, float]] = {}
    for label in labels:
        tp = sum(1 for g, p in zip(gold, pred) if label in g and label in p)
        fp = sum(1 for g, p in zip(gold, pred) if label not in g and label in p)
        fn = sum(1 for g, p in zip(gold, pred) if label in g and label not in p)
        tp_micro += tp
        fp_micro += fp
        fn_micro += fn
        per_label[label] = _prf(tp, fp, fn)

    micro_p, micro_r, micro_f1 = _prf(tp_micro, fp_micro, fn_micro)
    if per_label:
        macro_p = sum(v[0] for v in per_label.values()) / len(per_label)
        macro_r = sum(v[1] for v in per_label.values()) / len(per_label)
        macro_f1 = sum(v[2] for v in per_label.values()) / len(per_label)
    else:
        macro_p = macro_r = macro_f1 = 0.0
    return {
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
    }


def scan_threshold(
    probs: list[list[float]],
    gold: list[set[str]],
    candidates=DEFAULT_CANDIDATES,
) -> dict:
    """Scan a global threshold; ties choose the highest candidate."""
    scan: list[dict] = []
    for threshold in candidates:
        pred = []
        for row in probs:
            scores = {
                label: float(row[index])
                for index, label in enumerate(TOPIC_CLASSES)
                if index < len(row)
            }
            pred.append(set(labels_from_scores(scores, threshold)))
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
        "scan": scan,
    }
