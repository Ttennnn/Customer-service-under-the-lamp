"""Evidence-confidence threshold calibration job.

The gate in :mod:`app.agent.confidence` combines four fixed-weight signals
into a 0-1 score.  This job measures that score on the annotated RAG evaluation
set so a threshold can be picked with a defensible criterion rather than a
guess:

1. Every non-excluded case in ``tests/eval/rag_cases.json`` is analysed once,
   then retrieved with ``hybrid_rerank`` at ``top_k=10`` (the same analysis is
   reused for retrieval).
2. Positive label = ``answerable=true`` (A/B/C/E buckets); negative label =
   the D bucket (``D_absent`` / ``D_unanswerable``).
3. ``retrieval_error`` and ``rerank_failed`` runs are excluded and counted:
   RRF-fallback scores are not comparable with rerank scores.
4. ``select_threshold`` sweeps 0.00-1.00 in 0.01 steps and maximises
   Youden's J (TPR - FPR); ties prefer higher precision, then the threshold
   closest to 0.5.

Outputs ``confidence_calibration.txt`` / ``.json`` under the configured report
directory.  It never edits settings: the live acceptance task copies the
recommended value into ``Settings.evidence_confidence_threshold``.

Importing this module is offline: no DB, Milvus or LLM client is built until
:func:`main` actually calls the injected/real ``analyze``/``retrieve_hybrid``.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from app.agent.confidence import EVIDENCE_WEIGHTS, compute_evidence_confidence
from app.config import get_settings
from app.rag.query_understanding import analyze
from app.rag.retrieval import RetrievalStrategy, retrieve_hybrid

DEFAULT_CASES = (
    Path(__file__).resolve().parents[2] / "tests" / "eval" / "rag_cases.json"
)
DEFAULT_TOP_K = 10
THRESHOLD_STEP = 0.01
THRESHOLD_COUNT = 101
NEGATIVE_BUCKETS = frozenset({"D_absent", "D_unanswerable"})
RETRIEVAL_ERROR = "retrieval_error"
RERANK_FAILED = "rerank_failed"
REPORT_TXT = "confidence_calibration.txt"
REPORT_JSON = "confidence_calibration.json"


def load_cases(path: Path) -> list[dict]:
    """Load the ``cases`` array from *path* (dataset order preserved)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = data.get("cases") if isinstance(data, dict) else data
    if not isinstance(cases, list):
        raise ValueError(f"{path}: expected a 'cases' list")
    return cases


def case_label(case: dict) -> bool:
    """True for answerable A/B/C/E cases, False for D-bucket negatives."""
    bucket = str(case.get("bucket") or "")
    if bucket in NEGATIVE_BUCKETS or case.get("answerable") is False:
        return False
    return bool(case.get("answerable", True))


def _case_filters(case: dict) -> tuple[str | None, str | None]:
    filters = case.get("filters") or {}
    return filters.get("category"), filters.get("content_type")


def run_case(
    case: dict,
    *,
    analyze_fn=None,
    retrieve_fn=None,
    top_k: int = DEFAULT_TOP_K,
    valid_floor: float = 0.1,
    valid_cap: int = 3,
    margin_scale: float = 0.2,
) -> tuple[dict | None, str | None]:
    """Evaluate one case; return ``(record, exclusion_reason)``.

    ``analyze_fn`` is called exactly once per attempted case and the returned
    analysis is passed straight to retrieval.
    """
    query = str(case.get("query") or "")
    category, content_type = _case_filters(case)
    # Resolve at call time so tests (and callers) can inject fakes.
    if analyze_fn is None:
        analyze_fn = analyze
    if retrieve_fn is None:
        retrieve_fn = retrieve_hybrid

    try:
        analysis = analyze_fn(query)
        result = retrieve_fn(
            query,
            strategy=RetrievalStrategy.HYBRID_RERANK,
            analysis=analysis,
            category=category,
            content_type=content_type,
            top_k=top_k,
        )
    except Exception:  # noqa: BLE001 - a failed run is excluded, not fatal
        return None, RETRIEVAL_ERROR

    if bool(getattr(result, "rerank_failed", False)):
        return None, RERANK_FAILED

    confidence = compute_evidence_confidence(
        result.chunks,
        valid_floor=valid_floor,
        valid_cap=valid_cap,
        margin_scale=margin_scale,
    )
    return (
        {
            "case_id": str(case.get("id") or ""),
            "bucket": str(case.get("bucket") or ""),
            "score": confidence.score,
            "label": case_label(case),
            "signals": confidence.signals,
        },
        None,
    )


def _confusion(scored: list[tuple[float, bool]], threshold: float) -> dict:
    tp = fp = tn = fn = 0
    for score, label in scored:
        predicted = score >= threshold
        if predicted and label:
            tp += 1
        elif predicted:
            fp += 1
        elif label:
            fn += 1
        else:
            tn += 1
    positives = tp + fn
    negatives = fp + tn
    tpr = tp / positives
    fpr = fp / negatives
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tpr": tpr,
        "fpr": fpr,
        "precision": precision,
        "youden_j": tpr - fpr,
    }


def scan_thresholds(scored: list[tuple[float, bool]]) -> list[dict]:
    """Evaluate every threshold from 0.00 to 1.00 in 0.01 steps."""
    if not scored:
        raise ValueError("no scored samples to calibrate")
    positives = sum(1 for _, label in scored if label)
    negatives = len(scored) - positives
    if not positives or not negatives:
        raise ValueError(
            "calibration needs both positive and negative samples"
        )
    return [
        _confusion(scored, round(index * THRESHOLD_STEP, 2))
        for index in range(THRESHOLD_COUNT)
    ]


def select_threshold(scored: list[tuple[float, bool]]) -> dict:
    """Pick the Youden-J-optimal threshold with deterministic tie-breaks.

    Ties are broken by higher precision, then by a threshold closer to 0.5.
    The returned dict is one point of :func:`scan_thresholds`.
    """
    curve = scan_thresholds(scored)
    return max(
        curve,
        key=lambda point: (
            round(point["youden_j"], 10),
            round(point["precision"], 10),
            -abs(point["threshold"] - 0.5),
        ),
    )


def _describe(scores: list[float]) -> dict:
    if not scores:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(scores),
        "min": min(scores),
        "max": max(scores),
        "mean": statistics.fmean(scores),
        "median": statistics.median(scores),
    }


def build_report(
    records: list[dict],
    exclusions: Counter[str],
    *,
    dataset_size: int,
    dataset_excluded: int,
    cases_path: Path,
    top_k: int,
    valid_floor: float,
    valid_cap: int,
    margin_scale: float,
    generated_at: datetime | None = None,
) -> dict:
    """Assemble the JSON report (also used by the text renderer)."""
    scored = [(record["score"], record["label"]) for record in records]
    curve = scan_thresholds(scored)
    recommended = select_threshold(scored)
    positive_scores = [r["score"] for r in records if r["label"]]
    negative_scores = [r["score"] for r in records if not r["label"]]
    return {
        "generated_at": (
            generated_at or datetime.now(timezone.utc)
        ).isoformat(),
        "source": str(cases_path),
        "formula": {
            "weights": dict(EVIDENCE_WEIGHTS),
            "valid_floor": valid_floor,
            "valid_cap": valid_cap,
            "margin_scale": margin_scale,
            "top_k": top_k,
            "score": (
                "0.50*top1 + 0.20*min(valid_count/valid_cap,1) + "
                "0.20*min(margin/margin_scale,1) + 0.10*key_clause_hit"
            ),
        },
        "samples": {
            "dataset_size": dataset_size,
            "dataset_excluded": dataset_excluded,
            "attempted": len(records) + sum(exclusions.values()),
            "evaluable": len(records),
            "positives": len(positive_scores),
            "negatives": len(negative_scores),
            "excluded": dict(sorted(exclusions.items())),
        },
        "distribution": {
            "positive": _describe(positive_scores),
            "negative": _describe(negative_scores),
        },
        "scored": [
            {
                "case_id": record["case_id"],
                "bucket": record["bucket"],
                "score": record["score"],
                "label": record["label"],
            }
            for record in records
        ],
        "curve": curve,
        "recommended": recommended,
    }


def render_text(report: dict) -> str:
    """Render a human-readable summary plus the full sweep curve."""
    formula = report["formula"]
    samples = report["samples"]
    recommended = report["recommended"]
    lines = [
        "confidence_calibration",
        f"generated_at: {report['generated_at']}",
        f"source: {report['source']}",
        "",
        "formula:",
        f"  {formula['score']}",
        f"  weights: {formula['weights']}",
        f"  valid_floor={formula['valid_floor']} "
        f"valid_cap={formula['valid_cap']} "
        f"margin_scale={formula['margin_scale']} top_k={formula['top_k']}",
        "",
        "samples:",
        f"  dataset_size={samples['dataset_size']} "
        f"dataset_excluded={samples['dataset_excluded']} "
        f"attempted={samples['attempted']} "
        f"evaluable={samples['evaluable']} "
        f"positives={samples['positives']} negatives={samples['negatives']}",
        f"  excluded={samples['excluded'] or '{}'}",
        "",
        "distribution:",
    ]
    for name in ("positive", "negative"):
        stats = report["distribution"][name]
        lines.append(f"  {name}: {stats}")
    lines += [
        "",
        "recommended:",
        f"  threshold={recommended['threshold']:.2f} "
        f"tpr={recommended['tpr']:.3f} fpr={recommended['fpr']:.3f} "
        f"youden_j={recommended['youden_j']:.3f} "
        f"precision={recommended['precision']:.3f}",
        "  (copy into evidence_confidence_threshold; this job never edits config)",
        "",
        "curve (threshold,tp,fp,tn,fn,tpr,fpr,precision,j):",
    ]
    for point in report["curve"]:
        lines.append(
            f"  {point['threshold']:.2f},{point['tp']},{point['fp']},"
            f"{point['tn']},{point['fn']},{point['tpr']:.3f},"
            f"{point['fpr']:.3f},{point['precision']:.3f},"
            f"{point['youden_j']:.3f}"
        )
    return "\n".join(lines) + "\n"


def write_reports(report: dict, out_dir: Path) -> tuple[Path, Path]:
    """Write both report formats and return their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / REPORT_TXT
    json_path = out_dir / REPORT_JSON
    txt_path.write_text(render_text(report), encoding="utf-8")
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return txt_path, json_path


def _run(
    cases: list[dict],
    *,
    top_k: int,
    valid_floor: float,
    valid_cap: int,
    margin_scale: float,
    cases_path: Path,
    generated_at: datetime | None = None,
) -> dict:
    selected = [case for case in cases if not case.get("excluded")]
    exclusions: Counter[str] = Counter()
    records: list[dict] = []
    for case in selected:
        record, reason = run_case(
            case,
            top_k=top_k,
            valid_floor=valid_floor,
            valid_cap=valid_cap,
            margin_scale=margin_scale,
        )
        if reason is not None:
            exclusions[reason] += 1
        else:
            records.append(record)
    return build_report(
        records,
        exclusions,
        dataset_size=len(cases),
        dataset_excluded=len(cases) - len(selected),
        cases_path=cases_path,
        top_k=top_k,
        valid_floor=valid_floor,
        valid_cap=valid_cap,
        margin_scale=margin_scale,
        generated_at=generated_at,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: calibrate and write the confidence reports."""
    parser = argparse.ArgumentParser(
        description="校准 ch09 证据置信度阈值（Youden J）"
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES,
        help=f"评估用例 JSON（默认 {DEFAULT_CASES}）",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="报告输出目录（默认取 settings.ch09_report_dir）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"hybrid_rerank 最终条数（默认 {DEFAULT_TOP_K}）",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="同时把 JSON 报告打印到 stdout（仍会写文件）",
    )
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        parser.error("--top-k must be a positive integer")

    settings = get_settings()
    out_dir = args.out_dir or Path(settings.ch09_report_dir)
    cases = load_cases(args.cases)
    try:
        report = _run(
            cases,
            top_k=args.top_k,
            valid_floor=settings.evidence_valid_floor,
            valid_cap=settings.evidence_valid_cap,
            margin_scale=settings.evidence_margin_scale,
            cases_path=args.cases,
        )
    except ValueError as exc:
        print(f"calibrate_confidence: {exc}")
        return 2

    txt_path, json_path = write_reports(report, out_dir)
    recommended = report["recommended"]
    print(
        "recommended confidence threshold: "
        f"{recommended['threshold']:.2f} "
        f"(Youden J={recommended['youden_j']:.3f}, "
        f"TPR={recommended['tpr']:.3f}, FPR={recommended['fpr']:.3f}, "
        f"precision={recommended['precision']:.3f})"
    )
    print(f"reports: {txt_path} {json_path}")
    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
