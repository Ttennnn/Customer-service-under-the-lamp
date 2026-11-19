"""评估流水线:运行 RAG 评估并把指标写入 ``eval_runs`` 趋势。

``run_once`` 调用 :mod:`app.jobs.eval_rag` 只跑 ``hybrid_rerank`` 策略,然后读取
eval_rag 写出的 JSON sidecar,把 Recall@10 / MRR@10 / Faithfulness / D 桶拒答
准确率落库。``eval_runs`` 是趋势的唯一权威来源;``eval_trend.txt`` 只是给人
看的便捷副本。

导入本模块不会连接数据库、Milvus 或 LLM;只有 ``run_once`` / ``main`` 执行时
才解析 settings 或打开会话。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal

from app.config import get_settings
from app.db import session as db_session
from app.models import EvalRun

TRIGGER_CHOICES = ("手动", "定时")
DEFAULT_TRIGGERED_BY = "手动"
STRATEGY = "hybrid_rerank"
CASES_PATH = "tests/eval/rag_cases.json"
REPORT_NAME = "eval_flywheel.md"
TREND_NAME = "eval_trend.txt"
TREND_LIMIT = 10

# (metrics key, display label), in report order.
METRIC_LABELS = (
    ("recall_at_10", "Recall@10"),
    ("mrr_at_10", "MRR@10"),
    ("faithfulness", "Faithfulness"),
    ("d_refusal_accuracy", "D accuracy"),
)

Runner = Callable[[list[str]], int]


def load_metrics_from_report(path: Path) -> dict:
    """Extract the trend metrics from an eval_rag JSON sidecar.

    The sidecar is ``overall.hybrid_rerank`` in
    ``eval_rag._build_summary``. ``cases_loaded`` is kept so ``run_once`` can
    set ``EvalRun.dataset_size``; ``cases_evaluated``/``excluded`` describe the
    sample the metrics were computed on. A report without the
    ``hybrid_rerank`` block is rejected rather than stored as all-null.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("eval report must be a JSON object")
    block = (data.get("overall") or {}).get(STRATEGY)
    if not isinstance(block, dict):
        raise ValueError(f"eval report has no {STRATEGY} metrics")

    retrieval = block.get("retrieval") or {}
    generation = block.get("generation") or {}
    d_accuracy = block.get("d_accuracy") or {}
    return {
        "strategy": STRATEGY,
        "cases_loaded": int(data.get("cases_loaded") or data.get("cases") or 0),
        "cases_evaluated": int(data.get("cases") or 0),
        "excluded": int((data.get("excluded") or {}).get("count") or 0),
        "recall_at_10": retrieval.get("recall_at_10"),
        "mrr_at_10": retrieval.get("mrr_at_10"),
        "faithfulness": generation.get("faithfulness"),
        "d_refusal_accuracy": d_accuracy.get("accuracy"),
    }


def _default_runner(argv: list[str]) -> int:
    """Run the real RAG evaluation; imported lazily to keep module import light."""
    from app.jobs import eval_rag

    return eval_rag.main(argv)


def _report_path() -> Path:
    return Path(get_settings().ch09_report_dir) / REPORT_NAME


def run_once(
    triggered_by: Literal["手动", "定时"],
    limit: int | None = None,
    *,
    runner: Runner | None = None,
) -> dict:
    """Run one evaluation round and persist it as an ``EvalRun``.

    ``runner`` is injectable for tests; it must write the markdown report to
    the ``--out`` path and the JSON sidecar next to it. A non-zero exit or a
    missing/unreadable report means no row is inserted, so a failed run can
    never look like a real trend point.
    """
    if triggered_by not in TRIGGER_CHOICES:
        raise ValueError(f"unknown triggered_by: {triggered_by!r}")

    report_path = _report_path()
    argv = [
        "--cases",
        CASES_PATH,
        "--strategies",
        STRATEGY,
        "--out",
        str(report_path),
    ]
    if limit is not None:
        argv.extend(["--limit", str(limit)])

    run = runner or _default_runner
    try:
        code = run(argv)
    except Exception as exc:  # noqa: BLE001 - CLI reports, never crashes
        return {
            "ok": False,
            "error": f"eval_rag raised {type(exc).__name__}: {exc}",
        }
    if code != 0:
        return {"ok": False, "error": f"eval_rag exited with code {code}"}

    sidecar = report_path.with_suffix(".json")
    try:
        metrics = load_metrics_from_report(sidecar)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return {"ok": False, "error": f"cannot read {sidecar}: {exc}"}

    session = None
    try:
        session = db_session.get_session_factory()()
        row = EvalRun(
            triggered_by=triggered_by,
            dataset_size=metrics["cases_loaded"],
            metrics=metrics,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        run_id = row.id
    except Exception as exc:  # noqa: BLE001 - report the DB failure
        if session is not None:
            session.rollback()
        return {
            "ok": False,
            "error": f"cannot persist eval run: {type(exc).__name__}: {exc}",
        }
    finally:
        if session is not None:
            session.close()

    return {"ok": True, "run_id": run_id, "metrics": metrics}


def _fmt_metric(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _change(previous, current) -> str:
    """Return ↑ / ↓ ⚠ / → / — for *current* relative to *previous*."""
    if previous is None or current is None:
        return "—"
    try:
        before = float(previous)
        after = float(current)
    except (TypeError, ValueError):
        return "—"
    if after > before:
        return "↑"
    if after < before:
        return "↓ ⚠"
    return "→"


def _run_sort_key(run: EvalRun):
    return (
        getattr(run, "created_at", None) or datetime.min,
        getattr(run, "id", 0) or 0,
    )


def format_trend(runs: list[EvalRun]) -> str:
    """Render the latest runs with each metric compared to the previous run.

    A falling metric is explicitly flagged with ⚠. The first run has no
    baseline and shows ``—``; missing metric values show ``n/a``.
    """
    ordered = sorted(runs, key=_run_sort_key)
    lines = [f"# eval 趋势（最近 {TREND_LIMIT} 轮）", ""]
    if not ordered:
        lines.append("- 暂无 eval_runs 记录")
        return "\n".join(lines) + "\n"

    previous_metrics: dict | None = None
    for run in ordered:
        created_at = getattr(run, "created_at", None)
        stamp = (
            created_at.strftime("%Y-%m-%d %H:%M:%S")
            if isinstance(created_at, datetime)
            else "unknown"
        )
        lines.append(
            f"run #{getattr(run, 'id', '?')} | {stamp} | "
            f"{getattr(run, 'triggered_by', '?')} | "
            f"dataset={getattr(run, 'dataset_size', '?')}"
        )
        metrics = getattr(run, "metrics", None) or {}
        for key, label in METRIC_LABELS:
            current = metrics.get(key)
            if previous_metrics is None:
                lines.append(f"  {label:<13} {_fmt_metric(current)} —")
            else:
                before = previous_metrics.get(key)
                lines.append(
                    f"  {label:<13} {_fmt_metric(before)} → "
                    f"{_fmt_metric(current)} {_change(before, current)}"
                )
        lines.append("")
        previous_metrics = metrics

    return "\n".join(lines).rstrip() + "\n"


def _recent_runs(session, limit: int = TREND_LIMIT) -> list[EvalRun]:
    """Return up to *limit* newest rows in chronological order."""
    rows = (
        session.query(EvalRun).order_by(EvalRun.id.desc()).limit(limit).all()
    )
    return list(reversed(rows))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行 RAG 评估并把指标写入 eval_runs 趋势"
    )
    parser.add_argument(
        "--triggered-by",
        choices=TRIGGER_CHOICES,
        default=DEFAULT_TRIGGERED_BY,
        help=f"触发方式（默认 {DEFAULT_TRIGGERED_BY}）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只评估前 N 条用例（默认全部）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: run the eval, persist one row, print the trend."""
    args = _build_parser().parse_args(argv)

    result = run_once(args.triggered_by, limit=args.limit)
    if not result.get("ok"):
        print(f"eval_flywheel: {result.get('error')}")
        return 1
    print(
        f"eval_flywheel: saved run #{result['run_id']} "
        f"(triggered_by={args.triggered_by})"
    )

    session = None
    try:
        session = db_session.get_session_factory()()
        runs = _recent_runs(session)
    except Exception as exc:  # noqa: BLE001 - CLI reports and exits non-zero
        print(f"eval_flywheel: cannot read eval_runs: {exc}")
        return 1
    finally:
        if session is not None:
            session.close()

    trend = format_trend(runs)
    trend_path = Path(get_settings().ch09_report_dir) / TREND_NAME
    try:
        trend_path.parent.mkdir(parents=True, exist_ok=True)
        trend_path.write_text(trend, encoding="utf-8")
    except OSError as exc:
        print(f"eval_flywheel: cannot write {trend_path}: {exc}")
        return 1

    print(trend, end="")
    print(f"eval_flywheel: trend written to {trend_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
