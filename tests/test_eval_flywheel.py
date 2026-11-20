"""Offline tests for the ch09 eval flywheel job.

``run_once`` is exercised with an injected runner that only writes a report
and its JSON sidecar, so the tests never call Milvus, MySQL or an LLM. The
``session`` fixture supplies the file-backed SQLite database used by the
job's own session factory.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.jobs import eval_flywheel as job
from app.models import EvalRun


def _report() -> dict:
    return {
        "cases_loaded": 300,
        "cases": 298,
        "excluded": {"count": 2},
        "overall": {
            "hybrid_rerank": {
                "retrieval": {"recall_at_10": 0.9, "mrr_at_10": 0.8},
                "generation": {"faithfulness": 0.95},
                "d_accuracy": {"accuracy": 0.97},
            }
        },
    }


def _settings(monkeypatch, tmp_path):
    settings = SimpleNamespace(ch09_report_dir=str(tmp_path / "reports"))
    monkeypatch.setattr(job, "get_settings", lambda: settings)
    return settings


def _runner(report: dict):
    """Return ``(runner, calls)``; the runner writes both report files."""
    calls: list[list[str]] = []

    def run(argv):
        calls.append(list(argv))
        out = Path(argv[argv.index("--out") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("# report\n", encoding="utf-8")
        out.with_suffix(".json").write_text(
            json.dumps(report, ensure_ascii=False), encoding="utf-8"
        )
        return 0

    return run, calls


# --- load_metrics_from_report ------------------------------------------------


def test_load_metrics_from_report(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(_report()), encoding="utf-8")

    metrics = job.load_metrics_from_report(path)

    assert metrics["recall_at_10"] == 0.9
    assert metrics["mrr_at_10"] == 0.8
    assert metrics["faithfulness"] == 0.95
    assert metrics["d_refusal_accuracy"] == 0.97
    assert metrics["strategy"] == "hybrid_rerank"
    assert metrics["cases_loaded"] == 300
    assert metrics["cases_evaluated"] == 298
    assert metrics["excluded"] == 2


def test_load_metrics_rejects_report_without_strategy(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"overall": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="hybrid_rerank"):
        job.load_metrics_from_report(path)


# --- run_once ----------------------------------------------------------------


def test_run_once_inserts_eval_run(session, monkeypatch, tmp_path):
    _settings(monkeypatch, tmp_path)
    runner, calls = _runner(_report())

    out = job.run_once("手动", runner=runner)

    assert out["ok"] is True
    assert out["metrics"]["recall_at_10"] == 0.9
    row = session.query(EvalRun).one()
    assert row.triggered_by == "手动"
    assert row.dataset_size == 300
    assert row.metrics["recall_at_10"] == 0.9
    assert row.metrics["d_refusal_accuracy"] == 0.97
    assert out["run_id"] == row.id
    # Default dataset/strategy, report under ch09_report_dir.
    argv = calls[0]
    assert argv[argv.index("--cases") + 1] == "tests/eval/rag_cases.json"
    assert argv[argv.index("--strategies") + 1] == "hybrid_rerank"
    assert argv[argv.index("--out") + 1] == str(
        tmp_path / "reports" / "eval_flywheel.md"
    )
    assert "--limit" not in argv


def test_run_once_passes_limit(session, monkeypatch, tmp_path):
    _settings(monkeypatch, tmp_path)
    runner, calls = _runner(_report())

    out = job.run_once("定时", limit=5, runner=runner)

    assert out["ok"] is True
    argv = calls[0]
    assert argv[argv.index("--limit") + 1] == "5"
    assert session.query(EvalRun).one().triggered_by == "定时"


def test_failed_runner_does_not_insert(session, monkeypatch, tmp_path):
    _settings(monkeypatch, tmp_path)

    out = job.run_once("手动", runner=lambda argv: 2)

    assert out["ok"] is False
    assert "2" in out["error"]
    assert session.query(EvalRun).count() == 0


def test_missing_report_does_not_insert(session, monkeypatch, tmp_path):
    _settings(monkeypatch, tmp_path)

    def runner(argv):
        out = Path(argv[argv.index("--out") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("# no sidecar\n", encoding="utf-8")
        return 0

    out = job.run_once("手动", runner=runner)

    assert out["ok"] is False
    assert session.query(EvalRun).count() == 0


def test_stale_sidecar_is_removed_before_a_failed_run(session, monkeypatch, tmp_path):
    _settings(monkeypatch, tmp_path)
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True)
    (report_dir / "eval_flywheel.md").write_text("# stale\n", encoding="utf-8")
    (report_dir / "eval_flywheel.json").write_text(
        json.dumps(_report()), encoding="utf-8"
    )

    out = job.run_once("手动", runner=lambda argv: 2)

    assert out["ok"] is False
    assert not (report_dir / "eval_flywheel.json").exists()
    assert not (report_dir / "eval_flywheel.md").exists()
    assert session.query(EvalRun).count() == 0


def test_runner_that_writes_only_markdown_cannot_reuse_stale_sidecar(
    session, monkeypatch, tmp_path
):
    _settings(monkeypatch, tmp_path)
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True)
    (report_dir / "eval_flywheel.json").write_text(
        json.dumps(_report()), encoding="utf-8"
    )

    def runner(argv):
        out = Path(argv[argv.index("--out") + 1])
        out.write_text("# fresh markdown only\n", encoding="utf-8")
        return 0

    out = job.run_once("手动", runner=runner)

    assert out["ok"] is False
    assert session.query(EvalRun).count() == 0


def test_structurally_wrong_json_is_reported_not_raised(
    session, monkeypatch, tmp_path
):
    _settings(monkeypatch, tmp_path)

    def runner(argv):
        out = Path(argv[argv.index("--out") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("# report\n", encoding="utf-8")
        # ``overall`` is a list: .get() on it raises AttributeError.
        out.with_suffix(".json").write_text(
            json.dumps({"cases_loaded": 1, "overall": ["bad"]}), encoding="utf-8"
        )
        return 0

    out = job.run_once("手动", runner=runner)

    assert out["ok"] is False
    assert "AttributeError" in out["error"] or "cannot read" in out["error"]
    assert session.query(EvalRun).count() == 0


def test_unreadable_report_does_not_insert(session, monkeypatch, tmp_path):
    _settings(monkeypatch, tmp_path)

    def runner(argv):
        out = Path(argv[argv.index("--out") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("# report\n", encoding="utf-8")
        out.with_suffix(".json").write_text("{not json", encoding="utf-8")
        return 0

    out = job.run_once("手动", runner=runner)

    assert out["ok"] is False
    assert session.query(EvalRun).count() == 0


# --- format_trend ------------------------------------------------------------


def _metrics(recall, mrr, faith=0.9, accuracy=0.9) -> dict:
    return {
        "recall_at_10": recall,
        "mrr_at_10": mrr,
        "faithfulness": faith,
        "d_refusal_accuracy": accuracy,
    }


def test_format_trend_marks_up_down_and_flat(session):
    session.add_all(
        [
            EvalRun(
                triggered_by="手动",
                dataset_size=300,
                metrics=_metrics(0.8, 0.8),
            ),
            EvalRun(
                triggered_by="定时",
                dataset_size=300,
                metrics=_metrics(0.9, 0.7),
            ),
            EvalRun(
                triggered_by="定时",
                dataset_size=300,
                metrics=_metrics(0.9, 0.7),
            ),
        ]
    )
    session.commit()
    runs = session.query(EvalRun).order_by(EvalRun.id).all()

    text = job.format_trend(runs)

    assert "Recall@10" in text
    assert "MRR@10" in text
    assert "Faithfulness" in text
    assert "D accuracy" in text
    assert "↑" in text
    assert "↓ ⚠" in text
    assert "→" in text
    assert "run #" in text


def test_format_trend_handles_empty_and_missing_metrics(session):
    assert "暂无" in job.format_trend([])

    session.add(
        EvalRun(triggered_by="手动", dataset_size=0, metrics={"recall_at_10": None})
    )
    session.commit()
    text = job.format_trend(session.query(EvalRun).all())
    assert "n/a" in text
    assert "—" in text


# --- main --------------------------------------------------------------------


def _seed_runs(session, count: int = 12) -> None:
    session.add_all(
        EvalRun(
            triggered_by="定时",
            dataset_size=100 + number,
            metrics=_metrics(0.5 + number / 100, 0.5),
        )
        for number in range(count)
    )
    session.commit()


def test_main_runs_once_and_writes_latest_trend(
    session, monkeypatch, tmp_path
):
    _settings(monkeypatch, tmp_path)
    _seed_runs(session)
    calls = {}

    def fake_run_once(triggered_by, limit=None):
        calls["triggered_by"] = triggered_by
        calls["limit"] = limit
        return {"ok": True, "run_id": 99, "metrics": {}}

    monkeypatch.setattr(job, "run_once", fake_run_once)

    code = job.main(["--triggered-by", "定时", "--limit", "5"])

    assert code == 0
    assert calls == {"triggered_by": "定时", "limit": 5}
    text = (tmp_path / "reports" / "eval_trend.txt").read_text(encoding="utf-8")
    assert "run #12" in text
    assert "run #3" in text
    assert "run #2" not in text
    assert "暂无" not in text


def test_main_reports_run_failure(session, monkeypatch, tmp_path):
    _settings(monkeypatch, tmp_path)
    monkeypatch.setattr(
        job,
        "run_once",
        lambda triggered_by, limit=None: {"ok": False, "error": "boom"},
    )

    code = job.main([])

    assert code == 1
    assert not (tmp_path / "reports" / "eval_trend.txt").exists()
