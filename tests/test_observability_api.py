"""观测总览 / 后台摘要 API(ch09 Task 10)。

三块数据互不连坐:任一产物缺失或读取失败只影响自己的 ``present=false``,
其余块照常返回。所有测试都用 tmp_path 报告目录 + 文件级 SQLite,不联网。
"""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.api.admin as admin_api
import app.api.observability as obs_api
from app.config import get_settings
from app.main import app
from app.models import EvalRun, ReviewQueue


@pytest.fixture
def client(session):
    with TestClient(app) as c:
        yield c


@pytest.fixture
def report_dir(tmp_path):
    path = tmp_path / "reports"
    path.mkdir()
    return path


@pytest.fixture
def settings_override(report_dir):
    settings = SimpleNamespace(
        ch09_report_dir=str(report_dir),
        evidence_confidence_threshold=0.4,
    )
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        yield settings
    finally:
        app.dependency_overrides.pop(get_settings, None)


def _add_eval_run(session, metrics=None, triggered_by="定时"):
    row = EvalRun(
        triggered_by=triggered_by,
        dataset_size=10,
        metrics=metrics
        or {
            "recall_at_10": 0.8,
            "mrr_at_10": 0.6,
            "faithfulness": 0.9,
            "d_refusal_accuracy": 1.0,
        },
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def _add_review(session, question, status="待审", count=1):
    row = ReviewQueue(
        normalized_question=question,
        review_status=status,
        occurrence_count=count,
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


# ---------------------------------------------------------------------------
# 三块独立性 / cost
# ---------------------------------------------------------------------------


def test_overview_blocks_are_independent(client, session, settings_override):
    _add_eval_run(session)

    data = client.get("/api/observability/overview").json()

    assert data["cost"]["present"] is False
    assert "hint" in data["cost"]
    assert data["eval"]["present"] is True
    assert data["calibration"]["present"] is False
    assert "error" in data["calibration"] or "hint" in data["calibration"]


def test_overview_cost_block_present_when_json_exists(
    client, session, settings_override, report_dir
):
    _add_eval_run(session)
    payload = {
        "generated_at": "2026-11-19T00:00:00+00:00",
        "window_days": 7,
        "rows": [{"intent": "退款", "requests": 2, "total_tokens": 100}],
    }
    (report_dir / "cost_by_intent.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    (report_dir / "confidence_calibration.json").write_text(
        json.dumps({"recommended": {"threshold": 0.4}}), encoding="utf-8"
    )

    data = client.get("/api/observability/overview").json()

    assert data["cost"]["present"] is True
    assert data["cost"]["window_days"] == 7
    assert data["cost"]["rows"][0]["intent"] == "退款"


def test_overview_cost_block_reports_read_error(
    client, session, settings_override, report_dir
):
    _add_eval_run(session)
    (report_dir / "cost_by_intent.json").write_text("{not json", encoding="utf-8")
    (report_dir / "confidence_calibration.json").write_text(
        json.dumps({"recommended": {"threshold": 0.4}}), encoding="utf-8"
    )

    data = client.get("/api/observability/overview").json()

    assert data["cost"]["present"] is False
    assert "error" in data["cost"]
    assert data["eval"]["present"] is True


# ---------------------------------------------------------------------------
# eval 趋势
# ---------------------------------------------------------------------------


def test_overview_eval_returns_latest_ten_chronologically(
    client, session, settings_override
):
    for index in range(12):
        _add_eval_run(session, metrics={"recall_at_10": index / 10})

    data = client.get("/api/observability/overview").json()

    runs = data["eval"]["runs"]
    assert data["eval"]["present"] is True
    assert len(runs) == 10
    # ids 3..12: newest 10, returned oldest -> newest for the trend page.
    assert [run["metrics"]["recall_at_10"] for run in runs] == [
        index / 10 for index in range(2, 12)
    ]
    assert isinstance(runs[0]["created_at"], str)
    assert runs[0]["triggered_by"] == "定时"


def test_overview_eval_db_failure_only_breaks_its_block(
    client, settings_override, monkeypatch
):
    def broken_factory():
        raise RuntimeError("mysql is down")

    monkeypatch.setattr(
        "app.db.session.get_session_factory", broken_factory
    )

    data = client.get("/api/observability/overview").json()

    assert data["eval"]["present"] is False
    assert "mysql is down" in data["eval"]["error"]
    assert data["cost"]["present"] is False
    assert data["calibration"]["present"] is False


def test_overview_eval_with_no_rows_is_absent(client, session, settings_override):
    data = client.get("/api/observability/overview").json()

    assert data["eval"]["present"] is False
    assert "hint" in data["eval"]
    assert data["eval"]["runs"] == []


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


def _write_calibration(report_dir, threshold):
    (report_dir / "confidence_calibration.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-11-19T00:00:00+00:00",
                "recommended": {
                    "threshold": threshold,
                    "youden_j": 0.5,
                    "tpr": 0.8,
                    "fpr": 0.3,
                    "precision": 0.7,
                },
            }
        ),
        encoding="utf-8",
    )


def test_overview_calibration_needs_update_when_recommended_differs(
    client, session, settings_override, report_dir
):
    _add_eval_run(session)
    _write_calibration(report_dir, 0.42)
    settings_override.evidence_confidence_threshold = 0.0

    data = client.get("/api/observability/overview").json()

    calibration = data["calibration"]
    assert calibration["present"] is True
    assert calibration["recommended"]["threshold"] == 0.42
    assert calibration["current_threshold"] == 0.0
    assert calibration["needs_update"] is True


def test_overview_calibration_no_update_when_thresholds_match(
    client, session, settings_override, report_dir
):
    _add_eval_run(session)
    _write_calibration(report_dir, 0.42)
    settings_override.evidence_confidence_threshold = 0.42

    data = client.get("/api/observability/overview").json()

    assert data["calibration"]["present"] is True
    assert data["calibration"]["needs_update"] is False


def test_overview_calibration_missing_file_has_hint(
    client, session, settings_override
):
    _add_eval_run(session)

    data = client.get("/api/observability/overview").json()

    assert data["calibration"]["present"] is False
    assert "hint" in data["calibration"]


# ---------------------------------------------------------------------------
# admin summary
# ---------------------------------------------------------------------------


def test_admin_summary_counts_and_highest_occurrence(client, session):
    _add_review(session, "退款多久到账", status="待审", count=8)
    _add_review(session, "怎么开发票", status="待审", count=2)
    _add_review(session, "运费谁承担", status="通过", count=3)
    _add_review(session, "能改地址吗", status="驳回", count=1)

    r = client.get("/api/admin/summary")

    assert r.status_code == 200
    body = r.json()
    assert body["counts"] == {"待审": 2, "通过": 1, "驳回": 1}
    assert body["highest"]["normalized_question"] == "退款多久到账"
    assert body["highest"]["occurrence_count"] == 8
    assert "2 条待审" in body["conclusion"]
    assert "8 次" in body["conclusion"]
    assert body["links"]["review"] == "/review"
    assert body["links"]["observability"] == "/observability"


def test_admin_summary_highest_is_pending_only(client, session):
    _add_review(session, "已通过的旧高频", status="通过", count=99)
    _add_review(session, "待审低频", status="待审", count=3)

    body = client.get("/api/admin/summary").json()

    assert body["highest"]["normalized_question"] == "待审低频"
    assert body["highest"]["review_status"] == "待审"
    assert "最高频待审问题出现 3 次" in body["conclusion"]


def test_admin_summary_no_pending_explains_approved_and_rejected(client, session):
    _add_review(session, "已通过", status="通过", count=5)
    _add_review(session, "已驳回", status="驳回", count=2)

    body = client.get("/api/admin/summary").json()

    assert body["highest"] is None
    assert "暂无待审知识缺口" in body["conclusion"]
    assert "已通过 1 条" in body["conclusion"]
    assert "已驳回 1 条" in body["conclusion"]


def test_admin_summary_empty_queue(client, session):
    body = client.get("/api/admin/summary").json()

    assert body["counts"] == {"待审": 0, "通过": 0, "驳回": 0}
    assert body["highest"] is None
    assert "还没有" in body["conclusion"]


# ---------------------------------------------------------------------------
# page routes
# ---------------------------------------------------------------------------


def test_observability_page_serves_file_with_no_store(
    client, session, monkeypatch, tmp_path
):
    page = tmp_path / "observability.html"
    page.write_text("<html>observability</html>", encoding="utf-8")
    monkeypatch.setattr(obs_api, "PAGE", page)

    r = client.get("/observability")

    assert r.status_code == 200
    assert "observability" in r.text
    assert r.headers["cache-control"] == "no-store"


def test_observability_page_missing_is_404(client, session, monkeypatch, tmp_path):
    monkeypatch.setattr(obs_api, "PAGE", tmp_path / "missing.html")

    assert client.get("/observability").status_code == 404


def test_admin_page_serves_file_with_no_store(
    client, session, monkeypatch, tmp_path
):
    page = tmp_path / "admin.html"
    page.write_text("<html>admin</html>", encoding="utf-8")
    monkeypatch.setattr(admin_api, "PAGE", page)

    r = client.get("/admin")

    assert r.status_code == 200
    assert "admin" in r.text
    assert r.headers["cache-control"] == "no-store"


def test_admin_page_missing_is_404(client, session, monkeypatch, tmp_path):
    monkeypatch.setattr(admin_api, "PAGE", tmp_path / "missing.html")

    assert client.get("/admin").status_code == 404
