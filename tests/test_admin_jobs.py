"""白名单后台作业运行器(ch09 Task 10)。

运行器只接受三个固定作业名,后台线程执行 ``app/jobs/*.main()``,同一作业
互斥;捕获 stdout/stderr 到 ``output_tail`` 供页面轮询。测试全部 monkeypatch
作业 callable,不启动真实 Langfuse/Milvus/LLM。
"""

import sys
import threading
import time

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import admin_jobs

TERMINAL_TIMEOUT = 3.0


@pytest.fixture
def client(session):
    with TestClient(app) as c:
        yield c


def _wait_until_done(name: str, timeout: float = TERMINAL_TIMEOUT) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = admin_jobs.job_status(name)
        if state is not None and not state["running"]:
            return state
        time.sleep(0.01)
    raise AssertionError(f"job {name!r} did not finish within {timeout}s")


# ---------------------------------------------------------------------------
# service: whitelist / mutex / capture
# ---------------------------------------------------------------------------


def test_unknown_job_is_rejected():
    assert admin_jobs.start_job("rm -rf /") is None
    assert admin_jobs.start_job("cost-report; rm -rf /") is None
    assert admin_jobs.job_status("rm -rf /") is None


def test_whitelist_has_exactly_the_three_jobs():
    assert set(admin_jobs.ALLOWED_JOBS) == {
        "cost-report",
        "eval-flywheel",
        "calibrate-confidence",
    }


def test_same_job_is_mutex(monkeypatch):
    gate = threading.Event()
    monkeypatch.setitem(
        admin_jobs.ALLOWED_JOBS, "cost-report", lambda: gate.wait(1)
    )

    first = admin_jobs.start_job("cost-report")
    second = admin_jobs.start_job("cost-report")

    assert first is not None
    assert second is None
    assert first["running"] is True

    gate.set()
    _wait_until_done("cost-report")


def test_different_jobs_may_run_in_parallel(monkeypatch):
    gate = threading.Event()
    monkeypatch.setitem(
        admin_jobs.ALLOWED_JOBS, "cost-report", lambda: gate.wait(1)
    )
    monkeypatch.setitem(
        admin_jobs.ALLOWED_JOBS, "eval-flywheel", lambda: gate.wait(1)
    )

    first = admin_jobs.start_job("cost-report")
    second = admin_jobs.start_job("eval-flywheel")

    assert first is not None
    assert second is not None

    gate.set()
    _wait_until_done("cost-report")
    _wait_until_done("eval-flywheel")


def test_job_output_tail_is_captured(monkeypatch):
    monkeypatch.setitem(
        admin_jobs.ALLOWED_JOBS,
        "cost-report",
        lambda: (print("cost report done"), print("warn", file=sys.stderr)),
    )

    assert admin_jobs.start_job("cost-report") is not None
    state = _wait_until_done("cost-report")

    assert state["running"] is False
    assert state["exit_code"] == 0
    assert "cost report done" in state["output_tail"]
    assert "warn" in state["output_tail"]
    assert state["last_error"] is None
    assert state["started_at"] is not None
    assert state["finished_at"] is not None


def test_output_tail_is_truncated_to_4000_chars(monkeypatch):
    def noisy():
        print("START-MARKER")
        print("x" * 5000)
        print("END-MARKER")

    monkeypatch.setitem(admin_jobs.ALLOWED_JOBS, "cost-report", noisy)

    assert admin_jobs.start_job("cost-report") is not None
    state = _wait_until_done("cost-report")

    assert len(state["output_tail"]) <= 4000
    assert "START-MARKER" not in state["output_tail"]
    assert "END-MARKER" in state["output_tail"]


def test_capture_registration_failure_does_not_wedge_the_job(monkeypatch):
    def broken_capture(state):
        raise RuntimeError("stdout proxy unavailable")

    monkeypatch.setattr(admin_jobs, "_register_capture", broken_capture)
    monkeypatch.setitem(admin_jobs.ALLOWED_JOBS, "cost-report", lambda: 0)

    assert admin_jobs.start_job("cost-report") is not None
    state = _wait_until_done("cost-report")

    assert state["running"] is False
    assert state["exit_code"] != 0
    assert "stdout proxy unavailable" in state["last_error"]
    # The job lock was released, so the same job can be started again.
    assert admin_jobs.start_job("cost-report") is not None
    _wait_until_done("cost-report")


def test_failed_job_records_exit_code_and_error(monkeypatch):
    def boom():
        raise RuntimeError("langfuse is down")

    monkeypatch.setitem(admin_jobs.ALLOWED_JOBS, "calibrate-confidence", boom)

    assert admin_jobs.start_job("calibrate-confidence") is not None
    state = _wait_until_done("calibrate-confidence")

    assert state["running"] is False
    assert state["exit_code"] != 0
    assert "langfuse is down" in state["last_error"]


def test_job_status_before_first_run_is_idle(monkeypatch):
    # Drop any state left by an earlier test so "never run" is observable.
    admin_jobs._states.pop("eval-flywheel", None)
    state = admin_jobs.job_status("eval-flywheel")

    assert state == {
        "running": False,
        "started_at": None,
        "finished_at": None,
        "exit_code": None,
        "output_tail": "",
        "last_error": None,
    }


# ---------------------------------------------------------------------------
# REST
# ---------------------------------------------------------------------------


def test_run_unknown_job_is_404(client):
    assert client.post("/api/admin/jobs/rm-rf/run").status_code == 404
    assert client.get("/api/admin/jobs/rm-rf/status").status_code == 404


def test_run_job_returns_202_and_status_polls(client, monkeypatch):
    monkeypatch.setitem(
        admin_jobs.ALLOWED_JOBS, "cost-report", lambda: print("cost done")
    )

    r = client.post("/api/admin/jobs/cost-report/run")

    assert r.status_code == 202
    assert r.json()["running"] is True

    _wait_until_done("cost-report")
    status = client.get("/api/admin/jobs/cost-report/status")

    assert status.status_code == 200
    body = status.json()
    assert body["running"] is False
    assert "cost done" in body["output_tail"]


def test_run_same_job_while_running_is_409(client, monkeypatch):
    gate = threading.Event()
    monkeypatch.setitem(
        admin_jobs.ALLOWED_JOBS, "cost-report", lambda: gate.wait(1)
    )

    assert client.post("/api/admin/jobs/cost-report/run").status_code == 202
    assert client.post("/api/admin/jobs/cost-report/run").status_code == 409

    gate.set()
    _wait_until_done("cost-report")
