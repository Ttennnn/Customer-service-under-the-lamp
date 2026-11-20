"""Whitelisted make-backed job runner for the ch10 acceptance pages."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.core import paths

TAIL_LIMIT = 4000


@dataclass(frozen=True)
class JobSpec:
    name: str
    argv: tuple[str, ...]
    heavy: bool
    description: str
    env: dict = field(default_factory=dict)


def _job(name: str, target: str, heavy: bool, description: str, **env) -> JobSpec:
    return JobSpec(
        name=name,
        argv=("make", target),
        heavy=heavy,
        description=description,
        env=dict(env),
    )


JOBS: dict[str, JobSpec] = {
    "ch10-golden": _job(
        "ch10-golden", "ch10-golden", False, "黄金样例闸"
    ),
    "ch10-corpus": _job(
        "ch10-corpus", "ch10-corpus", True, "语料流水线（捞取/清洗/预标/造数）"
    ),
    "ch10-dataset": _job(
        "ch10-dataset", "ch10-dataset", True, "切分 + 训练集增强"
    ),
    "ch10-train": _job("ch10-train", "ch10-train", True, "全参微调"),
    "ch10-export": _job(
        "ch10-export", "ch10-export", True, "ONNX 导出与阈值重扫"
    ),
    "classifier-up": _job(
        "classifier-up", "classifier-up", False, "启动分类服务"
    ),
    "classifier-down": _job(
        "classifier-down", "classifier-down", False, "停止分类服务"
    ),
    "ch10-eval": _job("ch10-eval", "ch10-eval", False, "测试集评测"),
    "ch10-threshold-scan": _job(
        "ch10-threshold-scan",
        "ch10-threshold-scan",
        False,
        "阈值扫描重演",
    ),
    "classify-pool": _job(
        "classify-pool", "classify-pool", False, "旁路批量归类"
    ),
    "classify-pool-force": _job(
        "classify-pool-force",
        "classify-pool",
        False,
        "旁路批量归类（不足一批也跑）",
        FORCE="1",
    ),
}

_states: dict[str, dict] = {}
_procs: dict[str, subprocess.Popen] = {}
_locks: dict[str, threading.Lock] = {}
_lock_guard = threading.Lock()
_states_lock = threading.Lock()


def _platform_is_windows() -> bool:
    return os.name == "nt"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _lock_for(name: str) -> threading.Lock:
    with _lock_guard:
        return _locks.setdefault(name, threading.Lock())


def _log_path(name: str) -> Path:
    return paths.LOG_DIR / f"{name}.log"


def _finish(name: str, code: int) -> None:
    with _states_lock:
        state = _states.get(name)
        if state is None or not state["running"]:
            return
        state["running"] = False
        state["finished_at"] = _now()
        state["exit_code"] = code
    lock = _locks.get(name)
    if lock is not None and lock.locked():
        lock.release()


def _refresh_tail(name: str, state: dict) -> None:
    path = _log_path(name)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    state["output_tail"] = text[-TAIL_LIMIT:]


def start_job(name: str) -> dict | None:
    spec = JOBS.get(name)
    if spec is None:
        return None
    lock = _lock_for(name)
    if not lock.acquire(blocking=False):
        return None

    paths.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _log_path(name)
    log_file = log_path.open("ab")
    env = dict(os.environ)
    env.update(spec.env)
    kwargs: dict = {
        "cwd": str(paths.ROOT),
        "stdout": log_file,
        "stderr": subprocess.STDOUT,
        "env": env,
    }
    if _platform_is_windows():
        kwargs["creationflags"] = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
    else:
        kwargs["start_new_session"] = True
    try:
        process = subprocess.Popen(list(spec.argv), **kwargs)
    except Exception:  # noqa: BLE001 - release and report as start failure
        log_file.close()
        lock.release()
        return None
    log_file.close()

    state = {
        "running": True,
        "started_at": _now(),
        "finished_at": None,
        "exit_code": None,
        "output_tail": "",
        "log_path": str(log_path),
        "heavy": spec.heavy,
        "description": spec.description,
    }
    with _states_lock:
        _states[name] = state
        _procs[name] = process
    return dict(state)


def job_status(name: str) -> dict | None:
    spec = JOBS.get(name)
    if spec is None:
        return None
    with _states_lock:
        state = _states.get(name)
        if state is None:
            return {
                "running": False,
                "started_at": None,
                "finished_at": None,
                "exit_code": None,
                "output_tail": "",
                "log_path": str(_log_path(name)),
                "heavy": spec.heavy,
                "description": spec.description,
            }
        snapshot = dict(state)
    process = _procs.get(name)
    if snapshot["running"] and process is not None:
        code = process.poll()
        if code is not None:
            _finish(name, int(code))
            with _states_lock:
                snapshot = dict(_states[name])
    _refresh_tail(name, snapshot)
    with _states_lock:
        if name in _states:
            _states[name]["output_tail"] = snapshot["output_tail"]
    snapshot["heavy"] = spec.heavy
    snapshot["description"] = spec.description
    return snapshot


def stop_job(name: str) -> bool:
    if name not in JOBS:
        return False
    process = _procs.get(name)
    if process is None or process.poll() is not None:
        return False
    pid = process.pid
    if _platform_is_windows():
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    return True
