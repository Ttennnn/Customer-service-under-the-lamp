"""白名单后台作业运行器(ch09 Task 10)。

前端只能传固定的作业名,不能传命令片段;``ALLOWED_JOBS`` 把名字映射到
进程内的 Python callable(对应的 ``app/jobs/*.main`` 薄包装),运行器不拼
shell。每个作业在守护线程里执行,同一作业由一把非阻塞锁互斥,不同作业可
并行。

标准输出/错误在作业线程内被路由到 ``output_tail``(保留最后 4000 字符)。
路由按 ``threading.get_ident()`` 分桶,所以并行作业不会串台;只有存在活动
作业时才临时替换 ``sys.stdout``/``sys.stderr``,最后一个作业结束后恢复。
"""

from __future__ import annotations

import io
import sys
import threading
import traceback
from datetime import datetime, timezone
from typing import Callable

#: ``output_tail`` 保留的最大字符数。
TAIL_LIMIT = 4000


# --- callable map ----------------------------------------------------------
#
# 薄包装按需导入 jobs 模块并显式传 ``[]``:如果直接 ``main()``,argparse 会
# 解析服务器进程自己的 argv(例如 uvicorn 的 --port),把作业跑挂。


def _cost_report_main() -> int:
    from app.jobs.cost_by_intent import main

    return main([])


def _eval_flywheel_main() -> int:
    from app.jobs.eval_flywheel import main

    return main([])


def _calibrate_confidence_main() -> int:
    from app.jobs.calibrate_confidence import main

    return main([])


ALLOWED_JOBS: dict[str, Callable[[], int]] = {
    "cost-report": _cost_report_main,
    "eval-flywheel": _eval_flywheel_main,
    "calibrate-confidence": _calibrate_confidence_main,
}


# --- state -----------------------------------------------------------------

_states: dict[str, dict] = {}
_states_lock = threading.Lock()

_job_locks: dict[str, threading.Lock] = {}
_job_locks_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _idle_state() -> dict:
    return {
        "running": False,
        "started_at": None,
        "finished_at": None,
        "exit_code": None,
        "output_tail": "",
        "last_error": None,
    }


def _lock_for(name: str) -> threading.Lock:
    with _job_locks_lock:
        return _job_locks.setdefault(name, threading.Lock())


# --- stdout/stderr capture -------------------------------------------------

_capture_lock = threading.Lock()
_active_buffers: dict[int, "_JobOutputBuffer"] = {}
_routed_stdout: "_ThreadRoutedStream | None" = None
_routed_stderr: "_ThreadRoutedStream | None" = None
_saved_stdout = None
_saved_stderr = None


class _JobOutputBuffer:
    """Append text to one job's live ``output_tail`` (tail-truncated)."""

    def __init__(self, state: dict):
        self._state = state

    def write(self, text: str) -> None:
        if not text:
            return
        with _states_lock:
            self._state["output_tail"] = (
                self._state["output_tail"] + text
            )[-TAIL_LIMIT:]


class _ThreadRoutedStream(io.TextIOBase):
    """Route writes from an active job thread into that job's buffer.

    Output is also teed to the process stream saved at install time so server
    logs still show what the job printed.
    """

    def __init__(self, target):
        self._target = target

    def write(self, text: str) -> int:
        with _capture_lock:
            buffer = _active_buffers.get(threading.get_ident())
        if buffer is not None:
            buffer.write(text)
        if self._target is not None:
            try:
                self._target.write(text)
            except Exception:  # noqa: BLE001 - logging must not kill the job
                pass
        return len(text)

    def flush(self) -> None:
        if self._target is not None:
            try:
                self._target.flush()
            except Exception:  # noqa: BLE001
                pass

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    @property
    def encoding(self) -> str:
        return "utf-8"


def _register_capture(state: dict) -> None:
    global _routed_stdout, _routed_stderr, _saved_stdout, _saved_stderr
    with _capture_lock:
        if _routed_stdout is None:
            _saved_stdout, _saved_stderr = sys.stdout, sys.stderr
            _routed_stdout = _ThreadRoutedStream(_saved_stdout)
            _routed_stderr = _ThreadRoutedStream(_saved_stderr)
            sys.stdout = _routed_stdout
            sys.stderr = _routed_stderr
        _active_buffers[threading.get_ident()] = _JobOutputBuffer(state)


def _release_capture() -> None:
    global _routed_stdout, _routed_stderr, _saved_stdout, _saved_stderr
    with _capture_lock:
        _active_buffers.pop(threading.get_ident(), None)
        if not _active_buffers and _routed_stdout is not None:
            sys.stdout = _saved_stdout
            sys.stderr = _saved_stderr
            _routed_stdout = _routed_stderr = None
            _saved_stdout = _saved_stderr = None


# --- runner ----------------------------------------------------------------


def _run_job(name: str, func: Callable[[], int], lock: threading.Lock) -> None:
    state = _states[name]
    _register_capture(state)
    exit_code = 0
    last_error = None
    try:
        result = func()
        if isinstance(result, int) and not isinstance(result, bool):
            exit_code = result
    except SystemExit as exc:  # argparse errors from a job
        code = exc.code
        exit_code = code if isinstance(code, int) else 1
        if exit_code:
            last_error = f"job exited with status {exit_code}"
    except BaseException:  # noqa: BLE001 - a job must never leave "running" set
        exit_code = 1
        last_error = traceback.format_exc(limit=10)[-TAIL_LIMIT:]
    finally:
        _release_capture()
        with _states_lock:
            state["running"] = False
            state["finished_at"] = _now()
            state["exit_code"] = exit_code
            state["last_error"] = last_error
        lock.release()


def start_job(name: str) -> dict | None:
    """Start an allow-listed job.

    Returns a snapshot of the new state, or ``None`` when the name is unknown
    or that same job is already running. No shell command is ever built.
    """
    func = ALLOWED_JOBS.get(name)
    if not callable(func):
        return None

    lock = _lock_for(name)
    if not lock.acquire(blocking=False):
        return None

    state = {
        "running": True,
        "started_at": _now(),
        "finished_at": None,
        "exit_code": None,
        "output_tail": "",
        "last_error": None,
    }
    with _states_lock:
        _states[name] = state
        snapshot = dict(state)

    try:
        thread = threading.Thread(
            target=_run_job,
            args=(name, func, lock),
            name=f"admin-job-{name}",
            daemon=True,
        )
        thread.start()
    except Exception:
        with _states_lock:
            _states[name] = _idle_state()
        lock.release()
        return None

    return snapshot


def job_status(name: str) -> dict | None:
    """Return a snapshot of ``name``'s state, or ``None`` for unknown jobs."""
    if name not in ALLOWED_JOBS:
        return None
    with _states_lock:
        state = _states.get(name)
        return dict(state) if state is not None else _idle_state()
