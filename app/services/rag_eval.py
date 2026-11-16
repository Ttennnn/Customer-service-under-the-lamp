"""Background runner for the four-strategy RAG evaluation page.

The evaluation is a CLI job (``app.jobs.eval_rag``); this service runs it in a
daemon thread, exposes its state for polling, and reads the JSON sidecar the
job writes next to the Markdown report.  A single run is allowed at a time.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

CASE_PATH = Path("tests/eval/rag_cases.json")
REPORT_PATH = Path("docs/superpowers/reports/2026-11-16-ch04-rag-eval.md")
JSON_PATH = REPORT_PATH.with_suffix(".json")
STRATEGIES = "dense,bm25,hybrid,hybrid_rerank"

_lock = threading.Lock()
_state: dict = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "last_error": None,
    "exit_code": None,
    "report_path": str(REPORT_PATH),
    "json_path": str(JSON_PATH),
}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def status() -> dict:
    """Return a snapshot of the current run state."""
    with _lock:
        return dict(_state)


def start_eval() -> dict | None:
    """Start a full four-strategy evaluation, or return None if one is running."""
    with _lock:
        if _state["running"]:
            return None
        _state.update(
            running=True,
            started_at=_now(),
            finished_at=None,
            last_error=None,
            exit_code=None,
        )
        snapshot = dict(_state)

    thread = threading.Thread(target=_run, name="rag-eval", daemon=True)
    thread.start()
    return snapshot


def _run() -> None:
    from app.jobs import eval_rag

    try:
        code = eval_rag.main(
            [
                "--cases",
                str(CASE_PATH),
                "--strategies",
                STRATEGIES,
                "--out",
                str(REPORT_PATH),
            ]
        )
        with _lock:
            _state["exit_code"] = code
            if code != 0:
                _state["last_error"] = f"eval_rag exited with code {code}"
    except Exception as exc:  # noqa: BLE001 - surfaced on the page, not raised
        logger.exception("rag evaluation failed")
        with _lock:
            _state["last_error"] = f"{type(exc).__name__}: {exc}"
            _state["exit_code"] = 1
    finally:
        with _lock:
            _state["running"] = False
            _state["finished_at"] = _now()


def report() -> dict | None:
    """Read the JSON sidecar, or return None when no report exists yet."""
    if not JSON_PATH.is_file():
        return None
    try:
        return json.loads(JSON_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.exception("cannot read rag evaluation report")
        return {"error": f"cannot read report: {exc}"}


def report_markdown_path() -> Path:
    return REPORT_PATH
