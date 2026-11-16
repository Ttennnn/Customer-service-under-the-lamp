"""In-process build state machine for the ``/kb`` knowledge-base page.

The page has no job queue: at most one daemon thread rebuilds the *document*
domain (``build_knowledge(reset=True)``, so ``mined_qa`` rows survive) while
the UI polls :func:`status`.  State is deliberately process-local; running the
API with multiple workers would need an external queue.

``status`` keeps the document-domain progress bar honest by counting
``mined_qa`` chunks separately instead of folding them into ``pending`` /
``done`` / ``total``.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy import func, or_, select

from app.config import get_settings
from app.db import session as db_session
from app.jobs.build_knowledge import BuildReport, build_knowledge
from app.models import KnowledgeChunk
from app.rag.embedding import get_embedder
from app.rag.milvus_store import get_vector_store

logger = logging.getLogger(__name__)

MINED_QA = "mined_qa"


@dataclass
class BuildState:
    running: bool = False
    started_at: datetime | None = None
    finished_at: datetime | None = None
    last_report: BuildReport | None = None
    last_error: str | None = None


_state = BuildState()
_lock = threading.Lock()
# Set means "the last started build has finished" (or none ever started), so
# ``wait_for_build`` never blocks forever when nothing is happening.
_done = threading.Event()
_done.set()


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def is_running() -> bool:
    with _lock:
        return _state.running


def start_build() -> datetime | None:
    """Start one daemon build.

    Returns the build's ``started_at`` timestamp, or ``None`` when a build is
    already running.  If the thread cannot start at all the state machine is
    rolled back (and the completion event set) before the error is re-raised,
    so one failed ``Thread.start`` cannot wedge every later build.
    """
    global _state
    with _lock:
        if _state.running:
            return None
        started_at = datetime.now()
        _state = BuildState(running=True, started_at=started_at)
        _done.clear()

    thread = threading.Thread(target=_run_build, name="kb-build", daemon=True)
    try:
        thread.start()
    except Exception as exc:
        logger.exception("failed to start knowledge build thread")
        with _lock:
            _state.running = False
            _state.finished_at = datetime.now()
            _state.last_error = str(exc) or type(exc).__name__
            _done.set()
        raise
    return started_at


def wait_for_build(timeout: float | None = None) -> bool:
    """Block until the current (or last) build finishes; False on timeout."""
    return _done.wait(timeout)


def snapshot() -> dict:
    """Return build state only (no database counts)."""
    with _lock:
        return {
            "running": _state.running,
            "started_at": _iso(_state.started_at),
            "finished_at": _iso(_state.finished_at),
            "last_report": asdict(_state.last_report) if _state.last_report else None,
            "last_error": _state.last_error,
        }


def _count(session, *conditions) -> int:
    stmt = select(func.count()).select_from(KnowledgeChunk).where(*conditions)
    return int(session.execute(stmt).scalar_one())


def status(session) -> dict:
    """Count document-domain and mined chunks separately, then merge state."""
    mined = KnowledgeChunk.content_type == MINED_QA
    # SQL ``col != value`` is NULL (not true) for NULL content_type, so older
    # rows without metadata must be matched explicitly too.
    document = or_(
        KnowledgeChunk.content_type != MINED_QA,
        KnowledgeChunk.content_type.is_(None),
    )
    pending = KnowledgeChunk.vectorize_status == "pending"
    done = KnowledgeChunk.vectorize_status == "done"

    doc_pending = _count(session, document, pending)
    doc_done = _count(session, document, done)
    mined_pending = _count(session, mined, pending)
    mined_done = _count(session, mined, done)

    state = snapshot()
    return {
        "pending": doc_pending,
        "done": doc_done,
        "total": doc_pending + doc_done,
        "mined_total": mined_pending + mined_done,
        "mined_pending": mined_pending,
        "mined_done": mined_done,
        "running": state["running"],
        "started_at": state["started_at"],
        "finished_at": state["finished_at"],
        "last_report": state["last_report"],
        "last_error": state["last_error"],
    }


def reset_state_for_tests() -> None:
    """Clear the module-level state between tests."""
    global _state
    with _lock:
        _state = BuildState()
        # Same invariant as a finished build: state and completion event move
        # together while holding the lock, so start_build cannot interleave.
        _done.set()


def _run_build() -> None:
    """Thread body: own DB session, record outcome, never raise."""
    session = None
    try:
        session = db_session.get_session_factory()()
        settings = get_settings()
        report = build_knowledge(
            session,
            Path(settings.knowledge_docs_dir),
            get_embedder(),
            get_vector_store(),
            reset=True,
        )
    except Exception as exc:
        logger.exception("knowledge base build failed")
        with _lock:
            _state.last_error = str(exc) or type(exc).__name__
    else:
        with _lock:
            _state.last_report = report
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                logger.exception("failed to close the knowledge build session")
        with _lock:
            _state.running = False
            _state.finished_at = datetime.now()
            # Keep the completion event update under the same lock as the
            # state transition. Setting it after releasing the lock would let
            # a concurrent start_build clear it for build B before this older
            # build's stray set() woke build B's waiters early.
            _done.set()
