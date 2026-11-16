import threading
from datetime import datetime
from pathlib import Path

import pytest

from app.jobs.build_knowledge import BuildReport
from app.models import KnowledgeChunk
from app.services import kb_build


def _row(session, status="pending", content_type="policy"):
    c = KnowledgeChunk(
        category="c",
        questions="q",
        answer="a",
        content_type=content_type,
        vectorize_status=status,
    )
    session.add(c)
    session.commit()
    return c


def test_status_splits_document_and_mined(session):
    _row(session, "pending")
    _row(session, "done")
    _row(session, "pending", "mined_qa")
    _row(session, "done", "mined_qa")
    s = kb_build.status(session)
    assert (s["pending"], s["done"], s["total"]) == (1, 1, 2)
    assert (s["mined_pending"], s["mined_done"], s["mined_total"]) == (1, 1, 2)


def test_status_returns_exact_keys(session):
    assert set(kb_build.status(session)) == {
        "pending",
        "done",
        "total",
        "mined_total",
        "mined_pending",
        "mined_done",
        "running",
        "started_at",
        "finished_at",
        "last_report",
        "last_error",
    }
    assert set(kb_build.snapshot()) == {
        "running",
        "started_at",
        "finished_at",
        "last_report",
        "last_error",
    }


def test_null_content_type_counts_as_document(session):
    _row(session, "pending", None)
    s = kb_build.status(session)
    assert (s["pending"], s["total"], s["mined_total"]) == (1, 1, 0)


def test_start_build_success(monkeypatch, session):
    kb_build.reset_state_for_tests()
    monkeypatch.setattr(
        kb_build,
        "get_settings",
        lambda: type("S", (), {"knowledge_docs_dir": "knowledge"})(),
    )
    monkeypatch.setattr(kb_build, "get_embedder", lambda: object())
    monkeypatch.setattr(kb_build, "get_vector_store", lambda: object())

    calls = []

    def fake_build(session, docs_dir, embedder, store, *, reset=False):
        calls.append((docs_dir, reset))
        return BuildReport(documents=1, chunks=2, inserted=2, reused=0, vectorized=2)

    monkeypatch.setattr(kb_build, "build_knowledge", fake_build)
    assert kb_build.start_build() is not None
    assert kb_build.wait_for_build(2) is True
    # A regression to reset=False would leak stale document chunks; the docs
    # dir must come from settings, not a hard-coded path.
    assert calls == [(Path("knowledge"), True)]
    s = kb_build.status(session)
    assert s["running"] is False and s["last_report"]["documents"] == 1
    assert s["last_error"] is None
    assert s["started_at"] is not None and s["finished_at"] is not None


def test_start_build_rejects_concurrent(monkeypatch, session):
    kb_build.reset_state_for_tests()
    gate = threading.Event()
    monkeypatch.setattr(
        kb_build,
        "get_settings",
        lambda: type("S", (), {"knowledge_docs_dir": "knowledge"})(),
    )
    monkeypatch.setattr(kb_build, "get_embedder", lambda: object())
    monkeypatch.setattr(kb_build, "get_vector_store", lambda: object())

    def blocking(*a, **k):
        gate.wait(5)
        return BuildReport(0, 0, 0, 0, 0)

    monkeypatch.setattr(kb_build, "build_knowledge", blocking)
    assert kb_build.start_build() is not None
    assert kb_build.start_build() is None
    gate.set()
    assert kb_build.wait_for_build(2) is True


def test_done_event_tracks_build_lifecycle(monkeypatch):
    """Regression: completion event and running state must move together."""
    kb_build.reset_state_for_tests()
    gate = threading.Event()
    monkeypatch.setattr(
        kb_build,
        "get_settings",
        lambda: type("S", (), {"knowledge_docs_dir": "knowledge"})(),
    )
    monkeypatch.setattr(kb_build, "get_embedder", lambda: object())
    monkeypatch.setattr(kb_build, "get_vector_store", lambda: object())

    def blocking(*a, **k):
        gate.wait(5)
        return BuildReport(0, 0, 0, 0, 0)

    monkeypatch.setattr(kb_build, "build_knowledge", blocking)

    assert kb_build._done.is_set() is True  # reset: nothing to wait for
    assert kb_build.start_build() is not None
    assert kb_build._done.is_set() is False  # build running: waiters block
    assert kb_build.wait_for_build(0) is False
    gate.set()
    assert kb_build.wait_for_build(2) is True
    assert kb_build._done.is_set() is True  # completed build wakes waiters


def test_build_failure_is_visible(monkeypatch, session):
    kb_build.reset_state_for_tests()
    monkeypatch.setattr(
        kb_build,
        "get_settings",
        lambda: type("S", (), {"knowledge_docs_dir": "knowledge"})(),
    )
    monkeypatch.setattr(kb_build, "get_embedder", lambda: object())
    monkeypatch.setattr(kb_build, "get_vector_store", lambda: object())

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(kb_build, "build_knowledge", boom)
    assert kb_build.start_build() is not None
    assert kb_build.wait_for_build(2) is True
    assert "boom" in kb_build.status(session)["last_error"]
    assert kb_build.is_running() is False


def test_thread_start_failure_does_not_wedge_state(monkeypatch):
    """A failed ``Thread.start`` must roll back instead of blocking every build."""
    kb_build.reset_state_for_tests()
    monkeypatch.setattr(
        kb_build,
        "get_settings",
        lambda: type("S", (), {"knowledge_docs_dir": "knowledge"})(),
    )
    monkeypatch.setattr(kb_build, "get_embedder", lambda: object())
    monkeypatch.setattr(kb_build, "get_vector_store", lambda: object())

    class BoomThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("thread start failed")

    real_thread = threading.Thread
    monkeypatch.setattr(kb_build.threading, "Thread", BoomThread)
    with pytest.raises(RuntimeError, match="thread start failed"):
        kb_build.start_build()

    snap = kb_build.snapshot()
    assert kb_build.is_running() is False
    assert kb_build.wait_for_build(0) is True
    assert snap["finished_at"] is not None
    assert "thread start failed" in snap["last_error"]

    # The state machine recovers: a later build starts normally.
    monkeypatch.setattr(kb_build.threading, "Thread", real_thread)
    monkeypatch.setattr(
        kb_build,
        "build_knowledge",
        lambda *a, **k: BuildReport(0, 0, 0, 0, 0),
    )
    assert kb_build.start_build() is not None
    assert kb_build.wait_for_build(2) is True
    assert kb_build.is_running() is False
