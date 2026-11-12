import threading

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

    def fake_build(session, docs_dir, embedder, store, *, reset=False):
        return BuildReport(documents=1, chunks=2, inserted=2, reused=0, vectorized=2)

    monkeypatch.setattr(kb_build, "build_knowledge", fake_build)
    assert kb_build.start_build() is True
    assert kb_build.wait_for_build(2) is True
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
    assert kb_build.start_build() is True
    assert kb_build.start_build() is False
    gate.set()
    assert kb_build.wait_for_build(2) is True


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
    assert kb_build.start_build() is True
    assert kb_build.wait_for_build(2) is True
    assert "boom" in kb_build.status(session)["last_error"]
    assert kb_build.is_running() is False
