from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


def _settings(tmp_path):
    from app.config import Settings

    return Settings(
        _env_file=None,
        openai_api_key="x",
        knowledge_docs_dir=str(tmp_path / "knowledge"),
    )


def test_files_lists_effective_docs_with_sources(monkeypatch, tmp_path):
    import app.api.kb as kb

    root = tmp_path / "knowledge"
    (root / "uploads").mkdir(parents=True)
    (root / "seed.md").write_text(
        "---\ncontent_type: policy\n---\n# Seed\n\nS\n", encoding="utf-8"
    )
    (root / "uploads" / "seed.md").write_text(
        "---\ncontent_type: policy\n---\n# Upload\n\nU\n", encoding="utf-8"
    )
    monkeypatch.setattr(kb, "get_settings", lambda: _settings(tmp_path))
    r = TestClient(app).get("/kb/api/files")
    assert r.status_code == 200
    files = r.json()["files"]
    assert files == [
        {
            "name": "seed.md",
            "source": "upload",
            "size": (root / "uploads" / "seed.md").stat().st_size,
            "modified_at": files[0]["modified_at"],
        }
    ]
    assert isinstance(files[0]["modified_at"], str)


def test_files_marks_root_docs_as_seed(monkeypatch, tmp_path):
    import app.api.kb as kb

    root = tmp_path / "knowledge"
    root.mkdir(parents=True)
    (root / "a.md").write_text("# A\n", encoding="utf-8")
    monkeypatch.setattr(kb, "get_settings", lambda: _settings(tmp_path))
    r = TestClient(app).get("/kb/api/files")
    assert r.status_code == 200
    files = r.json()["files"]
    assert files == [
        {
            "name": "a.md",
            "source": "seed",
            "size": (root / "a.md").stat().st_size,
            "modified_at": files[0]["modified_at"],
        }
    ]


def test_status_returns_counts_and_state(monkeypatch, session):
    import app.api.kb as kb
    from app.models import KnowledgeChunk

    session.add(
        KnowledgeChunk(
            category="c",
            questions="q",
            answer="a",
            content_type="policy",
            vectorize_status="done",
        )
    )
    session.commit()
    monkeypatch.setattr(kb, "get_settings", lambda: _settings(Path("/tmp")))
    r = TestClient(app).get("/kb/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["done"] == 1 and body["total"] == 1 and body["mined_total"] == 0


def test_status_database_error_is_500(monkeypatch):
    import app.api.kb as kb

    def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(kb, "get_settings", lambda: _settings(Path("/tmp")))
    monkeypatch.setattr(kb.db_session, "get_session_factory", lambda: boom)
    r = TestClient(app).get("/kb/api/status")
    assert r.status_code == 500


def test_build_returns_202_and_409(monkeypatch):
    import app.api.kb as kb

    monkeypatch.setattr(kb.kb_build, "start_build", lambda: True)
    r = TestClient(app).post("/kb/api/build")
    assert r.status_code == 202
    assert r.json() == {"status": "running"}
    monkeypatch.setattr(kb.kb_build, "start_build", lambda: False)
    r = TestClient(app).post("/kb/api/build")
    assert r.status_code == 409
    assert r.json() == {"detail": "a build is already running"}
