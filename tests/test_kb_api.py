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


def _client(monkeypatch, tmp_path):
    import app.api.kb as kb

    monkeypatch.setattr(kb, "get_settings", lambda: _settings(tmp_path))
    return TestClient(app)


def test_upload_plain_md_injects_frontmatter(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post(
        "/kb/api/upload",
        files={"files": ("new.md", b"# New\n\nBody\n", "text/markdown")},
        data={"content_type": "policy", "category": "售后"},
    )
    assert r.status_code == 200
    saved = r.json()["saved"][0]
    assert saved["content_type"] == "policy" and saved["name"] == "new.md"
    p = tmp_path / "knowledge" / "uploads" / "new.md"
    text = p.read_text(encoding="utf-8")
    assert text.startswith("---\ncontent_type: policy\ncategory: 售后\n---")
    assert p.read_bytes().startswith(
        "---\ncontent_type: policy\ncategory: 售后\n---\n\n# New".encode()
    )
    from app.rag.docs import load_document

    assert load_document(p).title == "New"


def test_upload_existing_frontmatter_wins(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    body = "---\ncontent_type: faq\ncategory: 商品\n---\n# FAQ\n\n## 尺码怎么选?\n\n按脚长。\n"
    r = c.post(
        "/kb/api/upload",
        files={"files": ("faq.md", body.encode(), "text/markdown")},
        data={"content_type": "policy"},
    )
    assert r.json()["saved"][0]["content_type"] == "faq"


def test_upload_rejects_bad_files(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    for name, data in [
        ("a.txt", b"# A\n"),
        ("bad.md", b"\xff\xfe"),
        ("noh1.md", b"no heading\n"),
    ]:
        r = c.post(
            "/kb/api/upload",
            files={"files": (name, data, "text/markdown")},
            data={"content_type": "policy"},
        )
        assert r.status_code == 200 and r.json()["saved"] == []
    assert not any((tmp_path / "knowledge" / "uploads").glob("*.md"))


def test_upload_rejects_path_traversal_and_invalid_type(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post(
        "/kb/api/upload",
        files={"files": ("../x.md", b"# X\n", "text/markdown")},
        data={"content_type": "policy"},
    )
    assert r.json()["saved"] == []
    r = c.post(
        "/kb/api/upload",
        files={"files": ("x.md", b"# X\n", "text/markdown")},
        data={"content_type": "blog"},
    )
    assert r.status_code == 400


def test_upload_rejects_oversized_file(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    data = b"# X\n" + b"x" * (2 * 1024 * 1024)
    r = c.post(
        "/kb/api/upload",
        files={"files": ("big.md", data, "text/markdown")},
        data={"content_type": "policy"},
    )
    assert r.json()["saved"] == []
    assert (
        "2 MiB" in r.json()["rejected"][0]["reason"]
        or "size" in r.json()["rejected"][0]["reason"].lower()
    )


def test_upload_rejects_too_many_files(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    files = [("files", (f"f{i}.md", b"# X\n", "text/markdown")) for i in range(21)]
    r = c.post("/kb/api/upload", files=files, data={"content_type": "policy"})
    assert r.status_code == 400


def test_upload_rejects_newline_category(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post(
        "/kb/api/upload",
        files={"files": ("x.md", b"# X\n", "text/markdown")},
        data={"content_type": "policy", "category": "a\nb"},
    )
    assert r.status_code == 400


def test_upload_conflicts_with_running_build(monkeypatch, tmp_path):
    import app.api.kb as kb

    c = _client(monkeypatch, tmp_path)
    monkeypatch.setattr(kb.kb_build, "is_running", lambda: True)
    r = c.post(
        "/kb/api/upload",
        files={"files": ("x.md", b"# X\n", "text/markdown")},
        data={"content_type": "policy"},
    )
    assert r.status_code == 409


def test_upload_accepts_bom_and_overwrites_same_name(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    first = "﻿# One\n\nfirst\n".encode("utf-8")
    second = "# Two\n\nsecond\n"
    r = c.post(
        "/kb/api/upload",
        files={"files": ("same.md", first, "text/markdown")},
        data={"content_type": "policy"},
    )
    assert r.status_code == 200 and r.json()["saved"][0]["name"] == "same.md"
    r = c.post(
        "/kb/api/upload",
        files={"files": ("same.md", second.encode(), "text/markdown")},
        data={"content_type": "manual"},
    )
    assert r.status_code == 200 and r.json()["saved"][0]["content_type"] == "manual"
    text = (tmp_path / "knowledge" / "uploads" / "same.md").read_text(encoding="utf-8")
    assert text.startswith("---\ncontent_type: manual\n---\n\n# Two\n")
    assert "# One" not in text


def test_upload_cleans_temp_file_on_validation_failure(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post(
        "/kb/api/upload",
        files={"files": ("noh1.md", b"no heading\n", "text/markdown")},
        data={"content_type": "policy"},
    )
    assert r.status_code == 200 and r.json()["saved"] == []
    uploads = tmp_path / "knowledge" / "uploads"
    assert list(uploads.glob(".tmp-*")) == []
