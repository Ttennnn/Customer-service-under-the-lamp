from types import SimpleNamespace

import pytest

from app.models import KnowledgeChunk
from app.jobs.build_knowledge import build_knowledge
from app.rag.ingest import MinedQA, insert_mined_qa


class FakeEmbedder:
    def embed(self, texts):
        return [[0.0] * 1024 for _ in texts]


class FakeStore:
    def __init__(self):
        self.items = []
        self.deleted = []

    def upsert(self, items):
        self.items.extend(items)

    def delete_ids(self, ids):
        self.deleted.extend(ids)


def _docs(tmp_path):
    d = tmp_path / "knowledge"
    d.mkdir()
    (d / "a.md").write_text(
        "---\ncontent_type: policy\n---\n# 退货政策\n\n## 退货时限\n\n签收后 7 天内。\n",
        encoding="utf-8",
    )
    (d / "b.md").write_text(
        "---\ncontent_type: policy\n---\n# 运费\n\n## 邮费\n\n满 99 元包邮。\n",
        encoding="utf-8",
    )
    return d


def _settings(docs_dir):
    return SimpleNamespace(
        knowledge_docs_dir=str(docs_dir),
        chunk_max_chars=500,
        chunk_overlap_chars=80,
        embedding_batch_size=16,
    )


def _wire_build_cli(monkeypatch, session, docs_dir):
    """Point the build CLI at in-memory fakes instead of real services."""
    from app.jobs import build_knowledge as job

    embedder = FakeEmbedder()
    store = FakeStore()
    monkeypatch.setattr(job, "get_settings", lambda: _settings(docs_dir))
    monkeypatch.setattr(job, "get_session_factory", lambda: (lambda: session))
    monkeypatch.setattr(job, "get_embedder", lambda: embedder)
    monkeypatch.setattr(job, "get_vector_store", lambda: store)
    return job, embedder, store


def test_build_twice_is_idempotent(session, tmp_path):
    d = _docs(tmp_path)
    e = FakeEmbedder()
    s = FakeStore()
    r1 = build_knowledge(session, d, e, s)
    r2 = build_knowledge(session, d, e, s)
    assert r1.inserted == 2 and r1.vectorized == 2
    assert r1.documents == 2 and r1.chunks == 2 and r1.reused == 0
    assert r2.inserted == 0 and r2.reused == 2 and r2.vectorized == 0
    assert session.query(KnowledgeChunk).count() == 2
    assert len(s.items) == 2


def test_no_vectorize_leaves_pending(session, tmp_path):
    d = _docs(tmp_path)
    r = build_knowledge(
        session, d, FakeEmbedder(), FakeStore(), do_vectorize=False
    )
    assert r.vectorized == 0
    assert (
        session.query(KnowledgeChunk)
        .filter_by(vectorize_status="pending")
        .count()
        == 2
    )


def test_reset_preserves_mined(session, tmp_path):
    d = _docs(tmp_path)
    build_knowledge(session, d, FakeEmbedder(), FakeStore())
    insert_mined_qa(session, [MinedQA("挖矿问", "挖矿答", "售后")])
    session.commit()
    r = build_knowledge(session, d, FakeEmbedder(), FakeStore(), reset=True)
    assert r.inserted == 2
    assert (
        session.query(KnowledgeChunk)
        .filter_by(content_type="mined_qa")
        .count()
        == 1
    )


def test_reset_all_wins_over_reset(session, tmp_path):
    d = _docs(tmp_path)
    build_knowledge(session, d, FakeEmbedder(), FakeStore())
    insert_mined_qa(session, [MinedQA("挖矿问", "挖矿答", "售后")])
    session.commit()
    r = build_knowledge(
        session, d, FakeEmbedder(), FakeStore(), reset=True, reset_all=True
    )
    assert r.inserted == 2
    assert (
        session.query(KnowledgeChunk)
        .filter_by(content_type="mined_qa")
        .count()
        == 0
    )


def test_max_chunks_limits_vectorization(session, tmp_path):
    d = _docs(tmp_path)
    r = build_knowledge(
        session, d, FakeEmbedder(), FakeStore(), max_chunks=1
    )
    assert r.inserted == 2 and r.vectorized == 1
    assert (
        session.query(KnowledgeChunk)
        .filter_by(vectorize_status="pending")
        .count()
        == 1
    )


def test_neighbor_links_stay_within_each_document(session, tmp_path):
    d = tmp_path / "knowledge"
    d.mkdir()
    (d / "a.md").write_text(
        "---\ncontent_type: policy\n---\n# A\n\n## 一\n\n甲。\n\n## 二\n\n乙。\n",
        encoding="utf-8",
    )
    (d / "b.md").write_text(
        "---\ncontent_type: policy\n---\n# B\n\n## 三\n\n丙。\n",
        encoding="utf-8",
    )

    build_knowledge(session, d, FakeEmbedder(), FakeStore(), do_vectorize=False)
    rows = session.query(KnowledgeChunk).order_by(KnowledgeChunk.id).all()
    assert [row.questions for row in rows] == ["一", "二", "三"]
    assert rows[0].prev_chunk_id is None
    assert rows[0].next_chunk_id == rows[1].id
    assert rows[1].prev_chunk_id == rows[0].id
    # The first document ends its own chain; no link crosses into b.md.
    assert rows[1].next_chunk_id is None
    assert rows[2].prev_chunk_id is None
    assert rows[2].next_chunk_id is None


def test_reset_all_without_yes_returns_2_and_deletes_nothing(
    session, tmp_path, monkeypatch, capsys
):
    d = _docs(tmp_path)
    build_knowledge(session, d, FakeEmbedder(), FakeStore())

    from app.jobs import build_knowledge as job

    monkeypatch.setattr(
        job, "get_settings", lambda: pytest.fail("must not read settings")
    )
    monkeypatch.setattr(
        job, "get_session_factory", lambda: pytest.fail("must not open a session")
    )
    monkeypatch.setattr(
        job, "get_vector_store", lambda: pytest.fail("must not touch Milvus")
    )

    assert job.main(["--docs-dir", str(d), "--reset-all"]) == 2
    out = capsys.readouterr().out
    assert "--reset-all" in out and "--yes" in out
    assert session.query(KnowledgeChunk).count() == 2


def test_cli_reset_all_with_yes_clears_everything(
    session, tmp_path, monkeypatch, capsys
):
    d = _docs(tmp_path)
    build_knowledge(session, d, FakeEmbedder(), FakeStore())
    insert_mined_qa(session, [MinedQA("挖矿问", "挖矿答", "售后")])
    session.commit()

    job, _, store = _wire_build_cli(monkeypatch, session, d)
    monkeypatch.setattr(
        job, "get_embedder", lambda: pytest.fail("--no-vectorize must not embed")
    )

    code = job.main(
        ["--docs-dir", str(d), "--reset-all", "--yes", "--no-vectorize"]
    )
    assert code == 0
    session.expire_all()
    assert (
        session.query(KnowledgeChunk)
        .filter_by(content_type="mined_qa")
        .count()
        == 0
    )
    assert session.query(KnowledgeChunk).count() == 2
    out = capsys.readouterr().out
    assert "documents=2" in out and "inserted=2" in out and "vectorized=0" in out


def test_vectorize_cli_processes_pending(session, tmp_path, monkeypatch, capsys):
    d = _docs(tmp_path)
    build_knowledge(
        session, d, FakeEmbedder(), FakeStore(), do_vectorize=False
    )

    from app.jobs import vectorize as job

    embedder = FakeEmbedder()
    store = FakeStore()
    monkeypatch.setattr(job, "get_settings", lambda: _settings(d))
    monkeypatch.setattr(job, "get_session_factory", lambda: (lambda: session))
    monkeypatch.setattr(job, "get_embedder", lambda: embedder)
    monkeypatch.setattr(job, "get_vector_store", lambda: store)

    assert job.main([]) == 0
    assert "processed=2" in capsys.readouterr().out
    session.expire_all()
    assert (
        session.query(KnowledgeChunk)
        .filter_by(vectorize_status="pending")
        .count()
        == 0
    )
    assert len(store.items) == 2
