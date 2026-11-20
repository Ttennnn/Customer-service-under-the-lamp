import pytest

from app.models import KnowledgeChunk
from app.rag.milvus_store import VectorItem
from app.rag.vectorize import (
    vector_id_for,
    vectorization_text,
    vectorize_all,
    vectorize_pending,
    vectorize_rows,
)


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(texts)
        return [[0.1] * 1024 for _ in texts]


class FakeStore:
    def __init__(self):
        self.items = []

    def upsert(self, items):
        self.items.extend(items)


def _chunk(session, q="问", a="答", cat="售后", status="pending"):
    c = KnowledgeChunk(
        category=cat,
        questions=q,
        answer=a,
        content_type="faq",
        section_path="文档 > 节",
        vectorize_status=status,
    )
    session.add(c)
    session.commit()
    session.refresh(c)
    return c


def test_vectorize_pending_sets_done_and_vector_id(session):
    c = _chunk(session)
    e = FakeEmbedder()
    s = FakeStore()
    r = vectorize_pending(session, e, s)
    session.expire_all()
    row = session.get(KnowledgeChunk, c.id)
    assert r.processed == 1 and row.vectorize_status == "done"
    assert row.vector_id == vector_id_for(c.id)
    assert e.calls == [["售后\n问\n答"]]
    assert isinstance(s.items[0], VectorItem)
    assert s.items[0].vector_id == vector_id_for(c.id)
    assert s.items[0].chunk_id == c.id and len(s.items[0].vector) == 1024


def test_vectorize_rows_only_publishes_given_rows(session):
    selected = _chunk(session, q="选中", a="A")
    _chunk(session, q="无关", a="B")  # also pending, must stay untouched

    e = FakeEmbedder()
    s = FakeStore()
    report = vectorize_rows(session, [selected], e, s)

    session.expire_all()
    assert report.processed == 1
    assert session.get(KnowledgeChunk, selected.id).vectorize_status == "done"
    other = session.query(KnowledgeChunk).filter_by(questions="无关").one()
    assert other.vectorize_status == "pending"
    assert [item.chunk_id for item in s.items] == [selected.id]
    assert e.calls == [["售后\n选中\nA"]]


def test_vectorize_rows_validates_batch_size(session):
    with pytest.raises(ValueError, match="batch_size"):
        vectorize_rows(session, [], FakeEmbedder(), FakeStore(), batch_size=0)


def test_second_run_does_nothing(session):
    _chunk(session)
    e = FakeEmbedder()
    s = FakeStore()
    vectorize_pending(session, e, s)
    vectorize_pending(session, e, s)
    assert len(e.calls) == 1 and len(s.items) == 1


def test_embed_failure_leaves_pending_and_retry_works(session):
    c = _chunk(session)

    class Boom:
        def embed(self, texts):
            raise RuntimeError("api down")

    try:
        vectorize_pending(session, Boom(), FakeStore())
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected failure")
    session.expire_all()
    assert session.get(KnowledgeChunk, c.id).vectorize_status == "pending"
    vectorize_pending(session, FakeEmbedder(), FakeStore())


def test_max_chunks_leaves_rest_pending(session):
    c1 = _chunk(session, q="q1")
    c2 = _chunk(session, q="q2")
    r = vectorize_pending(session, FakeEmbedder(), FakeStore(), max_chunks=1)
    session.expire_all()
    assert r.processed == 1
    assert session.get(KnowledgeChunk, c1.id).vectorize_status == "done"
    assert session.get(KnowledgeChunk, c2.id).vectorize_status == "pending"


def test_text_and_id_helpers():
    c = KnowledgeChunk(category="C", questions="Q", answer="A")
    assert vectorization_text(c) == "C\nQ\nA"
    assert vector_id_for(42) == "chunk_42"


def test_batches_are_embedded_and_committed_in_groups(session):
    chunks = [_chunk(session, q=f"q{i}") for i in range(5)]
    e = FakeEmbedder()
    s = FakeStore()
    r = vectorize_pending(session, e, s, batch_size=2)
    assert r.processed == 5
    assert [len(call) for call in e.calls] == [2, 2, 1]
    assert [item.chunk_id for item in s.items] == [c.id for c in chunks]


def test_pending_rows_are_processed_in_id_order(session):
    late = _chunk(session, q="late")
    _chunk(session, q="done", status="done")
    early = _chunk(session, q="early")
    e = FakeEmbedder()
    vectorize_pending(session, e, FakeStore())
    assert e.calls == [["售后\nlate\n答", "售后\nearly\n答"]]
    assert late.id < early.id


def test_store_failure_leaves_rows_pending(session):
    c = _chunk(session)

    class BoomStore:
        def upsert(self, items):
            raise RuntimeError("milvus down")

    with pytest.raises(RuntimeError, match="milvus down"):
        vectorize_pending(session, FakeEmbedder(), BoomStore())
    session.expire_all()
    assert session.get(KnowledgeChunk, c.id).vectorize_status == "pending"


def test_failure_in_later_batch_keeps_committed_batches_done(session):
    chunks = [_chunk(session, q=f"q{i}") for i in range(3)]

    class FlakyEmbedder:
        def __init__(self):
            self.calls = 0

        def embed(self, texts):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("api down")
            return [[0.1] * 1024 for _ in texts]

    store = FakeStore()
    with pytest.raises(RuntimeError, match="api down"):
        vectorize_pending(session, FlakyEmbedder(), store, batch_size=2)

    session.expire_all()
    assert session.get(KnowledgeChunk, chunks[0].id).vectorize_status == "done"
    assert session.get(KnowledgeChunk, chunks[1].id).vectorize_status == "done"
    assert session.get(KnowledgeChunk, chunks[2].id).vectorize_status == "pending"

    r = vectorize_pending(session, FakeEmbedder(), store, batch_size=2)
    session.expire_all()
    assert r.processed == 1
    assert session.get(KnowledgeChunk, chunks[2].id).vectorize_status == "done"
    assert [item.chunk_id for item in store.items] == [c.id for c in chunks]


def test_wrong_dimension_fails_before_upsert_and_stays_pending(session):
    c = _chunk(session)

    class ShortEmbedder:
        def embed(self, texts):
            return [[0.1] * 3 for _ in texts]

    s = FakeStore()
    with pytest.raises(ValueError, match="1024"):
        vectorize_pending(session, ShortEmbedder(), s)
    assert s.items == []
    session.expire_all()
    assert session.get(KnowledgeChunk, c.id).vectorize_status == "pending"


def test_empty_pending_returns_zero_report(session):
    r = vectorize_pending(session, FakeEmbedder(), FakeStore())
    assert (r.processed, r.failed) == (0, 0)


def test_vectorize_pending_writes_bm25_text_and_metadata(session):
    c = _chunk(session)
    s = FakeStore()
    vectorize_pending(session, FakeEmbedder(), s)
    item = s.items[0]
    assert item.text == "售后\n问\n答"
    assert item.category == "售后" and item.content_type == "faq"
    assert item.section_path == "文档 > 节"


def test_vectorize_all_reprocesses_done_rows(session):
    c = _chunk(session, status="done")
    e = FakeEmbedder()
    s = FakeStore()
    r = vectorize_all(session, e, s)
    assert r.processed == 1 and s.items[0].chunk_id == c.id
