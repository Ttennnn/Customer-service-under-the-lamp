import pytest

from app.models import KnowledgeChunk
from app.rag.splitter import Chunk
from app.rag.ingest import (
    ingest_document_chunks,
    insert_mined_qa,
    chunk_fingerprint,
    MinedQA,
    reset_document_domain,
    reset_all,
)


def _chunk(q="问", a="答", cat="售后", path="文档 > 节"):
    return Chunk(category=cat, questions=q, answer=a, section_path=path,
                 content_type="faq", is_key_clause=False)


class FakeStore:
    def __init__(self):
        self.deleted = []

    def delete_ids(self, ids):
        self.deleted.append(ids)


def test_ingest_twice_is_idempotent_and_links(session):
    chunks = [_chunk(q="q1"), _chunk(q="q2"), _chunk(q="q3")]
    rows1, r1 = ingest_document_chunks(session, chunks)
    rows2, r2 = ingest_document_chunks(session, chunks)
    assert (r1.inserted, r1.reused) == (3, 0) and (r2.inserted, r2.reused) == (0, 3)
    assert [r.id for r in rows1] == [r.id for r in rows2]
    session.expire_all()
    assert rows1[0].prev_chunk_id is None and rows1[0].next_chunk_id == rows1[1].id
    assert rows1[1].prev_chunk_id == rows1[0].id and rows1[1].next_chunk_id == rows1[2].id
    assert rows1[2].next_chunk_id is None


def test_reset_document_keeps_mined(session):
    doc_rows, _ = ingest_document_chunks(session, [_chunk(q="doc")])
    doc_rows[0].vector_id = "chunk_doc"; session.commit()
    mined = insert_mined_qa(session, [MinedQA("挖矿问", "挖矿答", "售后")])
    mined[0].vector_id = "chunk_mined"; session.commit()
    store = FakeStore(); n = reset_document_domain(session, store)
    assert n == 1 and store.deleted == [["chunk_doc"]]
    session.expire_all()
    ids = [r.id for r in session.query(KnowledgeChunk).all()]
    assert ids == [mined[0].id]


def test_reset_all_clears_everything(session):
    ingest_document_chunks(session, [_chunk(q="doc")])
    insert_mined_qa(session, [MinedQA("挖矿问", "挖矿答", "售后")])
    assert reset_all(session, FakeStore()) == 2
    assert session.query(KnowledgeChunk).count() == 0


def test_chunk_fingerprint_is_the_content_identity_tuple():
    chunk = _chunk(q="q", a="a", cat="c", path="p")
    assert chunk_fingerprint(chunk) == ("faq", "p", "c", "q", "a")

    empty = Chunk(category="c", questions="q", answer="a", section_path=None,
                  content_type=None, is_key_clause=False)
    assert chunk_fingerprint(empty) == ("", "", "c", "q", "a")
    assert chunk_fingerprint(_chunk(q="different")) != chunk_fingerprint(chunk)


def test_ingest_returns_rows_in_input_order_and_pending(session):
    chunks = [_chunk(q=f"q{i}") for i in range(3)]
    rows, report = ingest_document_chunks(session, chunks)
    assert (report.inserted, report.reused) == (3, 0)
    assert [r.questions for r in rows] == ["q0", "q1", "q2"]
    assert all(r.vectorize_status == "pending" for r in rows)
    assert all(r.content_type == "faq" for r in rows)
    assert all(r.section_path == "文档 > 节" for r in rows)
    assert all(r.is_key_clause is False for r in rows)


def test_ingest_truncates_ddl_lengths_and_stays_idempotent(session):
    leaf = "末节"
    path = " > ".join(["长" * 400, "中" * 400, leaf])
    chunk = _chunk(q="long", cat="类" * 300, path=path)
    rows, report = ingest_document_chunks(session, [chunk])
    row_id = rows[0].id
    session.expire_all()
    row = session.get(KnowledgeChunk, row_id)
    assert len(row.section_path) <= 512
    assert row.section_path.endswith(leaf)
    assert len(row.category) == 255

    _, report2 = ingest_document_chunks(session, [chunk])
    assert (report.inserted, report.reused) == (1, 0)
    assert (report2.inserted, report2.reused) == (0, 1)


def test_insert_mined_qa_dedupes_and_sets_fields(session):
    first = insert_mined_qa(session, [MinedQA("挖矿问", "挖矿答", "售后")])
    second = insert_mined_qa(
        session,
        [MinedQA("挖矿问", "挖矿答", "售后"), MinedQA("问2", "答2", "售后")],
    )
    assert [r.id for r in second[:1]] == [first[0].id]
    rows = session.query(KnowledgeChunk).order_by(KnowledgeChunk.id).all()
    assert len(rows) == 2
    row = rows[0]
    assert row.content_type == "mined_qa"
    assert row.section_path is None
    assert row.is_key_clause is False
    assert row.vectorize_status == "pending"
    assert row.prev_chunk_id is None and row.next_chunk_id is None


def test_reset_document_domain_includes_null_content_type(session):
    ingest_document_chunks(session, [_chunk(q="doc")])
    session.add(
        KnowledgeChunk(
            category="c", questions="nullq", answer="a",
            content_type=None, vectorize_status="pending",
        )
    )
    session.commit()
    assert reset_document_domain(session, FakeStore()) == 2
    assert session.query(KnowledgeChunk).count() == 0


def test_reset_all_batches_vector_deletes(session):
    session.add_all(
        [
            KnowledgeChunk(
                category="c", questions=f"q{i}", answer="a", content_type="faq",
                section_path="p", vectorize_status="done", vector_id=f"v{i}",
            )
            for i in range(501)
        ]
    )
    session.commit()
    store = FakeStore()
    assert reset_all(session, store) == 501
    assert [len(batch) for batch in store.deleted] == [500, 1]
    assert store.deleted[0] == [f"v{i}" for i in range(500)]
    assert store.deleted[1] == ["v500"]
    assert session.query(KnowledgeChunk).count() == 0


def test_reset_document_store_failure_keeps_mysql_rows(session):
    rows, _ = ingest_document_chunks(session, [_chunk(q="doc")])
    rows[0].vector_id = "chunk_doc"
    session.commit()

    class BoomStore:
        def delete_ids(self, ids):
            raise RuntimeError("milvus down")

    with pytest.raises(RuntimeError, match="milvus down"):
        reset_document_domain(session, BoomStore())
    session.expire_all()
    assert session.query(KnowledgeChunk).count() == 1


def test_reset_all_store_failure_keeps_mysql_rows(session):
    rows, _ = ingest_document_chunks(session, [_chunk(q="doc")])
    rows[0].vector_id = "chunk_doc"
    session.commit()
    insert_mined_qa(session, [MinedQA("挖矿问", "挖矿答", "售后")])

    class BoomStore:
        def delete_ids(self, ids):
            raise RuntimeError("milvus down")

    with pytest.raises(RuntimeError, match="milvus down"):
        reset_all(session, BoomStore())
    session.expire_all()
    assert session.query(KnowledgeChunk).count() == 2
