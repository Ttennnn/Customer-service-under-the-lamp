"""审核 API 与知识库写回(Task 7)。

所有测试都用 Fake embedder/store 或 monkeypatch 掉写回函数,不联网、不建
真实 Milvus 集合。审批写回的事务边界是:chunk 先 pending 落库并向量化成功,
之后才把审核状态置为「通过」。
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import KnowledgeChunk, LowConfidenceQuestion, ReviewQueue
from app.services import kb_build, knowledge_approval
from app.services.knowledge_approval import (
    ApprovalBusy,
    InvalidApprovedAnswer,
    ReviewNotFound,
    ReviewNotPending,
    approve_review,
    reject_review,
)


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[0.1] * 1024 for _ in texts]


class FakeStore:
    def __init__(self, error=None):
        self.items = []
        self.deleted = []
        self.error = error

    def upsert(self, items):
        if self.error is not None:
            raise self.error
        self.items.extend(items)

    def delete_ids(self, ids):
        self.deleted.extend(ids)


@pytest.fixture
def client(session):
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _never_build(monkeypatch):
    """Default: no knowledge build is running; tests opt in when needed."""
    monkeypatch.setattr(kb_build, "is_running", lambda: False)


def _review(
    session,
    q="退款多久到账",
    status="待审",
    count=1,
    answer=None,
):
    row = ReviewQueue(
        normalized_question=q,
        review_status=status,
        occurrence_count=count,
        ai_suggested_answer=answer,
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


# ---------------------------------------------------------------------------
# service: approve / reject
# ---------------------------------------------------------------------------


def test_approve_writes_chunk_then_marks_approved(session, monkeypatch):
    review = _review(session)
    calls = []
    monkeypatch.setattr(
        knowledge_approval,
        "_write_and_vectorize",
        lambda *a, **k: calls.append("write"),
    )

    out = approve_review(session, review.id, "1-3 个工作日")

    assert calls == ["write"]
    assert out.review_status == "通过"
    assert out.approved_answer == "1-3 个工作日"


def test_approve_vectorize_failure_keeps_pending(session, monkeypatch):
    review = _review(session)

    def fail(*args, **kwargs):
        raise RuntimeError("milvus down")

    monkeypatch.setattr(knowledge_approval, "_write_and_vectorize", fail)

    with pytest.raises(RuntimeError, match="milvus down"):
        approve_review(session, review.id, "答案")

    session.expire_all()
    stored = session.get(ReviewQueue, review.id)
    assert stored.review_status == "待审"
    assert stored.approved_answer is None


def test_approve_writes_mined_chunk_and_vectorizes(session):
    review = _review(session)
    embedder = FakeEmbedder()
    store = FakeStore()

    out = approve_review(
        session,
        review.id,
        "1-3 个工作日",
        embedder=embedder,
        store=store,
    )

    assert out.review_status == "通过"
    chunk = session.query(KnowledgeChunk).one()
    assert chunk.content_type == "mined_qa"
    assert chunk.category == "通用"
    assert chunk.questions == "退款多久到账"
    assert chunk.answer == "1-3 个工作日"
    assert chunk.vectorize_status == "done"
    assert embedder.calls == [["通用\n退款多久到账\n1-3 个工作日"]]
    assert store.items[0].chunk_id == chunk.id


def test_vectorize_failure_removes_chunk_then_retry_succeeds(session):
    review = _review(session)
    with pytest.raises(RuntimeError, match="milvus down"):
        approve_review(
            session,
            review.id,
            "1-3 个工作日",
            embedder=FakeEmbedder(),
            store=FakeStore(error=RuntimeError("milvus down")),
        )

    session.expire_all()
    assert session.get(ReviewQueue, review.id).review_status == "待审"
    # A failed writeback must not leave a publishable chunk behind.
    assert session.query(KnowledgeChunk).count() == 0

    approved = approve_review(
        session,
        review.id,
        "1-3 个工作日",
        embedder=FakeEmbedder(),
        store=FakeStore(),
    )

    assert approved.review_status == "通过"
    session.expire_all()
    assert session.query(KnowledgeChunk).count() == 1
    assert session.query(KnowledgeChunk).one().vectorize_status == "done"


# --- regression: approval must never publish a superseded/unapproved answer ---


def test_failed_old_answer_then_retry_publishes_only_new_answer(session):
    """A failed OLD approval must not leak into the retried NEW approval."""
    review = _review(session)

    with pytest.raises(RuntimeError, match="milvus down"):
        approve_review(
            session,
            review.id,
            "旧答案",
            embedder=FakeEmbedder(),
            store=FakeStore(error=RuntimeError("milvus down")),
        )
    session.expire_all()
    assert session.query(KnowledgeChunk).count() == 0

    embedder = FakeEmbedder()
    store = FakeStore()
    approved = approve_review(
        session, review.id, "新答案", embedder=embedder, store=store
    )

    assert approved.review_status == "通过"
    session.expire_all()
    chunks = session.query(KnowledgeChunk).all()
    assert len(chunks) == 1
    assert chunks[0].answer == "新答案"
    assert chunks[0].vectorize_status == "done"
    # The only text embedded/upserted is the NEW answer.
    assert embedder.calls == [["通用\n退款多久到账\n新答案"]]
    assert [item.text for item in store.items] == ["通用\n退款多久到账\n新答案"]
    assert store.deleted == []


def test_approving_a_does_not_vectorize_unrelated_pending_b(session):
    review = _review(session)
    unrelated = KnowledgeChunk(
        category="通用",
        questions="怎么开发票",
        answer="B 的旧答案",
        section_path=None,
        content_type="mined_qa",
        is_key_clause=False,
        vectorize_status="pending",
    )
    session.add(unrelated)
    session.commit()
    session.refresh(unrelated)

    embedder = FakeEmbedder()
    store = FakeStore()
    approve_review(session, review.id, "A 的答案", embedder=embedder, store=store)

    session.expire_all()
    # Only A was embedded/upserted; B is untouched and still pending.
    assert embedder.calls == [["通用\n退款多久到账\nA 的答案"]]
    assert [item.chunk_id for item in store.items] == [
        session.query(KnowledgeChunk).filter_by(answer="A 的答案").one().id
    ]
    assert session.get(KnowledgeChunk, unrelated.id).vectorize_status == "pending"


def test_unrelated_failing_pending_b_does_not_block_a(session):
    """B's bad pending row is irrelevant: scoped vectorize only touches A."""
    review = _review(session)
    bad = KnowledgeChunk(
        category="通用",
        questions="会失败的问题",
        answer="B",
        section_path=None,
        content_type="mined_qa",
        is_key_clause=False,
        vectorize_status="pending",
    )
    session.add(bad)
    session.commit()

    class ShortEmbedder(FakeEmbedder):
        """Would fail if asked for B: returns too few vectors for any batch > 0."""

        def embed(self, texts):
            self.calls.append(list(texts))
            if any("会失败的问题" in text for text in texts):
                return []
            return [[0.1] * 1024 for _ in texts]

    embedder = ShortEmbedder()
    approved = approve_review(
        session, review.id, "A 的答案", embedder=embedder, store=FakeStore()
    )

    assert approved.review_status == "通过"
    session.expire_all()
    assert session.get(ReviewQueue, review.id).review_status == "通过"
    assert session.get(KnowledgeChunk, bad.id).vectorize_status == "pending"


def test_stale_same_question_done_row_is_deleted_with_its_vector(session):
    """A done-but-unapproved chunk is removed (Milvus vector first)."""
    review = _review(session)
    stale = KnowledgeChunk(
        category="通用",
        questions="退款多久到账",
        answer="旧答案",
        section_path=None,
        content_type="mined_qa",
        is_key_clause=False,
        vectorize_status="done",
        vector_id="chunk_999",
    )
    session.add(stale)
    session.commit()
    session.refresh(stale)

    store = FakeStore()
    approved = approve_review(
        session, review.id, "新答案", embedder=FakeEmbedder(), store=store
    )

    assert approved.review_status == "通过"
    session.expire_all()
    assert store.deleted == ["chunk_999"]
    chunks = session.query(KnowledgeChunk).all()
    assert len(chunks) == 1
    # The stale answer/vector is gone; only the new approved chunk remains.
    # (SQLite may reuse the deleted row id, so assert on content, not id.)
    assert chunks[0].answer == "新答案"
    assert chunks[0].vectorize_status == "done"
    assert session.query(KnowledgeChunk).filter_by(answer="旧答案").count() == 0


def test_approve_blank_answer_is_rejected_before_any_write(session):
    review = _review(session)

    with pytest.raises(InvalidApprovedAnswer):
        approve_review(session, review.id, "   \n ")

    session.expire_all()
    assert session.get(ReviewQueue, review.id).review_status == "待审"
    assert session.query(KnowledgeChunk).count() == 0


def test_approve_missing_and_non_pending(session):
    with pytest.raises(ReviewNotFound):
        approve_review(session, 9999, "答案")

    review = _review(session, status="通过")
    with pytest.raises(ReviewNotPending):
        approve_review(session, review.id, "答案")


def test_approve_is_busy_while_a_build_runs(session, monkeypatch):
    review = _review(session)
    monkeypatch.setattr(kb_build, "is_running", lambda: True)

    with pytest.raises(ApprovalBusy):
        approve_review(session, review.id, "答案")


def test_approve_is_busy_when_the_module_lock_is_held(session):
    review = _review(session)
    assert knowledge_approval._lock.acquire(blocking=False)
    try:
        with pytest.raises(ApprovalBusy):
            approve_review(session, review.id, "答案")
    finally:
        knowledge_approval._lock.release()


def test_reject_marks_rejected_and_guards_transitions(session):
    review = _review(session)

    out = reject_review(session, review.id)

    assert out.review_status == "驳回"
    with pytest.raises(ReviewNotPending):
        reject_review(session, review.id)
    with pytest.raises(ReviewNotFound):
        reject_review(session, 9999)


# ---------------------------------------------------------------------------
# REST: queue / detail
# ---------------------------------------------------------------------------


def test_queue_defaults_to_pending_and_orders_by_occurrence(client, session):
    _review(session, q="低频", count=1)
    _review(session, q="高频", count=9)
    _review(session, q="已通过", status="通过", count=99)

    r = client.get("/api/review/queue")

    assert r.status_code == 200
    items = r.json()["items"]
    assert [item["normalized_question"] for item in items] == ["高频", "低频"]
    assert items[0]["occurrence_count"] == 9
    assert items[0]["review_status"] == "待审"
    assert items[0]["ai_suggested_answer"] is None
    assert isinstance(items[0]["created_at"], str)


def test_queue_status_filter(client, session):
    _review(session, q="待审项", count=1)
    _review(session, q="驳回项", status="驳回", count=1)

    r = client.get("/api/review/queue", params={"status": "驳回"})

    assert r.status_code == 200
    assert [item["normalized_question"] for item in r.json()["items"]] == ["驳回项"]


def test_detail_returns_review_and_linked_low_confidence_rows(client, session):
    review = _review(session, answer="AI 建议")
    snapshot = [
        {
            "section_path": "售后 > 退款",
            "rerank_score": 0.42,
            "answer": "3 个工作日",
        }
    ]
    session.add_all(
        [
            LowConfidenceQuestion(
                raw_question="退款多久到账",
                source="self_check",
                reason="low score",
                retrieved_chunks=snapshot,
                matched_review_id=review.id,
            ),
            LowConfidenceQuestion(
                raw_question="未归并的问题",
                source="retrieval_low_conf",
            ),
        ]
    )
    session.commit()

    r = client.get(f"/api/review/{review.id}")

    assert r.status_code == 200
    body = r.json()
    assert body["review"]["id"] == review.id
    assert body["review"]["normalized_question"] == "退款多久到账"
    assert body["review"]["ai_suggested_answer"] == "AI 建议"
    assert len(body["questions"]) == 1
    question = body["questions"][0]
    assert question["raw_question"] == "退款多久到账"
    assert question["source"] == "self_check"
    assert question["reason"] == "low score"
    assert question["retrieved_chunks"] == snapshot
    assert isinstance(question["created_at"], str)


def test_detail_missing_returns_404(client):
    assert client.get("/api/review/9999").status_code == 404


# ---------------------------------------------------------------------------
# REST: approve / reject
# ---------------------------------------------------------------------------


def test_approve_api_success_marks_approved(client, session, monkeypatch):
    review = _review(session, answer="AI 建议")
    calls = []
    monkeypatch.setattr(
        knowledge_approval,
        "_write_and_vectorize",
        lambda *a, **k: calls.append("write"),
    )

    r = client.post(
        f"/api/review/{review.id}/approve",
        json={"approved_answer": "1-3 个工作日"},
    )

    assert r.status_code == 200
    assert calls == ["write"]
    assert r.json()["review_status"] == "通过"
    assert r.json()["approved_answer"] == "1-3 个工作日"
    session.expire_all()
    stored = session.get(ReviewQueue, review.id)
    assert stored.review_status == "通过"
    assert stored.approved_answer == "1-3 个工作日"


def test_approve_api_vectorize_failure_is_500_and_keeps_pending(session, monkeypatch):
    review = _review(session)

    def fail(*args, **kwargs):
        raise RuntimeError("milvus down")

    monkeypatch.setattr(knowledge_approval, "_write_and_vectorize", fail)

    with TestClient(app, raise_server_exceptions=False) as client:
        r = client.post(
            f"/api/review/{review.id}/approve",
            json={"approved_answer": "答案"},
        )

    assert r.status_code == 500
    session.expire_all()
    assert session.get(ReviewQueue, review.id).review_status == "待审"


def test_approve_non_pending_returns_409(client, session):
    review = _review(session, status="通过")

    r = client.post(
        f"/api/review/{review.id}/approve",
        json={"approved_answer": "答案"},
    )

    assert r.status_code == 409


def test_approve_blank_answer_is_422(client, session):
    review = _review(session)

    r = client.post(
        f"/api/review/{review.id}/approve",
        json={"approved_answer": "   "},
    )

    assert r.status_code == 422


def test_approve_missing_is_404(client):
    r = client.post("/api/review/9999/approve", json={"approved_answer": "答案"})
    assert r.status_code == 404


def test_approve_while_build_running_is_409(client, session, monkeypatch):
    review = _review(session)
    monkeypatch.setattr(kb_build, "is_running", lambda: True)

    r = client.post(
        f"/api/review/{review.id}/approve",
        json={"approved_answer": "答案"},
    )

    assert r.status_code == 409


def test_reject_api_marks_rejected_and_non_pending_is_409(client, session):
    review = _review(session)

    r = client.post(f"/api/review/{review.id}/reject")
    assert r.status_code == 200
    assert r.json()["review_status"] == "驳回"

    session.expire_all()
    assert session.get(ReviewQueue, review.id).review_status == "驳回"

    assert client.post(f"/api/review/{review.id}/reject").status_code == 409


def test_reject_missing_is_404(client):
    assert client.post("/api/review/9999/reject").status_code == 404


# ---------------------------------------------------------------------------
# page route
# ---------------------------------------------------------------------------


def test_review_page_serves_file_with_no_store(client, monkeypatch, tmp_path):
    import app.api.review as review_api

    page = tmp_path / "review.html"
    page.write_text("<html>review queue</html>", encoding="utf-8")
    monkeypatch.setattr(review_api, "PAGE", page)

    r = client.get("/review")

    assert r.status_code == 200
    assert "review queue" in r.text
    assert r.headers["cache-control"] == "no-store"


def test_review_page_missing_is_404(client, monkeypatch, tmp_path):
    import app.api.review as review_api

    monkeypatch.setattr(review_api, "PAGE", tmp_path / "missing.html")

    assert client.get("/review").status_code == 404
