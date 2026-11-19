"""审核队列的人工审核与知识库写回(Task 7)。

审核通过时,先把核准答案作为一条 ``mined_qa`` chunk 写入 ``knowledge_chunks``
(``content_type='mined_qa'``、``category='通用'``),再同步向量化;只有向量化
成功后才把 ``ReviewQueue.review_status`` 置为「通过」并写回 ``approved_answer``。
向量化失败时状态保持「待审」,已经落库的 chunk 保持 ``pending``,重试会因
指纹去重复用同一行并继续向量化,不会重复建行。

``insert_mined_qa`` / ``vectorize_pending`` 各自提交事务,因此本模块在写回
期间持有一把进程内模块锁,并拒绝在知识库构建进行时写回(single-process
假设,与 ch09 其他批处理一致)。``embedder`` / ``store`` 可从外部注入;默认
在真正需要时才从进程级工厂取,模块导入不会连接 Milvus 或嵌入服务。
"""

from __future__ import annotations

import threading

from app.models import ReviewQueue
from app.rag.embedding import get_embedder
from app.rag.ingest import MinedQA, insert_mined_qa
from app.rag.milvus_store import get_vector_store
from app.rag.vectorize import vectorize_pending
from app.services import kb_build

CATEGORY = "通用"

# One in-process writeback at a time.  The underlying helpers commit on their
# own, so a second writer cannot be given a larger transaction to roll back.
_lock = threading.Lock()


class ReviewNotFound(LookupError):
    """The requested review-queue row does not exist (HTTP 404)."""


class ReviewNotPending(RuntimeError):
    """Only 待审 rows may be approved or rejected (HTTP 409)."""


class InvalidApprovedAnswer(ValueError):
    """The approved answer is blank (HTTP 422)."""


class ApprovalBusy(RuntimeError):
    """A writeback or knowledge build is already in progress (HTTP 409)."""


def _write_and_vectorize(session, review, approved_answer, embedder, store):
    """Write the QA chunk, then vectorize it; return the reused/new chunk."""
    chunks = insert_mined_qa(
        session,
        [
            MinedQA(
                question=review.normalized_question,
                answer=approved_answer,
                category=CATEGORY,
            )
        ],
    )
    if embedder is None:
        embedder = get_embedder()
    if store is None:
        store = get_vector_store()
    vectorize_pending(session, embedder, store)
    return chunks[0]


def _acquire_lock() -> None:
    if not _lock.acquire(blocking=False):
        raise ApprovalBusy("another review writeback is in progress")
    try:
        if kb_build.is_running():
            raise ApprovalBusy("a knowledge base build is running")
    except Exception:
        _lock.release()
        raise


def _get_pending(session, review_id: int) -> ReviewQueue:
    review = session.get(ReviewQueue, review_id)
    if review is None:
        raise ReviewNotFound(f"review_queue id {review_id} not found")
    if review.review_status != "待审":
        raise ReviewNotPending(
            f"review_queue id {review_id} is {review.review_status}, not 待审"
        )
    return review


def approve_review(
    session,
    review_id: int,
    approved_answer: str,
    *,
    embedder=None,
    store=None,
    settings=None,
) -> ReviewQueue:
    """Approve one 待审 row after its answer is written and vectorized.

    ``settings`` is accepted for parity with other ch09 services; the default
    embedder/store factories read the process-wide settings themselves.
    """
    answer = (approved_answer or "").strip()
    if not answer:
        raise InvalidApprovedAnswer("approved_answer must not be blank")

    _acquire_lock()
    try:
        review = _get_pending(session, review_id)
        try:
            _write_and_vectorize(session, review, answer, embedder, store)
        except Exception:
            # Keep the row 待审; a pending extracted chunk is safe to retry
            # because insert_mined_qa reuses it by fingerprint.
            session.rollback()
            raise
        review.review_status = "通过"
        review.approved_answer = answer
        session.commit()
        return review
    finally:
        _lock.release()


def reject_review(session, review_id: int) -> ReviewQueue:
    """Mark one 待审 row as 驳回; no knowledge-base write happens."""
    _acquire_lock()
    try:
        review = _get_pending(session, review_id)
        review.review_status = "驳回"
        session.commit()
        return review
    finally:
        _lock.release()
