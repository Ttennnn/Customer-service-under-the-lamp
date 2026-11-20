"""审核队列的人工审核与知识库写回(Task 7)。

审核通过时,先把同一标准化问题下**过期的** ``mined_qa`` chunk 清掉,再把
核准答案作为一条新的 ``mined_qa`` chunk 写入 ``knowledge_chunks``
(``content_type='mined_qa'``、``category='通用'``),然后**只向量化这一条**。
只有向量化成功后才把 ``ReviewQueue.review_status`` 置为「通过」并写回
``approved_answer``。

清掉旧行是审核写回正确性的关键:

- 有一次审批失败后,旧答案的 chunk 会以 ``pending`` 留在库里;若这次用新答案
  重试,旧 chunk 必须删除,否则旧的全局向量化逻辑会在后续某次运行时把旧答案
  也发布出去。
- 若上次在「向量化成功、状态未写回」之间崩溃,旧 chunk 可能是 ``done``;删除
  前先删对应的 Milvus 向量,避免孤儿向量,然后删 MySQL 行。

向量化只作用于本次审批的 chunk(``vectorize_rows``),不会顺带发布其他待审
chunk。失败时 best-effort 删掉本次新 chunk 的向量和行,状态保持「待审」,
原异常照常抛出;调用方可以安全重试。

``insert_mined_qa`` / ``vectorize_rows`` 各自提交事务,因此本模块在写回
期间持有一把进程内模块锁,并拒绝在知识库构建进行时写回(single-process
假设,与 ch09 其他批处理一致)。``embedder`` / ``store`` 可从外部注入;默认
在真正需要时才从进程级工厂取,模块导入不会连接 Milvus 或嵌入服务。
"""

from __future__ import annotations

import logging
import threading

from app.models import KnowledgeChunk, ReviewQueue
from app.rag.embedding import get_embedder
from app.rag.ingest import MINED_CONTENT_TYPE, MinedQA, insert_mined_qa
from app.rag.milvus_store import get_vector_store
from app.rag.vectorize import vectorize_rows
from app.services import kb_build

logger = logging.getLogger(__name__)

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


def _stale_chunks(session, normalized_question: str) -> list[KnowledgeChunk]:
    """Existing ``mined_qa`` chunks for the same standardized question.

    Every matching row predates the chunk this approval is about to insert, so
    they are all stale: a prior failed answer (``pending``) or a previously
    vectorized-but-never-approved answer (``done``).
    """
    return (
        session.query(KnowledgeChunk)
        .filter(
            KnowledgeChunk.content_type == MINED_CONTENT_TYPE,
            KnowledgeChunk.questions == normalized_question,
        )
        .order_by(KnowledgeChunk.id)
        .all()
    )


def _remove_stale_chunks(session, normalized_question: str, store) -> int:
    """Delete same-question mined chunks (vector first) and commit.

    Milvus vectors are removed before the MySQL rows so a failed vector
    deletion never leaves an orphan behind.  Returns how many rows were
    removed.
    """
    rows = _stale_chunks(session, normalized_question)
    if not rows:
        return 0
    if store is None and any(row.vector_id for row in rows):
        store = get_vector_store()
    vector_ids = [row.vector_id for row in rows if row.vector_id]
    if vector_ids:
        store.delete_ids(vector_ids)
    for row in rows:
        session.delete(row)
    session.commit()
    return len(rows)


def _cleanup_new_chunks(session, chunk_ids: list[int], store) -> None:
    """Best-effort removal of the chunks written by a failed writeback."""
    try:
        session.rollback()
        rows = (
            session.query(KnowledgeChunk)
            .filter(KnowledgeChunk.id.in_(chunk_ids))
            .all()
            if chunk_ids
            else []
        )
        vector_ids = [row.vector_id for row in rows if row.vector_id]
        if vector_ids:
            store.delete_ids(vector_ids)
        for row in rows:
            session.delete(row)
        session.commit()
    except Exception:  # noqa: BLE001 - cleanup must not mask the real error
        session.rollback()
        logger.warning(
            "failed to clean up approved chunk(s) %s after vectorize error",
            chunk_ids,
            exc_info=True,
        )


def _write_and_vectorize(session, review, approved_answer, embedder, store):
    """Replace stale chunks, write the approved QA chunk, vectorize only it."""
    if embedder is None:
        embedder = get_embedder()
    if store is None:
        store = get_vector_store()

    _remove_stale_chunks(session, review.normalized_question, store)
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
    chunk_ids = [chunk.id for chunk in chunks]
    try:
        vectorize_rows(session, chunks, embedder, store)
    except Exception:
        _cleanup_new_chunks(session, chunk_ids, store)
        raise
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
            # Keep the row 待审.  The chunk this attempt created was already
            # removed by _write_and_vectorize; any older same-question chunk
            # is gone too, so a retry starts from a clean slate.
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
