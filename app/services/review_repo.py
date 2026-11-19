"""待审队列与低置信问题池的读写(Task 6 飞轮批处理)。

本模块只做查询与写入,不提交事务:调用方(``process_pending`` 或审核
API)拥有 transaction,决定何时 commit/rollback。``increment_occurrence``
与 ``set_matched_review`` 只 flush,便于同一事务内完成「命中/新建 → 回写」
的原子操作。
"""

from app.models import LowConfidenceQuestion, ReviewQueue

NORMALIZED_QUESTION_MAX_CHARS = 512


class FlywheelError(RuntimeError):
    """Raised when a flywheel row is missing or LLM output is invalid."""


def list_unmatched_low_confidence(
    session, limit: int
) -> list[LowConfidenceQuestion]:
    """Low-confidence questions not yet merged into a review-queue gap."""
    limit = max(0, int(limit))
    if limit == 0:
        return []
    return (
        session.query(LowConfidenceQuestion)
        .filter(LowConfidenceQuestion.matched_review_id.is_(None))
        .order_by(
            LowConfidenceQuestion.created_at.asc(),
            LowConfidenceQuestion.id.asc(),
        )
        .limit(limit)
        .all()
    )


def list_review_candidates(session, limit: int = 200) -> list[ReviewQueue]:
    """All review-queue rows (any status), newest-updated first."""
    limit = max(0, int(limit))
    if limit == 0:
        return []
    return (
        session.query(ReviewQueue)
        .order_by(ReviewQueue.updated_at.desc(), ReviewQueue.id.desc())
        .limit(limit)
        .all()
    )


def insert_review_item(session, output) -> ReviewQueue:
    """Insert a new 待审 gap; truncate the standardized question to 512."""
    normalized = (getattr(output, "normalized_question", "") or "").strip()
    answer = getattr(output, "ai_suggested_answer", None)
    if isinstance(answer, str):
        answer = answer.strip() or None
    row = ReviewQueue(
        normalized_question=normalized[:NORMALIZED_QUESTION_MAX_CHARS],
        ai_suggested_answer=answer,
        occurrence_count=1,
        review_status="待审",
    )
    session.add(row)
    session.flush()
    return row


def increment_occurrence(session, review_id: int) -> ReviewQueue:
    """Bump ``occurrence_count`` and keep the existing review status."""
    row = session.get(ReviewQueue, review_id)
    if row is None:
        raise FlywheelError(f"review_queue id {review_id} not found")
    row.occurrence_count = (row.occurrence_count or 0) + 1
    session.flush()
    return row


def set_matched_review(session, lcq_id: int, review_id: int) -> None:
    """Point one low-confidence question at the gap it was merged into."""
    row = session.get(LowConfidenceQuestion, lcq_id)
    if row is None:
        raise FlywheelError(f"low_confidence_questions id {lcq_id} not found")
    row.matched_review_id = review_id
    session.flush()
