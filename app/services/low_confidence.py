"""Low-confidence question pool writes.

Only genuine knowledge gaps belong here. Callers must not record
infrastructure failures (timeouts, Milvus/embedding/self-check outages) as
low-confidence questions; those are handled as conservative refusals without
polluting the pool.
"""

from app.models import LowConfidenceQuestion


def record_low_confidence(
    session,
    conversation_id: int | None,
    raw_question: str,
    reason: str | None,
    source: str = "self_check",
) -> LowConfidenceQuestion:
    """Insert one low-confidence question and flush it.

    The caller owns the transaction and must commit. ``flush`` assigns the row
    id so callers can log or link it before committing.
    """
    row = LowConfidenceQuestion(
        conversation_id=conversation_id,
        raw_question=raw_question,
        reason=reason,
        source=source,
    )
    session.add(row)
    session.flush()
    return row
