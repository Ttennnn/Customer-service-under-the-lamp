"""Fabrication ledger writes (``faith_cases``).

One row per evaluation case, keyed by ``eval_id``. The evaluation job runs the
same case under several retrieval strategies; re-running a case updates the
existing row (``seen_count += 1``, ``last_seen_at`` refreshed, status reopened)
instead of inserting one row per strategy, so the ledger stays a per-question
view of recurring fabrications. ``resolved_at`` is intentionally preserved when
a case recurs, so the original resolution time survives for recurrence
analysis.
"""

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select

from app.models import FaithCase


@dataclass(frozen=True)
class FaithCasePayload:
    """Everything needed to insert or refresh one ``faith_cases`` row."""

    eval_id: str
    bucket: str
    query: str
    strategy: str
    answer: str
    reason: str
    citations: list[dict]
    judge_model: str


def upsert_faith_case(session, payload: FaithCasePayload) -> FaithCase:
    """Insert a new case or refresh the existing row for ``payload.eval_id``.

    New rows start ``未解决`` with ``seen_count=1`` and identical
    ``first_seen_at``/``last_seen_at``. Existing rows refresh the strategy,
    answer, reason, citations and judge model, increment ``seen_count``, bump
    ``last_seen_at`` and reopen to ``未解决`` while keeping ``resolved_at``.
    The caller owns the transaction and must commit; this function flushes so
    the returned row has an id.
    """
    now = datetime.now()
    row = session.execute(
        select(FaithCase).where(FaithCase.eval_id == payload.eval_id)
    ).scalar_one_or_none()

    if row is None:
        row = FaithCase(
            eval_id=payload.eval_id,
            bucket=payload.bucket,
            query=payload.query,
            strategy=payload.strategy,
            answer=payload.answer,
            reason=payload.reason,
            citations=payload.citations,
            judge_model=payload.judge_model,
            status="未解决",
            seen_count=1,
            first_seen_at=now,
            last_seen_at=now,
        )
        session.add(row)
    else:
        row.strategy = payload.strategy
        row.answer = payload.answer
        row.reason = payload.reason
        row.citations = payload.citations
        row.judge_model = payload.judge_model
        row.seen_count = (row.seen_count or 0) + 1
        row.last_seen_at = now
        row.status = "未解决"

    session.flush()
    return row
