"""飞轮批处理:标准化 + 查重 + 待审(Task 6)。

``process_pending`` 串行扫描尚未归并的低置信问题,每条都重新读取候选
缺口(因此同一批里刚新建的行能被后续条目命中),然后:

- LLM 返回有效候选 id → ``occurrence_count += 1``,审核状态保持不变;
- LLM 返回 null → 新建一条「待审」缺口,occurrence_count=1;
- 越界/编造 id、坏 JSON、模型异常 → 跳过该条,``matched_review_id`` 保持
  NULL,``failed += 1``;下一次运行会重试。

只有全部条目处理完且没有失败时,任务退出码才是 0。
"""

import logging
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, field_validator

from app.config import get_settings
from app.models import LowConfidenceQuestion, ReviewQueue
from app.prompts.flywheel import FLYWHEEL_NORMALIZE_PROMPT, render_candidates
from app.services.review_repo import (
    FlywheelError,
    increment_occurrence,
    insert_review_item,
    list_review_candidates,
    list_unmatched_low_confidence,
    set_matched_review,
)

logger = logging.getLogger(__name__)

__all__ = [
    "FlywheelError",
    "FlywheelOutput",
    "FlywheelReport",
    "fetch_unmatched_low_conf",
    "list_review_candidates",
    "normalize_and_match",
    "process_pending",
]

DEFAULT_CANDIDATE_LIMIT = 200


class FlywheelOutput(BaseModel):
    """One LLM normalization/dedup decision.

    Positional construction is supported (``FlywheelOutput(q, id, answer)``)
    because the job/tests treat it as a small value object; pydantic itself
    still handles validation and structured-output schema generation.
    """

    normalized_question: str
    matched_question_id: int | None = None
    ai_suggested_answer: str = ""

    @field_validator("matched_question_id", mode="before")
    @classmethod
    def _reject_boolean_id(cls, value):
        """Reject ``True``/``False`` instead of coercing them to 1/0.

        pydantic accepts ``bool`` for ``int`` because ``bool`` subclasses
        ``int``; a structured LLM emitting ``true`` must be treated as bad
        output, not as a match against candidate id 1.
        """
        if isinstance(value, bool):
            raise ValueError(
                "matched_question_id must be an integer or null, not a boolean"
            )
        return value

    def __init__(
        self,
        normalized_question: str | None = None,
        matched_question_id: int | None = None,
        ai_suggested_answer: str = "",
        **data,
    ) -> None:
        if normalized_question is not None:
            data["normalized_question"] = normalized_question
        data.setdefault("matched_question_id", matched_question_id)
        data.setdefault("ai_suggested_answer", ai_suggested_answer)
        super().__init__(**data)


@dataclass
class FlywheelReport:
    scanned: int = 0
    created: int = 0
    merged: int = 0
    failed: int = 0


def _resolve_candidate_limit() -> int:
    """Candidate cap from settings, falling back to 200 if config is absent."""
    try:
        return int(get_settings().flywheel_candidate_limit)
    except Exception:
        return DEFAULT_CANDIDATE_LIMIT


def fetch_unmatched_low_conf(
    session, limit: int
) -> list[LowConfidenceQuestion]:
    """Return low-confidence questions still missing a review-queue link."""
    return list_unmatched_low_confidence(session, limit)


def _coerce_output(result) -> FlywheelOutput:
    """Accept a pydantic model or a plain dict from the structured LLM."""
    if isinstance(result, dict):
        data = result
    else:
        data = {}
        for field in (
            "normalized_question",
            "matched_question_id",
            "ai_suggested_answer",
        ):
            if hasattr(result, field):
                data[field] = getattr(result, field)
    normalized = data.get("normalized_question")
    normalized = normalized.strip() if isinstance(normalized, str) else ""
    matched = data.get("matched_question_id")
    # ``bool`` is an ``int`` subclass; ``True`` must not silently become id 1.
    if isinstance(matched, bool):
        raise FlywheelError(
            f"flywheel output has a boolean matched_question_id: {matched!r}"
        )
    answer = data.get("ai_suggested_answer")
    answer = answer.strip() if isinstance(answer, str) else ""
    return FlywheelOutput(
        normalized_question=normalized,
        matched_question_id=matched,
        ai_suggested_answer=answer,
    )


def normalize_and_match(
    llm,
    raw_question: str,
    candidates: list[ReviewQueue],
) -> FlywheelOutput:
    """Run one structured LLM call and validate the returned match id.

    Raises :class:`FlywheelError` for a model/JSON failure, a blank
    standardized question, or a ``matched_question_id`` that is not in the
    candidate list (hallucinated/out-of-range id).
    """
    candidate_ids = {candidate.id for candidate in candidates}
    human = HumanMessage(
        content=(
            f"用户原话：{raw_question}\n\n"
            f"候选缺口列表（共 {len(candidates)} 条）：\n"
            f"{render_candidates(candidates)}"
        )
    )
    try:
        structured = llm.with_structured_output(FlywheelOutput, method="json_mode")
        result = structured.invoke(
            [SystemMessage(content=FLYWHEEL_NORMALIZE_PROMPT), human]
        )
    except Exception as exc:
        raise FlywheelError(f"flywheel normalization failed: {exc}") from exc

    try:
        output = _coerce_output(result)
    except Exception as exc:
        raise FlywheelError(f"flywheel output failed validation: {exc}") from exc
    if not output.normalized_question:
        raise FlywheelError("flywheel output has a blank normalized_question")

    matched = output.matched_question_id
    if matched is not None:
        try:
            matched = int(matched)
        except (TypeError, ValueError) as exc:
            raise FlywheelError(
                f"flywheel output has non-integer matched_question_id: {matched!r}"
            ) from exc
        if matched not in candidate_ids:
            raise FlywheelError(
                f"flywheel output matched unknown candidate id {matched}"
            )

    return FlywheelOutput(
        normalized_question=output.normalized_question,
        matched_question_id=matched,
        ai_suggested_answer=output.ai_suggested_answer,
    )


def process_pending(
    session,
    llm,
    limit: int = 50,
    candidate_limit: int | None = None,
) -> FlywheelReport:
    """Normalize/dedup unmatched low-confidence rows, one at a time.

    The caller's session is used for the whole batch; each item is committed
    independently so a failure only rolls back that item. Candidate rows are
    re-read before every item, which lets a row created earlier in the same
    batch match a later question.
    """
    candidate_limit = (
        _resolve_candidate_limit() if candidate_limit is None else candidate_limit
    )
    pending = fetch_unmatched_low_conf(session, limit)
    report = FlywheelReport(scanned=len(pending))

    for row in pending:
        try:
            candidates = list_review_candidates(session, candidate_limit)
            if candidate_limit > 0 and len(candidates) == candidate_limit:
                logger.info(
                    "flywheel candidate list truncated at limit=%s; newest "
                    "rows only were considered",
                    candidate_limit,
                )
            output = normalize_and_match(llm, row.raw_question, candidates)

            if output.matched_question_id is not None:
                review = increment_occurrence(session, output.matched_question_id)
                report.merged += 1
            else:
                review = insert_review_item(session, output)
                report.created += 1

            set_matched_review(session, row.id, review.id)
            session.commit()
        except Exception as exc:  # noqa: BLE001 - one bad item must not stop the batch
            session.rollback()
            report.failed += 1
            logger.warning(
                "flywheel item failed (lcq_id=%s): %s: %s",
                getattr(row, "id", None),
                type(exc).__name__,
                exc,
            )

    return report
