"""Background append-only conversation summaries (chapter 7).

When the runtime layer (L2) exceeds its token budget, :func:`schedule_summary`
submits a job to a small module-level thread pool.  The job re-reads the
conversation anchors, compresses the newly eligible range ``(summary_upto,
layer1_from]`` into one segment and appends it to ``conversation_summaries``.

Segments are append-only: old segments and the existing projection are never
rewritten or re-summarized, so a fact only ever goes through one lossy
compression.  The old projection is passed to the model as background only.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.config import Settings, build_llm
from app.db import session as db_session
from app.models import Message
from app.services.history import (
    append_summary_segment,
    get_conversation,
    list_dialog_messages,
)

_logger = logging.getLogger(__name__)

#: Lower bound of a usable segment, in characters (runtime validation).
SUMMARY_MIN_CHARS = 20
#: Hard ceiling of a segment, in characters (runtime validation).
SUMMARY_MAX_CHARS = 400
#: Default prompt target, in Chinese characters.
SUMMARY_DEFAULT_MAX_CHARS = 200

_SUMMARY_SYSTEM_PROMPT_TEMPLATE = (
    "你是电商客服会话的摘要助手。请把【本段新对话】压缩成一小段事实与诉求摘要。\n"
    "红线：\n"
    "1. 只提炼事实与诉求：问过哪款商品/型号、报过的订单号/手机号、明确的退款/退货/"
    "售后诉求、尚未解决的问题、已确认的处理结论。\n"
    "2. 对话里没出现的内容一个字都不许编造；不确定就不写。\n"
    "3. 寒暄、闲聊、客套、重复确认一律不要保留。\n"
    "4. 长度控制在 50–{max_chars} 个汉字左右，不要超过 {max_chars} 字。\n"
    "5. 只输出 JSON：{{\"summary\": \"<这一段的事实与诉求>\"}}，不要解释，不要 Markdown。\n"
    "6. 【旧摘要】只作背景，帮助理解指代；不要把旧摘要的内容重新概括进新段，"
    "不要合并、改写旧摘要。\n"
    "7. 如果整段对话只有问候、寒暄、客套，没有任何事实或明确诉求，summary 留空"
    "（空字符串），不要为了凑长度列举业务类别。\n"
    "8. 同一事实被用户更正后（订单号、手机号等），只保留最新值，不重复被更正的旧值。\n"
)

#: Default system prompt (200-character target); kept importable for eval jobs.
SUMMARY_SYSTEM_PROMPT = _SUMMARY_SYSTEM_PROMPT_TEMPLATE.format(
    max_chars=SUMMARY_DEFAULT_MAX_CHARS
)


class SegmentSummary(BaseModel):
    """Structured-output schema for one summary segment."""

    summary: str


@dataclass
class SummaryResult:
    """Outcome of one :func:`run_summary` attempt."""

    status: str
    seq: int | None = None
    from_msg_id: int | None = None
    upto_msg_id: int | None = None
    chars: int = 0
    elapsed_ms: int = 0
    reason: str = ""


_executor: ThreadPoolExecutor | None = None
_inflight: set[int] = set()
_lock = threading.Lock()

#: ``[订单选择] 订单 1001`` audit rows are rendered as a normal user statement.
_ORDER_AUDIT_RE = re.compile(r"^\[订单选择\]\s*订单\s*(\d+)\s*$")


def _get_executor() -> ThreadPoolExecutor:
    """Return the module executor, creating it lazily.

    Recreated after :func:`shutdown_summary_executor` (which sets it back to
    ``None``).  The caller must hold ``_lock``.
    """
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="ch07-summary"
        )
    return _executor


def _clear_inflight(conversation_id: int) -> None:
    """Drop the in-flight marker for one conversation (idempotent)."""
    with _lock:
        _inflight.discard(conversation_id)


def _run_and_clear(conversation_id: int, settings: Settings) -> None:
    """Run one job and always drop the conversation's in-flight marker."""
    try:
        run_summary(conversation_id, settings)
    except Exception:  # noqa: BLE001 - background job must never crash silently
        _logger.exception("summary failed conv=%d", conversation_id)
    finally:
        _clear_inflight(conversation_id)


def schedule_summary(conversation_id: int, settings: Settings) -> bool:
    """Submit a summary job for *conversation_id* without blocking.

    Returns ``True`` when a new job was submitted and ``False`` when one is
    already in flight for the same conversation.
    """
    with _lock:
        if conversation_id in _inflight:
            _logger.info("summary skip reason=inflight conv=%d", conversation_id)
            return False
        _inflight.add(conversation_id)
        try:
            future = _get_executor().submit(
                _run_and_clear, conversation_id, settings
            )
        except RuntimeError:
            # The executor was shut down concurrently; do not leak the marker.
            _inflight.discard(conversation_id)
            raise

    # Registered outside ``_lock`` so an already-done/cancelled future cannot
    # invoke the callback (and re-acquire the lock) on this thread.  This also
    # clears markers for queued jobs cancelled by ``shutdown`` without ever
    # starting; ``_run_and_clear``'s ``finally`` stays as a second, idempotent
    # cleanup path for jobs that do run.
    future.add_done_callback(
        lambda _future: _clear_inflight(conversation_id)
    )
    return True


def shutdown_summary_executor() -> None:
    """Shut the module executor down; safe to call repeatedly."""
    global _executor
    with _lock:
        executor, _executor = _executor, None
    if executor is not None:
        executor.shutdown(wait=False, cancel_futures=True)


def build_summary_prompt(
    prior_summary: str,
    dialogue: str,
    max_chars: int = SUMMARY_DEFAULT_MAX_CHARS,
) -> list[BaseMessage]:
    """Build the two-message prompt for one segment summary.

    *max_chars* is the configured segment target; the model only sees the
    ``50–{max_chars}`` guidance, never the runtime 20/400 validation bounds.
    """
    return [
        SystemMessage(
            content=_SUMMARY_SYSTEM_PROMPT_TEMPLATE.format(max_chars=max_chars)
        ),
        HumanMessage(
            content=(
                f"【旧摘要，仅作背景，不要合并】\n{prior_summary}\n\n"
                f"【本段新对话】\n{dialogue}"
            )
        ),
    ]


def _render_row(row: Message) -> str:
    """Render one dialog row as a ``用户：``/``客服：`` line.

    ``[订单选择]`` rows are UI audit records, not user speech; they are
    rendered as a system fact so the model does not attribute them to the user.
    """
    content = row.content or ""
    if row.role == "user":
        match = _ORDER_AUDIT_RE.match(content)
        if match:
            return f"系统事实：用户选择了订单 {match.group(1)}"
        return f"用户：{content}"
    return f"客服：{content}"


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _skip(reason: str, started: float, *, from_id=None, upto_id=None) -> SummaryResult:
    _logger.info(
        "summary skip reason=%s from=%s upto=%s", reason, from_id, upto_id
    )
    return SummaryResult(
        status="skip",
        from_msg_id=from_id,
        upto_msg_id=upto_id,
        elapsed_ms=_elapsed_ms(started),
        reason=reason,
    )


def run_summary(
    conversation_id: int,
    settings: Settings,
    *,
    session: Session | None = None,
    llm=None,
) -> SummaryResult:
    """Summarize ``(summary_upto, layer1_from]`` into one appended segment.

    Opens (and closes) its own DB session when *session* is ``None`` so the
    background worker is independent of the request/thread that triggered it.
    """
    started = time.perf_counter()
    own_session = session is None
    if own_session:
        session = db_session.get_session_factory()()

    try:
        conversation = get_conversation(session, conversation_id)
        if conversation is None:
            return _skip("missing_conversation", started)

        old_upto = conversation.summary_upto_msg_id
        layer1_from = conversation.layer1_from_msg_id
        if layer1_from is None:
            return _skip("no_range", started)

        rows = [
            row
            for row in list_dialog_messages(session, conversation_id)
            if row.id <= layer1_from
            and (old_upto is None or row.id > old_upto)
        ]
        if not rows:
            return _skip("no_range", started)

        from_id = rows[0].id
        upto_id = rows[-1].id
        if old_upto is not None and upto_id <= old_upto:
            return _skip(
                "boundary_moved", started, from_id=from_id, upto_id=upto_id
            )

        effective_cap = max(
            50, min(SUMMARY_MAX_CHARS, settings.summary_max_chars)
        )
        dialogue = "\n".join(_render_row(row) for row in rows)
        prompt = build_summary_prompt(
            conversation.summary or "", dialogue, effective_cap
        )

        if llm is None:
            llm = build_llm(
                settings,
                temperature=0.0,
                model=settings.summary_model or None,
            )

        _logger.info(
            "summary start conv=%d from=%d upto=%d", conversation_id, from_id, upto_id
        )
        try:
            response = llm.with_structured_output(
                SegmentSummary, method="json_mode"
            ).invoke(prompt)
        except OutputParserException:
            # The model answered but the payload was not valid JSON/shape.
            _logger.exception(
                "summary invalid_json conv=%d from=%d upto=%d",
                conversation_id,
                from_id,
                upto_id,
            )
            session.rollback()
            return _skip(
                "invalid_json", started, from_id=from_id, upto_id=upto_id
            )
        except Exception:
            _logger.exception(
                "summary failed conv=%d from=%d upto=%d",
                conversation_id,
                from_id,
                upto_id,
            )
            session.rollback()
            return SummaryResult(
                status="error",
                from_msg_id=from_id,
                upto_msg_id=upto_id,
                elapsed_ms=_elapsed_ms(started),
                reason="exception",
            )

        summary = getattr(response, "summary", None)
        if not isinstance(summary, str) or not summary.strip():
            return _skip("too_short", started, from_id=from_id, upto_id=upto_id)
        summary = summary.strip()
        if len(summary) < SUMMARY_MIN_CHARS:
            return _skip("too_short", started, from_id=from_id, upto_id=upto_id)
        if len(summary) > effective_cap:
            return _skip("too_long", started, from_id=from_id, upto_id=upto_id)

        segment = append_summary_segment(
            session, conversation_id, from_id, upto_id, summary
        )
        elapsed_ms = _elapsed_ms(started)
        _logger.info(
            "summary done conv=%d seq=%d from=%d upto=%d chars=%d elapsed_ms=%d",
            conversation_id,
            segment.seq,
            from_id,
            upto_id,
            len(summary),
            elapsed_ms,
        )
        return SummaryResult(
            status="done",
            seq=segment.seq,
            from_msg_id=from_id,
            upto_msg_id=upto_id,
            chars=len(summary),
            elapsed_ms=elapsed_ms,
        )
    finally:
        if own_session:
            session.close()
