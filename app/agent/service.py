"""Conversation-aware chat service: SSE mapping, DB audit, low-confidence pool.

``stream_chat_agent`` drives one turn of the parent workflow graph:

1. Resolve the conversation first (explicit id, forced new, or latest
   in-progress) and pin the checkpointer thread to ``conv:{id}``.
2. Read the thread snapshot once. It supplies both the pending-interrupt check
   and the prior checkpoint messages used to prepare this turn's context. A
   pending order picker re-emits the ``conversation`` and ``interrupt`` frames
   and ends the turn without writing a user message or running the graph; a
   refresh/new message therefore never wedges the thread.
3. Persist the user turn *before* starting the graph, so the audit trail
   survives a graph failure, then prepare the layered context plan with its
   ``db-{id}`` message identity and inject the plan into the turn inputs.
4. Emit the ``conversation`` frame, then consume only
   ``stream_mode="custom"`` frames. On langgraph 1.2.20 with
   ``subgraphs=True`` each item is ``{"type": "custom", "ns": ...,
   "data": <node frame>}`` — read ``item["data"]``, not ``item[1]``.
5. Forward ``token`` / ``tool`` / ``sources`` frames as raw SSE payload
   strings: plain text for tokens, compact JSON for the rest. Unknown frames
   are logged and dropped.
6. Read the final State after the stream.  If the graph paused on an
   ``interrupt``, emit the ``interrupt`` frame and stop: no options, no
   assistant answer.  Otherwise emit ``options`` when the graph suggested
   actions, persist the assistant answer, and record a low-confidence
   question only for genuine knowledge gaps.  Infrastructure and rerank
   failures never enter the pool.

``resume_chat_agent`` continues the resolved conversation's paused thread with
``Command(resume=...)``. The resume value is *not* sent as a ``HumanMessage``:
``fetch_order`` already appends the ``【订单选择器】`` ``HumanMessage`` on
resume and the ticket confirmation is a control decision, not user text.
MySQL still gets an audit-only user row (``[订单选择] 订单 <id>`` or
``[工单确认] 确认提交`` / ``[工单确认] 取消``) and ``prepare_context`` runs for
logging/summary triggering only, because the paused turn's plan fields already
live in the checkpoint.

Graph exceptions are converted into a single ``{"error": ...}`` frame; the
generator never re-raises.
"""

import dataclasses
import json
import logging
from collections.abc import Iterator

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from sqlalchemy.orm import Session

from app.config import Settings
from app.observability import _state_graph, tracing_enabled
from app.services.history import (
    ConversationNotFound,
    append_message,
    resolve_conversation,
)
from app.services.low_confidence import record_low_confidence
from app.services.session_context import ContextPlan, prepare_context

logger = logging.getLogger(__name__)

ERROR_TEXT = "客服服务暂时不可用,请稍后重试"
CONVERSATION_NOT_FOUND_TEXT = "会话不存在"
_GATE_GAP_REASONS = frozenset({"empty", "below_threshold"})
_FORWARDED_FRAME_TYPES = frozenset({"token", "tool", "sources"})


class NoPendingInterrupt(Exception):
    """Raised when a resume is requested but the thread is not paused."""


def _dumps(payload: dict) -> str:
    """Serialize an SSE status frame exactly like the legacy chat service."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _thread_config(
    conversation_id: int,
    *,
    user_id: str | None = None,
    settings: Settings | None = None,
) -> dict:
    """Config pinning the checkpointer thread to one conversation.

    When both ``user_id`` and ``settings`` are supplied *and* tracing is
    enabled, the Langfuse LangChain metadata keys are added so the trace is
    grouped by conversation session and attributed to the user.  Callers that
    only need to read a snapshot (``has_pending_interrupt``) keep the legacy
    two-argument path.
    """
    config: dict = {"configurable": {"thread_id": f"conv:{conversation_id}"}}
    if (
        user_id is not None
        and settings is not None
        and tracing_enabled(settings)
    ):
        config["metadata"] = {
            "langfuse_session_id": f"conv:{conversation_id}",
            "langfuse_user_id": user_id,
        }
    return config


def _conversation_frame(conversation_id: int) -> str:
    """Serialize the frame that tells the frontend which session is active."""
    return _dumps({"type": "conversation", "conversation_id": conversation_id})


def _snapshot_messages(snapshot) -> list:
    """Return the checkpoint's message list without touching it."""
    values = getattr(snapshot, "values", None) or {}
    messages = values.get("messages") if isinstance(values, dict) else None
    return list(messages or [])


def _last_human_index(messages) -> int:
    """Index of the last user message, or 0 when the turn has none yet."""
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return index
    return 0


def _truncate_user_input(message: str, settings: Settings) -> str:
    """Cap the model-facing user text at ``max_user_input_tokens``.

    The DB audit row always keeps the full original message; only the graph
    input (and the ``user_message`` state field derived from it) is capped, so
    a long paste cannot push the prompt past the model window. Returns the
    message unchanged when it fits.
    """
    limit = max(0, settings.max_user_input_tokens)
    if len(message) <= limit:
        return message
    logger.warning("用户输入超限已截断 chars=%d limit=%d", len(message), limit)
    return message[:limit]


def _reset_inputs(
    message: str,
    message_id: int,
    conversation_id: int,
    session_id: str,
    settings: Settings,
    plan: ContextPlan,
) -> dict:
    """Build one turn's inputs with every transient field reset explicitly.

    ``rerank_failed`` in particular must be reset: it is a bool rather than a
    reducer-backed channel, so a stale ``True`` would leak into the next turn.
    The ch06 slots (``order_id`` / ``order_data``) and routing fields are reset
    for the same reason: a new turn must re-derive them from its own message.
    ``created_ticket_no`` is reset too, otherwise a ticket number from an
    earlier turn would survive a later cancel/unrelated turn (Task 10 reads it).

    The user message carries the MySQL ``db-{id}`` identity so the layered
    window and the summary job can anchor on persisted rows. The context plan
    fields are injected as this turn's inputs (never into a resumed Command).
    """
    return {
        "messages": [HumanMessage(content=message, id=f"db-{message_id}")],
        "user_message": message,
        "resolved_query": "",
        "intent": "",
        "intent_confidence": None,
        "route": "",
        "session_id": session_id,
        "order_id": "",
        "order_data": None,
        "expanded_queries": [],
        "evidence": [],
        "sources": [],
        "confidence": None,
        "gate_passed": False,
        "gate_reason": "",
        "rerank_failed": False,
        "answer": "",
        "created_ticket_no": "",
        "suggested_actions": [],
        "step_count": 0,
        "token_estimate": 0,
        "exhausted": False,
        "conversation_id": conversation_id,
        "token_budget": settings.agent_token_budget,
        "max_steps": settings.agent_max_steps,
        "summary": plan.summary,
        "summary_upto_msg_id": plan.summary_upto_msg_id,
        "layer1_from_msg_id": plan.layer1_from_msg_id,
        "prompt_history": plan.prompt_history,
        "history_l2": plan.history_l2,
        "history_l1": plan.history_l1,
        "turn_start_index": plan.turn_start_index,
        "l2_tokens": plan.l2_tokens,
        "l1_tokens": plan.l1_tokens,
        "context_budget": dataclasses.asdict(plan.budget),
    }


def _interrupt_payload(snapshot) -> dict | None:
    """Return the first pending interrupt's payload, or ``None`` if none.

    A non-dict payload still counts as pending (callers must stop), but there
    is nothing to merge into the frame, so an empty mapping is returned.
    """
    interrupts = getattr(snapshot, "interrupts", None) or ()
    if not interrupts:
        return None
    value = getattr(interrupts[0], "value", None)
    if isinstance(value, dict):
        return value
    logger.warning("ch06 ignoring non-dict interrupt payload: %r", value)
    return {}


def _pending_interrupt(graph, config: dict) -> dict | None:
    """Read the thread snapshot and return its pending interrupt payload."""
    return _interrupt_payload(_state_graph(graph).get_state(config))


def has_pending_interrupt(graph, conversation_id: int) -> bool:
    """Whether the conversation's thread is paused on an ``interrupt``."""
    return _pending_interrupt(graph, _thread_config(conversation_id)) is not None


def _interrupt_frame(payload: dict) -> str:
    """Serialize the order-picker frame the frontend renders as cards."""
    return _dumps({"type": "interrupt", **payload})


def _finish_turn(
    session: Session,
    conversation_id: int,
    state_values: dict,
    question: str,
) -> Iterator[str]:
    """Emit the ticket fallback and options, persist the answer, pool gaps.

    A write turn can succeed in the tool while the model's final answer omits
    the created ticket number. When the graph recorded ``created_ticket_no``
    and the answer does not already contain it, the number is appended as a
    plain-text token frame (before ``options``, so the browser receives it)
    and the augmented answer — not the original — is persisted.
    """
    answer = state_values.get("answer") or ""
    ticket_no = state_values.get("created_ticket_no") or ""
    if ticket_no and ticket_no not in answer:
        suffix = f"\n\n工单已创建，工单号：{ticket_no}"
        answer += suffix
        yield suffix

    actions = state_values.get("suggested_actions") or []
    if actions:
        yield _dumps({"type": "options", "options": list(actions)})

    append_message(session, conversation_id, "assistant", answer)
    session.commit()

    gate_reason = state_values.get("gate_reason")
    if gate_reason in _GATE_GAP_REASONS:
        try:
            record_low_confidence(
                session,
                conversation_id,
                question,
                gate_reason,
                source="confidence_gate",
            )
            session.commit()
        except Exception:  # noqa: BLE001 - a pool write must not break chat
            logger.exception("failed to record low-confidence question")
            session.rollback()


def _iter_forwarded_frames(graph, inputs, config: dict) -> Iterator[str]:
    """Yield SSE payloads for the node frames the client understands."""
    for item in graph.stream(
        inputs,
        config,
        stream_mode="custom",
        subgraphs=True,
        version="v2",
    ):
        if not isinstance(item, dict) or item.get("type") != "custom":
            logger.warning("ch06 ignoring non-custom stream item: %r", item)
            continue
        frame = item.get("data")
        if not isinstance(frame, dict):
            logger.warning("ch06 ignoring malformed custom frame: %r", frame)
            continue
        frame_type = frame.get("type")
        if frame_type not in _FORWARDED_FRAME_TYPES:
            logger.warning("ch06 ignoring unknown custom frame type=%r", frame_type)
            continue
        if frame_type == "token":
            text = frame.get("text") or ""
            if text:
                yield text
            continue
        yield _dumps(frame)


def stream_chat_agent(
    graph,
    session: Session,
    user_id: str,
    message: str,
    *,
    settings: Settings,
    conversation_id: int | None = None,
    new_conversation: bool = False,
) -> Iterator[str]:
    """Run one chat turn on the resolved conversation's thread.

    The conversation is resolved *before* the checkpoint is read: a new
    conversation, an explicit id, or the latest in-progress conversation each
    map to its own ``conv:{id}`` thread. A pending order-picker ``interrupt``
    on that thread re-emits the ``conversation`` and ``interrupt`` frames and
    ends the turn without running the graph or writing a user row. After a
    normal stream a pending interrupt likewise ends the turn before options
    or an assistant answer are written.

    The pre-turn snapshot is read once and reused for both the pending check
    and the prior checkpoint messages. Yields plain text for assistant tokens
    and compact JSON strings for ``conversation`` / ``tool`` / ``sources`` /
    ``options`` / ``interrupt`` status frames. On a graph exception it yields
    a single ``{"error": ...}`` frame and returns; the caller (API layer)
    appends ``[DONE]`` and never sees a 500 once streaming started.
    """
    try:
        conversation = resolve_conversation(
            session, user_id, conversation_id, new_conversation
        )
    except ConversationNotFound:
        logger.info(
            "ch07 chat rejected unknown conversation user_id=%s conversation_id=%s",
            user_id,
            conversation_id,
        )
        yield _dumps({"error": CONVERSATION_NOT_FOUND_TEXT})
        return

    cid = conversation.id
    config = _thread_config(cid, user_id=user_id, settings=settings)

    try:
        snapshot = _state_graph(graph).get_state(config)
    except Exception:  # noqa: BLE001 - pending check failure becomes one frame
        logger.exception("ch07 chat pending check failed conversation_id=%s", cid)
        yield _dumps({"error": ERROR_TEXT})
        return

    pending = _interrupt_payload(snapshot)
    if pending is not None:
        logger.info(
            "ch07 interrupt re-emitted conversation_id=%s kind=%s",
            cid,
            pending.get("kind"),
        )
        yield _conversation_frame(cid)
        yield _interrupt_frame(pending)
        return

    # Audit the user turn before the graph starts: a later graph crash must
    # not erase what the user asked. The DB row keeps the full text; only the
    # model-facing copy is truncated to the configured input budget.
    prompt_message = _truncate_user_input(message, settings)
    user_message = append_message(session, cid, "user", message)
    session.commit()

    prior = _snapshot_messages(snapshot)
    plan = prepare_context(
        session,
        conversation,
        prior,
        len(prior),
        settings,
        current_msg_id=user_message.id,
    )
    inputs = _reset_inputs(
        prompt_message, user_message.id, cid, user_id, settings, plan
    )

    yield _conversation_frame(cid)

    try:
        yield from _iter_forwarded_frames(graph, inputs, config)
        snapshot = _state_graph(graph).get_state(config)
    except Exception:  # noqa: BLE001 - stream errors become one SSE frame
        logger.exception("ch07 chat graph failed conversation_id=%s", cid)
        yield _dumps({"error": ERROR_TEXT})
        return

    pending = _interrupt_payload(snapshot)
    if pending is not None:
        logger.info(
            "ch07 interrupt raised conversation_id=%s kind=%s",
            cid,
            pending.get("kind"),
        )
        yield _interrupt_frame(pending)
        return

    state_values = dict(getattr(snapshot, "values", None) or {})
    yield from _finish_turn(session, cid, state_values, message)


def _resume_audit_text(resume_value: str | dict) -> str:
    """Describe the resume value for the MySQL audit-only user row."""
    if isinstance(resume_value, str):
        return f"[订单选择] 订单 {resume_value}"
    if isinstance(resume_value, dict) and resume_value.get("confirmed") is True:
        return "[工单确认] 确认提交"
    return "[工单确认] 取消"


def resume_chat_agent(
    graph,
    session: Session,
    user_id: str,
    resume_value: str | dict,
    *,
    settings: Settings,
    conversation_id: int | None = None,
) -> Iterator[str]:
    """Resume the resolved conversation's paused thread.

    ``resume_value`` is either the order id picked in the order-picker
    (``str``) or the ticket confirmation decision (``{"confirmed": bool}``).
    Raises ``NoPendingInterrupt`` when the thread is not paused, so the API
    layer can answer 409 instead of sending a stray ``Command(resume=...)``.
    The value is audit-only in MySQL; the graph receives ``Command`` and keeps
    the ``【订单选择器】`` ``HumanMessage`` it appends itself, so no duplicate
    ``HumanMessage`` enters the checkpoint.

    ``prepare_context`` runs for logging and summary triggering only: the
    resumed turn reuses the plan fields already stored in the checkpoint, so
    the plan is never injected into the ``Command``.
    """
    try:
        conversation = resolve_conversation(
            session, user_id, conversation_id, False
        )
    except ConversationNotFound:
        logger.info(
            "ch07 resume rejected unknown conversation user_id=%s conversation_id=%s",
            user_id,
            conversation_id,
        )
        yield _dumps({"error": CONVERSATION_NOT_FOUND_TEXT})
        return

    cid = conversation.id
    config = _thread_config(cid, user_id=user_id, settings=settings)
    snapshot = _state_graph(graph).get_state(config)
    pending = _interrupt_payload(snapshot)
    if pending is None:
        raise NoPendingInterrupt(f"conversation {cid} has no pending interrupt")

    audit_text = _resume_audit_text(resume_value)
    audit_message = append_message(session, cid, "user", audit_text)
    session.commit()

    prior = _snapshot_messages(snapshot)
    prepare_context(
        session,
        conversation,
        prior,
        _last_human_index(prior),
        settings,
        current_msg_id=audit_message.id,
    )

    yield _conversation_frame(cid)

    try:
        yield from _iter_forwarded_frames(
            graph, Command(resume=resume_value), config
        )
        snapshot = _state_graph(graph).get_state(config)
    except Exception:  # noqa: BLE001 - stream errors become one SSE frame
        logger.exception("ch07 resume graph failed conversation_id=%s", cid)
        yield _dumps({"error": ERROR_TEXT})
        return

    # A resume may itself pause on another interrupt (e.g. a second picker);
    # re-emit it instead of writing an empty assistant answer.
    pending = _interrupt_payload(snapshot)
    if pending is not None:
        logger.info(
            "ch07 interrupt raised on resume conversation_id=%s kind=%s",
            cid,
            pending.get("kind"),
        )
        yield _interrupt_frame(pending)
        return

    state_values = dict(getattr(snapshot, "values", None) or {})
    question = state_values.get("user_message") or audit_text
    yield from _finish_turn(session, cid, state_values, question)
