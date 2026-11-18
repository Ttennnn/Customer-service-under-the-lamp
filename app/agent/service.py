"""``/ch06/chat`` streaming service: SSE mapping, DB audit, low-confidence pool.

``stream_chat_agent`` drives one turn of the parent workflow graph:

1. Check the checkpointer thread for a pending interrupt *before* touching
   the database.  A pending order picker is re-emitted and the turn ends
   without writing a user message or running the graph; a refresh/new message
   therefore never wedges the thread.
2. Resolve (or create) the conversation and persist the user turn *before*
   starting the graph, so the audit trail survives a graph failure.
3. Reset every transient state field for the new turn. ``messages`` carries
   only the new ``HumanMessage``; with a checkpointer the ``add_messages``
   reducer appends it to the thread's existing history.
4. Consume only ``stream_mode="custom"`` frames. On langgraph 1.2.20 with
   ``subgraphs=True`` each item is ``{"type": "custom", "ns": ..., "data":
   <node frame>}`` — read ``item["data"]``, not ``item[1]``.
5. Forward ``token`` / ``tool`` / ``sources`` frames as raw SSE payload
   strings: plain text for tokens, compact JSON for the rest. Unknown frames
   are logged and dropped.
6. Read the final State after the stream.  If the graph paused on an
   ``interrupt``, emit the ``interrupt`` frame and stop: no options, no
   assistant answer.  Otherwise emit ``options`` when the graph suggested
   actions, persist the assistant answer, and record a low-confidence
   question only for genuine knowledge gaps.  Infrastructure and rerank
   failures never enter the pool.

``resume_chat_agent`` continues a paused thread with ``Command(resume=...)``.
The resume value is *not* sent as a ``HumanMessage``: ``fetch_order`` already
appends the ``[订单选择器]`` ``SystemMessage`` on resume.  MySQL still gets an
audit-only user row ``[订单选择] 订单 <id>``.

Graph exceptions are converted into a single ``{"error": ...}`` frame; the
generator never re-raises.
"""

import json
import logging
from collections.abc import Iterator

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from sqlalchemy.orm import Session

from app.config import Settings
from app.services.history import append_message, get_or_create_conversation
from app.services.low_confidence import record_low_confidence

logger = logging.getLogger(__name__)

ERROR_TEXT = "客服服务暂时不可用,请稍后重试"
_GATE_GAP_REASONS = frozenset({"empty", "below_threshold"})
_FORWARDED_FRAME_TYPES = frozenset({"token", "tool", "sources"})


class NoPendingInterrupt(Exception):
    """Raised when a resume is requested but the thread is not paused."""


def _dumps(payload: dict) -> str:
    """Serialize an SSE status frame exactly like the legacy chat service."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _thread_config(user_id: str) -> dict:
    """Config pinning the checkpointer thread to the frontend session id."""
    return {"configurable": {"thread_id": f"user:{user_id}"}}


def _reset_inputs(
    message: str,
    conversation_id: int,
    session_id: str,
    settings: Settings,
) -> dict:
    """Build one turn's inputs with every transient field reset explicitly.

    ``rerank_failed`` in particular must be reset: it is a bool rather than a
    reducer-backed channel, so a stale ``True`` would leak into the next turn.
    The ch06 slots (``order_id`` / ``order_data``) and routing fields are reset
    for the same reason: a new turn must re-derive them from its own message.
    """
    return {
        "messages": [HumanMessage(content=message)],
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
        "suggested_actions": [],
        "step_count": 0,
        "token_estimate": 0,
        "exhausted": False,
        "conversation_id": conversation_id,
        "token_budget": settings.agent_token_budget,
        "max_steps": settings.agent_max_steps,
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
    return _interrupt_payload(graph.get_state(config))


def has_pending_interrupt(graph, user_id: str) -> bool:
    """Whether the user's thread is paused on an ``interrupt``."""
    return _pending_interrupt(graph, _thread_config(user_id)) is not None


def _interrupt_frame(payload: dict) -> str:
    """Serialize the order-picker frame the frontend renders as cards."""
    return _dumps({"type": "interrupt", **payload})


def _finish_turn(
    session: Session,
    conversation_id: int,
    state_values: dict,
    question: str,
) -> Iterator[str]:
    """Emit trailing options, persist the answer and record a knowledge gap."""
    actions = state_values.get("suggested_actions") or []
    if actions:
        yield _dumps({"type": "options", "options": list(actions)})

    answer = state_values.get("answer") or ""
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
) -> Iterator[str]:
    """Run one ``/ch06/chat`` turn and yield raw SSE payload strings.

    If the thread is paused on an order-picker ``interrupt``, the frame is
    re-emitted and the graph is not run: the new message is dropped rather
    than classified against a thread waiting for a resume value. After a
    normal stream a pending interrupt likewise ends the turn before options
    or an assistant answer are written.

    Yields plain text for assistant tokens and compact JSON strings for
    ``tool`` / ``sources`` / ``options`` / ``interrupt`` status frames. On a
    graph exception it yields a single ``{"error": ...}`` frame and returns;
    the caller (API layer) appends ``[DONE]`` and never sees a 500 once
    streaming started.
    """
    config = _thread_config(user_id)

    try:
        pending = _pending_interrupt(graph, config)
    except Exception:  # noqa: BLE001 - pending check failure becomes one frame
        logger.exception("ch06 chat pending check failed user_id=%s", user_id)
        yield _dumps({"error": ERROR_TEXT})
        return

    if pending is not None:
        logger.info(
            "ch06 interrupt re-emitted user_id=%s kind=%s",
            user_id,
            pending.get("kind"),
        )
        yield _interrupt_frame(pending)
        return

    conversation = get_or_create_conversation(session, user_id)
    cid = conversation.id

    # Audit the user turn before the graph starts: a later graph crash must
    # not erase what the user asked.
    append_message(session, cid, "user", message)
    session.commit()

    inputs = _reset_inputs(message, cid, user_id, settings)

    try:
        yield from _iter_forwarded_frames(graph, inputs, config)
        snapshot = graph.get_state(config)
    except Exception:  # noqa: BLE001 - stream errors become one SSE frame
        logger.exception("ch06 chat graph failed user_id=%s", user_id)
        yield _dumps({"error": ERROR_TEXT})
        return

    pending = _interrupt_payload(snapshot)
    if pending is not None:
        logger.info(
            "ch06 interrupt raised user_id=%s kind=%s",
            user_id,
            pending.get("kind"),
        )
        yield _interrupt_frame(pending)
        return

    state_values = dict(getattr(snapshot, "values", None) or {})
    yield from _finish_turn(session, cid, state_values, message)


def resume_chat_agent(
    graph,
    session: Session,
    user_id: str,
    order_id: str,
    *,
    settings: Settings,
) -> Iterator[str]:
    """Resume a paused thread with the user's order pick.

    Raises ``NoPendingInterrupt`` when the thread is not paused, so the API
    layer can answer 409 instead of sending a stray ``Command(resume=...)``.
    The pick is audit-only in MySQL (``[订单选择] 订单 <id>``); the graph
    receives ``Command`` and keeps the ``[订单选择器]`` SystemMessage it
    appends itself, so no duplicate ``HumanMessage`` enters the checkpoint.
    """
    config = _thread_config(user_id)
    pending = _pending_interrupt(graph, config)
    if pending is None:
        raise NoPendingInterrupt(f"user {user_id} has no pending interrupt")

    conversation = get_or_create_conversation(session, user_id)
    cid = conversation.id
    audit_text = f"[订单选择] 订单 {order_id}"
    append_message(session, cid, "user", audit_text)
    session.commit()

    try:
        yield from _iter_forwarded_frames(graph, Command(resume=order_id), config)
        snapshot = graph.get_state(config)
    except Exception:  # noqa: BLE001 - stream errors become one SSE frame
        logger.exception("ch06 resume graph failed user_id=%s", user_id)
        yield _dumps({"error": ERROR_TEXT})
        return

    # A resume may itself pause on another interrupt (e.g. a second picker);
    # re-emit it instead of writing an empty assistant answer.
    pending = _interrupt_payload(snapshot)
    if pending is not None:
        logger.info(
            "ch06 interrupt raised on resume user_id=%s kind=%s",
            user_id,
            pending.get("kind"),
        )
        yield _interrupt_frame(pending)
        return

    state_values = dict(getattr(snapshot, "values", None) or {})
    question = state_values.get("user_message") or audit_text
    yield from _finish_turn(session, cid, state_values, question)
