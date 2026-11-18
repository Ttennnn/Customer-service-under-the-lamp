"""``/ch05/chat`` streaming service: SSE mapping, DB audit, low-confidence pool.

``stream_chat_agent`` drives one turn of the parent workflow graph:

1. Resolve (or create) the conversation and persist the user turn *before*
   starting the graph, so the audit trail survives a graph failure.
2. Reset every transient state field for the new turn. ``messages`` carries
   only the new ``HumanMessage``; with a checkpointer the ``add_messages``
   reducer appends it to the thread's existing history.
3. Consume only ``stream_mode="custom"`` frames. On langgraph 1.2.20 with
   ``subgraphs=True`` each item is ``{"type": "custom", "ns": ..., "data":
   <node frame>}`` — read ``item["data"]``, not ``item[1]``.
4. Forward ``token`` / ``tool`` / ``sources`` frames as raw SSE payload
   strings: plain text for tokens, compact JSON for the rest. Unknown frames
   are logged and dropped.
5. Read the final State after the stream, emit ``options`` when the graph
   suggested actions, persist the assistant answer, and record a
   low-confidence question only for genuine knowledge gaps. Infrastructure
   and rerank failures never enter the pool.

Graph exceptions are converted into a single ``{"error": ...}`` frame; the
generator never re-raises.
"""

import json
import logging
from collections.abc import Iterator

from langchain_core.messages import HumanMessage
from sqlalchemy.orm import Session

from app.config import Settings
from app.services.history import append_message, get_or_create_conversation
from app.services.low_confidence import record_low_confidence

logger = logging.getLogger(__name__)

ERROR_TEXT = "客服服务暂时不可用,请稍后重试"
_GATE_GAP_REASONS = frozenset({"empty", "below_threshold"})
_FORWARDED_FRAME_TYPES = frozenset({"token", "tool", "sources"})


def _dumps(payload: dict) -> str:
    """Serialize an SSE status frame exactly like the legacy chat service."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _thread_config(user_id: str) -> dict:
    """Config pinning the checkpointer thread to the frontend session id."""
    return {"configurable": {"thread_id": f"user:{user_id}"}}


def _reset_inputs(
    message: str, conversation_id: int, settings: Settings
) -> dict:
    """Build one turn's inputs with every transient field reset explicitly.

    ``rerank_failed`` in particular must be reset: it is a bool rather than a
    reducer-backed channel, so a stale ``True`` would leak into the next turn.
    """
    return {
        "messages": [HumanMessage(content=message)],
        "user_message": message,
        "resolved_message": message,
        "intent": "",
        "route": "",
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


def _iter_forwarded_frames(graph, inputs: dict, config: dict) -> Iterator[str]:
    """Yield SSE payloads for the node frames the client understands."""
    for item in graph.stream(
        inputs,
        config,
        stream_mode="custom",
        subgraphs=True,
        version="v2",
    ):
        if not isinstance(item, dict) or item.get("type") != "custom":
            logger.warning("ch05 ignoring non-custom stream item: %r", item)
            continue
        frame = item.get("data")
        if not isinstance(frame, dict):
            logger.warning("ch05 ignoring malformed custom frame: %r", frame)
            continue
        frame_type = frame.get("type")
        if frame_type not in _FORWARDED_FRAME_TYPES:
            logger.warning("ch05 ignoring unknown custom frame type=%r", frame_type)
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
    """Run one ``/ch05/chat`` turn and yield raw SSE payload strings.

    Yields plain text for assistant tokens and compact JSON strings for
    ``tool`` / ``sources`` / ``options`` status frames. On a graph exception
    it yields a single ``{"error": ...}`` frame and returns; the caller (API
    layer) appends ``[DONE]`` and never sees a 500 once streaming started.
    """
    conversation = get_or_create_conversation(session, user_id)
    cid = conversation.id

    # Audit the user turn before the graph starts: a later graph crash must
    # not erase what the user asked.
    append_message(session, cid, "user", message)
    session.commit()

    config = _thread_config(user_id)
    inputs = _reset_inputs(message, cid, settings)

    try:
        yield from _iter_forwarded_frames(graph, inputs, config)
        state_values = dict(graph.get_state(config).values or {})
    except Exception:  # noqa: BLE001 - stream errors become one SSE frame
        logger.exception("ch05 chat graph failed user_id=%s", user_id)
        yield _dumps({"error": ERROR_TEXT})
        return

    # Options come from the final State, so they always trail the answer
    # tokens and the frontend can render them as a separate button row.
    actions = state_values.get("suggested_actions") or []
    if actions:
        yield _dumps({"type": "options", "options": list(actions)})

    answer = state_values.get("answer") or ""
    append_message(session, cid, "assistant", answer)
    session.commit()

    gate_reason = state_values.get("gate_reason")
    if gate_reason in _GATE_GAP_REASONS:
        try:
            record_low_confidence(
                session,
                cid,
                message,
                gate_reason,
                source="confidence_gate",
            )
            session.commit()
        except Exception:  # noqa: BLE001 - a pool write must not break chat
            logger.exception("failed to record low-confidence question")
            session.rollback()
