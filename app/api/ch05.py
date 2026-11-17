"""Ch05 agent-workflow API: streaming chat and button-driven ticket creation.

The LangGraph workflow is built lazily on the first ``/ch05/chat`` request and
cached on ``app.state``.  Building it at import/startup time would make every
existing ``with TestClient(app)`` test construct real LLM clients and open the
real SQLite checkpoint file, so the graph stays ``None`` until a request needs
it.  The checkpointer handle is closed by the application lifespan.
"""

from __future__ import annotations

import json
import logging
import threading

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from sqlalchemy.orm import Session

from app.agent.graph import build_graph, open_sqlite_checkpointer
from app.agent.service import stream_chat_agent
from app.config import Settings, build_llm, get_settings
from app.db.session import get_db
from app.schemas import ChatRequest, TicketRequest
from app.services.history import get_or_create_conversation
from app.tools.context import current_conversation_id
from app.tools.executor import execute_tool
from app.tools.ticket import TICKET_NO_PATTERN, create_ticket

logger = logging.getLogger(__name__)

api_router = APIRouter(prefix="/ch05")

ERROR_TEXT = "客服服务暂时不可用,请稍后重试"
TICKET_CREATE_FAILED = "工单创建失败,请稍后重试"


def _build_and_cache_graph(app, settings: Settings):
    """Create the checkpointer + graph once and cache them on ``app.state``."""
    handle = None
    try:
        handle = open_sqlite_checkpointer(settings.langgraph_checkpoint_db)
        llm = build_llm(settings)
        intent_llm = build_llm(settings, temperature=0.0)
        graph = build_graph(
            llm,
            intent_llm,
            checkpointer=handle.saver,
            settings=settings,
        )
    except Exception:  # noqa: BLE001 - construction failure surfaced as 503
        if handle is not None:
            try:
                handle.close()
            except Exception:  # noqa: BLE001 - keep the original failure
                logger.exception("failed to close ch05 checkpointer after build error")
        logger.exception("failed to build ch05 graph")
        raise HTTPException(status_code=503, detail=ERROR_TEXT) from None

    app.state.ch05_checkpoint_handle = handle
    app.state.ch05_graph = graph
    return graph


def get_ch05_graph_dep(
    request: Request, settings: Settings = Depends(get_settings)
):
    """Return the cached workflow graph, building it on first use.

    Double-checked locking: the fast path reads without the lock, and the
    slow path re-checks inside the lock so concurrent first requests build
    exactly one graph/checkpointer.
    """
    state = request.app.state
    graph = getattr(state, "ch05_graph", None)
    if graph is not None:
        return graph

    lock = getattr(state, "ch05_graph_lock", None)
    if lock is None:
        lock = threading.Lock()
        state.ch05_graph_lock = lock

    with lock:
        graph = getattr(state, "ch05_graph", None)
        if graph is not None:
            return graph
        return _build_and_cache_graph(request.app, settings)


def close_ch05_graph(app) -> None:
    """Close the lazily-opened checkpointer handle, if one exists."""
    handle = getattr(app.state, "ch05_checkpoint_handle", None)
    if handle is None:
        return
    try:
        handle.close()
    except Exception:  # noqa: BLE001 - shutdown must not crash the process
        logger.exception("failed to close ch05 checkpointer")
    finally:
        app.state.ch05_checkpoint_handle = None
        app.state.ch05_graph = None


@api_router.post("/chat", response_class=EventSourceResponse)
def ch05_chat(
    req: ChatRequest,
    settings: Settings = Depends(get_settings),
    graph=Depends(get_ch05_graph_dep),
    session: Session = Depends(get_db),
):
    """Stream one workflow turn as SSE, always terminating with ``[DONE]``.

    ``stream_chat_agent`` converts graph failures into an ``error`` frame and
    returns normally.  This wrapper additionally guards conversation/audit DB
    failures (which happen before the graph starts) so a started stream never
    ends with a 500 or hangs the client.
    """
    try:
        for payload in stream_chat_agent(
            graph,
            session,
            req.session_id,
            req.message,
            settings=settings,
        ):
            yield ServerSentEvent(raw_data=payload)
    except Exception:  # noqa: BLE001 - stream already in flight, end cleanly
        logger.exception("ch05 chat stream failed")
        yield ServerSentEvent(
            raw_data=json.dumps({"error": ERROR_TEXT}, ensure_ascii=False)
        )
    yield ServerSentEvent(raw_data="[DONE]")


@api_router.post("/tickets")
def ch05_ticket(req: TicketRequest, session: Session = Depends(get_db)):
    """Create a ticket for the current conversation via the ch02 tool.

    ``retries=0`` is deliberate: ticket creation is not idempotent, so a
    timeout must surface as a 502 for a manual retry instead of silently
    creating a duplicate ticket.  The conversation status is never changed;
    human transfer is a frontend-only simulation.
    """
    conversation = get_or_create_conversation(session, req.session_id)
    token = current_conversation_id.set(conversation.id)
    try:
        result = execute_tool(
            create_ticket,
            {"description": req.description, "ticket_type": req.ticket_type},
            timeout_s=5,
            retries=0,
        )
    finally:
        current_conversation_id.reset(token)

    if not TICKET_NO_PATTERN.match(result or ""):
        logger.warning("ch05 ticket creation failed: %r", result)
        raise HTTPException(status_code=502, detail=TICKET_CREATE_FAILED)

    return {"ticket_no": result, "message": "工单已创建"}
