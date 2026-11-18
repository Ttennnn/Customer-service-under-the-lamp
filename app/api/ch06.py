"""Ch06 dispatcher API: chat/resume SSE, refund tickets and page routes.

The parent workflow graph is built lazily on the first ``/ch06/*`` request and
cached on ``app.state``.  Building it at import/startup time would make every
``with TestClient(app)`` test construct real LLM clients and open the real
SQLite checkpoint file, so the graph stays ``None`` until a request needs it.
The checkpointer handle is closed by the application lifespan.

``/ch06/chat/resume`` must answer 409 *before* an SSE response starts when the
thread is not paused.  The pending check therefore runs in a regular
dependency (``require_pending_interrupt``), not inside the streaming
generator: dependency failures still map to a normal HTTP status, whereas an
exception raised while iterating a generator would already have sent 200.
The resume generator opens its own DB session so the stream does not depend
on a request-scoped session staying open.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from sqlalchemy.orm import Session

from app.agent.graph import build_graph, open_sqlite_checkpointer
from app.agent.service import (
    NoPendingInterrupt,
    has_pending_interrupt,
    resume_chat_agent,
    stream_chat_agent,
)
from app.config import Settings, build_llm, get_settings
from app.db import session as db_session
from app.db.session import get_db
from app.schemas import ChatRequest, RefundRequest, ResumeRequest
from app.services.history import get_or_create_conversation
from app.tools.business import list_user_orders
from app.tools.context import current_conversation_id
from app.tools.executor import execute_tool
from app.tools.ticket import TICKET_NO_PATTERN, create_ticket

logger = logging.getLogger(__name__)

api_router = APIRouter(prefix="/ch06")
page_router = APIRouter()

CH06_PAGE = Path(__file__).resolve().parents[1] / "static" / "ch06.html"

ERROR_TEXT = "客服服务暂时不可用,请稍后重试"
NO_PENDING_INTERRUPT = "当前没有待选择的订单"
REFUND_ORDER_INVALID = "订单不存在或不属于当前会话"
REFUND_CREATE_FAILED = "退款单创建失败,请稍后重试"


def _build_and_cache_graph(app, settings: Settings):
    """Create the checkpointer + graph once and cache them on ``app.state``."""
    handle = None
    try:
        handle = open_sqlite_checkpointer(settings.ch06_checkpoint_db)
        llm = build_llm(settings)
        intent_llm = build_llm(
            settings,
            temperature=0.0,
            model=settings.intent_model or None,
        )
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
                logger.exception("failed to close ch06 checkpointer after build error")
        logger.exception("failed to build ch06 graph")
        raise HTTPException(status_code=503, detail=ERROR_TEXT) from None

    app.state.ch06_checkpoint_handle = handle
    app.state.ch06_graph = graph
    return graph


def get_ch06_graph_dep(
    request: Request, settings: Settings = Depends(get_settings)
):
    """Return the cached workflow graph, building it on first use.

    Double-checked locking: the fast path reads without the lock, and the
    slow path re-checks inside the lock so concurrent first requests build
    exactly one graph/checkpointer.
    """
    state = request.app.state
    graph = getattr(state, "ch06_graph", None)
    if graph is not None:
        return graph

    lock = getattr(state, "ch06_graph_lock", None)
    if lock is None:
        lock = threading.Lock()
        state.ch06_graph_lock = lock

    with lock:
        graph = getattr(state, "ch06_graph", None)
        if graph is not None:
            return graph
        return _build_and_cache_graph(request.app, settings)


def close_ch06_graph(app) -> None:
    """Close the lazily-opened checkpointer handle, if one exists."""
    handle = getattr(app.state, "ch06_checkpoint_handle", None)
    if handle is None:
        return
    try:
        handle.close()
    except Exception:  # noqa: BLE001 - shutdown must not crash the process
        logger.exception("failed to close ch06 checkpointer")
    finally:
        app.state.ch06_checkpoint_handle = None
        app.state.ch06_graph = None


def require_pending_interrupt(
    req: ResumeRequest,
    graph=Depends(get_ch06_graph_dep),
):
    """Answer 409 when the thread is not paused, before any SSE starts.

    This runs as a dependency so a missing interrupt surfaces as a normal
    HTTP 409 instead of a 200 stream with an error frame.
    """
    try:
        pending = has_pending_interrupt(graph, req.session_id)
    except Exception:  # noqa: BLE001 - checkpoint read failure becomes 503
        logger.exception("ch06 resume pending check failed user_id=%s", req.session_id)
        raise HTTPException(status_code=503, detail=ERROR_TEXT) from None

    if not pending:
        raise HTTPException(status_code=409, detail=NO_PENDING_INTERRUPT)
    return graph


@page_router.get("/ch06", response_class=FileResponse)
def ch06_page() -> FileResponse:
    """Serve the self-contained ch06 chat page."""
    return FileResponse(CH06_PAGE, media_type="text/html")


@api_router.post("/chat", response_class=EventSourceResponse)
def ch06_chat(
    req: ChatRequest,
    settings: Settings = Depends(get_settings),
    graph=Depends(get_ch06_graph_dep),
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
        logger.exception("ch06 chat stream failed")
        yield ServerSentEvent(
            raw_data=json.dumps({"error": ERROR_TEXT}, ensure_ascii=False)
        )
    yield ServerSentEvent(raw_data="[DONE]")


@api_router.post("/chat/resume", response_class=EventSourceResponse)
def ch06_chat_resume(
    req: ResumeRequest,
    settings: Settings = Depends(get_settings),
    graph=Depends(require_pending_interrupt),
):
    """Resume a paused thread as SSE, always terminating with ``[DONE]``.

    The 409/503 checks already ran in ``require_pending_interrupt``.  The
    generator owns its DB session because the request-scoped dependency
    session is not guaranteed to outlive the streamed response.
    """
    session = db_session.get_session_factory()()
    try:
        for payload in resume_chat_agent(
            graph,
            session,
            req.session_id,
            req.order_id,
            settings=settings,
        ):
            yield ServerSentEvent(raw_data=payload)
    except NoPendingInterrupt:
        # The thread was resumed or reset between the dependency check and
        # the generator start; answer with a normal error frame.
        logger.warning(
            "ch06 resume lost pending interrupt user_id=%s", req.session_id
        )
        yield ServerSentEvent(
            raw_data=json.dumps(
                {"error": NO_PENDING_INTERRUPT}, ensure_ascii=False
            )
        )
    except Exception:  # noqa: BLE001 - stream already in flight, end cleanly
        logger.exception("ch06 resume stream failed")
        yield ServerSentEvent(
            raw_data=json.dumps({"error": ERROR_TEXT}, ensure_ascii=False)
        )
    finally:
        session.close()
    yield ServerSentEvent(raw_data="[DONE]")


@api_router.post("/refunds")
def ch06_refund(req: RefundRequest, session: Session = Depends(get_db)):
    """Create a refund ticket for an order in the user's demo catalogue.

    The order id and reason are validated before any DB write so an invalid
    form never leaves a dirty ticket.  ``retries=0`` is deliberate: ticket
    creation is not idempotent, so a timeout surfaces as 502 for a manual
    retry instead of silently creating a duplicate refund.
    """
    orders = list_user_orders(req.session_id)
    if req.order_id not in {order.get("order_id") for order in orders}:
        raise HTTPException(status_code=422, detail=REFUND_ORDER_INVALID)

    conversation = get_or_create_conversation(session, req.session_id)
    token = current_conversation_id.set(conversation.id)
    try:
        result = execute_tool(
            create_ticket,
            {
                "description": f"订单 {req.order_id} 退款申请，原因：{req.reason}",
                "ticket_type": "退款",
            },
            timeout_s=5,
            retries=0,
        )
    finally:
        current_conversation_id.reset(token)

    if not TICKET_NO_PATTERN.match(result or ""):
        logger.warning("ch06 refund ticket creation failed: %r", result)
        raise HTTPException(status_code=502, detail=REFUND_CREATE_FAILED)

    return {"ticket_no": result, "message": "退款单已创建"}
