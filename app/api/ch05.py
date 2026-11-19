"""Ch05 compatibility layer.

Chapter 6 replaced the ch05 workflow with ``/ch06``.  The old page and chat
endpoint now answer 307 redirects so bookmarks and legacy clients keep
working, and the graph lazy-load/close machinery moved to ``app.api.ch06``.
``POST /ch05/tickets`` stays: the chat page still creates complaint and
after-sale tickets through the ch02 tool.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.api.ch06 import CH06_PAGE, resolve_request_conversation
from app.db.session import get_db
from app.schemas import TicketRequest
from app.tools.builtin.tickets import TICKET_NO_PATTERN
from app.tools.engine import execute
from app.tools.registry import builtin_specs

logger = logging.getLogger(__name__)

api_router = APIRouter(prefix="/ch05")
page_router = APIRouter()

TICKET_CREATE_FAILED = "工单创建失败,请稍后重试"


@page_router.get("/ch05", response_class=FileResponse)
def ch05_page() -> FileResponse:
    """Serve the latest ch06 chat page at the legacy /ch05 URL."""
    return FileResponse(
        CH06_PAGE,
        media_type="text/html",
        headers={"Cache-Control": "no-store"},
    )


@api_router.post("/chat")
def ch05_chat() -> RedirectResponse:
    """Redirect legacy chat clients to the ch06 streaming endpoint.

    No body model is declared on purpose: the redirect must fire without
    validating a payload this layer no longer understands.
    """
    return RedirectResponse(url="/ch06/chat", status_code=307)


@api_router.post("/tickets")
def ch05_ticket(req: TicketRequest, session: Session = Depends(get_db)):
    """Create a ticket for the current conversation through the engine.

    The button click is the trusted confirmation, so ``confirmed=True`` is
    passed to the write permission gate; no request field can override it.
    The engine sets the conversation contextvar itself, writes the audit row,
    and forces zero retries for writes (ticket creation is not idempotent, so
    a timeout must surface as a 502 for a manual retry instead of silently
    creating a duplicate). The conversation status is never changed; human
    transfer is a frontend-only simulation.
    """
    conversation = resolve_request_conversation(
        session, req.session_id, req.conversation_id
    )
    run = execute(
        builtin_specs()["create_ticket"],
        {"description": req.description, "ticket_type": req.ticket_type},
        conversation_id=conversation.id,
        confirmed=True,
    )

    if run.status != "ok" or not TICKET_NO_PATTERN.match(str(run.data or "")):
        logger.warning("ch05 ticket creation failed: %r", run.content)
        raise HTTPException(status_code=502, detail=TICKET_CREATE_FAILED)

    return {"ticket_no": run.data, "message": "工单已创建"}
