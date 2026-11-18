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
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas import TicketRequest
from app.services.history import get_or_create_conversation
from app.tools.context import current_conversation_id
from app.tools.executor import execute_tool
from app.tools.ticket import TICKET_NO_PATTERN, create_ticket

logger = logging.getLogger(__name__)

api_router = APIRouter(prefix="/ch05")
page_router = APIRouter()

TICKET_CREATE_FAILED = "工单创建失败,请稍后重试"


@page_router.get("/ch05")
def ch05_page() -> RedirectResponse:
    """Redirect the retired ch05 page to the ch06 chat page."""
    return RedirectResponse(url="/ch06", status_code=307)


@api_router.post("/chat")
def ch05_chat() -> RedirectResponse:
    """Redirect legacy chat clients to the ch06 streaming endpoint.

    No body model is declared on purpose: the redirect must fire without
    validating a payload this layer no longer understands.
    """
    return RedirectResponse(url="/ch06/chat", status_code=307)


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
