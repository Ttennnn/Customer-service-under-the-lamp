"""Ch07 API aliases: the same chat/resume/refund handlers under ``/ch07``.

Chapter 7 is a context-management evolution of the ch06 dispatcher rather
than a separate workflow. These routes therefore reuse the ch06 handlers and
the one lazily-built graph cached on ``app.state.ch06_*`` so ``/ch06`` and
``/ch07`` share a single checkpointer and thread. ``page_router`` serves the
multi-conversation frontend page at ``/ch07``.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from fastapi.sse import EventSourceResponse
from sqlalchemy.orm import Session

from app.api.ch06 import (
    ch06_chat,
    ch06_chat_resume,
    ch06_refund,
    get_ch06_graph_dep,
    require_pending_interrupt,
)
from app.config import Settings, get_settings
from app.db.session import get_db
from app.schemas import ChatRequest, RefundRequest, ResumeRequest

api_router = APIRouter(prefix="/ch07")
page_router = APIRouter()

CH07_PAGE = Path(__file__).resolve().parents[1] / "static" / "ch07.html"


@page_router.get("/ch07", response_class=FileResponse)
def ch07_page() -> FileResponse:
    """Serve the self-contained multi-conversation ch07 chat page."""
    return FileResponse(
        CH07_PAGE,
        media_type="text/html",
        headers={"Cache-Control": "no-store"},
    )


@api_router.post("/chat", response_class=EventSourceResponse)
def ch07_chat(
    req: ChatRequest,
    settings: Settings = Depends(get_settings),
    graph=Depends(get_ch06_graph_dep),
    session: Session = Depends(get_db),
):
    """Stream one context-managed turn, identical to ``POST /ch06/chat``."""
    yield from ch06_chat(req, settings, graph, session)


@api_router.post("/chat/resume", response_class=EventSourceResponse)
def ch07_chat_resume(
    req: ResumeRequest,
    settings: Settings = Depends(get_settings),
    graph=Depends(require_pending_interrupt),
):
    """Resume the paused conversation thread, identical to ch06."""
    yield from ch06_chat_resume(req, settings, graph)


@api_router.post("/refunds")
def ch07_refund(req: RefundRequest, session: Session = Depends(get_db)):
    """Create a refund ticket in the requested conversation."""
    return ch06_refund(req, session)
