"""Read-only conversation list and dialog history APIs.

The list is scoped to one user and returns the first user question as a
preview plus whether any summary exists. The history endpoint returns only
user/assistant rows oldest first; ``[订单选择]`` audit rows written by the
order picker are UI bookkeeping, not dialog, so they are filtered out.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.history import (
    get_conversation,
    list_conversations,
    list_dialog_messages,
)

api_router = APIRouter(prefix="/api/conversations")

CONVERSATION_NOT_FOUND = "会话不存在"
ORDER_AUDIT_PREFIX = "[订单选择]"


@api_router.get("")
def get_conversations(user_id: str, session: Session = Depends(get_db)):
    """List ``user_id``'s conversations newest first."""
    return {"conversations": list_conversations(session, user_id)}


@api_router.get("/{conversation_id}/messages")
def get_conversation_messages(
    conversation_id: int,
    user_id: str | None = None,
    session: Session = Depends(get_db),
):
    """Return one conversation's dialog, 404 when missing or not owned."""
    conversation = get_conversation(session, conversation_id, user_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail=CONVERSATION_NOT_FOUND)

    messages = [
        {
            "id": message.id,
            "role": message.role,
            "content": message.content,
            "created_at": message.created_at,
        }
        for message in list_dialog_messages(session, conversation.id)
        if not (message.content or "").startswith(ORDER_AUDIT_PREFIX)
    ]
    return {"conversation_id": conversation.id, "messages": messages}
