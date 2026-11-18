"""DB-backed conversation history: persistence and reconstruction.

Stores conversation messages in the ``messages`` table and rebuilds
``langchain_core`` message objects from them for prompt assembly.
"""

import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from sqlalchemy import func

from app.models import Conversation, ConversationSummary, Message

#: ``[订单选择] 订单 1001`` rows are UI audit records, not user speech.
#: Shared by the summary job and the layered-window rebuild so both render the
#: same system fact instead of attributing the pick to the user.
ORDER_AUDIT_RE = re.compile(r"^\[订单选择\]\s*订单\s*(\d+)\s*$")


class ConversationNotFound(Exception):
    """Raised when a conversation does not exist or is not owned by the user."""


def get_or_create_conversation(session, user_id: str) -> Conversation:
    """Return the most recent in-progress conversation for ``user_id``.

    If none exists, create a new ``Conversation`` (status ``"进行中"``),
    commit it and return it.
    """
    conversation = (
        session.query(Conversation)
        .filter(Conversation.user_id == user_id, Conversation.status == "进行中")
        .order_by(Conversation.id.desc())
        .first()
    )
    if conversation is not None:
        return conversation

    conversation = Conversation(user_id=user_id, status="进行中")
    session.add(conversation)
    session.commit()
    session.refresh(conversation)
    return conversation


def append_message(
    session,
    conversation_id: int,
    role: str,
    content: str | None = None,
    tool_calls: list | None = None,
    tool_call_id: str | None = None,
) -> Message:
    """Append a message row and flush so its id is assigned.

    The caller is responsible for committing the transaction.
    """
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        tool_calls=tool_calls,
        tool_call_id=tool_call_id,
    )
    session.add(message)
    session.flush()
    return message


def load_history(session, conversation_id: int) -> list[BaseMessage]:
    """Rebuild langchain messages from the DB, oldest first.

    ``Message.tool_calls`` is a JSON column, so SQLAlchemy already
    deserializes it to a Python list on read; it is passed straight into
    ``AIMessage`` without ``json.loads``.
    """
    rows = (
        session.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.id.asc())
        .all()
    )

    messages: list[BaseMessage] = []
    for row in rows:
        if row.role == "user":
            messages.append(HumanMessage(content=row.content or ""))
        elif row.role == "assistant":
            if row.tool_calls:
                messages.append(
                    AIMessage(content=row.content or "", tool_calls=row.tool_calls)
                )
            else:
                messages.append(AIMessage(content=row.content or ""))
        elif row.role == "tool":
            messages.append(
                ToolMessage(content=row.content or "", tool_call_id=row.tool_call_id)
            )
        else:
            raise ValueError(f"未知消息角色: {row.role!r}")
    return messages


def create_conversation(session, user_id: str) -> Conversation:
    """Create and commit a fresh in-progress conversation for ``user_id``."""
    conversation = Conversation(user_id=user_id, status="进行中")
    session.add(conversation)
    session.commit()
    session.refresh(conversation)
    return conversation


def get_conversation(
    session, conversation_id: int, user_id: str | None = None
) -> Conversation | None:
    """Return a conversation by id, optionally scoped to ``user_id``."""
    query = session.query(Conversation).filter(Conversation.id == conversation_id)
    if user_id is not None:
        query = query.filter(Conversation.user_id == user_id)
    return query.first()


def resolve_conversation(
    session,
    user_id: str,
    conversation_id: int | None,
    new_conversation: bool,
) -> Conversation:
    """Pick the conversation for this turn.

    ``new_conversation`` always creates one; an explicit id must exist and
    belong to ``user_id``; otherwise fall back to the latest in-progress
    conversation (creating one if needed).
    """
    if new_conversation:
        return create_conversation(session, user_id)
    if conversation_id is not None:
        conversation = get_conversation(session, conversation_id, user_id)
        if conversation is None:
            raise ConversationNotFound(conversation_id)
        return conversation
    return get_or_create_conversation(session, user_id)


def list_conversations(session, user_id: str) -> list[dict]:
    """List a user's conversations newest first with a first-question preview."""
    conversations = (
        session.query(Conversation)
        .filter(Conversation.user_id == user_id)
        .order_by(Conversation.id.desc())
        .all()
    )

    rows: list[dict] = []
    for conversation in conversations:
        first = (
            session.query(Message)
            .filter(
                Message.conversation_id == conversation.id,
                Message.role == "user",
            )
            .order_by(Message.id.asc())
            .first()
        )
        rows.append(
            {
                "id": conversation.id,
                "preview": (first.content or "")[:30] if first is not None else "",
                "has_summary": bool(conversation.summary),
                "created_at": conversation.created_at,
                "updated_at": conversation.updated_at,
            }
        )
    return rows


def list_dialog_messages(session, conversation_id: int) -> list[Message]:
    """Return user/assistant messages for a conversation, oldest first."""
    return (
        session.query(Message)
        .filter(
            Message.conversation_id == conversation_id,
            Message.role.in_(("user", "assistant")),
        )
        .order_by(Message.id.asc())
        .all()
    )


def previous_message_id(
    session, conversation_id: int, before_id: int
) -> int | None:
    """Return the id of the message immediately before ``before_id``."""
    row = (
        session.query(Message.id)
        .filter(
            Message.conversation_id == conversation_id,
            Message.id < before_id,
        )
        .order_by(Message.id.desc())
        .first()
    )
    return row[0] if row is not None else None


def update_layer1_from(
    session, conversation_id: int, boundary: int | None
) -> None:
    """Move the layer-1 anchor without touching any other column."""
    conversation = session.get(Conversation, conversation_id)
    if conversation is None:
        raise ConversationNotFound(conversation_id)
    conversation.layer1_from_msg_id = boundary
    session.commit()


def append_summary_segment(
    session,
    conversation_id: int,
    from_msg_id: int,
    upto_msg_id: int,
    content: str,
) -> ConversationSummary:
    """Append a summary segment and project it onto the conversation.

    Segments are append-only: existing rows are never rewritten. The
    conversation keeps the concatenated projection and advances
    ``summary_upto_msg_id`` to the new segment's upper bound.
    """
    max_seq = (
        session.query(func.max(ConversationSummary.seq))
        .filter(ConversationSummary.conversation_id == conversation_id)
        .scalar()
    )
    seq = (max_seq or 0) + 1

    conversation = session.get(Conversation, conversation_id)
    if conversation is None:
        raise ConversationNotFound(conversation_id)

    segment = ConversationSummary(
        conversation_id=conversation_id,
        seq=seq,
        from_msg_id=from_msg_id,
        upto_msg_id=upto_msg_id,
        content=content,
    )
    session.add(segment)

    segment_text = f"第{seq}段:{content}"
    if conversation.summary:
        conversation.summary = f"{conversation.summary}\n\n{segment_text}"
    else:
        conversation.summary = segment_text
    conversation.summary_upto_msg_id = upto_msg_id

    session.commit()
    session.refresh(segment)
    return segment


def list_summary_segments(
    session, conversation_id: int
) -> list[ConversationSummary]:
    """Return a conversation's summary segments in append order."""
    return (
        session.query(ConversationSummary)
        .filter(ConversationSummary.conversation_id == conversation_id)
        .order_by(ConversationSummary.id.asc())
        .all()
    )


def update_conversation_summary(
    session, conversation_id: int, summary: str, upto_msg_id: int
) -> None:
    """Replace the conversation's projected summary and coverage anchor."""
    conversation = session.get(Conversation, conversation_id)
    if conversation is None:
        raise ConversationNotFound(conversation_id)
    conversation.summary = summary
    conversation.summary_upto_msg_id = upto_msg_id
    session.commit()
