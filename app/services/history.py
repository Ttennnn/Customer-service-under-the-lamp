"""DB-backed conversation history: persistence and reconstruction.

Stores conversation messages in the ``messages`` table and rebuilds
``langchain_core`` message objects from them for prompt assembly.
"""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from app.models import Conversation, Message


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
    return messages
