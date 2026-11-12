from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, DateTime, Enum, ForeignKey, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("conversations.id"), index=True
    )
    role: Mapped[str] = mapped_column(Enum("user", "assistant", "tool", name="message_role"))
    content: Mapped[Optional[str]] = mapped_column(Text)
    tool_calls: Mapped[Optional[dict]] = mapped_column(JSON)
    tool_call_id: Mapped[Optional[str]] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
