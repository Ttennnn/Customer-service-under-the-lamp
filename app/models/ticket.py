from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Enum, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Ticket(Base):
    __tablename__ = "tickets"

    ticket_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("conversations.id"), index=True
    )
    description: Mapped[str] = mapped_column(Text)
    ticket_type: Mapped[str] = mapped_column(
        Enum("售后", "投诉", "咨询", name="ticket_type")
    )
    status: Mapped[str] = mapped_column(
        Enum("待处理", "已处理", name="ticket_status"), default="待处理"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
