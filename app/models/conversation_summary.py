from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class ConversationSummary(Base):
    __tablename__ = "conversation_summaries"
    __table_args__ = (
        UniqueConstraint("conversation_id", "seq", name="uk_conv_seq"),
        Index("idx_conv_upto", "conversation_id", "upto_msg_id"),
    )

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    conversation_id: Mapped[int] = mapped_column(BigInteger)
    seq: Mapped[int] = mapped_column(Integer)
    from_msg_id: Mapped[int] = mapped_column(BigInteger)
    upto_msg_id: Mapped[int] = mapped_column(BigInteger)
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
