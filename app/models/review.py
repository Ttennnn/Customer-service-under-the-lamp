from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum,
    Integer,
    JSON,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class ReviewQueue(Base):
    """Standardized knowledge gap awaiting human review (ch09 flywheel)."""

    __tablename__ = "review_queue"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    normalized_question: Mapped[str] = mapped_column(String(512))
    ai_suggested_answer: Mapped[Optional[str]] = mapped_column(Text)
    occurrence_count: Mapped[int] = mapped_column(
        Integer, default=1, server_default=text("1")
    )
    review_status: Mapped[str] = mapped_column(
        Enum("待审", "通过", "驳回", name="review_status"),
        default="待审",
        server_default="待审",
        index=True,
    )
    approved_answer: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class EvalRun(Base):
    """One automated evaluation round, kept to show metric trends."""

    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    triggered_by: Mapped[str] = mapped_column(
        Enum("定时", "手动", name="eval_triggered_by"),
        default="定时",
        server_default="定时",
    )
    dataset_size: Mapped[int] = mapped_column(Integer)
    metrics: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), index=True
    )
