from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class TopicClassification(Base):
    """One side-channel multi-label classification of a low-confidence question."""

    __tablename__ = "topic_classifications"
    __table_args__ = (
        UniqueConstraint("question_id", name="uk_question_id"),
    )

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    question_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        ForeignKey("low_confidence_questions.id"),
        nullable=False,
    )
    labels: Mapped[list] = mapped_column(JSON, nullable=False)
    classified_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
