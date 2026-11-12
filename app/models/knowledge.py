from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    category: Mapped[str] = mapped_column(String(255), index=True)
    questions: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    section_path: Mapped[Optional[str]] = mapped_column(String(512))
    content_type: Mapped[Optional[str]] = mapped_column(String(32))
    is_key_clause: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=text("0")
    )
    prev_chunk_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("knowledge_chunks.id", ondelete="SET NULL")
    )
    next_chunk_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("knowledge_chunks.id", ondelete="SET NULL")
    )
    vector_id: Mapped[Optional[str]] = mapped_column(String(64))
    vectorize_status: Mapped[str] = mapped_column(
        Enum("pending", "done", name="knowledge_vectorize_status"),
        default="pending",
        server_default="pending",
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class QaExtractionStaging(Base):
    __tablename__ = "qa_extraction_staging"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    batch_no: Mapped[str] = mapped_column(String(64), index=True)
    source_ref: Mapped[Optional[str]] = mapped_column(String(255))
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        Enum("extracted", "kept", "discarded", name="qa_extraction_status"),
        default="extracted",
        server_default="extracted",
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
