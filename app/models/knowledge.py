from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    JSON,
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


class LowConfidenceQuestion(Base):
    __tablename__ = "low_confidence_questions"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    conversation_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("conversations.id")
    )
    raw_question: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(
        Enum(
            "retrieval_low_conf",
            "self_check",
            "user_feedback",
            # ch05 confidence gate: genuine knowledge gaps only; infra and
            # rerank failures never enter the pool.
            "confidence_gate",
            name="low_confidence_source",
        ),
        index=True,
    )
    reason: Mapped[Optional[str]] = mapped_column(Text)
    # ch09: retrieval snapshot at pool time (Top N chunks) and the review_queue
    # gap this raw question was merged into during dedup.
    retrieved_chunks: Mapped[Optional[list]] = mapped_column(JSON)
    matched_review_id: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        ForeignKey("review_queue.id", ondelete="SET NULL"),
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), index=True
    )


class FaithCase(Base):
    __tablename__ = "faith_cases"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    eval_id: Mapped[str] = mapped_column(String(16), unique=True)
    bucket: Mapped[str] = mapped_column(String(24))
    query: Mapped[str] = mapped_column(String(512))
    strategy: Mapped[str] = mapped_column(
        String(24), default="hybrid_rerank", server_default="hybrid_rerank"
    )
    answer: Mapped[str] = mapped_column(Text)
    reason: Mapped[str] = mapped_column(Text)
    citations: Mapped[Optional[list]] = mapped_column(JSON)
    judge_model: Mapped[Optional[str]] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(
        Enum("未解决", "已解决", "无需解决", name="faith_case_status"),
        default="未解决",
        server_default="未解决",
        index=True,
    )
    seen_count: Mapped[int] = mapped_column(
        Integer, default=1, server_default="1"
    )
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now()
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), index=True
    )
    resolution: Mapped[Optional[str]] = mapped_column(String(300))
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
