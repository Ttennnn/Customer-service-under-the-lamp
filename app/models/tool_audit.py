from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Enum, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class ToolAuditLog(Base):
    __tablename__ = "tool_audit_logs"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    conversation_id: Mapped[int | None] = mapped_column(BigInteger, index=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(64))
    tool_name: Mapped[str] = mapped_column(String(128), index=True)
    tool_source: Mapped[str] = mapped_column(Enum("builtin", "mcp", name="tool_source"))
    mcp_server: Mapped[str | None] = mapped_column(String(64))
    arguments: Mapped[dict | None] = mapped_column(JSON)
    result_summary: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        Enum("成功", "失败", "超时", "校验拦下", "权限拒绝", name="tool_audit_status"),
        index=True,
    )
    error_message: Mapped[str | None] = mapped_column(String(512))
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
