"""工具调用审计写入。

审计表不挂外键,写入使用独立 session(工具可能在子线程执行,不能复用
调用方会话)。审计永远不能反拦工具执行:这里把所有异常吞掉并记日志。
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from app.config import get_settings
from app.db import session as db_session
from app.models import ToolAuditLog

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)

ERROR_MESSAGE_MAX_CHARS = 512


def _json_safe(arguments: object) -> object:
    """Round-trip *arguments* through JSON with ``default=str``."""
    return json.loads(json.dumps(arguments, default=str, ensure_ascii=False))


def record_tool_audit(
    *,
    conversation_id: int | None,
    tool_call_id: str | None,
    tool_name: str,
    tool_source: str,
    status: str,
    mcp_server: str | None = None,
    arguments: object = None,
    result_summary: str | None = None,
    error_message: str | None = None,
    retry_count: int = 0,
    duration_ms: int | None = None,
    settings: Settings | None = None,
) -> None:
    """Write one audit row, swallowing every failure."""
    try:
        if settings is None:
            settings = get_settings()

        if result_summary is not None:
            result_summary = result_summary[: settings.tool_audit_result_max_chars]
        if arguments is not None:
            arguments = _json_safe(arguments)
        if error_message is not None:
            error_message = str(error_message)[:ERROR_MESSAGE_MAX_CHARS]

        session = db_session.get_session_factory()()
        try:
            session.add(
                ToolAuditLog(
                    conversation_id=conversation_id,
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    tool_source=tool_source,
                    mcp_server=mcp_server,
                    arguments=arguments,
                    result_summary=result_summary,
                    status=status,
                    error_message=error_message,
                    retry_count=retry_count,
                    duration_ms=duration_ms,
                )
            )
            session.commit()
        finally:
            session.close()
    except Exception:  # noqa: BLE001 - 审计失败只记日志,绝不向上抛
        logger.exception("写入工具审计失败: tool=%s status=%s", tool_name, status)
