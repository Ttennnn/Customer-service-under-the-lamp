"""工具权限策略。

本侧唯一真相是 ``WRITE_TOOLS``:内置工具即便在规格里声明了 write,只要
名字不在白名单里也不能被当作写操作;MCP 工具一律按只读处理,忽略 Server
的 ``destructiveHint`` 等元数据。
"""

from __future__ import annotations

from app.tools.spec import ToolSpec

WRITE_TOOLS = frozenset({"create_ticket"})


def effective_permission(spec: ToolSpec) -> str:
    """Return the permission the system actually enforces for *spec*."""
    if spec.source == "mcp":
        return "read"
    return "write" if spec.name in WRITE_TOOLS else "read"
