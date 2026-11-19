"""兼容 shim:工单工具已迁移到 :mod:`app.tools.builtin.tickets`。

保留本模块是为了让 ``app.api.ch05``、``app.api.ch06``、
``app.tools.registry`` 以及既有测试的导入路径继续可用。
"""

from app.tools.builtin.tickets import TICKET_NO_PATTERN, create_ticket

__all__ = ["TICKET_NO_PATTERN", "create_ticket"]
