"""兼容 shim:FAQ 工具已迁移到 :mod:`app.tools.builtin.faq`。

保留本模块是为了让 ``app.services.chat``、``app.agent.graph``、
``app.jobs.eval_rag`` 以及既有测试的导入路径继续可用。
"""

from app.tools.builtin.faq import (
    ERROR_TEXT,
    MISS_TEXT,
    order_for_prompt,
    query_faq,
)

__all__ = ["ERROR_TEXT", "MISS_TEXT", "order_for_prompt", "query_faq"]
