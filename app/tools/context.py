"""工具执行的会话上下文。

用 ``contextvars.ContextVar`` 在请求/会话执行链中传递「当前会话 id」,
供需要感知会话上下文的工具(如 ``create_ticket``)读取。默认值为 ``None``,
表示当前没有关联的会话。
"""

from contextvars import ContextVar

current_conversation_id: ContextVar[int | None] = ContextVar(
    "current_conversation_id", default=None
)
