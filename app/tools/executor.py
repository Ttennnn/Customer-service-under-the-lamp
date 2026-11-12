"""工具执行器。

所有业务工具统一经由 ``execute_tool`` 执行,提供以下保障:

1. **参数校验**:执行前先用 ``tool.args_schema`` 校验参数,非法参数立即返回
   错误串,不进入执行流程。
2. **超时控制**:在 daemon 子线程中执行 ``tool.invoke(args)``,主线程用
   ``queue.Queue.get(timeout=...)`` 等待结果;超时后 daemon 线程不会阻塞进程退出
   (不使用 ``ThreadPoolExecutor``,其非 daemon 线程会拖住进程)。
3. **contextvar 传播**:开线程前 ``contextvars.copy_context()``,线程内
   ``ctx.run(...)``,把调用方的 contextvar(如 ``current_conversation_id``)
   传播进子线程,否则 ``create_ticket`` 在子线程里永远读到 ``None``。
4. **重试与兜底**:超时或异常最多重试 ``retries`` 次,最终仍失败返回
   「工具执行失败」串。

任何路径都返回 ``str``,绝不抛异常。注意:裸参数 dict 直接传入
``tool.invoke`` 时,langchain-core 1.x 会返回裸 ``str``(不包装成
``ToolMessage``),因此直接返回该字符串,不取 ``.content``。
"""

import contextvars
import queue
import threading

from langchain_core.tools import BaseTool
from pydantic import ValidationError


def execute_tool(
    tool: BaseTool,
    args: dict,
    timeout_s: float = 5,
    retries: int = 1,
) -> str:
    """执行工具并返回字符串结果。

    先校验参数,再在 daemon 子线程执行;超时或异常重试 ``retries`` 次,
    仍失败返回「工具执行失败」。
    """
    # 1. 参数校验(只校验一次,不进入执行)。
    schema = getattr(tool, "args_schema", None)
    if schema is not None:
        try:
            schema.model_validate(args)
        except ValidationError as exc:
            return f"参数校验错误:{exc}"

    # 2. 执行 + 超时/异常重试。
    attempts = retries + 1
    for _ in range(attempts):
        result_q: queue.Queue = queue.Queue()
        # 在开线程前拷贝当前上下文,以便把 contextvar 传播进子线程。
        ctx = contextvars.copy_context()

        def _run() -> None:
            try:
                result_q.put(("ok", ctx.run(lambda: tool.invoke(args))))
            except Exception as exc:  # noqa: BLE001 - 兜底,不向外抛
                result_q.put(("err", exc))

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        try:
            status, payload = result_q.get(timeout=timeout_s)
        except queue.Empty:
            # 超时:daemon 线程仍在后台跑,主流程进入下一次重试。
            continue

        if status == "ok":
            return str(payload)

    # 3. 重试耗尽仍失败 → 兜底错误串。
    return "工具执行失败"
