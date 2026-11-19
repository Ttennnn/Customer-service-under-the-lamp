"""同步代码调用异步协程的桥。

工具执行引擎是同步的,而 MCP 工具只提供异步接口。这里用一个常驻的
daemon 事件循环线程承载所有 MCP 协程:主线程通过
``asyncio.run_coroutine_threadsafe`` 投递并在 ``future.result(timeout)``
上等待,从而复用引擎既有的超时/重试语义。

同一时刻只存在一个 loop 线程;``shutdown_async_bridge`` 供测试与
应用 lifespan 收尾。从 loop 线程自身调用 ``run_async`` 会直接报错,
避免 ``future.result()`` 等待自己造成死锁。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")

_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_lock = threading.Lock()


def _ensure_loop() -> asyncio.AbstractEventLoop:
    """Return the bridge loop, starting its daemon thread on first use."""
    global _loop, _thread

    with _lock:
        if _loop is not None and _loop.is_running():
            return _loop

        loop = asyncio.new_event_loop()
        started = threading.Event()

        def _run() -> None:
            asyncio.set_event_loop(loop)
            started.set()
            try:
                loop.run_forever()
            finally:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                loop.close()

        thread = threading.Thread(target=_run, name="tool-async-bridge", daemon=True)
        # 先登记再启动,run_async 的 loop 线程自检才能命中。
        _loop = loop
        _thread = thread
        thread.start()
        started.wait()
        return loop


def run_async(coro: Coroutine[Any, Any, T], timeout: float | None = None) -> T:
    """Run *coro* on the bridge loop and block for its result.

    Raises builtin :class:`TimeoutError` when *timeout* expires; the pending
    future is cancelled first so the coroutine does not keep running.
    """
    if _thread is not None and threading.current_thread() is _thread:
        coro.close()
        raise RuntimeError(
            "run_async() cannot be called from the async bridge event loop thread; "
            "it would deadlock"
        )

    loop = _ensure_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result(timeout)
    except concurrent.futures.TimeoutError:
        if not future.cancel():
            # 恰好在超时边界完成:取走结果/异常,避免 "never retrieved" 警告。
            future.exception()
        raise TimeoutError(f"异步调用超时（{timeout}s）") from None


def shutdown_async_bridge() -> None:
    """Stop the bridge loop and join its thread (idempotent)."""
    global _loop, _thread

    with _lock:
        loop, thread = _loop, _thread
        _loop = None
        _thread = None

    if loop is not None and loop.is_running():
        try:
            loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:
            # loop 已在并发关闭路径中停止。
            pass
    if thread is not None and thread is not threading.current_thread():
        thread.join(timeout=5)
