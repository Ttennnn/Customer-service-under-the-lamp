"""统一工具执行引擎。

所有工具调用走同一条管道:校验 → 权限门 → 执行(超时/重试) →
结果分诊 → 格式化 → 审计。对外保持同步:内置工具在 daemon 子线程执行,
主线程用 ``queue.Queue.get(timeout=...)`` 等待;开线程前把
``current_conversation_id`` 写入 contextvar 再 ``copy_context()``,保证
工具线程能读到会话上下文。MCP 工具走 :func:`run_async`,把
``spec.tool.ainvoke`` 投递到常驻事件循环线程,超时/重试语义保持一致。

任何路径都返回 :class:`ToolRun`,不向调用方抛异常。审计失败也在引擎层
再兜一层 try/except(与 :func:`record_tool_audit` 内部的兜底构成双重防线)。
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import queue
import threading
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING

import httpx

from app.config import get_settings
from app.tools.async_bridge import run_async
from app.tools.audit import record_tool_audit
from app.tools.context import current_conversation_id
from app.tools.formatting import classify_result, format_result
from app.tools.mcp_client import normalize_mcp_result
from app.tools.policy import effective_permission
from app.tools.registry import get_all_specs
from app.tools.spec import ToolRun, ToolSpec
from app.tools.validation import validate_arguments

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)

# 只有超时和网络/传输类异常值得重试;业务异常重试也不会变好。
_TRANSIENT_ERRORS = (
    TimeoutError,
    ConnectionError,
    OSError,
    httpx.TransportError,
    asyncio.TimeoutError,
)

VALIDATION_SUFFIX = "，请修正参数或向用户追问"


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _audit(
    spec: ToolSpec,
    arguments: object,
    *,
    conversation_id: int | None,
    tool_call_id: str | None,
    status: str,
    content: str | None = None,
    error_message: str | None = None,
    retry_count: int = 0,
    duration_ms: int | None = None,
    settings: Settings | None = None,
) -> None:
    """Best-effort audit; never lets a failure escape the engine."""
    try:
        record_tool_audit(
            conversation_id=conversation_id,
            tool_call_id=tool_call_id,
            tool_name=spec.name,
            tool_source=spec.source,
            mcp_server=spec.mcp_server,
            arguments=arguments,
            result_summary=content,
            status=status,
            error_message=error_message,
            retry_count=retry_count,
            duration_ms=duration_ms,
            settings=settings,
        )
    except Exception:  # noqa: BLE001 - 双重防线,审计不能反拦执行
        logger.exception("审计调用失败: tool=%s status=%s", spec.name, status)


def _invoke_sync(
    spec: ToolSpec, arguments: object, timeout: float, ctx: contextvars.Context
) -> object:
    """Run ``spec.tool.invoke`` in a daemon thread, raising TimeoutError on expiry."""
    result_q: queue.Queue = queue.Queue(maxsize=1)

    def _run() -> None:
        try:
            result_q.put(("ok", ctx.run(lambda: spec.tool.invoke(arguments))))
        except Exception as exc:  # noqa: BLE001 - 结果交回主线程分诊
            result_q.put(("err", exc))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    try:
        status, payload = result_q.get(timeout=timeout)
    except queue.Empty as exc:
        # daemon 线程可能还在跑;写操作不重试,这是「超时未必没执行」的防线。
        raise TimeoutError(f"工具 {spec.name} 执行超时") from exc

    if status == "err":
        raise payload
    return payload


def _invoke(
    spec: ToolSpec, arguments: object, timeout: float, ctx: contextvars.Context
) -> object:
    """Dispatch to the async bridge for MCP tools, sync thread otherwise."""
    if spec.source == "mcp":
        raw = run_async(spec.tool.ainvoke(arguments), timeout=timeout)
        # MCP 适配器返回 content-block 列表,先还原成内层 JSON 再分诊/格式化。
        return normalize_mcp_result(raw)
    return _invoke_sync(spec, arguments, timeout, ctx)


def _execute_with_retries(
    spec: ToolSpec,
    arguments: object,
    *,
    timeout: float,
    max_retries: int,
) -> tuple[object, str | None, Exception | None, int]:
    """Return ``(raw, failure_kind, exc, retry_count)``.

    ``failure_kind`` is ``None`` on success, ``"timeout"`` or ``"error"``
    otherwise. ``retry_count`` is always ``attempts - 1`` of the final try.
    """
    attempts = max_retries + 1
    for attempt in range(attempts):
        # 每次尝试独立拷贝上下文:超时后的 daemon 线程仍在跑,同一个
        # Context 不能被两个线程并发进入。
        ctx = contextvars.copy_context()
        try:
            raw = _invoke(spec, arguments, timeout, ctx)
        except TimeoutError:
            if attempt < attempts - 1:
                time.sleep(0.2 * attempt)
                continue
            return None, "timeout", None, attempt
        except Exception as exc:  # noqa: BLE001 - 分诊后如实回灌
            if isinstance(exc, _TRANSIENT_ERRORS) and attempt < attempts - 1:
                time.sleep(0.2 * attempt)
                continue
            return None, "error", exc, attempt
        return raw, None, None, attempt

    # ``range`` 至少执行一次,理论不可达;保留兜底以满足「绝不抛异常」。
    return None, "error", RuntimeError("no attempt executed"), max_retries


def execute(
    spec: ToolSpec,
    arguments: object,
    *,
    conversation_id: int | None,
    tool_call_id: str | None = None,
    confirmed: bool = False,
    settings: Settings | None = None,
    retries: int | None = None,
    timeout: float | None = None,
) -> ToolRun:
    """Execute *spec* through the unified pipeline.

    ``timeout`` overrides the timeout resolved from the spec/settings; it lets
    a caller (e.g. legacy ``/chat``) keep its historical per-tool budget.
    """
    settings = settings or get_settings()
    started = time.monotonic()

    # 1. 参数校验:不合法立即返回,不执行、不进入权限门。
    errors = validate_arguments(spec.json_schema, arguments)
    if errors:
        content = "；".join(errors)
        if VALIDATION_SUFFIX.strip("，") not in content:
            content += VALIDATION_SUFFIX
        duration = _elapsed_ms(started)
        _audit(
            spec,
            arguments,
            conversation_id=conversation_id,
            tool_call_id=tool_call_id,
            status="校验拦下",
            content=content,
            error_message=content,
            duration_ms=duration,
            settings=settings,
        )
        return ToolRun(
            status="validation_error",
            content=content,
            duration_ms=duration,
            error_message=content,
        )

    # 2. 权限门:写操作必须由可信路径显式 confirmed=True。
    permission = effective_permission(spec)
    if permission == "write" and not confirmed:
        content = f"该操作需要用户确认：{spec.name}"
        duration = _elapsed_ms(started)
        _audit(
            spec,
            arguments,
            conversation_id=conversation_id,
            tool_call_id=tool_call_id,
            status="权限拒绝",
            content=content,
            error_message=content,
            duration_ms=duration,
            settings=settings,
        )
        return ToolRun(
            status="permission_denied",
            content=content,
            duration_ms=duration,
            error_message=content,
        )

    # 3. 执行:写操作恒 0 次重试;只读默认 settings.tool_max_retries。
    is_write = permission == "write"
    if is_write:
        max_retries = 0
    elif retries is not None:
        max_retries = max(0, retries)
    else:
        max_retries = max(0, settings.tool_max_retries)

    if timeout is None:
        timeout = spec.timeout or (
            settings.mcp_tool_timeout_s
            if spec.source == "mcp"
            else settings.tool_default_timeout_s
        )

    token = current_conversation_id.set(conversation_id)
    try:
        raw, failure, exc, retry_count = _execute_with_retries(
            spec, arguments, timeout=timeout, max_retries=max_retries
        )
    finally:
        current_conversation_id.reset(token)
    duration = _elapsed_ms(started)

    if failure == "timeout":
        content = f"工具执行超时：{spec.name}（已重试 {retry_count} 次）"
        _audit(
            spec,
            arguments,
            conversation_id=conversation_id,
            tool_call_id=tool_call_id,
            status="超时",
            content=content,
            error_message=content,
            retry_count=retry_count,
            duration_ms=duration,
            settings=settings,
        )
        return ToolRun(
            status="timeout",
            content=content,
            duration_ms=duration,
            retry_count=retry_count,
            error_message=content,
        )

    if failure == "error":
        content = f"工具暂时不可用：{spec.name}（{type(exc).__name__}）"
        _audit(
            spec,
            arguments,
            conversation_id=conversation_id,
            tool_call_id=tool_call_id,
            status="失败",
            content=content,
            error_message=content,
            retry_count=retry_count,
            duration_ms=duration,
            settings=settings,
        )
        return ToolRun(
            status="error",
            content=content,
            duration_ms=duration,
            retry_count=retry_count,
            error_message=content,
        )

    # 4. 分诊 + 格式化。
    classifier = spec.classify or (lambda value: classify_result(spec, value))
    kind = classifier(raw)

    if kind == "empty":
        content = f"未查询到相关结果：{spec.name}"
        _audit(
            spec,
            arguments,
            conversation_id=conversation_id,
            tool_call_id=tool_call_id,
            status="成功",
            content=content,
            retry_count=retry_count,
            duration_ms=duration,
            settings=settings,
        )
        return ToolRun(
            status="empty",
            content=content,
            duration_ms=duration,
            retry_count=retry_count,
        )

    if kind == "error":
        content = f"工具执行失败：{raw}"
        _audit(
            spec,
            arguments,
            conversation_id=conversation_id,
            tool_call_id=tool_call_id,
            status="失败",
            content=content,
            error_message=content,
            retry_count=retry_count,
            duration_ms=duration,
            settings=settings,
        )
        return ToolRun(
            status="error",
            content=content,
            data=raw,
            duration_ms=duration,
            retry_count=retry_count,
            error_message=content,
        )

    if spec.format_result is not None:
        raw = spec.format_result(raw)
    content, data = format_result(spec, raw, settings.agent_tool_result_max_chars)
    _audit(
        spec,
        arguments,
        conversation_id=conversation_id,
        tool_call_id=tool_call_id,
        status="成功",
        content=content,
        retry_count=retry_count,
        duration_ms=duration,
        settings=settings,
    )
    return ToolRun(
        status="ok",
        content=content,
        data=data,
        duration_ms=duration,
        retry_count=retry_count,
    )


def _call_field(tool_call: object, key: str, default=None):
    if isinstance(tool_call, Mapping):
        return tool_call.get(key, default)
    return getattr(tool_call, key, default)


def execute_tool_call(
    tool_call: dict,
    conversation_id: int | None,
    *,
    confirmed: bool = False,
    specs: dict[str, ToolSpec] | None = None,
    settings: Settings | None = None,
    retries: int | None = None,
    timeout: float | None = None,
) -> ToolRun:
    """Execute a model-emitted tool call dict through the unified pipeline.

    ``retries`` and ``timeout`` override the spec/settings defaults and are
    forwarded to :func:`execute`, so callers can pin a per-call budget.
    """
    settings = settings or get_settings()
    started = time.monotonic()

    name = _call_field(tool_call, "name")
    arguments = _call_field(tool_call, "args", {}) or {}
    tool_call_id = _call_field(tool_call, "id")

    resolved = specs if specs is not None else get_all_specs(settings)
    spec = resolved.get(name)

    if spec is None:
        content = f"未知工具：{name}"
        duration = _elapsed_ms(started)
        try:
            record_tool_audit(
                conversation_id=conversation_id,
                tool_call_id=tool_call_id,
                tool_name=str(name),
                tool_source="builtin",
                mcp_server=None,
                arguments=arguments,
                result_summary=content,
                status="失败",
                error_message=content,
                retry_count=0,
                duration_ms=duration,
                settings=settings,
            )
        except Exception:  # noqa: BLE001 - 审计失败不阻断
            logger.exception("未知工具审计失败: tool=%s", name)
        return ToolRun(
            status="error",
            content=content,
            duration_ms=duration,
            error_message=content,
        )

    return execute(
        spec,
        arguments,
        conversation_id=conversation_id,
        tool_call_id=tool_call_id,
        confirmed=confirmed,
        settings=settings,
        retries=retries,
        timeout=timeout,
    )


def deny_validation(
    spec: ToolSpec,
    arguments: object,
    *,
    conversation_id: int | None,
    tool_call_id: str | None,
    error_message: str,
) -> ToolRun:
    """Audit a ``校验拦下`` outcome and return it without executing."""
    started = time.monotonic()
    duration = _elapsed_ms(started)
    _audit(
        spec,
        arguments,
        conversation_id=conversation_id,
        tool_call_id=tool_call_id,
        status="校验拦下",
        content=error_message,
        error_message=error_message,
        duration_ms=duration,
    )
    return ToolRun(
        status="validation_error",
        content=error_message,
        duration_ms=duration,
        error_message=error_message,
    )


def deny_permission(
    spec: ToolSpec,
    arguments: object,
    *,
    conversation_id: int | None,
    tool_call_id: str | None,
    error_message: str,
) -> ToolRun:
    """Audit a ``权限拒绝`` outcome and return it without executing."""
    started = time.monotonic()
    duration = _elapsed_ms(started)
    _audit(
        spec,
        arguments,
        conversation_id=conversation_id,
        tool_call_id=tool_call_id,
        status="权限拒绝",
        content=error_message,
        error_message=error_message,
        duration_ms=duration,
    )
    return ToolRun(
        status="permission_denied",
        content=error_message,
        duration_ms=duration,
        error_message=error_message,
    )
