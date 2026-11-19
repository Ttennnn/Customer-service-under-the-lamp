"""统一执行引擎测试。

覆盖:参数校验拦截、权限门、超时重试语义(读重试/写不重试)、
业务空结果不重试、审计失败不阻断执行、contextvar 传播到工具线程、
未知工具审计,以及 ``deny_*`` 辅助入口。
"""

import time

import pytest
from langchain.tools import tool

from app.models import ToolAuditLog
from app.tools.context import current_conversation_id
from app.tools.engine import deny_permission, deny_validation, execute, execute_tool_call
from app.tools.registry import register, reset_registry
from app.tools.spec import ToolSpec

calls: list[str] = []


@tool
def echo(value: str) -> str:
    """Echo value."""
    calls.append(value)
    return value


def _spec(tool=echo, name="echo", permission="read", timeout=0.2, classify=None):
    return ToolSpec(
        name=name,
        description="Echo.",
        json_schema=tool.args_schema.model_json_schema(),
        tool=tool,
        permission=permission,
        source="builtin",
        timeout=timeout,
        classify=classify,
    )


@pytest.fixture(autouse=True)
def _clear_calls():
    calls.clear()
    yield
    calls.clear()


def test_validation_error_audits_and_does_not_execute(session):
    spec = _spec()
    run = execute(spec, {}, conversation_id=1)
    assert run.status == "validation_error"
    assert "缺少必填参数 value" in run.content
    row = session.query(ToolAuditLog).one()
    assert row.status == "校验拦下"
    assert calls == []


def test_unconfirmed_write_is_denied_and_audited(session):
    spec = _spec(name="create_ticket", permission="write")
    run = execute(spec, {"value": "x"}, conversation_id=1)
    assert run.status == "permission_denied"
    row = session.query(ToolAuditLog).one()
    assert row.status == "权限拒绝"
    assert calls == []


def test_read_timeout_retries_once(session):
    @tool
    def slow(value: str) -> str:
        """Slow echo."""
        time.sleep(0.2)
        return value

    spec = _spec(tool=slow, timeout=0.05)
    run = execute(spec, {"value": "x"}, conversation_id=1, retries=1)
    assert run.status == "timeout"
    assert run.retry_count == 1
    assert "已重试 1 次" in run.content
    row = session.query(ToolAuditLog).one()
    assert row.status == "超时" and row.retry_count == 1


def test_write_timeout_never_retries(session):
    @tool
    def slow_write(value: str) -> str:
        """Slow write."""
        time.sleep(0.2)
        return value

    spec = _spec(tool=slow_write, name="create_ticket", permission="write", timeout=0.05)
    run = execute(spec, {"value": "x"}, conversation_id=1, confirmed=True, retries=2)
    assert run.status == "timeout"
    assert run.retry_count == 0
    assert session.query(ToolAuditLog).one().retry_count == 0


def test_empty_result_is_success_and_not_retried(session):
    @tool
    def empty(value: str) -> str:
        """Return nothing."""
        return ""

    spec = _spec(tool=empty)
    run = execute(spec, {"value": "x"}, conversation_id=1)
    assert run.status == "empty"
    assert "未查询到相关结果" in run.content
    assert session.query(ToolAuditLog).one().status == "成功"


def test_audit_failure_does_not_block_execution(session, monkeypatch):
    monkeypatch.setattr(
        "app.tools.engine.record_tool_audit",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("db down")),
    )
    run = execute(_spec(), {"value": "ok"}, conversation_id=1)
    assert run.status == "ok"
    assert run.content == "ok"


def test_conversation_id_reaches_tool_thread(session):
    @tool
    def read_cid(value: str) -> str:
        """Read cid."""
        return str(current_conversation_id.get())

    spec = _spec(tool=read_cid)
    run = execute(spec, {"value": "x"}, conversation_id=42)
    assert run.content == "42"


def test_non_transient_exception_is_error_and_not_retried(session):
    @tool
    def boom(value: str) -> str:
        """Explode."""
        raise ValueError("bad")

    spec = _spec(tool=boom, timeout=0.2)
    run = execute(spec, {"value": "x"}, conversation_id=1, retries=2)
    assert run.status == "error"
    assert "工具暂时不可用" in run.content
    assert "ValueError" in run.content
    assert run.retry_count == 0
    row = session.query(ToolAuditLog).one()
    assert row.status == "失败"


def test_unknown_tool_is_audited_and_returns_error(session):
    run = execute_tool_call(
        {"name": "no_such_tool", "args": {"x": 1}, "id": "call-9"},
        5,
        specs={"echo": _spec()},
    )
    assert run.status == "error"
    assert "未知工具" in run.content and "no_such_tool" in run.content
    row = session.query(ToolAuditLog).one()
    assert row.status == "失败"
    assert row.tool_source == "builtin"
    assert row.tool_name == "no_such_tool"
    assert row.tool_call_id == "call-9"


def test_deny_validation_audits_without_executing(session):
    spec = _spec()
    run = deny_validation(
        spec,
        {"value": "x"},
        conversation_id=1,
        tool_call_id="call-1",
        error_message="一轮只能处理一个建工单请求",
    )
    assert run.status == "validation_error"
    assert run.content == "一轮只能处理一个建工单请求"
    row = session.query(ToolAuditLog).one()
    assert row.status == "校验拦下"
    assert row.error_message == "一轮只能处理一个建工单请求"
    assert calls == []


def test_deny_permission_audits_without_executing(session):
    spec = _spec(name="create_ticket", permission="write")
    run = deny_permission(
        spec,
        {"value": "x"},
        conversation_id=1,
        tool_call_id="call-2",
        error_message="用户取消了本次建工单，请不要再发起",
    )
    assert run.status == "permission_denied"
    assert run.content == "用户取消了本次建工单，请不要再发起"
    row = session.query(ToolAuditLog).one()
    assert row.status == "权限拒绝"
    assert calls == []


def test_builtin_write_outside_allowlist_rejected_by_register():
    reset_registry()
    try:
        with pytest.raises(ValueError):
            register(_spec(permission="write"))
    finally:
        reset_registry()
