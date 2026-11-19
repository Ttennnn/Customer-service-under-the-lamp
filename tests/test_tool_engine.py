"""统一执行引擎测试。

覆盖:参数校验拦截、权限门、超时重试语义(读重试/写不重试)、
业务空结果不重试、审计失败不阻断执行、contextvar 传播到工具线程、
未知工具审计,以及 ``deny_*`` 辅助入口。
"""

import asyncio
import json
import time

import pytest
from langchain.tools import tool
from langchain_core.tools import StructuredTool

from app.config import Settings
from app.models import ToolAuditLog
from app.tools.context import current_conversation_id
from app.tools.engine import deny_permission, deny_validation, execute, execute_tool_call
from app.tools.mcp_client import _spec_from_mcp_tool
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


# --- MCP async branch ---

MCP_SCHEMA = {
    "type": "object",
    "properties": {"tracking_no": {"type": "string"}},
    "required": ["tracking_no"],
}


def _mcp_spec(tool_obj, timeout=2.0):
    return ToolSpec(
        name="query_logistics",
        description="MCP logistics.",
        json_schema=MCP_SCHEMA,
        tool=tool_obj,
        permission="read",
        source="mcp",
        mcp_server="logistics",
        timeout=timeout,
    )


class _AsyncMCPTool:
    """Returns the adapter's real shape: a list of text content blocks."""

    def __init__(self, payload=None):
        self.calls = []
        self.thread_name = None
        self.payload = payload or {
            "tracking_no": "SF1",
            "carrier": "顺丰速运",
            "status_code": "TRANSIT",
            "traces": [{"desc": "运输中"}],
        }

    async def ainvoke(self, arguments):
        import threading

        self.calls.append(arguments)
        self.thread_name = threading.current_thread().name
        payload = dict(self.payload)
        if "tracking_no" in payload:
            payload["tracking_no"] = arguments["tracking_no"]
        return [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]


def _adapter_structured_tool(payload: str = '{"found": false}'):
    """Build a tool the way langchain-mcp-adapters does (dict full inputSchema)."""

    async def _call(**kwargs):
        return ([{"type": "text", "text": payload}], None)

    return StructuredTool(
        name="query_logistics",
        description="",
        args_schema=MCP_SCHEMA,
        coroutine=_call,
        response_format="content_and_artifact",
    )


class _SlowAsyncMCPTool:
    async def ainvoke(self, arguments):
        await asyncio.sleep(5)
        return "{}"


def test_mcp_tool_goes_through_async_bridge(session, monkeypatch):
    monkeypatch.setattr(
        "app.tools.engine._invoke_sync",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("sync path")),
    )
    tool_obj = _AsyncMCPTool()
    run = execute(_mcp_spec(tool_obj), {"tracking_no": "SF1"}, conversation_id=1)

    assert run.status == "ok"
    assert tool_obj.calls == [{"tracking_no": "SF1"}]
    assert tool_obj.thread_name != "MainThread"
    assert run.data["status_code"] == "运输中"
    assert session.query(ToolAuditLog).one().tool_source == "mcp"


def test_mcp_read_timeout_is_retried(session):
    spec = _mcp_spec(_SlowAsyncMCPTool(), timeout=0.05)
    run = execute(spec, {"tracking_no": "SF1"}, conversation_id=1, retries=1)
    assert run.status == "timeout"
    assert run.retry_count == 1
    row = session.query(ToolAuditLog).one()
    assert row.status == "超时" and row.retry_count == 1


def test_mcp_content_block_found_false_is_empty(session):
    run = execute(
        _mcp_spec(_AsyncMCPTool({"found": False})),
        {"tracking_no": "SF1"},
        conversation_id=1,
    )
    assert run.status == "empty"
    assert "未查询到相关结果" in run.content


def test_mcp_adapter_schema_rejects_missing_and_wrong_type(session):
    settings = Settings(_env_file=None, openai_api_key="test")
    spec = _spec_from_mcp_tool(_adapter_structured_tool(), "logistics", settings)
    assert spec.json_schema["type"] == "object"
    assert spec.json_schema["required"] == ["tracking_no"]

    missing = execute(spec, {}, conversation_id=1)
    assert missing.status == "validation_error"
    assert "缺少必填参数 tracking_no" in missing.content

    wrong_type = execute(spec, {"tracking_no": 123}, conversation_id=1)
    assert wrong_type.status == "validation_error"
    assert "类型应为 string" in wrong_type.content


def test_real_adapter_structured_tool_result_is_normalized(session):
    settings = Settings(_env_file=None, openai_api_key="test")
    spec = _spec_from_mcp_tool(_adapter_structured_tool(), "logistics", settings)
    run = execute(spec, {"tracking_no": "SF1"}, conversation_id=1)
    assert run.status == "empty"
