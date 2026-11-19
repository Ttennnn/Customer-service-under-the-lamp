"""工具审计写入测试。"""

from app.config import Settings
from app.models import ToolAuditLog
from app.tools.audit import record_tool_audit


def _record(**overrides):
    kwargs = {
        "conversation_id": 7,
        "tool_call_id": "call-1",
        "tool_name": "query_order",
        "tool_source": "builtin",
        "status": "成功",
        "arguments": {"order_id": "1001"},
        "result_summary": "订单 1001 已签收",
        "error_message": None,
        "retry_count": 0,
        "duration_ms": 12,
    }
    kwargs.update(overrides)
    record_tool_audit(**kwargs)


def test_record_tool_audit_writes_row(session):
    _record()

    row = session.query(ToolAuditLog).one()
    assert row.conversation_id == 7
    assert row.tool_call_id == "call-1"
    assert row.tool_name == "query_order"
    assert row.tool_source == "builtin"
    assert row.status == "成功"
    assert row.arguments == {"order_id": "1001"}
    assert row.result_summary == "订单 1001 已签收"
    assert row.retry_count == 0
    assert row.duration_ms == 12


def test_result_summary_truncated_to_setting(session):
    settings = Settings(
        _env_file=None, openai_api_key="test", tool_audit_result_max_chars=10
    )
    _record(result_summary="x" * 100, settings=settings)

    row = session.query(ToolAuditLog).one()
    assert len(row.result_summary) == 10


def test_arguments_with_non_json_values_are_serialized(session):
    _record(arguments={"when": object()})

    row = session.query(ToolAuditLog).one()
    assert row.arguments["when"].startswith("<object object at")


def test_record_tool_audit_swallows_db_errors(monkeypatch):
    monkeypatch.setattr(
        "app.db.session.get_session_factory",
        lambda: (_ for _ in ()).throw(RuntimeError("db down")),
    )
    # 不抛异常即通过。
    _record()
