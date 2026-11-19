from app.models import ToolAuditLog


def test_tool_audit_log_roundtrip(session):
    row = ToolAuditLog(
        conversation_id=7,
        tool_call_id="call-1",
        tool_name="query_order",
        tool_source="builtin",
        mcp_server=None,
        arguments={"order_id": "1001"},
        result_summary="订单 1001 已签收",
        status="成功",
        error_message=None,
        retry_count=0,
        duration_ms=12,
    )
    session.add(row)
    session.commit()

    loaded = session.query(ToolAuditLog).one()
    assert loaded.status == "成功"
    assert loaded.tool_source == "builtin"
    assert loaded.arguments == {"order_id": "1001"}
    assert loaded.conversation_id == 7
