"""工具结果分类、字段白名单、枚举翻译与截断。"""

import json

from app.tools.builtin.orders import query_order
from app.tools.formatting import (
    FIELD_ALLOWLISTS,
    classify_result,
    format_result,
)
from app.tools.registry import builtin_specs
from app.tools.spec import ToolSpec


def test_query_faq_content_is_not_json_wrapped():
    spec = builtin_specs()["query_faq"]
    content, data = format_result(spec, "[1] 章节:x\n问:y\n答:z", 2000)
    assert content.startswith("[1]")
    assert data == "[1] 章节:x\n问:y\n答:z"


def test_create_ticket_ok_and_error_classification():
    spec = builtin_specs()["create_ticket"]
    assert classify_result(spec, "T20261119001") == "ok"
    assert classify_result(spec, "创建工单失败:缺少会话上下文。") == "error"
    assert classify_result(spec, "创建工单失败:工单号生成冲突,请重试。") == "error"


def test_create_ticket_format_is_compact_json_and_data_is_raw():
    spec = builtin_specs()["create_ticket"]
    content, data = format_result(spec, "T20261119001", 2000)
    assert content == '{"ticket_no": "T20261119001"}'
    assert data == "T20261119001"


def test_mcp_found_false_is_empty():
    spec = builtin_specs()["query_order"]
    assert classify_result(spec, '{"found": false}') == "empty"


def test_query_order_null_is_empty_and_formats_safely():
    spec = builtin_specs()["query_order"]
    for raw in (None, "", "null"):
        assert classify_result(spec, raw) == "empty"
    content, data = format_result(spec, "null", 2000)
    assert data is None
    assert "未查询到相关结果" in content


def test_logistics_status_code_is_translated():
    spec = ToolSpec(
        name="query_logistics",
        description="",
        json_schema={},
        tool=query_order,
        permission="read",
        source="mcp",
        mcp_server="logistics",
    )
    content, data = format_result(
        spec, '{"tracking_no":"SF1","status_code":"TRANSIT","traces":[]}', 2000
    )
    assert "运输中" in content
    assert "TRANSIT" not in content
    assert data["status_code"] == "运输中"


def test_logistics_nested_trace_status_codes_are_translated():
    spec = ToolSpec(
        name="query_logistics",
        description="",
        json_schema={},
        tool=query_order,
        permission="read",
        source="mcp",
        mcp_server="logistics",
    )
    raw = json.dumps(
        {
            "tracking_no": "SF1",
            "traces": [
                {"desc": "已揽收", "status_code": "PICKED_UP"},
                {"desc": "运输中", "status_code": "TRANSIT"},
            ],
        }
    )
    content, data = format_result(spec, raw, 2000)

    assert "PICKED_UP" not in content
    assert "TRANSIT" not in content
    assert [trace["status_code"] for trace in data["traces"]] == ["已揽收", "运输中"]
    assert [trace["desc"] for trace in data["traces"]] == ["已揽收", "运输中"]


def test_known_tool_filters_fields_outside_allowlist():
    spec = builtin_specs()["query_order"]
    raw = json.dumps(
        {
            "order_id": "1001",
            "status": "已签收",
            "internal_secret": "do-not-leak",
        }
    )
    content, data = format_result(spec, raw, 2000)
    assert "internal_secret" not in content
    assert "do-not-leak" not in content
    assert json.loads(content) == {"order_id": "1001", "status": "已签收"}
    assert "internal_secret" not in data


def test_unknown_tool_is_serialized_without_field_filtering():
    spec = ToolSpec(
        name="some_mcp_tool",
        description="",
        json_schema={},
        tool=query_order,
        permission="read",
        source="mcp",
    )
    payload = {"any_field": "keep-me", "nested": {"n": 1}}
    raw = json.dumps(payload)
    content, data = format_result(spec, raw, 2000)
    assert content == raw
    assert data == raw

    content, data = format_result(spec, payload, 2000)
    assert json.loads(content) == payload
    assert data == payload


def test_non_json_string_and_object_are_handled():
    spec = ToolSpec(
        name="some_mcp_tool",
        description="",
        json_schema={},
        tool=query_order,
        permission="read",
        source="mcp",
    )
    content, data = format_result(spec, "plain text", 2000)
    assert content == "plain text"
    assert data == "plain text"

    content, data = format_result(spec, {"n": 1}, 2000)
    assert json.loads(content) == {"n": 1}
    assert data == {"n": 1}


def test_faq_is_returned_verbatim_without_truncation():
    spec = builtin_specs()["query_faq"]
    raw = "[1] 章节:x\n答:" + "答" * 50
    content, data = format_result(spec, raw, 10)
    assert content == raw
    assert data == raw


def test_truncates_and_appends_marker():
    spec = ToolSpec(
        name="some_mcp_tool",
        description="",
        json_schema={},
        tool=query_order,
        permission="read",
        source="mcp",
    )
    raw = "答" * 50
    content, data = format_result(spec, raw, 10)
    assert content == "答" * 10 + "…（结果已截断）"
    assert data == raw


def test_allowlists_match_design():
    assert FIELD_ALLOWLISTS["query_order"] == (
        "order_id",
        "items",
        "amount",
        "status",
        "tracking_no",
        "created_at",
        "signed_at",
    )
    assert FIELD_ALLOWLISTS["query_logistics"] == (
        "tracking_no",
        "carrier",
        "status_code",
        "traces",
    )
