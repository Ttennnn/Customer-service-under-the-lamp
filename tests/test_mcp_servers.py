"""MCP server 工具函数测试。

只测模块级纯函数(不启动进程、不访问网络):mock 数据稳定、
内部枚举码合法、必需字段齐全。
"""

import json

from mcp_servers.aftersales_server import query_return_status, query_warranty
from mcp_servers.logistics_server import query_logistics

STATUS_CODES = {"PICKED_UP", "TRANSIT", "ARRIVED", "DELIVERING", "DELIVERED"}
WARRANTY_STATUSES = {"IN_WARRANTY", "OUT_OF_WARRANTY", "UNKNOWN"}
RETURN_STATUSES = {"APPLIED", "PICKING", "IN_TRANSIT", "RECEIVED", "REFUNDING", "DONE"}


def test_logistics_mock_has_internal_status_code(monkeypatch):
    monkeypatch.setenv("MOCK_DELAY_SECONDS", "0")
    data = json.loads(query_logistics("SF1234567890"))
    assert data["tracking_no"] == "SF1234567890"
    assert data["status_code"] in STATUS_CODES
    assert data["traces"]


def test_logistics_mock_is_stable_by_tracking_no(monkeypatch):
    monkeypatch.setenv("MOCK_DELAY_SECONDS", "0")
    assert json.loads(query_logistics("SF1234567890")) == json.loads(
        query_logistics("SF1234567890")
    )


def test_aftersales_tools_are_stable_by_order(monkeypatch):
    monkeypatch.setenv("MOCK_DELAY_SECONDS", "0")
    a = json.loads(query_warranty("1001"))
    b = json.loads(query_warranty("1001"))
    assert a == b
    assert a["warranty_status"] in WARRANTY_STATUSES
    assert json.loads(query_return_status("1001"))["status"] in RETURN_STATUSES


def test_warranty_fields(monkeypatch):
    monkeypatch.setenv("MOCK_DELAY_SECONDS", "0")
    data = json.loads(query_warranty("1001"))
    assert data["order_id"] == "1001"
    assert set(data) >= {
        "order_id",
        "product_name",
        "warranty_status",
        "purchase_date",
        "expire_date",
        "coverage",
    }


def test_return_status_fields(monkeypatch):
    monkeypatch.setenv("MOCK_DELAY_SECONDS", "0")
    data = json.loads(query_return_status("1001"))
    assert data["order_id"] == "1001"
    assert set(data) >= {"order_id", "return_no", "status", "progress", "updated_at"}


def test_mock_delay_env_is_read_at_call_time(monkeypatch):
    monkeypatch.setenv("MOCK_DELAY_SECONDS", "0")
    assert json.loads(query_logistics("SF1"))["tracking_no"] == "SF1"
    monkeypatch.setenv("MOCK_DELAY_SECONDS", "0.01")
    assert json.loads(query_logistics("SF1"))["tracking_no"] == "SF1"
