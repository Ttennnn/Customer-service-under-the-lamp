import json
import re

from app.tools.builtin.orders import query_order, query_product
from app.tools.business import DEMO_ORDERS, list_user_orders, order_snapshot


def test_list_user_orders_returns_three_demo_orders():
    orders = list_user_orders("s1")
    assert [o["order_id"] for o in orders] == ["1001", "1002", "1003"]
    assert orders[0]["status"] == "已签收"
    assert orders[0]["signed_at"]


def test_query_order_matches_demo_order():
    payload = json.loads(query_order.invoke({"order_id": "1001"}))
    demo = list_user_orders("s1")[0]
    assert payload["order_id"] == "1001"
    assert payload["status"] == demo["status"]
    assert payload["signed_at"] == demo["signed_at"]


def test_query_order_unknown_id_keeps_contract():
    payload = json.loads(query_order.invoke({"order_id": "9999"}))
    assert payload["order_id"] == "9999"
    for key in ("items", "amount", "status", "tracking_no", "created_at", "signed_at"):
        assert key in payload


def test_query_order_is_stable_for_same_unknown_id():
    first = query_order.invoke({"order_id": "5678"})
    second = query_order.invoke({"order_id": "5678"})
    assert first == second


def test_query_order_status_is_one_of_known_values():
    payload = json.loads(query_order.invoke({"order_id": "5678"}))
    assert payload["status"] in {"待付款", "待发货", "运输中", "已签收"}


def test_list_user_orders_returns_copies():
    first = list_user_orders("s1")
    first[0]["status"] = "mutated"
    assert list_user_orders("s1")[0]["status"] != "mutated"


def test_query_order_returns_tracking_no():
    data = json.loads(query_order.invoke({"order_id": "1001"}))
    assert data["order_id"] == "1001"
    assert set(data) >= {"order_id", "items", "amount", "status", "tracking_no"}
    assert re.fullmatch(r"SF\d{10}", data["tracking_no"])


def test_query_product_keeps_contract():
    data = json.loads(query_product.invoke({"product_name": "无线蓝牙耳机"}))
    assert data["product_name"] == "无线蓝牙耳机"
    assert set(data) == {"product_name", "price", "stock", "specs"}


def test_order_snapshot_matches_demo_order():
    snapshot = order_snapshot("1001")
    assert snapshot == DEMO_ORDERS["1001"]
    assert snapshot is not DEMO_ORDERS["1001"]


def test_order_snapshot_unknown_id_generates_same_shape():
    payload = order_snapshot("9999")
    assert payload["order_id"] == "9999"
    for key in ("items", "amount", "status", "tracking_no", "created_at", "signed_at"):
        assert key in payload
    assert re.fullmatch(r"SF\d{10}", payload["tracking_no"])


def test_order_snapshot_is_stable_for_same_unknown_id():
    assert order_snapshot("5678") == order_snapshot("5678")


def test_order_snapshot_returns_none_for_empty_id():
    assert order_snapshot("") is None


def test_query_logistics_is_removed_from_builtin():
    import app.tools.business as business
    assert not hasattr(business, "query_logistics")


def test_builtin_specs_include_all_builtin_tools_but_not_logistics():
    from app.tools.registry import builtin_specs

    assert set(builtin_specs()) == {
        "query_order",
        "query_product",
        "query_faq",
        "create_ticket",
        "submit_refund",
        "suggest_handoff",
        "query_promotions",
    }


def test_builtin_specs_metadata_for_migrated_tools():
    from app.tools.registry import builtin_specs

    specs = builtin_specs()
    faq = specs["query_faq"]
    assert faq.permission == "read"
    assert faq.timeout == 30.0
    assert faq.exposed_to_agent is False
    assert faq.label == "查常见问题"

    ticket = specs["create_ticket"]
    assert ticket.permission == "write"
    assert ticket.inject_conversation is True
    assert ticket.label == "创建工单"

    refund = specs["submit_refund"]
    assert refund.permission == "read"
    assert refund.routes == frozenset({"refund_flow"})

    handoff = specs["suggest_handoff"]
    assert handoff.permission == "read"
    assert handoff.routes is None


def test_registry_keeps_legacy_exports_without_logistics():
    from app.tools.registry import ALL_TOOLS, TOOLS_BY_NAME, TOOL_LABELS
    assert [t.name for t in ALL_TOOLS] == ["query_order", "query_product", "query_faq", "create_ticket"]
    assert set(TOOLS_BY_NAME) == {t.name for t in ALL_TOOLS}
    assert set(TOOL_LABELS) == {"query_order", "query_product", "query_faq", "create_ticket"}

def test_tool_timeout_override():
    from app.tools.registry import tool_timeout
    assert tool_timeout("query_faq") == 25.0
    assert tool_timeout("query_order") == 5.0
