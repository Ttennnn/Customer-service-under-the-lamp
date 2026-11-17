import json
import re

import pytest

from app.tools.business import query_order, query_product, query_logistics


def test_query_order_returns_tracking_no():
    data = json.loads(query_order.invoke({"order_id": "1001"}))
    assert data["order_id"] == "1001"
    assert set(data) >= {"order_id", "items", "amount", "status", "tracking_no"}
    assert re.fullmatch(r"SF\d{10}", data["tracking_no"])


def test_query_logistics_uses_tracking_no():
    data = json.loads(query_logistics.invoke({"tracking_no": "SF1234567890"}))
    assert data["tracking_no"] == "SF1234567890"
    assert set(data) >= {"tracking_no", "carrier", "current_node", "traces"}


def test_order_tracking_no_flows_into_logistics():
    order = json.loads(query_order.invoke({"order_id": "1001"}))
    logistics = json.loads(query_logistics.invoke({"tracking_no": order["tracking_no"]}))
    assert logistics["tracking_no"] == order["tracking_no"]


def test_query_logistics_rejects_order_id_argument():
    with pytest.raises(Exception) as excinfo:
        query_logistics.invoke({"order_id": "1001"})
    assert "tracking_no" in str(excinfo.value)

def test_registry_has_five_tools_and_five_labels():
    from app.tools.registry import ALL_TOOLS, TOOLS_BY_NAME, TOOL_LABELS
    assert [t.name for t in ALL_TOOLS] == ["query_order", "query_product", "query_logistics", "query_faq", "create_ticket"]
    assert set(TOOLS_BY_NAME) == {t.name for t in ALL_TOOLS}
    assert set(TOOL_LABELS) == {"query_order", "query_product", "query_logistics", "query_faq", "create_ticket"}

def test_tool_timeout_override():
    from app.tools.registry import tool_timeout
    assert tool_timeout("query_faq") == 25.0
    assert tool_timeout("query_order") == 5.0
