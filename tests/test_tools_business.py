import json
from app.tools.business import query_order, query_product, query_logistics

def test_query_order_echoes_id():
    data = json.loads(query_order.invoke({"order_id": "1001"}))
    assert data["order_id"] == "1001"
    assert set(data) >= {"order_id", "items", "amount", "status"}

def test_query_logistics_has_node():
    data = json.loads(query_logistics.invoke({"order_id": "1001"}))
    assert data["order_id"] == "1001"
    assert set(data) >= {"logistics_no", "carrier", "current_node", "traces"}

def test_registry_has_three_business_tools_and_five_labels():
    from app.tools.registry import ALL_TOOLS, TOOLS_BY_NAME, TOOL_LABELS
    assert [t.name for t in ALL_TOOLS] == ["query_order", "query_product", "query_logistics"]
    assert set(TOOLS_BY_NAME) == {t.name for t in ALL_TOOLS}
    assert set(TOOL_LABELS) == {"query_order", "query_product", "query_logistics", "query_faq", "create_ticket"}
