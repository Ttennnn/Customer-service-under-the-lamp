from app.agent.routing import INTENT_ROUTES, route_for_intent


def test_intent_routes_cover_eight_intents():
    assert INTENT_ROUTES == {
        "物流": "business",
        "订单": "business",
        "商品咨询": "knowledge",
        "退款退货": "refund_flow",
        "售后": "refund_flow",
        "投诉": "escalate",
        "闲聊": "fallback_script",
        "其他": "fallback_script",
    }


def test_unknown_intent_falls_back_to_business():
    assert route_for_intent("未知") == "business"
