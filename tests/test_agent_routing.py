from app.agent.routing import INTENT_ROUTES, route_for_intent


def test_every_intent_maps_to_one_of_four_exits():
    assert INTENT_ROUTES == {
        "商品咨询": "knowledge",
        "退款退货": "knowledge",
        "物流": "business",
        "订单": "business",
        "售后": "business",
        "投诉": "complaint",
        "闲聊": "chitchat",
    }


def test_unknown_intent_falls_back_to_business():
    assert route_for_intent("未知") == "business"
