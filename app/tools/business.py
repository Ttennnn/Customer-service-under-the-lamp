"""Mock 业务工具。

提供客服机器人调用的三个模拟业务查询工具:
- query_order: 查询订单
- query_product: 查询商品
- query_logistics: 查询物流

所有工具都使用 ``langchain.tools.tool`` 装饰器,通过类型标注与 docstring
生成参数 schema;函数返回 JSON 字符串(``ensure_ascii=False``,中文原样输出),
并原样回显入参中的 ``order_id`` / ``product_name`` / ``tracking_no``。
"""

import copy
import json
import random
from datetime import datetime, timedelta

from langchain.tools import tool

_NOW = datetime.now()


def _days_ago(days: int) -> str:
    return (_NOW - timedelta(days=days)).strftime("%Y-%m-%d")


# 演示订单目录:退款子流程的订单选择器与 ``query_order`` 共用同一份数据。
# 日期按模块导入时间相对生成,保证「7 天无理由」等政策演示稳定。
DEMO_ORDERS: dict[str, dict] = {
    "1001": {
        "order_id": "1001",
        "items": [{"name": "无线蓝牙耳机", "qty": 1, "price": 299.0}],
        "amount": 299.0,
        "status": "已签收",
        "tracking_no": "SF1234567890",
        "created_at": _days_ago(7),
        "signed_at": _days_ago(3),
    },
    "1002": {
        "order_id": "1002",
        "items": [{"name": "智能手环", "qty": 1, "price": 199.0}],
        "amount": 199.0,
        "status": "运输中",
        "tracking_no": "SF2345678901",
        "created_at": _days_ago(2),
        "signed_at": None,
    },
    "1003": {
        "order_id": "1003",
        "items": [{"name": "移动电源", "qty": 2, "price": 89.0}],
        "amount": 178.0,
        "status": "待发货",
        "tracking_no": "SF3456789012",
        "created_at": _days_ago(1),
        "signed_at": None,
    },
}


def list_user_orders(session_id: str) -> list[dict]:
    """返回演示订单目录里的三张订单卡(深拷贝,调用方可安全修改)。"""
    return [copy.deepcopy(DEMO_ORDERS[order_id]) for order_id in ("1001", "1002", "1003")]


@tool
def query_order(order_id: str) -> str:
    """查询订单信息。

    根据订单号查询订单的商品明细、订单金额与订单状态。
    查询物流时请使用本工具返回的 tracking_no。

    Args:
        order_id: 要查询的订单号
    """
    if order_id in DEMO_ORDERS:
        return json.dumps(copy.deepcopy(DEMO_ORDERS[order_id]), ensure_ascii=False)

    item_names = ["无线蓝牙耳机", "智能手环", "移动电源", "机械键盘", "保温杯"]
    items = [
        {
            "name": random.choice(item_names),
            "qty": random.randint(1, 3),
            "price": round(random.uniform(50, 500), 2),
        }
        for _ in range(random.randint(1, 3))
    ]
    amount = round(sum(item["price"] * item["qty"] for item in items), 2)
    status = random.choice(["待付款", "待发货", "运输中", "已签收"])
    tracking_no = "SF" + "".join(str(random.randint(0, 9)) for _ in range(10))
    created_at = _days_ago(random.randint(1, 10))
    return json.dumps(
        {
            "order_id": order_id,
            "items": items,
            "amount": amount,
            "status": status,
            "tracking_no": tracking_no,
            "created_at": created_at,
            "signed_at": created_at if status == "已签收" else None,
        },
        ensure_ascii=False,
    )


@tool
def query_product(product_name: str) -> str:
    """查询某个订单商品的实时价格与库存快照。

    仅用于查询订单商品的实时价格、库存等交易信息;商品型号参数、规格、
    退换货政策等知识类问题应查询知识库(query_faq),不要用本工具回答。

    Args:
        product_name: 要查询的商品名称(订单中的商品)
    """
    price = round(random.uniform(10, 2000), 2)
    stock = random.randint(0, 500)
    specs = random.sample(
        ["标准版", "高配版", "旗舰版", "mini版", "套装版"],
        k=random.randint(1, 3),
    )
    return json.dumps(
        {"product_name": product_name, "price": price, "stock": stock, "specs": specs},
        ensure_ascii=False,
    )


@tool
def query_logistics(tracking_no: str) -> str:
    """查询物流信息。

    根据物流单号查询物流单号、承运商、当前物流节点与完整物流轨迹。
    物流单号必须来自 query_order 返回的 tracking_no，不要用订单号或猜测值代替。

    Args:
        tracking_no: 要查询的物流单号
    """
    carriers = ["顺丰速运", "中通快递", "圆通速递", "韵达快递"]
    nodes = ["已揽收", "运输中", "到达转运中心", "派送中", "已签收"]
    descs = ["快件已发出", "快件到达网点", "快件正在派送", "快件已签收"]

    carrier = random.choice(carriers)
    current_node = random.choice(nodes)
    traces = [
        {"node": random.choice(nodes), "desc": random.choice(descs)}
        for _ in range(random.randint(2, 5))
    ]
    return json.dumps(
        {
            "tracking_no": tracking_no,
            "carrier": carrier,
            "current_node": current_node,
            "traces": traces,
        },
        ensure_ascii=False,
    )
