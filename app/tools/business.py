"""Mock 业务工具。

提供客服机器人调用的三个模拟业务查询工具:
- query_order: 查询订单
- query_product: 查询商品
- query_logistics: 查询物流

所有工具都使用 ``langchain.tools.tool`` 装饰器,通过类型标注与 docstring
生成参数 schema;函数返回 JSON 字符串(``ensure_ascii=False``,中文原样输出),
并原样回显入参中的 ``order_id`` / ``product_name``。
"""

import json
import random

from langchain.tools import tool


@tool
def query_order(order_id: str) -> str:
    """查询订单信息。

    根据订单号查询订单的商品明细、订单金额与订单状态。

    Args:
        order_id: 要查询的订单号
    """
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
    return json.dumps(
        {"order_id": order_id, "items": items, "amount": amount, "status": status},
        ensure_ascii=False,
    )


@tool
def query_product(product_name: str) -> str:
    """查询商品信息。

    根据商品名称查询商品的价格、库存与规格。

    Args:
        product_name: 要查询的商品名称
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
def query_logistics(order_id: str) -> str:
    """查询订单物流信息。

    根据订单号查询物流单号、承运商、当前物流节点与完整物流轨迹。

    Args:
        order_id: 要查询的订单号
    """
    carriers = ["顺丰速运", "中通快递", "圆通速递", "韵达快递"]
    nodes = ["已揽收", "运输中", "到达转运中心", "派送中", "已签收"]
    descs = ["快件已发出", "快件到达网点", "快件正在派送", "快件已签收"]

    logistics_no = "SF" + "".join(str(random.randint(0, 9)) for _ in range(10))
    carrier = random.choice(carriers)
    current_node = random.choice(nodes)
    traces = [
        {"node": random.choice(nodes), "desc": random.choice(descs)}
        for _ in range(random.randint(2, 5))
    ]
    return json.dumps(
        {
            "order_id": order_id,
            "logistics_no": logistics_no,
            "carrier": carrier,
            "current_node": current_node,
            "traces": traces,
        },
        ensure_ascii=False,
    )
