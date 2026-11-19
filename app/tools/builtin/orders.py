"""订单与商品查询工具(内置插件)。

工具只做参数接收与 JSON 序列化;订单 mock 数据由
``app.tools.business.order_snapshot`` 提供,避免与图节点重复实现。
"""

import json
import random

from langchain.tools import tool

from app.tools.business import order_snapshot
from app.tools.spec import spec_from_tool


@tool
def query_order(order_id: str) -> str:
    """查询订单信息。

    根据订单号查询订单的商品明细、订单金额与订单状态。
    查询物流时请使用本工具返回的 tracking_no。

    Args:
        order_id: 要查询的订单号
    """
    return json.dumps(order_snapshot(order_id), ensure_ascii=False)


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


SPECS = [
    spec_from_tool(query_order, permission="read", label="查订单"),
    spec_from_tool(query_product, permission="read", label="查商品"),
]
