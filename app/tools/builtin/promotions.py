"""优惠活动查询工具(内置插件示例)。

演示「放入文件即可注册」的插件机制:本模块只暴露 ``SPECS``,注册中心
扫描 ``app.tools.builtin`` 时自动收集。
"""

import json

from langchain.tools import tool

from app.tools.spec import spec_from_tool


@tool
def query_promotions(category: str = "全部") -> str:
    """查询当前店铺的优惠活动。"""
    promotions = [
        {
            "name": "满 300 减 50",
            "category": "全场",
            "detail": "单笔订单实付满 300 元减 50 元",
            "valid_until": "2026-12-31",
        },
        {
            "name": "生鲜第二件半价",
            "category": "生鲜",
            "detail": "指定生鲜商品第二件享 5 折",
            "valid_until": "2026-12-15",
        },
        {
            "name": "数码新品立减 100",
            "category": "数码",
            "detail": "新品数码类商品下单立减 100 元",
            "valid_until": "2026-12-20",
        },
    ]
    if category and category != "全部":
        promotions = [p for p in promotions if p["category"] == category]
    return json.dumps(
        {"category": category, "count": len(promotions), "promotions": promotions},
        ensure_ascii=False,
    )


SPECS = [spec_from_tool(query_promotions, permission="read", label="查优惠")]
