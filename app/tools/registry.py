"""工具注册表。

集中登记客服机器人可用的业务工具,供后续 Agent / 路由统一查找。

当前只收录三个已实现的 mock 业务工具;Task 5/6 落地后会把
``query_faq`` 与 ``create_ticket`` 依次追加进 ``ALL_TOOLS``,
保持最终顺序 ``["query_order", "query_product", "query_logistics",
"query_faq", "create_ticket"]``。
"""

from langchain_core.tools import BaseTool

from app.tools.business import query_logistics, query_order, query_product

# 三个 mock 业务工具,按最终注册顺序排列。
ALL_TOOLS: list[BaseTool] = [query_order, query_product, query_logistics]

TOOLS_BY_NAME: dict[str, BaseTool] = {t.name: t for t in ALL_TOOLS}

# 五个工具的展示名(仅为中文标签字符串,不包含工具对象;
# query_faq / create_ticket 由 Task 5/6 补齐)。
TOOL_LABELS: dict[str, str] = {
    "query_order": "查订单",
    "query_product": "查商品",
    "query_logistics": "查物流",
    "query_faq": "查常见问题",
    "create_ticket": "创建工单",
}
