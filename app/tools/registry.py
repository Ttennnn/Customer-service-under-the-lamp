"""工具注册表。

集中登记客服机器人可用的业务工具,供后续 Agent / 路由统一查找。

当前已收录五个业务工具,最终顺序为 ``["query_order", "query_product",
"query_logistics", "query_faq", "create_ticket"]``。
"""

from langchain_core.tools import BaseTool

from app.tools.business import query_logistics, query_order, query_product
from app.tools.faq import query_faq
from app.tools.ticket import create_ticket

# 已实现的业务工具,按最终注册顺序排列。
ALL_TOOLS: list[BaseTool] = [
    query_order,
    query_product,
    query_logistics,
    query_faq,
    create_ticket,
]

TOOLS_BY_NAME: dict[str, BaseTool] = {t.name: t for t in ALL_TOOLS}

# 各工具的执行超时(秒);未列出的工具使用默认值。
# query_faq 覆盖改写 + 召回 + 重排的完整链路,给到 25 秒(低于前端 30 秒
# 无数据超时);配合 stream_chat 的 retries=0,超时只发生一次。
TOOL_TIMEOUTS: dict[str, float] = {"query_faq": 25.0}
DEFAULT_TOOL_TIMEOUT = 5.0


def tool_timeout(name: str) -> float:
    """Return the execution timeout in seconds for tool *name*."""
    return TOOL_TIMEOUTS.get(name, DEFAULT_TOOL_TIMEOUT)

# 五个工具的展示名(仅为中文标签字符串,不包含工具对象)。
TOOL_LABELS: dict[str, str] = {
    "query_order": "查订单",
    "query_product": "查商品",
    "query_logistics": "查物流",
    "query_faq": "查常见问题",
    "create_ticket": "创建工单",
}
