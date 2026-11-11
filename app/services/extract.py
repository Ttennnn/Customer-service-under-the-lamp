from langchain_core.messages import HumanMessage, SystemMessage

from app.schemas import AfterSaleExtraction

EXTRACT_SYSTEM_PROMPT = """\
你是电商售后信息抽取助手。请从用户描述的售后诉求中,抽取并仅以 JSON 格式输出以下三个字段:

- order_id:订单号(字符串)。用户未提及订单号时输出 null,不要编造。
- appeal_type:诉求类型(字符串)。必须是以下枚举之一:退款、退货退款、换货、仅退款、补发、维修、物流催单、投诉、其他。
- expected_solution:期望方案(字符串)。用户未提及期望方案时输出 null,不要编造。

只输出一个合法的 JSON 对象,不要输出任何解释、前后缀或 Markdown 代码块。输出示例:

{"order_id": "E20241111001", "appeal_type": "退款", "expected_solution": "全额退款"}

当订单号或期望方案未提及时,对应字段输出 null。"""


def extract_after_sale(llm, text: str) -> AfterSaleExtraction:
    """Extract structured after-sale fields from a user description.

    ``llm`` is expected to be a LangChain ``ChatOpenAI`` (or a compatible
    stand-in). DeepSeek only supports JSON output via
    ``response_format={"type": "json_object"}`` on the chat-completions
    endpoint, which LangChain exposes as ``method="json_mode"``; the default
    ``json_schema`` method would send an unsupported response format.
    """
    structured = llm.with_structured_output(AfterSaleExtraction, method="json_mode")
    messages = [SystemMessage(content=EXTRACT_SYSTEM_PROMPT), HumanMessage(content=text)]
    return structured.invoke(messages)
