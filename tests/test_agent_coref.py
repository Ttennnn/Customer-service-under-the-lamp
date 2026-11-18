from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agent.coref import COREF_SYSTEM_PROMPT, CorefRewrite, resolve_query
from app.services.context import render_recent_dialogue


class FakeStructured:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.messages = result, error, []

    def with_structured_output(self, schema, method=None):
        return self

    def invoke(self, messages):
        self.messages.extend(messages)
        if self.error:
            raise self.error
        return self.result


def test_render_recent_dialogue_roles_and_order_marker():
    text = render_recent_dialogue([
        HumanMessage(content="订单 1001 的物流到哪了"),
        AIMessage(content="已签收"),
        SystemMessage(content="[订单选择器] 用户选择了订单 1001"),
        ToolMessage(content="TOOL-SHOULD-NOT-APPEAR", tool_call_id="c1"),
    ])
    assert "订单 1001 的物流到哪了" in text
    assert "已签收" in text
    assert "[订单选择器]" in text
    assert "TOOL-SHOULD-NOT-APPEAR" not in text


def test_resolve_query_returns_llm_output():
    llm = FakeStructured(result=CorefRewrite(resolved_query="订单 1001 的商品能退吗"))
    assert resolve_query(llm, "它能退吗", [HumanMessage(content="订单 1001")]) == "订单 1001 的商品能退吗"


def test_resolve_query_falls_back_on_error_and_blank():
    assert resolve_query(FakeStructured(error=ValueError("x")), "它能退吗") == "它能退吗"
    assert resolve_query(FakeStructured(result=CorefRewrite(resolved_query="  ")), "它能退吗") == "它能退吗"


def test_coref_prompt_pins_passthrough_and_entities():
    assert "原样" in COREF_SYSTEM_PROMPT
    assert "订单号" in COREF_SYSTEM_PROMPT
    assert "不要引入" in COREF_SYSTEM_PROMPT


def test_resolve_query_includes_history():
    llm = FakeStructured(result=CorefRewrite(resolved_query="x"))
    resolve_query(llm, "它能退吗", [HumanMessage(content="订单 1001")])
    assert "订单 1001" in llm.messages[-1].content
