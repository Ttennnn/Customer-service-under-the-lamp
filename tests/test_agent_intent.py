from app.agent.intent import IntentResult, classify_intent


class StructuredLLM:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    def with_structured_output(self, schema, method=None):
        return self

    def invoke(self, messages):
        if self.error:
            raise self.error
        return self.result


def test_classify_valid_json():
    llm = StructuredLLM(result=IntentResult(intent="物流", reason="问快递"))
    assert classify_intent(llm, "我的快递到哪了").intent == "物流"


def test_classify_malformed_json_falls_back_to_after_sale():
    llm = StructuredLLM(error=ValueError("bad json"))
    out = classify_intent(llm, "我要投诉")
    assert out.intent == "售后"
    assert "fallback" in out.reason


def test_intent_prompt_lists_all_seven_labels():
    from app.agent.intent import INTENT_SYSTEM_PROMPT
    for label in ["物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊"]:
        assert label in INTENT_SYSTEM_PROMPT
    assert "只输出" in INTENT_SYSTEM_PROMPT and "JSON" in INTENT_SYSTEM_PROMPT
