from app.agent.intent import INTENT_SYSTEM_PROMPT, IntentResult, classify_intent


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


def test_prompt_has_nine_labels_and_json_fields():
    for label in ["物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "人工", "闲聊", "其他"]:
        assert label in INTENT_SYSTEM_PROMPT
    assert "confidence" in INTENT_SYSTEM_PROMPT
    assert "只输出" in INTENT_SYSTEM_PROMPT and "JSON" in INTENT_SYSTEM_PROMPT
    assert "reason" not in INTENT_SYSTEM_PROMPT


def test_prompt_has_boundary_examples():
    for text in ["这个还能退吗", "刚收到就坏了", "我要投诉", "你好，在吗", "帮我写一首诗"]:
        assert text in INTENT_SYSTEM_PROMPT


def test_classify_returns_intent_and_clamps_confidence():
    assert classify_intent(FakeStructured(IntentResult(intent="退款退货", confidence=0.91)), "x").confidence == 0.91
    assert classify_intent(FakeStructured(IntentResult(intent="物流", confidence=1.5)), "x").confidence == 1.0
    assert classify_intent(FakeStructured(IntentResult(intent="物流", confidence=-0.2)), "x").confidence == 0.0


def test_classify_falls_back_to_other():
    out = classify_intent(FakeStructured(error=ValueError("bad json")), "x")
    assert out.intent == "其他" and out.confidence == 0.0


def test_classify_can_raise_for_eval():
    import pytest
    with pytest.raises(ValueError):
        classify_intent(FakeStructured(error=ValueError("bad json")), "x", raise_on_error=True)


def test_classify_passes_history():
    llm = FakeStructured(IntentResult(intent="物流", confidence=0.8))
    classify_intent(llm, "那它到哪了", [("user", "订单 1001 的物流")])
    assert "订单 1001 的物流" in llm.messages[-1].content
