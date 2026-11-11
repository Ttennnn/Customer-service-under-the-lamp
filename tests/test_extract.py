from langchain_core.messages import SystemMessage

from app.schemas import AfterSaleExtraction, AppealType
from app.services.extract import EXTRACT_SYSTEM_PROMPT, extract_after_sale


class FakeStructuredLLM:
    def __init__(self):
        self.captured = None

    def invoke(self, messages):
        self.captured = messages
        return AfterSaleExtraction(
            order_id=None, appeal_type="退款", expected_solution="退货退款"
        )


class FakeLLM:
    def __init__(self):
        self.captured_method = None

    def with_structured_output(self, schema, **kwargs):
        self.captured_method = kwargs.get("method")
        self.fake = FakeStructuredLLM()
        return self.fake


def test_extract_returns_model_and_handles_missing_order_id():
    llm = FakeLLM()
    result = extract_after_sale(llm, "东西坏了要退款")
    assert result.appeal_type == AppealType.REFUND
    assert result.order_id is None
    assert llm.fake.captured[0].content  # system 提示已传入


def test_extract_passes_json_mode_and_builds_messages():
    llm = FakeLLM()
    result = extract_after_sale(llm, "订单 E123 要换货")
    assert llm.captured_method == "json_mode"
    assert isinstance(llm.fake.captured[0], SystemMessage)
    assert llm.fake.captured[0].content == EXTRACT_SYSTEM_PROMPT
    assert llm.fake.captured[1].content == "订单 E123 要换货"
    assert result.expected_solution == "退货退款"
