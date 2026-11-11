import pytest
from pydantic import ValidationError

from app.schemas import AfterSaleExtraction, AppealType, ChatRequest, ExtractRequest


def test_appeal_type_values():
    assert [m.value for m in AppealType] == ["退款", "退货退款", "换货", "仅退款", "补发", "维修", "物流催单", "投诉", "其他"]


def test_extraction_requires_appeal_type_but_nulls_others():
    m = AfterSaleExtraction.model_validate({"appeal_type": "退款"})
    assert m.order_id is None and m.expected_solution is None
    assert m.appeal_type == AppealType.REFUND


def test_chat_request_rejects_blank_message():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="s1", message="   ")


def test_chat_request_rejects_blank_session_id():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="   ", message="hello")


def test_chat_request_strips_values():
    req = ChatRequest(session_id="  s1  ", message="  hello  ")
    assert req.session_id == "s1"
    assert req.message == "hello"


def test_extract_request_rejects_blank_text():
    with pytest.raises(ValidationError):
        ExtractRequest(text=" \t ")
