import pytest
from pydantic import ValidationError

from app.schemas import (
    AfterSaleExtraction,
    AppealType,
    ChatRequest,
    ExtractRequest,
    RefundRequest,
    ResumeRequest,
    TicketRequest,
)


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


def test_ticket_request_accepts_refund():
    req = TicketRequest(session_id="s", description="x", ticket_type="退款")
    assert req.ticket_type == "退款"


def test_chat_request_defaults_and_conversation_fields():
    req = ChatRequest(session_id="s1", message="hello")
    assert req.conversation_id is None
    assert req.new_conversation is False

    req = ChatRequest(
        session_id="s1", message="hello", conversation_id=3, new_conversation=True
    )
    assert req.conversation_id == 3
    assert req.new_conversation is True


@pytest.mark.parametrize(
    "request_model, payload",
    [
        (ChatRequest, {"session_id": "s", "message": "x"}),
        (ResumeRequest, {"session_id": "s", "order_id": "1001"}),
        (
            RefundRequest,
            {"session_id": "s", "order_id": "1001", "reason": "质量问题"},
        ),
        (TicketRequest, {"session_id": "s", "description": "x"}),
    ],
)
def test_conversation_id_must_be_positive(request_model, payload):
    with pytest.raises(ValidationError):
        request_model(**payload, conversation_id=0)
    with pytest.raises(ValidationError):
        request_model(**payload, conversation_id=-1)

    assert request_model(**payload).conversation_id is None


def test_extract_request_rejects_blank_text():
    with pytest.raises(ValidationError):
        ExtractRequest(text=" \t ")


def test_resume_request_requires_exactly_one_of_order_or_ticket():
    with pytest.raises(ValidationError):
        ResumeRequest(session_id="s", order_id=None, confirmed=None)
    with pytest.raises(ValidationError):
        ResumeRequest(session_id="s", order_id="1001", confirmed=True)

    assert ResumeRequest(session_id="s", confirmed=True).confirmed is True
    assert ResumeRequest(session_id="s", confirmed=False).confirmed is False
    assert ResumeRequest(session_id="s", order_id="1001").order_id == "1001"


def test_resume_request_strips_provided_strings():
    req = ResumeRequest(session_id=" s ", order_id=" 1001 ")
    assert req.session_id == "s"
    assert req.order_id == "1001"

    with pytest.raises(ValidationError):
        ResumeRequest(session_id="s", order_id="   ")
