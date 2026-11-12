# tests/test_models.py
from sqlalchemy import JSON
from app.models import Conversation, Message, Faq, Ticket

def _cols(model):
    return {c.name: c for c in model.__table__.columns}

def test_conversation_status_enum():
    assert _cols(Conversation)["status"].type.enums == ["进行中", "已转人工", "已结束"]

def test_message_role_and_tool_calls():
    c = _cols(Message)
    assert c["role"].type.enums == ["user", "assistant", "tool"]
    assert isinstance(c["tool_calls"].type, JSON)
    assert c["content"].nullable is True

def test_message_fk_to_conversation():
    assert any(str(fk.target_fullname) == "conversations.id"
               for fk in Message.__table__.foreign_keys)

def test_ticket_enums_and_pk():
    c = _cols(Ticket)
    assert c["ticket_no"].primary_key
    assert c["ticket_type"].type.enums == ["售后", "投诉", "咨询"]
    assert c["status"].type.enums == ["待处理", "已处理"]

def test_faq_fields():
    c = _cols(Faq)
    assert c["question"].type.length == 512
    assert c["category"].index is True
