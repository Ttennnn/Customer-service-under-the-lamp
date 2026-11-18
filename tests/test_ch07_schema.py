from app.models import Conversation, ConversationSummary


def test_conversation_context_columns_roundtrip(session):
    conv = Conversation(user_id="u1", summary="第1段：订单 1001 退款", summary_upto_msg_id=42, layer1_from_msg_id=80)
    session.add(conv); session.commit(); session.refresh(conv)
    assert conv.summary == "第1段：订单 1001 退款"
    assert conv.summary_upto_msg_id == 42
    assert conv.layer1_from_msg_id == 80


def test_conversation_summary_unique_seq(session):
    from sqlalchemy.exc import IntegrityError
    import pytest
    session.add(ConversationSummary(conversation_id=1, seq=1, from_msg_id=1, upto_msg_id=2, content="a"))
    session.commit()
    session.add(ConversationSummary(conversation_id=1, seq=1, from_msg_id=3, upto_msg_id=4, content="b"))
    with pytest.raises(IntegrityError):
        session.commit()
