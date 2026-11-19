import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from app.models import Conversation
from app.services.history import get_or_create_conversation, append_message, load_history

def test_get_or_create_reuses_active(session):
    c1 = get_or_create_conversation(session, "u1")
    c2 = get_or_create_conversation(session, "u1")
    assert c1.id == c2.id and c1.status == "进行中"

def test_get_or_create_new_after_archived(session):
    c1 = get_or_create_conversation(session, "u1")
    c1.status = "已转人工"; session.commit()
    c2 = get_or_create_conversation(session, "u1")
    assert c2.id != c1.id

def test_load_history_rebuilds_tool_round(session):
    cid = get_or_create_conversation(session, "u1").id
    append_message(session, cid, "user", "物流到哪")
    # 旧历史参数只回放、不执行，用于验证兼容性
    append_message(session, cid, "assistant", tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}])
    append_message(session, cid, "tool", "物流结果", tool_call_id="c1")
    append_message(session, cid, "assistant", "已为您查到")
    session.commit()
    msgs = load_history(session, cid)
    assert isinstance(msgs[0], HumanMessage)
    assert isinstance(msgs[1], AIMessage) and msgs[1].tool_calls[0]["name"] == "query_order"
    assert isinstance(msgs[2], ToolMessage) and msgs[2].tool_call_id == "c1"
    assert isinstance(msgs[3], AIMessage) and msgs[3].content == "已为您查到"


from app.models import Conversation, ConversationSummary
from app.services.history import (
    ConversationNotFound, append_message, append_summary_segment, create_conversation,
    get_conversation, list_conversations, list_dialog_messages, previous_message_id,
    resolve_conversation, update_conversation_summary, update_layer1_from,
)


def test_resolve_conversation_new_existing_and_foreign(session):
    c1 = resolve_conversation(session, "u1", None, True)
    c2 = resolve_conversation(session, "u1", c1.id, False)
    assert c2.id == c1.id
    assert get_conversation(session, c1.id, "u2") is None
    with pytest.raises(ConversationNotFound):
        resolve_conversation(session, "u2", c1.id, False)


def test_list_conversations_newest_first_preview_and_summary(session):
    c1 = create_conversation(session, "u1")
    append_message(session, c1.id, "user", "订单 1001 想退款")
    append_message(session, c1.id, "assistant", "好的")
    update_conversation_summary(session, c1.id, "第1段：订单 1001 退款", 2)
    c2 = create_conversation(session, "u1")
    append_message(session, c2.id, "user", "物流到哪")
    rows = list_conversations(session, "u1")
    assert [row["id"] for row in rows] == [c2.id, c1.id]
    assert rows[1]["preview"].startswith("订单 1001")
    assert rows[1]["has_summary"] is True
    assert rows[0]["has_summary"] is False


def test_list_dialog_messages_and_previous_id(session):
    cid = create_conversation(session, "u1").id
    u1 = append_message(session, cid, "user", "问1")
    a1 = append_message(session, cid, "assistant", "答1")
    u2 = append_message(session, cid, "user", "问2")
    session.commit()
    rows = list_dialog_messages(session, cid)
    assert [row.content for row in rows] == ["问1", "答1", "问2"]
    assert previous_message_id(session, cid, u2.id) == a1.id


def test_append_summary_segment_appends_projection_and_advances_upto(session):
    cid = create_conversation(session, "u1").id
    seg1 = append_summary_segment(session, cid, 1, 4, "第1段：订单 1001")
    seg2 = append_summary_segment(session, cid, 5, 8, "第2段：已退款")
    assert (seg1.seq, seg2.seq) == (1, 2)
    conv = get_conversation(session, cid)
    assert conv.summary_upto_msg_id == 8
    assert "第1段：订单 1001" in conv.summary and "第2段：已退款" in conv.summary
    assert [s.content for s in session.query(ConversationSummary).order_by(ConversationSummary.seq)] == [
        "第1段：订单 1001", "第2段：已退款",
    ]


def test_update_layer1_from_only_moves_anchor(session):
    cid = create_conversation(session, "u1").id
    update_layer1_from(session, cid, 42)
    assert get_conversation(session, cid).layer1_from_msg_id == 42
