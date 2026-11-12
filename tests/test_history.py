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
    append_message(session, cid, "assistant", tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}])
    append_message(session, cid, "tool", "物流结果", tool_call_id="c1")
    append_message(session, cid, "assistant", "已为您查到")
    session.commit()
    msgs = load_history(session, cid)
    assert isinstance(msgs[0], HumanMessage)
    assert isinstance(msgs[1], AIMessage) and msgs[1].tool_calls[0]["name"] == "query_logistics"
    assert isinstance(msgs[2], ToolMessage) and msgs[2].tool_call_id == "c1"
    assert isinstance(msgs[3], AIMessage) and msgs[3].content == "已为您查到"
