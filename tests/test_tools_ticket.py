# tests/test_tools_ticket.py
import re
from app.db.seed import seed_faq
from app.models import Conversation, Ticket
from app.tools.context import current_conversation_id
from app.tools.ticket import create_ticket

def _make_conversation(session):
    c = Conversation(user_id="u1", status="进行中")
    session.add(c); session.commit(); session.refresh(c)
    return c.id

def test_create_ticket_writes_and_archives(session):
    cid = _make_conversation(session)
    current_conversation_id.set(cid)
    out = create_ticket.invoke({"description": "屏幕碎了", "ticket_type": "售后"})
    session.expire_all()
    assert re.match(r"T\d{8}\d{3}", out)
    t = session.query(Ticket).one()
    assert t.conversation_id == cid and t.status == "待处理"
    assert session.query(Conversation).get(cid).status == "已转人工"

def test_create_ticket_without_context_errors(session):
    current_conversation_id.set(None)
    out = create_ticket.invoke({"description": "x", "ticket_type": "咨询"})
    assert "失败" in out or "错误" in out

def test_registry_has_five_tools():
    from app.tools.registry import ALL_TOOLS
    assert [t.name for t in ALL_TOOLS] == ["query_order", "query_product", "query_logistics", "query_faq", "create_ticket"]
