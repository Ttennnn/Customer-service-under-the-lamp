# tests/test_tools_ticket.py
from types import SimpleNamespace

import pytest

from app.db.seed import seed_faq
from app.models import Conversation, Ticket
from app.tools.context import current_conversation_id
from app.tools.ticket import TICKET_NO_PATTERN, create_ticket

def _make_conversation(session):
    c = Conversation(user_id="u1", status="进行中")
    session.add(c); session.commit(); session.refresh(c)
    return c.id

def test_create_ticket_writes_only_ticket(session):
    cid = _make_conversation(session)
    current_conversation_id.set(cid)
    out = create_ticket.invoke({"description": "屏幕碎了", "ticket_type": "售后"})
    session.expire_all()
    assert TICKET_NO_PATTERN.match(out)
    t = session.query(Ticket).one()
    assert t.conversation_id == cid and t.status == "待处理"
    assert session.query(Conversation).get(cid).status == "进行中"

def test_create_ticket_without_context_errors(session):
    current_conversation_id.set(None)
    out = create_ticket.invoke({"description": "x", "ticket_type": "咨询"})
    assert "失败" in out or "错误" in out

@pytest.mark.parametrize("ticket_type", ["退款"])
def test_create_ticket_accepts_refund(session, ticket_type):
    cid = _make_conversation(session)
    current_conversation_id.set(cid)
    out = create_ticket.invoke(
        {"description": "商品有质量问题,申请退款", "ticket_type": ticket_type}
    )
    session.expire_all()
    assert TICKET_NO_PATTERN.match(out)
    assert session.query(Ticket).one().ticket_type == "退款"

def test_create_ticket_sleeps_after_commit(session, monkeypatch):
    import app.tools.builtin.tickets as tickets

    calls = []
    visible_at_sleep = []

    def fake_sleep(seconds):
        calls.append(seconds)
        # The ticket must already be committed when the demo delay runs.
        visible_at_sleep.append(session.query(Ticket).count())

    monkeypatch.setattr(tickets.time, "sleep", fake_sleep)
    monkeypatch.setattr(
        tickets,
        "get_settings",
        lambda: SimpleNamespace(demo_ticket_delay_seconds=0.5),
    )
    token = current_conversation_id.set(1)
    try:
        out = tickets.create_ticket.invoke(
            {"description": "屏幕碎了", "ticket_type": "售后"}
        )
    finally:
        current_conversation_id.reset(token)

    assert TICKET_NO_PATTERN.match(out)
    assert calls == [0.5]
    assert visible_at_sleep == [1]
    assert session.query(Ticket).count() == 1


def test_registry_keeps_legacy_exports_without_logistics():
    from app.tools.registry import ALL_TOOLS
    assert [t.name for t in ALL_TOOLS] == ["query_order", "query_product", "query_faq", "create_ticket"]
