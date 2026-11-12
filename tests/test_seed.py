# tests/test_seed.py
from app.db.seed import FAQ_SEED, seed_faq
from app.models import Conversation, Message

def test_faq_seed_has_return_policy_and_no_shipping():
    qs = " ".join(r["question"] for r in FAQ_SEED)
    answers = " ".join(r["answer"] for r in FAQ_SEED)
    assert "退货" in qs or "退货" in answers
    assert "邮费" not in qs and "邮费" not in answers

def test_seed_faq_inserts_rows(session):
    n = seed_faq(session)
    assert n == len(FAQ_SEED)
    session.commit()

def test_integer_pks_autoincrement_on_sqlite(session):
    conv = Conversation(user_id="u1")
    session.add(conv)
    session.flush()
    assert conv.id is not None

    conv2 = Conversation(user_id="u2")
    session.add(conv2)
    session.flush()
    assert conv2.id > conv.id

    msg = Message(conversation_id=conv.id, role="user", content="hi")
    session.add(msg)
    session.flush()
    assert msg.id is not None

    msg2 = Message(conversation_id=conv2.id, role="user", content="hi2")
    session.add(msg2)
    session.flush()
    assert msg2.id > msg.id
