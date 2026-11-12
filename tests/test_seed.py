# tests/test_seed.py
from app.db.seed import FAQ_SEED, seed_faq

def test_faq_seed_has_return_policy_and_no_shipping():
    qs = " ".join(r["question"] for r in FAQ_SEED)
    answers = " ".join(r["answer"] for r in FAQ_SEED)
    assert "退货" in qs or "退货" in answers
    assert "邮费" not in qs and "邮费" not in answers

def test_seed_faq_inserts_rows(session):
    n = seed_faq(session)
    assert n == len(FAQ_SEED)
    session.commit()
