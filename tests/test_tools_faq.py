from app.db.seed import seed_faq
from app.models import Faq
from app.tools.faq import query_faq

def test_query_faq_hits_return_policy(session):
    seed_faq(session); session.commit()
    out = query_faq.invoke({"keyword": "退货"})
    assert "退货" in out

def test_query_faq_misses_shipping(session):
    seed_faq(session); session.commit()
    out = query_faq.invoke({"keyword": "邮费"})
    assert "未找到" in out
