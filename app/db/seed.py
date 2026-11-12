"""Table initialization and FAQ seed data.

This module creates all tables (via ``Base.metadata.create_all``) and seeds the
``faq`` table with a small, curated set of Chinese Q&A entries used by the
chatbot's FAQ retrieval tool.

Note: the seed intentionally does NOT include any entry about 邮费 (shipping
fee) — that is a documented expected miss for the FAQ retrieval acceptance.
"""

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db.session import get_engine, get_session_factory
from app.models import Faq

FAQ_SEED: list[dict] = [
    {
        "question": "退货政策是怎样的？",
        "answer": "签收后 7 天内可申请退货，商品需保持完好、吊牌齐全，审核通过后我们会安排上门取件。",
        "category": "售后",
    },
    {
        "question": "如何申请换货？",
        "answer": "在订单详情页点击「申请售后」，选择换货并填写原因，客服审核通过后 1-2 个工作日内发出换货商品。",
        "category": "售后",
    },
    {
        "question": "退款多久能到账？",
        "answer": "退货签收并质检通过后，退款将在 1-3 个工作日内原路退回；到账时间以银行或第三方支付平台为准。",
        "category": "售后",
    },
    {
        "question": "一般多久能收到货？",
        "answer": "国内普通地区一般 3-5 天送达，偏远地区可能延长 2-3 天，具体以物流信息为准。",
        "category": "物流",
    },
    {
        "question": "下单后什么时候发货？",
        "answer": "现货商品付款后 48 小时内发货，预售商品按页面标注时间发货，发货后会短信通知您。",
        "category": "物流",
    },
]


def seed_faq(session: Session) -> int:
    """Insert all FAQ seed rows and return the number of inserted rows."""
    session.add_all([Faq(**row) for row in FAQ_SEED])
    session.flush()
    return len(FAQ_SEED)


def init_db(engine: Engine) -> None:
    """Create all tables from ``Base.metadata``."""
    Base.metadata.create_all(engine)


def main() -> None:
    """CLI entry point: create tables and seed FAQ data."""
    engine = get_engine()
    init_db(engine)
    session = get_session_factory()()
    try:
        seed_faq(session)
        session.commit()
    finally:
        session.close()


if __name__ == "__main__":
    main()
