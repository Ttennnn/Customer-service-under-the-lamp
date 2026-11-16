"""Create the pre-ch03 tables and seed demo FAQ data.

This module creates only the conversation-domain tables (``conversations``,
``messages``, ``faq``, ``tickets``) and seeds the ``faq`` table with a small,
curated set of Chinese Q&A entries.

It deliberately does **not** create ``knowledge_chunks`` or
``qa_extraction_staging``.  The ch03 schema is owned by
``sql/ch03_knowledge.sql`` and applied by ``python -m app.db.init_ch03``; run
``init_ch03`` before (or independently of) this seed on a fresh MySQL database.
Letting this seed create ORM-shaped knowledge tables would make ``init_ch03``
see the tables as already present and silently bypass the authoritative DDL.

The ``faq`` table remains as demo/seed data (retained by design) but no longer
feeds retrieval — ch03's ``query_faq`` reads the vector knowledge base.  The
seed intentionally contains no 邮费 (shipping fee) entry; that is no longer an
expected retrieval miss because shipping-fee knowledge now comes from
``knowledge/运费与配送说明.md``.
"""

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db.session import get_engine, get_session_factory
from app.models import Conversation, Faq, Message, Ticket

# The ch02-and-earlier table set.  Knowledge tables are intentionally absent:
# they must come from the authoritative DDL via ``python -m app.db.init_ch03``.
SEED_TABLES = (
    Conversation.__table__,
    Message.__table__,
    Faq.__table__,
    Ticket.__table__,
)

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
    """Create only the pre-ch03 tables listed in :data:`SEED_TABLES`.

    The ch03 knowledge tables are created by ``app.db.init_ch03`` from
    ``sql/ch03_knowledge.sql``, never by this seed path.
    """
    Base.metadata.create_all(engine, tables=list(SEED_TABLES))


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
