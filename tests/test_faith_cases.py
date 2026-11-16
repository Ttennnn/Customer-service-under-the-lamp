from datetime import datetime

from sqlalchemy import func, select

from app.models import FaithCase
from app.services.faith_cases import FaithCasePayload, upsert_faith_case


def make_payload(**overrides):
    values = {
        "eval_id": "A01",
        "bucket": "A_policy",
        "query": "退款多久",
        "strategy": "hybrid_rerank",
        "answer": "答案",
        "reason": "编造",
        "citations": [
            {
                "n": 1,
                "chunk_id": 3,
                "section_path": "退货政策 > 退款说明",
                "question": "退款多久",
                "answer": "1-3 个工作日",
            }
        ],
        "judge_model": "deepseek-chat",
    }
    values.update(overrides)
    return FaithCasePayload(**values)


def test_upsert_inserts_then_increments_and_reopens(session):
    p = FaithCasePayload(
        "A01",
        "A_policy",
        "退款多久",
        "hybrid_rerank",
        "答案",
        "编造",
        [{"n": 1}],
        "deepseek-chat",
    )
    first = upsert_faith_case(session, p)
    session.commit()
    assert first.seen_count == 1 and first.status == "未解决"
    first.status = "已解决"
    first.resolved_at = datetime(2026, 11, 1)
    session.commit()
    again = upsert_faith_case(session, p)
    session.commit()
    assert again.id == first.id and again.seen_count == 2
    assert again.status == "未解决" and again.resolved_at == datetime(2026, 11, 1)


def test_upsert_insert_sets_ledger_fields_and_flushes(session):
    row = upsert_faith_case(session, make_payload())

    assert isinstance(row, FaithCase)
    assert row.id is not None
    assert row.bucket == "A_policy"
    assert row.query == "退款多久"
    assert row.answer == "答案"
    assert row.reason == "编造"
    assert row.judge_model == "deepseek-chat"
    assert row.citations == make_payload().citations
    assert row.status == "未解决"
    assert row.seen_count == 1
    assert row.first_seen_at == row.last_seen_at
    assert row.resolved_at is None


def test_upsert_one_row_per_eval_id_across_strategies(session):
    first = upsert_faith_case(session, make_payload())
    session.commit()
    first_seen_at = first.first_seen_at

    later = upsert_faith_case(
        session,
        make_payload(
            strategy="dense",
            answer="不同策略的答案",
            reason="另一个编造点",
            citations=[{"n": 2, "chunk_id": 9}],
            judge_model="deepseek-reasoner",
        ),
    )
    session.commit()

    count = session.scalar(
        select(func.count()).select_from(FaithCase).where(FaithCase.eval_id == "A01")
    )
    assert count == 1
    assert later.id == first.id
    assert later.seen_count == 2
    assert later.strategy == "dense"
    assert later.answer == "不同策略的答案"
    assert later.reason == "另一个编造点"
    assert later.citations == [{"n": 2, "chunk_id": 9}]
    assert later.judge_model == "deepseek-reasoner"
    assert later.first_seen_at == first_seen_at
    assert later.last_seen_at >= first_seen_at


def test_upsert_reopens_clears_resolution_keeps_resolved_at(session):
    first = upsert_faith_case(session, make_payload())
    first.status = "已解决"
    first.resolution = "已补充 FAQ"
    first.resolved_at = datetime(2026, 11, 2)
    session.commit()

    again = upsert_faith_case(session, make_payload())
    session.commit()

    assert again.id == first.id
    assert again.status == "未解决"
    assert again.resolution is None
    assert again.resolved_at == datetime(2026, 11, 2)
