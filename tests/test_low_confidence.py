from app.models import LowConfidenceQuestion
from app.services.low_confidence import record_low_confidence


def test_record_low_confidence_inserts_self_check_row(session):
    row = record_low_confidence(session, None, "支持以旧换新吗", "证据不足")
    session.commit()
    assert row.source == "self_check" and row.raw_question.endswith("吗")


def test_record_low_confidence_flushes_and_returns_orm_row(session):
    row = record_low_confidence(session, None, "支持以旧换新吗", "只沾边")
    assert isinstance(row, LowConfidenceQuestion)
    assert row.id is not None
    assert row.conversation_id is None
    assert row.reason == "只沾边"


def test_record_low_confidence_keeps_conversation_id_and_custom_source(session):
    row = record_low_confidence(
        session, 42, "退款多久到账", None, source="retrieval_low_conf"
    )
    session.commit()
    assert row.conversation_id == 42
    assert row.source == "retrieval_low_conf"
    assert row.reason is None
