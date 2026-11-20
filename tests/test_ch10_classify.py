from app.models import (
    LowConfidenceQuestion,
    ReviewQueue,
    TopicClassification,
)
from scripts.ch10.classify_lib import run, select_unclassified, write_classifications


def _seed(session):
    review = ReviewQueue(normalized_question="买大了想退")
    session.add(review)
    session.commit()
    session.refresh(review)
    question = LowConfidenceQuestion(
        raw_question="买大了想退",
        source="self_check",
        matched_review_id=review.id,
    )
    session.add(question)
    session.commit()
    session.refresh(question)
    return question, review


def test_select_only_unclassified_matched(session):
    question, _ = _seed(session)
    rows = select_unclassified(session)
    assert [row["question_id"] for row in rows] == [question.id]
    assert rows[0]["text"] == "买大了想退"
    assert rows[0]["normalized"] is True


def test_write_is_idempotent(session):
    question, _ = _seed(session)
    items = [
        {"question_id": question.id, "labels": ["退换货", "尺码"]}
    ]
    assert write_classifications(session, items) == 1
    assert write_classifications(session, items) == 0
    assert select_unclassified(session) == []


def test_min_batch_and_force(session):
    _seed(session)
    calls = []
    out = run(
        session,
        "http://fake",
        min_batch=10,
        force=False,
        service=lambda texts, **kwargs: calls.append(texts),
    )
    assert out["skipped"] is True
    assert calls == []

    out2 = run(
        session,
        "http://fake",
        min_batch=10,
        force=True,
        service=lambda texts, **kwargs: [["尺码"]] * len(texts),
    )
    assert out2["classified"] == 1


def test_service_failure_writes_nothing(session):
    _seed(session)

    def boom(texts, **kwargs):
        raise RuntimeError("service down")

    out = run(
        session, "http://fake", min_batch=1, force=True, service=boom
    )
    assert out["ok"] is False
    assert session.query(TopicClassification).count() == 0


def test_reclassify_failure_keeps_existing(session):
    question, _ = _seed(session)
    write_classifications(
        session, [{"question_id": question.id, "labels": ["尺码"]}]
    )

    def boom(texts, **kwargs):
        raise RuntimeError("service down")

    out = run(
        session,
        "http://fake",
        min_batch=1,
        force=True,
        service=boom,
        reclassify=True,
    )
    assert out["ok"] is False
    assert session.query(TopicClassification).count() == 1


def test_reclassify_small_pool_keeps_existing(session):
    question, _ = _seed(session)
    write_classifications(
        session, [{"question_id": question.id, "labels": ["尺码"]}]
    )
    out = run(
        session,
        "http://fake",
        min_batch=10,
        force=False,
        service=lambda texts, **kwargs: [["尺码"]],
        reclassify=True,
    )
    assert out["skipped"] is True
    assert session.query(TopicClassification).count() == 1


def test_write_rejects_empty_labels(session):
    import pytest

    question, _ = _seed(session)
    with pytest.raises(ValueError):
        write_classifications(
            session, [{"question_id": question.id, "labels": []}]
        )
