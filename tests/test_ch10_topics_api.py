from fastapi.testclient import TestClient

from app.main import app
from app.models import (
    LowConfidenceQuestion,
    ReviewQueue,
    TopicClassification,
)


def test_distribution_counts_and_dedupes_samples(session):
    review = ReviewQueue(
        normalized_question="买大了想退", occurrence_count=3
    )
    session.add(review)
    session.commit()
    session.refresh(review)

    questions = [
        LowConfidenceQuestion(
            raw_question=f"买大了想退{i}",
            source="self_check",
            matched_review_id=review.id,
        )
        for i in range(2)
    ]
    session.add_all(questions)
    session.commit()
    for question in questions:
        session.add(
            TopicClassification(
                question_id=question.id, labels=["尺码", "退换货"]
            )
        )
    session.commit()

    body = TestClient(app).get("/api/topics/distribution").json()
    assert body["total"] == 2
    size = next(x for x in body["labels"] if x["label"] == "尺码")
    assert size["count"] == 2
    assert len(size["samples"]) == 1
    assert (
        next(x for x in body["labels"] if x["label"] == "退换货")["count"]
        == 2
    )


def test_questions_invalid_label_is_400_and_pagination():
    client = TestClient(app)
    assert (
        client.get(
            "/api/topics/questions", params={"label": "售后"}
        ).status_code
        == 400
    )
    assert (
        client.get(
            "/api/topics/questions",
            params={"label": "尺码", "page": 1, "size": 2},
        ).status_code
        == 200
    )
