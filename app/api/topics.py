"""Read-only topic distribution and per-class question list (ch10)."""

from __future__ import annotations

import math
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.taxonomy import LABEL2ID, TOPIC_CLASSES
from app.db.session import get_db
from app.models import LowConfidenceQuestion, ReviewQueue, TopicClassification

PAGE = Path(__file__).resolve().parents[1] / "static" / "topics.html"
QUESTIONS_PAGE = (
    Path(__file__).resolve().parents[1] / "static" / "topic-questions.html"
)

page_router = APIRouter()
api_router = APIRouter(prefix="/api/topics")


def _rows(session: Session):
    return (
        session.query(TopicClassification, LowConfidenceQuestion, ReviewQueue)
        .join(
            LowConfidenceQuestion,
            TopicClassification.question_id == LowConfidenceQuestion.id,
        )
        .join(
            ReviewQueue,
            LowConfidenceQuestion.matched_review_id == ReviewQueue.id,
        )
        .order_by(
            TopicClassification.classified_at.desc(),
            TopicClassification.id.desc(),
        )
        .all()
    )


def _text(question: LowConfidenceQuestion, review: ReviewQueue):
    normalized = (review.normalized_question or "").strip()
    if normalized:
        return normalized, True
    return (question.raw_question or "").strip(), False


def _iso(value):
    return value.isoformat() if hasattr(value, "isoformat") else value


def _pending(session: Session, rows) -> int:
    classified = {row[0].question_id for row in rows}
    matched = (
        session.query(LowConfidenceQuestion.id)
        .filter(LowConfidenceQuestion.matched_review_id.isnot(None))
        .all()
    )
    return sum(1 for (question_id,) in matched if question_id not in classified)


@page_router.get("/topics", response_class=FileResponse)
def topics_page() -> FileResponse:
    if not PAGE.is_file():
        raise HTTPException(status_code=404, detail="topics page is missing")
    return FileResponse(
        PAGE,
        media_type="text/html",
        headers={"Cache-Control": "no-store"},
    )


@page_router.get("/topics/questions", response_class=FileResponse)
def topic_questions_page() -> FileResponse:
    if not QUESTIONS_PAGE.is_file():
        raise HTTPException(
            status_code=404, detail="topic questions page is missing"
        )
    return FileResponse(
        QUESTIONS_PAGE,
        media_type="text/html",
        headers={"Cache-Control": "no-store"},
    )


@api_router.get("/distribution")
def distribution(session: Session = Depends(get_db)) -> dict:
    rows = _rows(session)
    counts = {label: 0 for label in TOPIC_CLASSES}
    samples = {label: [] for label in TOPIC_CLASSES}
    seen = {label: set() for label in TOPIC_CLASSES}
    for classification, question, review in rows:
        text, _ = _text(question, review)
        for label in classification.labels:
            if label not in counts:
                continue
            counts[label] += 1
            if text not in seen[label] and len(samples[label]) < 3:
                seen[label].add(text)
                samples[label].append(
                    {"text": text, "question_id": question.id}
                )

    last = max(
        (
            classification.classified_at
            for classification, _, _ in rows
            if classification.classified_at is not None
        ),
        default=None,
    )
    return {
        "total": len(rows),
        "pending": _pending(session, rows),
        "last_classified_at": _iso(last),
        "labels": [
            {
                "label": label,
                "count": counts[label],
                "samples": samples[label],
            }
            for label in TOPIC_CLASSES
        ],
    }


@api_router.get("/questions")
def questions(
    label: str,
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    session: Session = Depends(get_db),
) -> dict:
    if label not in LABEL2ID:
        raise HTTPException(
            status_code=400,
            detail=f"unknown topic label: {label}",
        )

    filtered = [
        row for row in _rows(session) if label in row[0].labels
    ]
    total = len(filtered)
    pages = max(1, math.ceil(total / size)) if total else 0
    start = (page - 1) * size
    items = []
    for classification, question, review in filtered[start : start + size]:
        text, normalized = _text(question, review)
        items.append(
            {
                "question_id": question.id,
                "text": text,
                "normalized": normalized,
                "raw_question": question.raw_question,
                "labels": list(classification.labels),
                "source": question.source,
                "occurrence_count": review.occurrence_count,
                "review_status": review.review_status,
                "created_at": _iso(question.created_at),
                "classified_at": _iso(classification.classified_at),
            }
        )
    return {
        "label": label,
        "total": total,
        "page": page,
        "pages": pages,
        "size": size,
        "items": items,
    }
