"""Side-channel batch classification of low-confidence questions."""

from __future__ import annotations

import time
from collections import Counter

import httpx
from sqlalchemy.exc import IntegrityError

from app.core.taxonomy import normalize_labels
from app.models import LowConfidenceQuestion, ReviewQueue, TopicClassification


def select_unclassified(
    session,
    limit: int | None = None,
    reclassify: bool = False,
) -> list[dict]:
    query = (
        session.query(
            LowConfidenceQuestion, ReviewQueue, TopicClassification
        )
        .join(
            ReviewQueue,
            LowConfidenceQuestion.matched_review_id == ReviewQueue.id,
        )
        .outerjoin(
            TopicClassification,
            TopicClassification.question_id == LowConfidenceQuestion.id,
        )
        .filter(LowConfidenceQuestion.matched_review_id.isnot(None))
        .filter(
            TopicClassification.id.is_(None)
            if not reclassify
            else True
        )
        .order_by(LowConfidenceQuestion.id.asc())
    )
    if limit is not None:
        query = query.limit(limit)

    rows: list[dict] = []
    for question, review, _ in query.all():
        normalized = (review.normalized_question or "").strip()
        if normalized:
            text, is_normalized = normalized, True
        else:
            text = (question.raw_question or "").strip()
            is_normalized = False
        rows.append(
            {
                "question_id": question.id,
                "review_id": review.id,
                "text": text,
                "normalized": is_normalized,
                "raw_question": question.raw_question,
                "source": question.source,
            }
        )
    return rows


def _existing_ids(session, ids: list[int]) -> set[int]:
    if not ids:
        return set()
    return {
        row[0]
        for row in session.query(TopicClassification.question_id)
        .filter(TopicClassification.question_id.in_(ids))
        .all()
    }


def write_classifications(session, items: list[dict]) -> int:
    if not items:
        return 0
    ids = [item["question_id"] for item in items]
    existing = _existing_ids(session, ids)
    new_items = [
        item for item in items if item["question_id"] not in existing
    ]
    if not new_items:
        return 0

    session.add_all(
        [
            TopicClassification(
                question_id=item["question_id"],
                labels=list(normalize_labels(item["labels"])),
            )
            for item in new_items
        ]
    )
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        existing = _existing_ids(session, ids)
        retry = [
            item
            for item in new_items
            if item["question_id"] not in existing
        ]
        if not retry:
            return 0
        session.add_all(
            [
                TopicClassification(
                    question_id=item["question_id"],
                    labels=list(normalize_labels(item["labels"])),
                )
                for item in retry
            ]
        )
        session.commit()
        return len(retry)
    return len(new_items)


def call_service(
    base_url: str,
    texts: list[str],
    batch_size: int = 32,
    timeout: float = 10.0,
) -> list[list[str]]:
    results: list[list[str]] = []
    for index in range(0, len(texts), batch_size):
        batch = texts[index : index + batch_size]
        response = httpx.post(
            f"{base_url.rstrip('/')}/classify",
            json={"texts": batch},
            timeout=timeout,
        )
        response.raise_for_status()
        items = response.json().get("items", [])
        if len(items) != len(batch):
            raise ValueError(
                f"service returned {len(items)} items for {len(batch)} texts"
            )
        for item in items:
            results.append(normalize_labels(item.get("labels", [])))
    return results


def run(
    session,
    base_url: str,
    min_batch: int = 10,
    force: bool = False,
    limit: int | None = None,
    service=None,
) -> dict:
    started = time.time()
    pending = select_unclassified(session, limit=limit)
    if not pending:
        return {
            "ok": True,
            "skipped": True,
            "reason": "no pending questions",
            "pending": 0,
            "classified": 0,
            "label_counts": {},
            "duration_s": time.time() - started,
        }
    if len(pending) < min_batch and not force:
        return {
            "ok": True,
            "skipped": True,
            "reason": "not enough pending",
            "pending": len(pending),
            "classified": 0,
            "label_counts": {},
            "duration_s": time.time() - started,
        }

    service = service or (
        lambda texts, **kwargs: call_service(base_url, texts, **kwargs)
    )
    items: list[dict] = []
    try:
        for index in range(0, len(pending), 32):
            batch = pending[index : index + 32]
            labels_batch = service([row["text"] for row in batch])
            if len(labels_batch) != len(batch):
                raise ValueError("service returned the wrong item count")
            for row, labels in zip(batch, labels_batch):
                items.append(
                    {
                        "question_id": row["question_id"],
                        "labels": list(labels),
                    }
                )
    except Exception as exc:  # noqa: BLE001 - no partial writes
        return {
            "ok": False,
            "skipped": False,
            "error": str(exc),
            "pending": len(pending),
            "classified": 0,
            "label_counts": {},
            "duration_s": time.time() - started,
        }

    classified = write_classifications(session, items)
    counts = Counter(
        label for item in items for label in item["labels"]
    )
    return {
        "ok": True,
        "skipped": False,
        "pending": len(pending),
        "classified": classified,
        "label_counts": dict(counts),
        "duration_s": time.time() - started,
    }
