"""待审队列审核 API 与 ``/review`` 页面路由(ch09 Task 7)。

JSON 接口只负责参数/状态码映射;写回知识库与状态流转在
:mod:`app.services.knowledge_approval`。审核页本身由 Task 11 提供,这里在
文件存在时按 no-store 提供静态 HTML。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models import LowConfidenceQuestion, ReviewQueue
from app.services import knowledge_approval

PAGE = Path(__file__).resolve().parents[1] / "static" / "review.html"

page_router = APIRouter()
api_router = APIRouter(prefix="/api/review")


class ApproveRequest(BaseModel):
    approved_answer: str


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _review_dict(row: ReviewQueue) -> dict:
    return {
        "id": row.id,
        "normalized_question": row.normalized_question,
        "ai_suggested_answer": row.ai_suggested_answer,
        "occurrence_count": row.occurrence_count,
        "review_status": row.review_status,
        "approved_answer": row.approved_answer,
        "created_at": _iso(row.created_at),
        "updated_at": _iso(row.updated_at),
    }


def _question_dict(row: LowConfidenceQuestion) -> dict:
    return {
        "id": row.id,
        "raw_question": row.raw_question,
        "source": row.source,
        "reason": row.reason,
        "created_at": _iso(row.created_at),
        "retrieved_chunks": row.retrieved_chunks,
    }


def _raise_for(error: Exception) -> None:
    """Map service exceptions onto the REST status codes from the design."""
    if isinstance(error, knowledge_approval.ReviewNotFound):
        raise HTTPException(status_code=404, detail=str(error)) from None
    if isinstance(error, knowledge_approval.ApprovalBusy):
        raise HTTPException(status_code=409, detail=str(error)) from None
    if isinstance(error, knowledge_approval.ReviewNotPending):
        raise HTTPException(status_code=409, detail=str(error)) from None
    if isinstance(error, knowledge_approval.InvalidApprovedAnswer):
        raise HTTPException(status_code=422, detail=str(error)) from None
    raise error


@page_router.get("/review", response_class=FileResponse)
def review_page() -> FileResponse:
    """Serve the review page when the (Task 11) HTML file exists."""
    if not PAGE.is_file():
        raise HTTPException(status_code=404, detail="review page is missing")
    return FileResponse(
        PAGE,
        media_type="text/html",
        headers={"Cache-Control": "no-store"},
    )


@api_router.get("/queue")
def list_queue(
    status: str | None = Query(default="待审"),
    session: Session = Depends(get_db),
) -> dict:
    """List review-queue rows; pending by default, most frequent first."""
    query = session.query(ReviewQueue)
    if status:
        query = query.filter(ReviewQueue.review_status == status)
    rows = query.order_by(
        ReviewQueue.occurrence_count.desc(), ReviewQueue.id.asc()
    ).all()
    return {"items": [_review_dict(row) for row in rows]}


@api_router.get("/{review_id}")
def get_review(
    review_id: int,
    session: Session = Depends(get_db),
) -> dict:
    """Return one queue row plus the low-confidence questions merged into it."""
    review = session.get(ReviewQueue, review_id)
    if review is None:
        raise HTTPException(status_code=404, detail="review not found")
    questions = (
        session.query(LowConfidenceQuestion)
        .filter(LowConfidenceQuestion.matched_review_id == review_id)
        .order_by(
            LowConfidenceQuestion.created_at.asc(),
            LowConfidenceQuestion.id.asc(),
        )
        .all()
    )
    return {
        "review": _review_dict(review),
        "questions": [_question_dict(row) for row in questions],
    }


@api_router.post("/{review_id}/approve")
def approve(
    review_id: int,
    req: ApproveRequest,
    session: Session = Depends(get_db),
) -> dict:
    """Write the approved answer back to the knowledge base, then mark 通过."""
    try:
        review = knowledge_approval.approve_review(
            session, review_id, req.approved_answer
        )
    except Exception as error:  # noqa: BLE001 - mapped by _raise_for
        _raise_for(error)
    return _review_dict(review)


@api_router.post("/{review_id}/reject")
def reject(
    review_id: int,
    session: Session = Depends(get_db),
) -> dict:
    """Mark one 待审 row as 驳回."""
    try:
        review = knowledge_approval.reject_review(session, review_id)
    except Exception as error:  # noqa: BLE001 - mapped by _raise_for
        _raise_for(error)
    return _review_dict(review)
