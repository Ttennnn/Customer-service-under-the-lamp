"""后台首页摘要 API 与 ``/admin`` 页面路由(ch09 Task 10)。

摘要只回答三件事:待审/通过/驳回各多少、最高频知识缺口是什么、一句话
结论指向哪个后台页。页面本身由 Task 11 提供,这里在文件存在时按 no-store
提供静态 HTML。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models import ReviewQueue

PAGE = Path(__file__).resolve().parents[1] / "static" / "admin.html"
ADMIN_JS = Path(__file__).resolve().parents[1] / "static" / "admin.js"

REVIEW_STATUSES = ("待审", "通过", "驳回")

page_router = APIRouter()
api_router = APIRouter(prefix="/api/admin")


def _counts(session: Session) -> dict[str, int]:
    counts = {status: 0 for status in REVIEW_STATUSES}
    rows = (
        session.query(ReviewQueue.review_status, func.count(ReviewQueue.id))
        .group_by(ReviewQueue.review_status)
        .all()
    )
    for status, count in rows:
        counts[str(status)] = int(count)
    return counts


def _highest(session: Session) -> ReviewQueue | None:
    return (
        session.query(ReviewQueue)
        .order_by(ReviewQueue.occurrence_count.desc(), ReviewQueue.id.asc())
        .first()
    )


def _conclusion(counts: dict[str, int], highest: ReviewQueue | None) -> str:
    if highest is None:
        return "还没有知识缺口,先聊几句再启动飞轮。"
    return (
        f"{counts['待审']} 条待审,最高频问题出现 "
        f"{highest.occurrence_count} 次。"
    )


@page_router.get("/static/admin.js", response_class=FileResponse)
def admin_js() -> FileResponse:
    """Serve the shared back-office navigation script."""
    if not ADMIN_JS.is_file():
        raise HTTPException(status_code=404, detail="admin.js is missing")
    return FileResponse(
        ADMIN_JS,
        media_type="application/javascript",
        headers={"Cache-Control": "no-store"},
    )


@page_router.get("/admin", response_class=FileResponse)
def admin_page() -> FileResponse:
    """Serve the admin home page when the (Task 11) HTML file exists."""
    if not PAGE.is_file():
        raise HTTPException(status_code=404, detail="admin page is missing")
    return FileResponse(
        PAGE,
        media_type="text/html",
        headers={"Cache-Control": "no-store"},
    )


@api_router.get("/summary")
def summary(session: Session = Depends(get_db)) -> dict:
    """Return counts, the highest-occurrence gap, and one-line conclusion."""
    counts = _counts(session)
    highest = _highest(session)
    return {
        "counts": counts,
        "highest": None
        if highest is None
        else {
            "id": highest.id,
            "normalized_question": highest.normalized_question,
            "occurrence_count": highest.occurrence_count,
            "review_status": highest.review_status,
        },
        "conclusion": _conclusion(counts, highest),
        "links": {
            "review": "/review",
            "observability": "/observability",
        },
    }
