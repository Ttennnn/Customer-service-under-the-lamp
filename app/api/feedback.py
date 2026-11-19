"""用户反馈 API(ch09 Task 8)。

``POST /api/feedback`` 接收前端对一轮回答的赞/踩。``up`` 只记录日志;
``down`` 把该轮检索快照一并写入低置信度问题池,供后续飞轮归一化与挖掘
复用。反馈属于用户体验路径:图/Langfuse 不可用时降级为无快照落池,绝不
返回 500。快照只在会话线程记录的本轮用户原话与本次反馈的问题一致时才
采用,避免把上一轮(或另一轮)的证据挂到当前问题下。

匹配优先使用父 State 的 ``raw_user_message``(未经截断的原话),旧 checkpoint
没有该字段时回退到模型侧 ``user_message``;这样长输入被截断后,前端发回的
完整原话仍能对上快照。快照条数取 ``settings.low_confidence_snapshot_top_k``。
"""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.ch06 import get_ch06_graph_dep
from app.config import Settings, get_settings
from app.db.session import get_db
from app.observability import _state_graph
from app.services.history import get_conversation
from app.services.low_confidence import record_low_confidence

logger = logging.getLogger(__name__)

api_router = APIRouter(prefix="/api/feedback")

CONVERSATION_NOT_FOUND = "会话不存在"
FEEDBACK_REASON = "用户反馈未解决"
SNAPSHOT_TOP_K = 3


class FeedbackRequest(BaseModel):
    session_id: str
    conversation_id: int
    rating: Literal["up", "down"]
    question: str


def get_feedback_graph_dep(
    request: Request, settings: Settings = Depends(get_settings)
):
    """Reuse the ch06 graph, but degrade to ``None`` when it cannot be built.

    Feedback must still be recorded (without a snapshot) when the graph or
    Langfuse is unavailable, so construction failures never surface as 500.
    """
    try:
        return get_ch06_graph_dep(request, settings)
    except Exception:  # noqa: BLE001 - feedback must not depend on the graph
        logger.warning(
            "feedback: ch06 graph unavailable; recording without snapshot",
            exc_info=True,
        )
        return None


def _latest_snapshot(
    graph, conversation_id: int, question: str, top_k: int = SNAPSHOT_TOP_K
):
    """Return the current turn's retrieval snapshot, or ``None``.

    The checkpointed state may belong to a later/older turn (or be missing
    entirely). The raw (untruncated) question is authoritative; older
    checkpoints without ``raw_user_message`` fall back to the model-facing
    ``user_message``. Only an exact match is trusted. Any failure reading the
    state degrades to ``None``.
    """
    if graph is None:
        return None
    try:
        state = _state_graph(graph).get_state(
            {"configurable": {"thread_id": f"conv:{conversation_id}"}}
        )
        values = getattr(state, "values", None) or {}
    except Exception:  # noqa: BLE001 - snapshot is best effort
        logger.warning("feedback: failed to read graph state", exc_info=True)
        return None

    raw_question = values.get("raw_user_message")
    matched_question = (
        raw_question if raw_question is not None else values.get("user_message")
    )
    if matched_question != question:
        return None
    chunks = values.get("retrieved_snapshot") or []
    if not chunks:
        return None
    limit = max(0, int(top_k))
    return list(chunks)[:limit]


def record_feedback(
    session, graph, req: FeedbackRequest, *, settings: Settings | None = None
) -> dict:
    """Validate ownership, then log ``up`` or pool ``down`` feedback."""
    conversation = get_conversation(session, req.conversation_id, req.session_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail=CONVERSATION_NOT_FOUND)

    if req.rating == "up":
        logger.info(
            "feedback up session=%s conversation=%s question=%s",
            req.session_id,
            conversation.id,
            req.question,
        )
        return {"status": "ignored"}

    top_k = (
        settings.low_confidence_snapshot_top_k
        if settings is not None
        else SNAPSHOT_TOP_K
    )
    snapshot = _latest_snapshot(graph, conversation.id, req.question, top_k)
    row = record_low_confidence(
        session,
        conversation.id,
        req.question,
        FEEDBACK_REASON,
        source="user_feedback",
        retrieved_chunks=snapshot,
    )
    session.commit()
    logger.info(
        "feedback down session=%s conversation=%s question=%s id=%s snapshot=%s",
        req.session_id,
        conversation.id,
        req.question,
        row.id,
        len(snapshot or []),
    )
    # ``low_confidence_id`` is the documented field; ``id`` stays for older
    # clients that already read it.
    return {"status": "recorded", "low_confidence_id": row.id, "id": row.id}


@api_router.post("")
def submit_feedback(
    req: FeedbackRequest,
    graph=Depends(get_feedback_graph_dep),
    session: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Record one up/down feedback for a conversation."""
    return record_feedback(session, graph, req, settings=settings)
