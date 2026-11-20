"""用户反馈 API(ch09 Task 8)。

``up`` 只记录日志,``down`` 把本轮检索快照一并写入低置信度池,供后续飞轮
挖掘使用。图状态只用于取快照:图不可用或读取失败时仍然落池(不 500)。
"""

import logging
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api import feedback as feedback_api
from app.api.feedback import FeedbackRequest, get_feedback_graph_dep, record_feedback
from app.main import app
from app.models import LowConfidenceQuestion
from app.services.history import get_or_create_conversation


def _graph(user_message="q", snapshot=None):
    return SimpleNamespace(
        get_state=lambda config: SimpleNamespace(
            values={
                "user_message": user_message,
                "retrieved_snapshot": snapshot or [],
            }
        )
    )


def _request(session_id="u1", conversation_id=None, rating="down", question="q"):
    return FeedbackRequest(
        session_id=session_id,
        conversation_id=conversation_id,
        rating=rating,
        question=question,
    )


@pytest.fixture
def client(session):
    app.dependency_overrides[get_feedback_graph_dep] = lambda: None
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# service
# ---------------------------------------------------------------------------


def test_up_only_logs_and_does_not_write(session, caplog):
    caplog.set_level(logging.INFO)
    cid = get_or_create_conversation(session, "u1").id

    out = record_feedback(
        session,
        _graph(),
        _request(conversation_id=cid, rating="up", question="q"),
    )

    assert out["status"] == "ignored"
    assert session.query(LowConfidenceQuestion).count() == 0
    assert "feedback up" in caplog.text


def test_down_writes_user_feedback_with_latest_snapshot(session):
    cid = get_or_create_conversation(session, "u1").id
    snap = [
        {
            "question": "支持以旧换新吗",
            "answer": "不支持",
            "rerank_score": 0.4,
            "section_path": "售后 > 换货",
            "is_key_clause": False,
        }
    ]

    out = record_feedback(
        session,
        _graph("支持以旧换新吗", snap),
        _request(
            conversation_id=cid,
            rating="down",
            question="支持以旧换新吗",
        ),
    )

    row = session.query(LowConfidenceQuestion).one()
    assert out["status"] == "recorded"
    assert out["low_confidence_id"] == row.id
    # Legacy field stays for older clients.
    assert out["id"] == row.id
    assert row.source == "user_feedback"
    assert row.reason == "用户反馈未解决"
    assert row.raw_question == "支持以旧换新吗"
    assert row.conversation_id == cid
    assert row.retrieved_chunks[0]["section_path"] == "售后 > 换货"


def test_snapshot_top_k_uses_settings(session):
    cid = get_or_create_conversation(session, "u1").id
    snap = [{"question": f"q{i}", "answer": "a"} for i in range(5)]
    settings = SimpleNamespace(low_confidence_snapshot_top_k=1)

    record_feedback(
        session,
        _graph("q", snap),
        _request(conversation_id=cid, question="q"),
        settings=settings,
    )

    row = session.query(LowConfidenceQuestion).one()
    assert [chunk["question"] for chunk in row.retrieved_chunks] == ["q0"]


def test_snapshot_is_read_through_bound_graph(session):
    cid = get_or_create_conversation(session, "u1").id
    snap = [{"question": "q", "answer": "a", "section_path": "s"}]
    inner = _graph("q", snap)

    class BoundGraph:
        bound = inner

    record_feedback(
        session,
        BoundGraph(),
        _request(conversation_id=cid, question="q"),
    )

    row = session.query(LowConfidenceQuestion).one()
    assert row.retrieved_chunks[0]["section_path"] == "s"


def test_snapshot_match_prefers_untruncated_raw_question(session):
    """A long question is truncated in user_message but kept raw in state."""
    cid = get_or_create_conversation(session, "u1").id
    long_question = "退款" * 300
    snap = [{"question": "退款多久到账", "answer": "1-3 个工作日"}]
    graph = SimpleNamespace(
        get_state=lambda config: SimpleNamespace(
            values={
                "raw_user_message": long_question,
                "user_message": long_question[:100],
                "retrieved_snapshot": snap,
            }
        )
    )

    record_feedback(
        session,
        graph,
        _request(conversation_id=cid, question=long_question),
    )

    row = session.query(LowConfidenceQuestion).one()
    assert row.retrieved_chunks == snap


def test_snapshot_falls_back_to_user_message_for_old_checkpoints(session):
    cid = get_or_create_conversation(session, "u1").id
    snap = [{"question": "q", "answer": "a"}]
    graph = SimpleNamespace(
        get_state=lambda config: SimpleNamespace(
            values={"user_message": "q", "retrieved_snapshot": snap}
        )
    )

    record_feedback(
        session,
        graph,
        _request(conversation_id=cid, question="q"),
    )

    row = session.query(LowConfidenceQuestion).one()
    assert row.retrieved_chunks == snap


def test_down_keeps_only_top_three_snapshot_chunks(session):
    cid = get_or_create_conversation(session, "u1").id
    snap = [
        {
            "question": f"q{i}",
            "answer": "a",
            "rerank_score": 1.0 - i / 10,
            "section_path": f"s{i}",
            "is_key_clause": False,
        }
        for i in range(5)
    ]

    record_feedback(
        session,
        _graph("q", snap),
        _request(conversation_id=cid, question="q"),
    )

    row = session.query(LowConfidenceQuestion).one()
    assert [chunk["section_path"] for chunk in row.retrieved_chunks] == [
        "s0",
        "s1",
        "s2",
    ]


def test_down_question_mismatch_stores_null_snapshot(session):
    cid = get_or_create_conversation(session, "u1").id
    snap = [
        {
            "question": "别的",
            "answer": "a",
            "rerank_score": 0.9,
            "section_path": "s",
            "is_key_clause": False,
        }
    ]

    record_feedback(
        session,
        _graph("另一轮", snap),
        _request(conversation_id=cid, rating="down", question="本轮问题"),
    )

    row = session.query(LowConfidenceQuestion).one()
    assert row.retrieved_chunks is None


def test_down_empty_snapshot_stores_null(session):
    cid = get_or_create_conversation(session, "u1").id

    record_feedback(
        session,
        _graph("q", []),
        _request(conversation_id=cid, question="q"),
    )

    row = session.query(LowConfidenceQuestion).one()
    assert row.retrieved_chunks is None


def test_down_without_graph_still_records(session):
    cid = get_or_create_conversation(session, "u1").id

    out = record_feedback(
        session, None, _request(conversation_id=cid, question="q")
    )

    row = session.query(LowConfidenceQuestion).one()
    assert out["status"] == "recorded"
    assert row.retrieved_chunks is None


def test_down_with_state_read_failure_still_records(session):
    cid = get_or_create_conversation(session, "u1").id

    def fail(config):
        raise RuntimeError("checkpoint unavailable")

    record_feedback(
        session,
        SimpleNamespace(get_state=fail),
        _request(conversation_id=cid, question="q"),
    )

    row = session.query(LowConfidenceQuestion).one()
    assert row.source == "user_feedback"
    assert row.retrieved_chunks is None


def test_wrong_owner_raises_404(session):
    cid = get_or_create_conversation(session, "owner").id

    with pytest.raises(HTTPException) as excinfo:
        record_feedback(
            session,
            _graph(),
            _request(session_id="intruder", conversation_id=cid, question="q"),
        )

    assert excinfo.value.status_code == 404
    assert session.query(LowConfidenceQuestion).count() == 0


# ---------------------------------------------------------------------------
# dependency: graph availability must never turn feedback into a 500
# ---------------------------------------------------------------------------


def test_feedback_graph_dep_reuses_ch06_graph(monkeypatch):
    sentinel = object()
    seen = []

    def fake(request, settings):
        seen.append((request, settings))
        return sentinel

    monkeypatch.setattr(feedback_api, "get_ch06_graph_dep", fake)
    request, settings = object(), object()

    assert get_feedback_graph_dep(request, settings) is sentinel
    assert seen == [(request, settings)]


def test_feedback_graph_dep_returns_none_when_build_fails(monkeypatch):
    def boom(request, settings):
        raise HTTPException(status_code=503, detail="graph unavailable")

    monkeypatch.setattr(feedback_api, "get_ch06_graph_dep", boom)

    assert get_feedback_graph_dep(object(), object()) is None


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------


def test_api_down_records_feedback(client, session):
    cid = get_or_create_conversation(session, "u1").id

    resp = client.post(
        "/api/feedback",
        json={
            "session_id": "u1",
            "conversation_id": cid,
            "rating": "down",
            "question": "支持以旧换新吗",
        },
    )

    assert resp.status_code == 200
    assert resp.json()["status"] == "recorded"
    row = session.query(LowConfidenceQuestion).one()
    assert resp.json()["low_confidence_id"] == row.id
    assert row.raw_question == "支持以旧换新吗"
    assert row.source == "user_feedback"


def test_api_up_does_not_write(client, session):
    cid = get_or_create_conversation(session, "u1").id

    resp = client.post(
        "/api/feedback",
        json={
            "session_id": "u1",
            "conversation_id": cid,
            "rating": "up",
            "question": "q",
        },
    )

    assert resp.status_code == 200
    assert resp.json()["status"] == "ignored"
    assert session.query(LowConfidenceQuestion).count() == 0


def test_wrong_owner_404(client, session):
    cid = get_or_create_conversation(session, "owner").id

    resp = client.post(
        "/api/feedback",
        json={
            "session_id": "intruder",
            "conversation_id": cid,
            "rating": "down",
            "question": "q",
        },
    )

    assert resp.status_code == 404


def test_api_accepts_down_when_graph_is_unavailable(client, session):
    cid = get_or_create_conversation(session, "u1").id

    resp = client.post(
        "/api/feedback",
        json={
            "session_id": "u1",
            "conversation_id": cid,
            "rating": "down",
            "question": "q",
        },
    )

    assert resp.status_code == 200
    assert resp.json()["status"] == "recorded"
    assert session.query(LowConfidenceQuestion).count() == 1


def test_api_rejects_unknown_rating(client, session):
    cid = get_or_create_conversation(session, "u1").id

    resp = client.post(
        "/api/feedback",
        json={
            "session_id": "u1",
            "conversation_id": cid,
            "rating": "sideways",
            "question": "q",
        },
    )

    assert resp.status_code == 422
