"""飞轮批处理:标准化、查重、待审队列写回(Task 6)。

所有测试都用 FakeLLM,不联网、不建真实模型。``process_pending`` 串行处理,
每条都重新查询候选,因此同一批里刚新建的缺口能被后续条目命中。
"""

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from app.models import LowConfidenceQuestion, ReviewQueue
from app.services.flywheel import (
    FlywheelError,
    FlywheelOutput,
    FlywheelReport,
    _resolve_candidate_limit,
    fetch_unmatched_low_conf,
    normalize_and_match,
    process_pending,
)
from app.services.review_repo import (
    increment_occurrence,
    insert_review_item,
    list_review_candidates,
    set_matched_review,
)


class FakeLLM:
    """Structured-output fake: returns *output* or raises *error*."""

    def __init__(self, output=None, error=None):
        self.output = output
        self.error = error
        self.calls = []
        self.schema = None
        self.method = None

    def with_structured_output(self, schema, method=None):
        self.schema = schema
        self.method = method
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return self.output


class SequenceLLM(FakeLLM):
    """Return one output/error per ``invoke`` call."""

    def __init__(self, steps):
        super().__init__()
        self.steps = list(steps)
        self.index = 0

    def invoke(self, messages):
        self.calls.append(messages)
        step = self.steps[self.index]
        self.index += 1
        if isinstance(step, Exception):
            raise step
        return step


@pytest.fixture
def low_conf_row(session):
    row = LowConfidenceQuestion(raw_question="退款多久到账", source="self_check")
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def test_new_question_creates_pending_review(session, low_conf_row):
    out = FlywheelOutput("退款多久到账", None, "1-3 个工作日")
    report = process_pending(session, FakeLLM(out))

    assert isinstance(report, FlywheelReport)
    assert (report.scanned, report.created, report.merged, report.failed) == (
        1,
        1,
        0,
        0,
    )
    row = session.query(ReviewQueue).one()
    assert row.review_status == "待审"
    assert row.occurrence_count == 1
    assert row.normalized_question == "退款多久到账"
    assert row.ai_suggested_answer == "1-3 个工作日"
    assert low_conf_row.matched_review_id == row.id


def test_match_rejected_row_increments_without_revival(session, low_conf_row):
    review = ReviewQueue(
        normalized_question="退款多久到账",
        review_status="驳回",
        occurrence_count=2,
    )
    session.add(review)
    session.commit()
    session.refresh(review)
    out = FlywheelOutput("退款多久到账", review.id, "")
    report = process_pending(session, FakeLLM(out))

    assert (report.created, report.merged, report.failed) == (0, 1, 0)
    session.refresh(review)
    assert review.review_status == "驳回"
    assert review.occurrence_count == 3
    assert review.ai_suggested_answer is None
    assert low_conf_row.matched_review_id == review.id
    assert session.query(ReviewQueue).count() == 1


def test_match_approved_row_keeps_approved_status(session, low_conf_row):
    review = ReviewQueue(
        normalized_question="退款多久到账",
        review_status="通过",
        occurrence_count=1,
        approved_answer="1-3 个工作日到账",
    )
    session.add(review)
    session.commit()
    session.refresh(review)

    report = process_pending(session, FakeLLM(FlywheelOutput("退款多久到账", review.id, "")))

    assert report.merged == 1
    session.refresh(review)
    assert review.review_status == "通过"
    assert review.occurrence_count == 2
    assert review.approved_answer == "1-3 个工作日到账"


def test_hallucinated_id_is_skipped(session, low_conf_row):
    out = FlywheelOutput("新问题", 999999, "")
    report = process_pending(session, FakeLLM(out))

    assert report.failed == 1
    assert report.created == 0 and report.merged == 0
    assert low_conf_row.matched_review_id is None
    assert session.query(ReviewQueue).count() == 0


def test_model_exception_is_skipped_and_retried_next_run(session):
    rows = [
        LowConfidenceQuestion(raw_question="q1", source="self_check"),
        LowConfidenceQuestion(raw_question="q2", source="self_check"),
    ]
    session.add_all(rows)
    session.commit()

    llm = SequenceLLM([RuntimeError("bad json"), FlywheelOutput("问题2", None, "")])
    first = process_pending(session, llm)

    assert (first.scanned, first.created, first.failed) == (2, 1, 1)
    q1, q2 = session.query(LowConfidenceQuestion).order_by(LowConfidenceQuestion.id).all()
    assert q1.matched_review_id is None
    assert q2.matched_review_id is not None

    # 下一轮只扫描未归并的那条,失败条目会被重试。
    second = process_pending(session, FakeLLM(FlywheelOutput("问题1", None, "")))
    assert (second.scanned, second.created) == (1, 1)
    session.refresh(q1)
    assert q1.matched_review_id is not None


def test_serial_same_batch_second_matches_first(session, monkeypatch):
    for q in ("q1", "q2"):
        session.add(LowConfidenceQuestion(raw_question=q, source="self_check"))
    session.commit()
    calls = []

    def fake_normalize(llm, raw_question, candidates):
        calls.append([(c.id, c.normalized_question) for c in candidates])
        if len(calls) == 1:
            return FlywheelOutput("标准问题1", None, "")
        return FlywheelOutput("标准问题2", candidates[0].id, "")

    monkeypatch.setattr("app.services.flywheel.normalize_and_match", fake_normalize)
    report = process_pending(session, object())

    assert (report.created, report.merged) == (1, 1)
    assert calls[1]
    assert calls[0] == []
    assert calls[1][0][1] == "标准问题1"


def test_process_pending_honours_limit(session):
    for index in range(3):
        session.add(LowConfidenceQuestion(raw_question=f"q{index}", source="self_check"))
    session.commit()

    report = process_pending(session, FakeLLM(FlywheelOutput("新问题", None, "")), limit=2)

    assert report.scanned == 2
    assert report.created == 2
    assert session.query(LowConfidenceQuestion).filter(
        LowConfidenceQuestion.matched_review_id.is_(None)
    ).count() == 1


def test_process_pending_uses_settings_candidate_limit(session, monkeypatch):
    session.add(LowConfidenceQuestion(raw_question="q1", source="self_check"))
    session.commit()
    seen = []

    def fake_normalize(llm, raw_question, candidates):
        seen.append(len(candidates))
        return FlywheelOutput("新问题", None, "")

    monkeypatch.setattr("app.services.flywheel.normalize_and_match", fake_normalize)
    monkeypatch.setattr(
        "app.services.flywheel.get_settings",
        lambda: SimpleNamespace(flywheel_candidate_limit=1),
    )
    process_pending(session, object())
    assert seen == [0]

    for index in range(3):
        session.add(
            ReviewQueue(normalized_question=f"已有{index}", review_status="待审")
        )
    session.commit()
    session.add(LowConfidenceQuestion(raw_question="q2", source="self_check"))
    session.commit()
    process_pending(session, object())
    assert seen == [0, 1]


def test_resolve_candidate_limit_falls_back_when_settings_fail(monkeypatch):
    def broken():
        raise RuntimeError("no api key")

    monkeypatch.setattr("app.services.flywheel.get_settings", broken)
    assert _resolve_candidate_limit() == 200


def test_insert_truncates_normalized_question_to_512(session):
    out = FlywheelOutput("问" * 600, None, "")
    row = insert_review_item(session, out)
    session.commit()

    assert len(row.normalized_question) == 512
    assert row.normalized_question == "问" * 512


def test_insert_treats_blank_suggested_answer_as_null(session):
    row = insert_review_item(session, FlywheelOutput("新问题", None, "   "))
    session.commit()
    assert row.ai_suggested_answer is None


def test_list_review_candidates_all_statuses_newest_first_and_limited(session):
    now = datetime(2026, 11, 19, 12, 0, 0)
    rows = [
        ReviewQueue(
            normalized_question="旧待审",
            review_status="待审",
            updated_at=now - timedelta(minutes=2),
        ),
        ReviewQueue(
            normalized_question="新驳回",
            review_status="驳回",
            updated_at=now,
        ),
        ReviewQueue(
            normalized_question="中通过",
            review_status="通过",
            updated_at=now - timedelta(minutes=1),
        ),
    ]
    session.add_all(rows)
    session.commit()

    candidates = list_review_candidates(session, limit=2)

    assert [row.normalized_question for row in candidates] == ["新驳回", "中通过"]
    assert {row.review_status for row in list_review_candidates(session, limit=10)} == {
        "待审",
        "通过",
        "驳回",
    }


def test_fetch_unmatched_low_conf_skips_matched_and_orders_oldest_first(session):
    matched = ReviewQueue(normalized_question="已归并")
    session.add(matched)
    session.commit()
    first = LowConfidenceQuestion(raw_question="first", source="self_check")
    second = LowConfidenceQuestion(raw_question="second", source="self_check")
    done = LowConfidenceQuestion(
        raw_question="done", source="self_check", matched_review_id=matched.id
    )
    session.add_all([first, second, done])
    session.commit()

    rows = fetch_unmatched_low_conf(session, limit=10)

    assert [row.raw_question for row in rows] == ["first", "second"]
    assert fetch_unmatched_low_conf(session, limit=1)[0].raw_question == "first"


def test_normalize_and_match_renders_candidate_fields_and_validates(session):
    candidate = ReviewQueue(
        normalized_question="退款多久到账",
        occurrence_count=4,
        review_status="驳回",
    )
    session.add(candidate)
    session.commit()
    session.refresh(candidate)

    llm = FakeLLM(FlywheelOutput("退款多久能到账", candidate.id, "1-3 个工作日"))
    output = normalize_and_match(llm, "退款啥时候到账", [candidate])

    assert output.matched_question_id == candidate.id
    assert llm.schema is FlywheelOutput
    assert llm.method == "json_mode"
    human = llm.calls[0][-1].content
    assert "退款啥时候到账" in human
    assert f"id={candidate.id}" in human
    assert "退款多久到账" in human
    assert "次数=4" in human or "次数: 4" in human
    assert "驳回" in human


@pytest.mark.parametrize(
    "output",
    [
        FlywheelOutput("标准问题", 12345, ""),
        FlywheelOutput("", None, ""),
        FlywheelOutput("   ", None, ""),
    ],
)
def test_normalize_and_match_rejects_invalid_output(session, output):
    candidate = ReviewQueue(normalized_question="已有问题", occurrence_count=1)
    session.add(candidate)
    session.commit()
    session.refresh(candidate)

    with pytest.raises(FlywheelError):
        normalize_and_match(FakeLLM(output), "原话", [candidate])


def test_normalize_and_match_wraps_unparseable_dict_as_flywheel_error(session):
    candidate = ReviewQueue(normalized_question="已有问题", occurrence_count=1)
    session.add(candidate)
    session.commit()
    session.refresh(candidate)

    llm = FakeLLM(
        {
            "normalized_question": "问题",
            "matched_question_id": "not-an-id",
            "ai_suggested_answer": "",
        }
    )
    with pytest.raises(FlywheelError):
        normalize_and_match(llm, "原话", [candidate])


def test_normalize_and_match_accepts_null_id_with_no_candidates(session):
    output = normalize_and_match(FakeLLM(FlywheelOutput("新问题", None, "")), "原话", [])
    assert output.normalized_question == "新问题"


def test_normalize_and_match_accepts_dict_output(session):
    candidate = ReviewQueue(normalized_question="已有问题", occurrence_count=1)
    session.add(candidate)
    session.commit()
    session.refresh(candidate)

    llm = FakeLLM(
        {
            "normalized_question": "标准化问题",
            "matched_question_id": candidate.id,
            "ai_suggested_answer": "",
        }
    )
    output = normalize_and_match(llm, "原话", [candidate])
    assert output.normalized_question == "标准化问题"
    assert output.matched_question_id == candidate.id


def test_repo_increment_and_set_matched_do_not_commit(session):
    review = ReviewQueue(normalized_question="问题", occurrence_count=1)
    lcq = LowConfidenceQuestion(raw_question="原话", source="self_check")
    session.add_all([review, lcq])
    session.commit()

    increment_occurrence(session, review.id)
    set_matched_review(session, lcq.id, review.id)
    session.rollback()

    session.refresh(review)
    session.refresh(lcq)
    assert review.occurrence_count == 1
    assert lcq.matched_review_id is None


def test_increment_occurrence_raises_for_missing_row(session):
    with pytest.raises(FlywheelError):
        increment_occurrence(session, 999999)


def test_build_llm_accepts_request_timeout():
    from app.config import Settings, build_llm

    settings = Settings(_env_file=None, openai_api_key="sk-test")
    llm = build_llm(settings, request_timeout=7)
    assert llm.request_timeout == 7.0
    assert build_llm(settings).request_timeout is None
