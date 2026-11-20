from app.models import LowConfidenceQuestion, ReviewQueue
from scripts.ch10.build_corpus import (
    build_real_samples,
    choose_text,
    clean_with_llm,
    fetch_real_questions,
)
from scripts.ch10.corpus_lib import Sample
from scripts.ch10.llm_io import CallBudget


def test_fetch_only_matched_and_join_normalized(session):
    review = ReviewQueue(normalized_question="支持以旧换新吗")
    session.add(review)
    session.commit()
    session.refresh(review)

    matched = LowConfidenceQuestion(
        raw_question="支持以旧换新不",
        source="self_check",
        matched_review_id=review.id,
    )
    unmatched = LowConfidenceQuestion(
        raw_question="没人管",
        source="self_check",
    )
    session.add_all([matched, unmatched])
    session.commit()

    rows = fetch_real_questions(session)
    assert [row["question_id"] for row in rows] == [matched.id]
    assert rows[0]["normalized_question"] == "支持以旧换新吗"


def test_choose_text_prefers_normalized_and_falls_back():
    assert choose_text(
        {"normalized_question": "开发票", "raw_question": "能开票不"}
    ) == ("开发票", True)
    assert choose_text(
        {"normalized_question": None, "raw_question": "能开票不"}
    ) == ("能开票不", False)


def test_build_real_samples_redacts_dedupes_and_keeps_refs():
    rows = [
        {
            "question_id": 1,
            "review_id": 1,
            "normalized_question": "手机号13800000000能改吗",
            "raw_question": "x",
            "source": "self_check",
            "created_at": None,
        },
        {
            "question_id": 2,
            "review_id": 1,
            "normalized_question": "手机号13800000000能改吗",
            "raw_question": "y",
            "source": "user_feedback",
            "created_at": None,
        },
    ]
    out = build_real_samples(rows)
    assert len(out) == 1
    assert "13800000000" not in out[0].text
    assert out[0].source_refs == (
        {"question_id": 1, "review_id": 1},
        {"question_id": 2, "review_id": 1},
    )


class FakeStructured:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = 0

    def invoke(self, messages):
        out = self.outputs[self.calls]
        self.calls += 1
        return out


class FakeLLM:
    def __init__(self, outputs):
        self.outputs = outputs

    def with_structured_output(self, schema, method="json_mode"):
        return FakeStructured(self.outputs)


def test_clean_with_llm_logs_changes():
    samples = [Sample(id="a", text="退活", origin="real")]
    llm = FakeLLM([{"items": [{"id": "a", "text": "退货"}]}])
    out, changes = clean_with_llm(
        llm, samples, CallBudget(5), batch_size=10
    )
    assert out[0].text == "退货"
    assert changes == [{"id": "a", "before": "退活", "after": "退货"}]


def test_clean_with_llm_keeps_batch_on_bad_ids():
    samples = [Sample(id="a", text="退活", origin="real")]
    llm = FakeLLM([{"items": [{"id": "zzz", "text": "退货"}]}])
    out, changes = clean_with_llm(
        llm, samples, CallBudget(5), batch_size=10
    )
    assert out[0].text == "退活"
    assert changes and "error" in changes[0]
