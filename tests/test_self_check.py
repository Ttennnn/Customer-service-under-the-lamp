import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from app.rag.self_check import EvidenceAssessment, assess_evidence


class FakeStructuredLLM:
    def __init__(self, payload):
        self.payload = payload
        self.captured_messages = None
        self.captured_schema = None
        self.captured_method = None

    def with_structured_output(self, schema, method=None):
        self.captured_schema = schema
        self.captured_method = method
        return self

    def invoke(self, messages):
        self.captured_messages = messages
        return self.payload


class FlakyLLM:
    def __init__(self, fail_times):
        self.fail_times = fail_times
        self.calls = 0

    def with_structured_output(self, schema, method=None):
        return self

    def invoke(self, messages):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError(f"boom {self.calls}")
        return EvidenceAssessment(useful=True, reason="ok")


def test_assess_evidence_returns_useful_false():
    llm = FakeStructuredLLM(EvidenceAssessment(useful=False, reason="只沾边"))
    a = assess_evidence(llm, "支持以旧换新吗", [{"n": 1, "answer": "退货政策"}])
    assert a.useful is False and a.reason == "只沾边"


def test_assess_evidence_retries_once_then_raises():
    llm = FlakyLLM(fail_times=2)
    with pytest.raises(RuntimeError):
        assess_evidence(llm, "q", [])
    assert llm.calls == 2


def test_assess_evidence_succeeds_after_one_retry():
    llm = FlakyLLM(fail_times=1)
    a = assess_evidence(llm, "q", [])
    assert a.useful is True and llm.calls == 2


def test_assess_evidence_with_zero_retries_raises_immediately():
    llm = FlakyLLM(fail_times=1)
    with pytest.raises(RuntimeError):
        assess_evidence(llm, "q", [], retries=0)
    assert llm.calls == 1


def test_assess_evidence_uses_json_mode_and_lists_question_and_sources():
    llm = FakeStructuredLLM(EvidenceAssessment(useful=True))
    assess_evidence(
        llm,
        "支持以旧换新吗",
        [
            {
                "n": 1,
                "section_path": "退货政策 > 退款说明",
                "answer": "七天无理由退货",
                "score": 0.9123,
            },
            {
                "n": 2,
                "section_path": "换货说明",
                "answer": "质量问题可换货",
                "score": 0.7788,
            },
        ],
    )

    assert llm.captured_schema is EvidenceAssessment
    assert llm.captured_method == "json_mode"
    assert isinstance(llm.captured_messages[0], SystemMessage)
    assert isinstance(llm.captured_messages[1], HumanMessage)

    prompt = llm.captured_messages[0].content + llm.captured_messages[1].content
    for field_name in ("(question):", "n=", "section_path:", "answer:", "score:"):
        assert field_name in prompt
    assert "支持以旧换新吗" in prompt
    assert "n=1" in prompt
    assert "退货政策 > 退款说明" in prompt
    assert "七天无理由退货" in prompt
    assert "0.9123" in prompt
    assert "n=2" in prompt
    assert "换货说明" in prompt
    assert "质量问题可换货" in prompt
    assert "0.7788" in prompt

    instruction = llm.captured_messages[0].content
    for phrase in ("证据不足", "只沾边", "互相矛盾", "需要库外知识", "useful=false", "JSON"):
        assert phrase in instruction


def test_assess_evidence_handles_missing_optional_source_fields():
    llm = FakeStructuredLLM(EvidenceAssessment(useful=True))
    assess_evidence(llm, "支持以旧换新吗", [{"n": 3, "answer": "退货政策"}])
    prompt = llm.captured_messages[0].content + llm.captured_messages[1].content
    assert "n=3" in prompt and "退货政策" in prompt
