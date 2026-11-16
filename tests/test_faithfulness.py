import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from app.services.faithfulness import FaithfulnessVerdict, judge_faithfulness


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
        return FaithfulnessVerdict(faithful=True, unsupported_claims=[])


def test_judge_returns_unfaithful_verdict():
    llm = FakeStructuredLLM(
        FaithfulnessVerdict(
            faithful=False,
            unsupported_claims=["承诺当天到账"],
            reason="证据没有",
        )
    )
    v = judge_faithfulness(
        llm, "退款多久", "当天到账", [{"n": 1, "answer": "1-3 个工作日"}]
    )
    assert v.faithful is False and v.unsupported_claims == ["承诺当天到账"]


def test_judge_passes_json_mode_and_lists_question_answer_and_evidence():
    llm = FakeStructuredLLM(FaithfulnessVerdict(faithful=True))
    judge_faithfulness(
        llm,
        "退款多久到账",
        "1-3 个工作日到账 [1]",
        [
            {
                "n": 1,
                "chunk_id": 7,
                "section_path": "退货政策 > 退款说明",
                "question": "退款多久到账",
                "answer": "验收通过后 1-3 个工作日内原路退回",
            }
        ],
    )

    assert llm.captured_schema is FaithfulnessVerdict
    assert llm.captured_method == "json_mode"
    assert isinstance(llm.captured_messages[0], SystemMessage)
    assert isinstance(llm.captured_messages[1], HumanMessage)

    prompt = llm.captured_messages[0].content + llm.captured_messages[1].content
    for field_name in (
        "(question):",
        "(answer):",
        "n=1",
        "chunk_id:7",
        "section_path:",
        "question:",
        "answer:",
    ):
        assert field_name in prompt
    assert "退款多久到账" in prompt
    assert "1-3 个工作日到账 [1]" in prompt
    assert "退货政策 > 退款说明" in prompt
    assert "验收通过后 1-3 个工作日内原路退回" in prompt


def test_judge_instruction_requires_evidence_only_and_json():
    llm = FakeStructuredLLM(FaithfulnessVerdict(faithful=True))
    judge_faithfulness(llm, "退款多久", "1-3 个工作日", [])

    instruction = llm.captured_messages[0].content
    for phrase in (
        "只依据",
        "逐句",
        "事实性陈述",
        "unsupported_claims",
        "faithful=false",
        "JSON",
    ):
        assert phrase in instruction
    # Empty evidence means no claim can be supported.
    assert "(无可用证据)" in llm.captured_messages[1].content


def test_judge_handles_missing_optional_citation_fields():
    llm = FakeStructuredLLM(FaithfulnessVerdict(faithful=True))
    judge_faithfulness(llm, "退款多久", "1-3 个工作日", [{"n": 3}])
    prompt = llm.captured_messages[0].content + llm.captured_messages[1].content
    assert "n=3" in prompt


def test_judge_forces_unfaithful_when_unsupported_claims_present():
    llm = FakeStructuredLLM(
        FaithfulnessVerdict(faithful=True, unsupported_claims=["当天到账"])
    )
    v = judge_faithfulness(llm, "退款多久", "当天到账", [])
    assert v.faithful is False
    assert v.unsupported_claims == ["当天到账"]


def test_judge_retries_once_then_raises():
    llm = FlakyLLM(fail_times=2)
    with pytest.raises(RuntimeError):
        judge_faithfulness(llm, "q", "a", [])
    assert llm.calls == 2


def test_judge_succeeds_after_one_retry():
    llm = FlakyLLM(fail_times=1)
    v = judge_faithfulness(llm, "q", "a", [])
    assert v.faithful is True and llm.calls == 2


def test_judge_with_zero_retries_raises_immediately():
    llm = FlakyLLM(fail_times=1)
    with pytest.raises(RuntimeError):
        judge_faithfulness(llm, "q", "a", [], retries=0)
    assert llm.calls == 1


def test_verdict_defaults():
    v = FaithfulnessVerdict(faithful=True)
    assert v.unsupported_claims == [] and v.reason == ""
