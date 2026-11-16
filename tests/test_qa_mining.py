import json

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from app.services.qa_mining import (QA_EXTRACTION_SYSTEM_PROMPT, ConversationTranscript,
                                    ExtractedQA, QaExtractionBatch, dedup_questions,
                                    extract_qa_batch, normalize_category,
                                    normalize_question)


class FakeLLM:
    def with_structured_output(self, schema, method=None):
        class S:
            def invoke(self, messages):
                return schema(items=[{"question": "邮费是多少", "answer": "满99包邮", "category": "物流"},
                                     {"question": "邮费是多少?", "answer": "满99包邮", "category": "乱七八糟"}])
        return S()


def test_extract_normalizes_category():
    out = extract_qa_batch(FakeLLM(), [ConversationTranscript("c1", [("user", "邮费")])])
    assert [x.question for x in out] == ["邮费是多少", "邮费是多少?"]
    assert out[1].category == "其他"


def test_normalize_question():
    assert normalize_question(" 邮费，是多少？ ") == "邮费是多少"


class FakeEmbedder:
    def embed(self, texts):
        # two clearly separated directions: 邮费 vs 尺码
        out = []
        for t in texts:
            v = [0.0] * 1024
            v[0 if "邮费" in t else 1] = 1.0
            out.append(v)
        return out


def test_dedup_exact_and_existing():
    keep = dedup_questions(["邮费是多少", "邮费是多少?", "尺码怎么选"], ["邮费是多少"], FakeEmbedder())
    assert keep == [False, False, True]


def test_dedup_near_duplicate(monkeypatch):
    class E:
        def embed(self, texts):
            return [[1.0] + [0.0] * 1023 if "邮费" in t else [0.0, 1.0] + [0.0] * 1022 for t in texts]
    assert dedup_questions(["邮费是多少", "邮费怎么算"], [], E(), threshold=0.9) == [True, False]


# --- supplementary contract tests ---

class CapturingLLM:
    def __init__(self, result):
        self.result = result
        self.schema = None
        self.method = None
        self.messages = None

    def with_structured_output(self, schema, method=None):
        llm = self

        class S:
            def invoke(self, messages):
                llm.messages = messages
                return llm.result
        self.schema = schema
        self.method = method
        return S()


def test_extract_uses_json_mode_and_serializes_transcripts():
    llm = CapturingLLM(QaExtractionBatch(items=[
        ExtractedQA(question="如何退货?", answer="7 天内申请", category="售后")
    ]))
    transcripts = [
        ConversationTranscript("conv-1", [("user", "我想退货"), ("assistant", "签收后 7 天内可申请")]),
    ]
    out = extract_qa_batch(llm, transcripts)

    assert llm.schema is QaExtractionBatch
    assert llm.method == "json_mode"
    assert isinstance(llm.messages[0], SystemMessage)
    assert llm.messages[0].content == QA_EXTRACTION_SYSTEM_PROMPT
    assert isinstance(llm.messages[1], HumanMessage)
    payload = json.loads(llm.messages[1].content)
    assert payload == [
        {"source_ref": "conv-1",
         "messages": [{"role": "user", "content": "我想退货"},
                      {"role": "assistant", "content": "签收后 7 天内可申请"}]}
    ]
    assert [x.category for x in out] == ["售后"]


def test_extract_drops_blank_question_or_answer():
    llm = CapturingLLM(QaExtractionBatch(items=[
        {"question": "  ", "answer": "有答案", "category": "商品"},
        {"question": "有问题", "answer": "   ", "category": "商品"},
        {"question": " 发票怎么开? ", "answer": " 下单时填写抬头 ", "category": " 商品 "},
    ]))
    out = extract_qa_batch(llm, [ConversationTranscript("c1", [("user", "开发票")])])
    assert [(x.question, x.answer, x.category) for x in out] == [
        ("发票怎么开?", "下单时填写抬头", "商品")
    ]


def test_prompt_is_json_mode_compatible():
    assert "JSON" in QA_EXTRACTION_SYSTEM_PROMPT
    for category in ("售后", "物流", "支付", "商品", "其他"):
        assert category in QA_EXTRACTION_SYSTEM_PROMPT
    assert '"items"' in QA_EXTRACTION_SYSTEM_PROMPT
    assert '{"items": []}' in QA_EXTRACTION_SYSTEM_PROMPT


def test_normalize_category_known_and_unknown():
    assert normalize_category("物流") == "物流"
    assert normalize_category(" 售后 ") == "售后"
    assert normalize_category("随便") == "其他"
    assert normalize_category("") == "其他"
    assert normalize_category(None) == "其他"


def test_normalize_question_keeps_chinese_removes_space_and_punctuation():
    assert normalize_question("运费！！怎么算？") == "运费怎么算"
    assert normalize_question("ＢＧＡ是什么") == "bga是什么"
    assert normalize_question("a b-c_d") == "abcd"


def test_dedup_returns_empty_for_no_candidates():
    class BoomEmbedder:
        def embed(self, texts):
            raise AssertionError("should not embed when there is nothing to compare")

    assert dedup_questions([], ["邮费是多少"], BoomEmbedder()) == []


def test_dedup_exact_match_normalizes_existing_too():
    keep = dedup_questions(["邮费是多少"], ["邮费，是多少？"], FakeEmbedder())
    assert keep == [False]


def test_dedup_handles_zero_norm_vectors_safely():
    class ZeroEmbedder:
        def embed(self, texts):
            return [[0.0] * 1024 for _ in texts]

    keep = dedup_questions(["尺码怎么选"], ["邮费是多少"], ZeroEmbedder(), threshold=0.92)
    assert keep == [True]


def test_dedup_rejects_vector_count_mismatch():
    class ShortEmbedder:
        def embed(self, texts):
            return [[1.0, 0.0]]

    with pytest.raises(ValueError):
        dedup_questions(["尺码怎么选", "怎么开发票"], [], ShortEmbedder())


def test_dedup_drops_later_near_duplicate_against_kept_candidate():
    class E:
        def embed(self, texts):
            return [[1.0, 0.0] if "邮费" in t else [0.0, 1.0] for t in texts]

    keep = dedup_questions(["邮费是多少", "邮费收费标准", "尺码怎么选"], [], E(), threshold=0.9)
    assert keep == [True, False, True]
