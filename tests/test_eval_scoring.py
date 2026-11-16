"""Pure scoring-helper tests for the evaluation harness (no network)."""

from app.jobs.eval_qa_mining import score_mining_case, summarize_dedup
from app.jobs.eval_rag import citation_validity, score_retrieval_case
from app.jobs.eval_retrieval import score_faq_case
from app.rag.retrieval import RetrievedChunk
from app.services.qa_mining import ExtractedQA


def make_chunk(chunk_id: int, section_path: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id, 0.9, "问", "答", section_path, "售后", "faq")


def test_recall_dedupes_same_section_and_handles_multi_hop():
    chunks = [make_chunk(1, "A > X"), make_chunk(2, "A > X"), make_chunk(3, "A > Y")]
    case = {"answerable": True, "ground_truth": {"section_paths": ["A > X", "A > Y"]}}
    s = score_retrieval_case(case, chunks[:2])
    assert s.recall_at_5 == 0.5 and s.mrr_at_10 == 1.0


def test_citation_validity_counts_unknown_numbers():
    assert citation_validity("见 [1] 和 [9]", [{"n": 1}]) == 0.5
    assert citation_validity("没有编号", [{"n": 1}]) is None


def test_score_hit_by_keyword():
    hits = [
        RetrievedChunk(
            1,
            0.7,
            "邮费怎么算?",
            "满 99 元包邮,未满收 8 元。",
            "物流/邮费",
            "物流",
            "faq",
        )
    ]
    assert score_faq_case(
        {"query": "邮费是多少", "expect_hit": True, "expect_any": ["包邮", "运费"]},
        hits,
    )
    assert not score_faq_case(
        {"query": "x", "expect_hit": True, "expect_any": ["发票"]}, hits
    )


def test_score_negative_case():
    assert score_faq_case({"query": "老板是谁", "expect_hit": False}, [])


def test_score_negative_case_fails_when_hits_present():
    hits = [RetrievedChunk(1, 0.7, "问", "答", None, "售后", "faq")]
    assert not score_faq_case({"query": "老板是谁", "expect_hit": False}, hits)


def test_score_hit_case_requires_a_hit():
    assert not score_faq_case(
        {"query": "邮费是多少", "expect_hit": True, "expect_any": ["包邮"]}, []
    )


def test_score_mining_case_requires_every_expected_item():
    got = [
        ExtractedQA(
            question="退货可以退吗?", answer="签收后 7 天内可申请退货", category="售后"
        ),
        ExtractedQA(
            question="退款多久到账?", answer="1-3 个工作日原路退回", category="售后"
        ),
    ]
    case = {
        "source_ref": "return-policy-1",
        "expected": [
            {"question_any": ["退货"], "answer_any": ["7 天"]},
            {"question_any": ["退款"], "answer_any": ["1-3", "工作日"]},
        ],
    }
    assert score_mining_case(case, got)
    missing = {"expected": [{"question_any": ["发票"], "answer_any": ["电子发票"]}]}
    assert not score_mining_case(missing, got)


def test_score_mining_case_does_not_cross_match_items():
    got = [
        ExtractedQA(
            question="退货怎么退?", answer="退款 1-3 个工作日到账", category="售后"
        ),
        ExtractedQA(
            question="发票怎么开?", answer="收货后 7 天内联系客服补开", category="商品"
        ),
    ]
    # The question keyword and the answer keyword must come from the same item.
    assert not score_mining_case(
        {"expected": [{"question_any": ["发票"], "answer_any": ["1-3"]}]}, got
    )


class _SameVectorEmbedder:
    def embed(self, texts):
        return [[1.0] + [0.0] * 1023 for _ in texts]


def test_summarize_dedup_collapses_exact_and_approximate_duplicates():
    kept, discarded, flags = summarize_dedup(
        ["邮费是多少", "邮费是多少?", "邮费怎么算"],
        _SameVectorEmbedder(),
        0.92,
    )
    assert (kept, discarded) == (1, 2)
    assert flags == [True, False, False]


def test_score_mining_case_accepts_a_single_expected_item():
    got = [
        ExtractedQA(question="邮费是多少", answer="满 99 元包邮", category="物流")
    ]
    assert score_mining_case(
        {"question_any": ["邮费"], "answer_any": ["包邮"]}, got
    )
