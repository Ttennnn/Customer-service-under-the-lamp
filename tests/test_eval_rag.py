"""Offline tests for the four-strategy RAG evaluation job.

Everything here runs without Milvus/MySQL/LLM: retrieval, query understanding,
the evidence gate and the faithfulness judge are injected as fakes, and the
faith-case writer is monkeypatched.
"""

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.jobs import eval_rag
from app.jobs.eval_rag import (
    _build_parser,
    build_eval_answer_messages,
    citation_validity,
    parse_strategies,
    score_retrieval_case,
    section_paths_in_order,
)
from app.prompts.customer_service import render_answer_system_prompt
from app.rag.query_understanding import QueryAnalysis
from app.rag.retrieval import RetrievalResult, RetrievalStrategy, RetrievedChunk
from app.rag.self_check import EvidenceAssessment
from app.services.faithfulness import FaithfulnessVerdict


def make_chunk(chunk_id: int, section_path: str | None, score: float = 0.9):
    return RetrievedChunk(
        chunk_id, score, "问", "答", section_path, "售后", "faq"
    )


def make_analysis(query: str, rewritten_by: str = "llm") -> QueryAnalysis:
    return QueryAnalysis(
        original=query,
        rewritten=query,
        keywords=["关键词"],
        rewritten_by=rewritten_by,
    )


def make_case(
    case_id: str = "A01",
    bucket: str = "A_policy",
    query: str = "退款多久到账?",
    answerable: bool = True,
    paths: list[str] | None = None,
) -> dict:
    return {
        "id": case_id,
        "bucket": bucket,
        "difficulty": "easy",
        "query": query,
        "answerable": answerable,
        "ground_truth": {"section_paths": paths if paths is not None else ["S1"]},
        "filters": None,
    }


def test_section_paths_in_order_dedupes_and_skips_missing():
    chunks = [
        make_chunk(1, "A > X"),
        make_chunk(2, None),
        make_chunk(3, "A > X"),
        make_chunk(4, "A > Y"),
    ]
    assert section_paths_in_order(chunks) == ["A > X", "A > Y"]


def test_score_retrieval_uses_unique_section_ranks():
    chunks = [
        make_chunk(1, "X"),
        make_chunk(2, "A"),
        make_chunk(3, "A"),
        make_chunk(4, "B"),
    ]
    case = make_case(paths=["A", "B"])
    score = score_retrieval_case(case, chunks)
    assert score.recall_at_5 == 1.0
    assert score.recall_at_10 == 1.0
    assert score.mrr_at_10 == 0.5
    assert score.first_relevant_rank == 2


def test_score_retrieval_without_ground_truth_does_not_divide_by_zero():
    score = score_retrieval_case(
        make_case(answerable=False, paths=[]), [make_chunk(1, "X")]
    )
    assert score.recall_at_5 == 0.0
    assert score.recall_at_10 == 0.0
    assert score.mrr_at_10 == 0.0
    assert score.first_relevant_rank is None


def test_citation_validity_ignores_out_of_range_and_counts_repeats():
    assert citation_validity("[1] [1]", [{"n": 1}]) == 1.0
    assert citation_validity("", [{"n": 1}]) is None
    assert citation_validity("[2]", [{"n": 1}]) == 0.0


def test_build_eval_answer_messages_uses_answer_prompt_and_numbers_evidence():
    messages = build_eval_answer_messages(
        "退款多久到账?",
        [
            {
                "n": 1,
                "section_path": "退货政策 > 退款说明",
                "question": "退款多久",
                "answer": "1-3 个工作日",
            }
        ],
    )
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == render_answer_system_prompt()
    assert isinstance(messages[1], HumanMessage)
    assert "退款多久到账?" in messages[1].content
    assert "[1] 章节:退货政策 > 退款说明" in messages[1].content


def test_build_eval_answer_messages_marks_empty_evidence():
    messages = build_eval_answer_messages("问题", [])
    assert "(无可用证据)" in messages[1].content


def test_parser_defaults_and_flags(tmp_path):
    args = _build_parser().parse_args(
        [
            "--out",
            str(tmp_path / "report.md"),
            "--limit",
            "3",
            "--retrieval-only",
            "--no-faith-cases",
        ]
    )
    assert args.strategies == "dense,bm25,hybrid,hybrid_rerank"
    assert args.limit == 3
    assert args.retrieval_only is True
    assert args.no_faith_cases is True


def test_parse_strategies_preserves_order_and_rejects_unknown():
    assert parse_strategies("dense, hybrid") == [
        RetrievalStrategy.DENSE,
        RetrievalStrategy.HYBRID,
    ]
    with pytest.raises(ValueError):
        parse_strategies("dense,nope")


class _NoGenerateLLM:
    def invoke(self, messages):  # pragma: no cover - must not be called
        raise AssertionError("generation must not run after a refusal")


class _AnswerLLM:
    def __init__(self, content: str):
        self.content = content
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(content=self.content)


def test_run_case_refusal_skips_generation(monkeypatch):
    chunk = make_chunk(1, "S1")
    monkeypatch.setattr(
        eval_rag,
        "retrieve_hybrid",
        lambda *a, **k: RetrievalResult(
            [chunk], RetrievalStrategy.DENSE, make_analysis("q")
        ),
    )
    calls = []

    def fake_assess(llm, question, sources):
        calls.append((question, sources))
        return EvidenceAssessment(useful=False, reason="证据不足")

    monkeypatch.setattr(eval_rag, "assess_evidence", fake_assess)
    run = eval_rag._run_case(
        make_case(), RetrievalStrategy.DENSE, make_analysis("q"), _NoGenerateLLM()
    )
    assert run.status == "refusal"
    assert run.answer is None
    assert calls and calls[0][1][0]["n"] == 1


def test_run_case_generates_then_judges_faithfulness(monkeypatch):
    chunk = make_chunk(1, "S1")
    monkeypatch.setattr(
        eval_rag,
        "retrieve_hybrid",
        lambda *a, **k: RetrievalResult(
            [chunk], RetrievalStrategy.HYBRID, make_analysis("q")
        ),
    )
    monkeypatch.setattr(
        eval_rag,
        "assess_evidence",
        lambda *a, **k: EvidenceAssessment(useful=True),
    )
    monkeypatch.setattr(
        eval_rag,
        "judge_faithfulness",
        lambda *a, **k: FaithfulnessVerdict(
            faithful=False, unsupported_claims=["当天到账"], reason="编造"
        ),
    )
    llm = _AnswerLLM("1-3 个工作日 [1]")
    run = eval_rag._run_case(
        make_case(), RetrievalStrategy.HYBRID, make_analysis("q"), llm
    )
    assert llm.calls == 1
    assert run.status == "answered"
    assert run.faithful is False
    assert run.faith_reason == "编造"
    assert run.citation_validity == 1.0
    assert run.citations[0]["chunk_id"] == 1


def test_run_case_detects_model_refusal_without_judging(monkeypatch):
    chunk = make_chunk(1, "S1")
    monkeypatch.setattr(
        eval_rag,
        "retrieve_hybrid",
        lambda *a, **k: RetrievalResult(
            [chunk], RetrievalStrategy.DENSE, make_analysis("q")
        ),
    )
    monkeypatch.setattr(
        eval_rag,
        "assess_evidence",
        lambda *a, **k: EvidenceAssessment(useful=True),
    )
    monkeypatch.setattr(
        eval_rag,
        "judge_faithfulness",
        lambda *a, **k: pytest.fail("model refusal must not be judged"),
    )
    run = eval_rag._run_case(
        make_case(),
        RetrievalStrategy.DENSE,
        make_analysis("q"),
        _AnswerLLM("知识库暂未收录该信息,建议联系人工客服。"),
    )
    assert run.status == "model_refusal"
    assert run.faithful is None


def test_run_case_records_self_check_error(monkeypatch):
    chunk = make_chunk(1, "S1")
    monkeypatch.setattr(
        eval_rag,
        "retrieve_hybrid",
        lambda *a, **k: RetrievalResult(
            [chunk], RetrievalStrategy.DENSE, make_analysis("q")
        ),
    )
    monkeypatch.setattr(
        eval_rag,
        "assess_evidence",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("self-check down")),
    )
    run = eval_rag._run_case(
        make_case(), RetrievalStrategy.DENSE, make_analysis("q"), _NoGenerateLLM()
    )
    assert run.status == "self_check_error"
    assert "self-check down" in run.error


def test_run_case_records_judge_error_without_faithfulness(monkeypatch):
    chunk = make_chunk(1, "S1")
    monkeypatch.setattr(
        eval_rag,
        "retrieve_hybrid",
        lambda *a, **k: RetrievalResult(
            [chunk], RetrievalStrategy.DENSE, make_analysis("q")
        ),
    )
    monkeypatch.setattr(
        eval_rag,
        "assess_evidence",
        lambda *a, **k: EvidenceAssessment(useful=True),
    )
    monkeypatch.setattr(
        eval_rag,
        "judge_faithfulness",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("judge down")),
    )
    run = eval_rag._run_case(
        make_case(),
        RetrievalStrategy.DENSE,
        make_analysis("q"),
        _AnswerLLM("答案 [1]"),
    )
    assert run.status == "judge_error"
    assert run.answer == "答案 [1]"
    assert run.citation_validity == 1.0
    assert run.faithful is None


def test_retrieval_summary_excludes_rerank_failures():
    from app.jobs.eval_rag import CaseRun, RetrievalScore

    def run(failed: bool) -> CaseRun:
        return CaseRun(
            case_id="A01",
            bucket="A_policy",
            query="q",
            answerable=True,
            strategy="hybrid_rerank",
            status="retrieval_only",
            score=RetrievalScore(1.0, 1.0, 1.0, 1),
            rerank_failed=failed,
        )

    summary = eval_rag._retrieval_summary(
        [run(False), run(True)], "hybrid_rerank"
    )
    assert summary["n"] == 1
    assert summary["recall_at_5"] == 1.0


def test_generation_summary_excludes_rerank_failures():
    from app.jobs.eval_rag import CaseRun

    def run(
        status: str,
        *,
        faithful: bool | None = None,
        validity: float | None = None,
        failed: bool = False,
    ) -> CaseRun:
        return CaseRun(
            case_id="A01",
            bucket="A_policy",
            query="q",
            answerable=True,
            strategy="hybrid_rerank",
            status=status,
            faithful=faithful,
            citation_validity=validity,
            rerank_failed=failed,
        )

    runs = [
        run("answered", faithful=True, validity=1.0),
        run("answered", faithful=False, validity=0.0, failed=True),
        run("refusal", failed=True),
    ]
    summary = eval_rag._generation_summary(runs, "hybrid_rerank")
    # Only the successful rerank run is evaluable: including the fallback runs
    # would report refusal_rate=1/3, faithfulness=0.5 and validity=0.5.
    assert summary["n"] == 1
    assert summary["refusal_rate"] == 0.0
    assert summary["faithfulness"] == 1.0
    assert summary["citation_validity"] == 1.0
    assert summary["model_refusals"] == 0


def _fake_settings():
    return SimpleNamespace(
        model_name="deepseek-chat",
        embedding_model="BAAI/bge-m3",
        rerank_model="BAAI/bge-reranker-v2-m3",
        hybrid_recall_top_k=50,
        hybrid_rrf_k=60,
    )


def _write_cases(path, cases):
    path.write_text(
        json.dumps({"version": 1, "cases": cases}, ensure_ascii=False),
        encoding="utf-8",
    )


def test_main_retrieval_only_writes_report_with_fallback_counts(
    monkeypatch, tmp_path
):
    cases_path = tmp_path / "cases.json"
    out = tmp_path / "report.md"
    cases = [
        make_case("A01", query="q1", paths=["S1"]),
        make_case("A02", query="q2", paths=["S1"]),
    ]
    _write_cases(cases_path, cases)

    def fake_retrieve(query, *, strategy, analysis, **kwargs):
        chunks = [make_chunk(1, "S1"), make_chunk(2, "S1")]
        failed = strategy is RetrievalStrategy.HYBRID_RERANK and query == "q2"
        return RetrievalResult(chunks, RetrievalStrategy(strategy), analysis, failed)

    monkeypatch.setattr(eval_rag, "build_llm", lambda *a, **k: object())
    monkeypatch.setattr(eval_rag, "get_settings", _fake_settings)
    monkeypatch.setattr(
        eval_rag,
        "analyze",
        lambda query, llm=None: make_analysis(
            query, "fallback" if query == "q1" else "llm"
        ),
    )
    monkeypatch.setattr(eval_rag, "retrieve_hybrid", fake_retrieve)
    monkeypatch.setattr(
        eval_rag,
        "get_session_factory",
        lambda: pytest.fail("--retrieval-only must not open a DB session"),
    )

    code = eval_rag.main(
        ["--cases", str(cases_path), "--out", str(out), "--retrieval-only"]
    )
    assert code == 0
    report = out.read_text(encoding="utf-8")
    assert "| dense |" in report
    assert "| hybrid_rerank |" in report
    assert "## Rerank fallback" in report
    assert "- fallback runs: 1" in report
    assert "- case ids: A02" in report
    assert "- cases with fallback rewrite: 1 / 2" in report
    assert "skipped (--retrieval-only)" in report


def test_main_writes_faith_cases_and_counts_new_vs_recurrence(
    monkeypatch, tmp_path
):
    cases_path = tmp_path / "cases.json"
    out = tmp_path / "report.md"
    _write_cases(cases_path, [make_case(paths=["S1"])])

    monkeypatch.setattr(eval_rag, "build_llm", lambda *a, **k: _AnswerLLM("答案 [1]"))
    monkeypatch.setattr(eval_rag, "get_settings", _fake_settings)
    monkeypatch.setattr(
        eval_rag,
        "analyze",
        lambda query, llm=None: make_analysis(query),
    )
    monkeypatch.setattr(
        eval_rag,
        "retrieve_hybrid",
        lambda query, *, strategy, analysis, **kwargs: RetrievalResult(
            [make_chunk(1, "S1")], RetrievalStrategy(strategy), analysis
        ),
    )
    monkeypatch.setattr(
        eval_rag,
        "assess_evidence",
        lambda *a, **k: EvidenceAssessment(useful=True),
    )
    monkeypatch.setattr(
        eval_rag,
        "judge_faithfulness",
        lambda *a, **k: FaithfulnessVerdict(faithful=False, reason="编造"),
    )

    class FakeSession:
        def commit(self):
            pass

        def rollback(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(
        eval_rag, "get_session_factory", lambda: lambda: FakeSession()
    )
    payloads = []

    def fake_upsert(session, payload):
        payloads.append(payload)
        return SimpleNamespace(seen_count=len(payloads))

    monkeypatch.setattr(eval_rag, "upsert_faith_case", fake_upsert)

    code = eval_rag.main(
        [
            "--cases",
            str(cases_path),
            "--out",
            str(out),
            "--strategies",
            "dense,bm25",
        ]
    )
    assert code == 0
    assert len(payloads) == 2
    first = payloads[0]
    assert first.eval_id == "A01"
    assert first.bucket == "A_policy"
    assert first.strategy == "dense"
    assert first.answer == "答案 [1]"
    assert first.reason == "编造"
    assert first.judge_model == "deepseek-chat"
    assert set(first.citations[0]) >= {
        "n",
        "chunk_id",
        "section_path",
        "question",
        "answer",
        "category",
        "content_type",
        "score",
    }
    report = out.read_text(encoding="utf-8")
    assert "- new: 1" in report
    assert "- recurrence: 1" in report


def test_main_no_faith_cases_skips_writes(monkeypatch, tmp_path):
    cases_path = tmp_path / "cases.json"
    out = tmp_path / "report.md"
    _write_cases(cases_path, [make_case(paths=["S1"])])

    monkeypatch.setattr(eval_rag, "build_llm", lambda *a, **k: _AnswerLLM("答案 [1]"))
    monkeypatch.setattr(eval_rag, "get_settings", _fake_settings)
    monkeypatch.setattr(
        eval_rag, "analyze", lambda query, llm=None: make_analysis(query)
    )
    monkeypatch.setattr(
        eval_rag,
        "retrieve_hybrid",
        lambda query, *, strategy, analysis, **kwargs: RetrievalResult(
            [make_chunk(1, "S1")], RetrievalStrategy(strategy), analysis
        ),
    )
    monkeypatch.setattr(
        eval_rag,
        "assess_evidence",
        lambda *a, **k: EvidenceAssessment(useful=True),
    )
    monkeypatch.setattr(
        eval_rag,
        "judge_faithfulness",
        lambda *a, **k: FaithfulnessVerdict(faithful=False, reason="编造"),
    )
    monkeypatch.setattr(
        eval_rag,
        "get_session_factory",
        lambda: pytest.fail("--no-faith-cases must not open a DB session"),
    )
    monkeypatch.setattr(
        eval_rag,
        "upsert_faith_case",
        lambda *a, **k: pytest.fail("--no-faith-cases must not write"),
    )

    code = eval_rag.main(
        [
            "--cases",
            str(cases_path),
            "--out",
            str(out),
            "--strategies",
            "dense",
            "--no-faith-cases",
        ]
    )
    assert code == 0
    assert "skipped (--no-faith-cases)" in out.read_text(encoding="utf-8")
