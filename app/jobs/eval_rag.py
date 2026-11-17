"""Four-strategy RAG evaluation job.

Runs every annotated case in ``tests/eval/rag_cases.json`` through ``dense``,
``bm25``, ``hybrid`` and ``hybrid_rerank`` with the **same** query analysis,
then measures section-level retrieval quality and, unless ``--retrieval-only``
is given, answer quality:

1. ``assess_evidence`` gates generation for every case (including the
   unanswerable D bucket); ``useful=false`` becomes a refusal.
2. Otherwise the answer is generated from the same citation payload the chat
   tool emits, with the tool-free answer system prompt and no tools bound.
3. ``judge_faithfulness`` scores the generated answer and citation validity is
   measured against the evidence numbers actually present.
4. Unfaithful answers are upserted into ``faith_cases`` (unless
   ``--no-faith-cases``), one row per ``eval_id`` as in Task 11.

Reporting rules that are easy to get wrong:

* Retrieval is scored on **unique section paths in first-occurrence order**;
  Recall@K is hit ground-truth paths / total ground-truth paths and MRR@10 is
  the reciprocal rank of the first relevant *section*.
* D-bucket cases have no ground truth: they never enter Recall/MRR and are
  reported separately via refusal accuracy.
* ``(case, strategy)`` runs with ``rerank_failed=True`` are excluded from all
  ``hybrid_rerank`` retrieval and generation aggregates (Recall/MRR,
  Faithfulness, Refusal rate, Citation validity) and reported as fallbacks,
  never counted as successful reranks.
* ``self_check_error`` and ``judge_error`` are excluded from Faithfulness and
  reported separately. A model that self-declares inability is
  ``model_refusal``, separate from the evidence-gate ``refusal``.
* A strategy with zero evaluable runs reports ``n/a`` rather than a fake 0.

This is a manual evaluation job. Importing the module never touches the
network or the database; only :func:`main` resolves settings, services, and
(only when there are unfaithful answers to record) a DB session.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.config import build_llm, get_settings
from app.db.session import get_session_factory
from app.prompts.customer_service import render_answer_system_prompt
from app.rag.query_understanding import QueryAnalysis, analyze
from app.rag.retrieval import RetrievalStrategy, RetrievedChunk, retrieve_hybrid
from app.rag.self_check import assess_evidence
from app.services.faith_cases import FaithCasePayload, upsert_faith_case
from app.services.faithfulness import judge_faithfulness
from app.tools.context import CitationCollector
from app.tools.faq import order_for_prompt

TOP_K = 10
D_BUCKET = "D_absent"
_DEFAULT_CASES = (
    Path(__file__).resolve().parents[2] / "tests" / "eval" / "rag_cases.json"
)
_DEFAULT_OUT = Path(
    "docs/superpowers/reports/2026-11-16-ch04-rag-eval.md"
)
_DEFAULT_STRATEGIES = "dense,bm25,hybrid,hybrid_rerank"
_CITATION_RE = re.compile(r"\[(\d+)\]")

# Conservative markers for a model that declares it cannot answer. Kept small
# so a normal "无法确认具体到账时间,以银行实际为准" caveat is not misread as
# a refusal.
MODEL_REFUSAL_MARKERS = (
    "知识库暂未收录",
    "知识库未收录",
    "知识库没有收录",
    "暂未收录",
    "无法回答",
    "无法确认这个问题",
    "未找到相关",
)

_GENERATION_ATTEMPT_STATUSES = frozenset(
    {"refusal", "answered", "model_refusal", "generation_error", "judge_error"}
)


@dataclass(frozen=True)
class RetrievalScore:
    """Section-level retrieval results for one case/strategy run."""

    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float
    first_relevant_rank: int | None


@dataclass(frozen=True)
class CaseRun:
    """One ``(case, strategy)`` execution, including its generation outcome."""

    case_id: str
    bucket: str
    query: str
    answerable: bool
    strategy: str
    status: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    score: RetrievalScore | None = None
    rerank_failed: bool = False
    analysis_by: str = "llm"
    citations: list[dict] = field(default_factory=list)
    answer: str | None = None
    faithful: bool | None = None
    faith_reason: str = ""
    unsupported_claims: list[str] = field(default_factory=list)
    citation_validity: float | None = None
    refusal_reason: str = ""
    error: str = ""


@dataclass
class FaithStats:
    """Outcome of the ``faith_cases`` ledger phase."""

    enabled: bool = True
    new: int = 0
    recurrence: int = 0
    errors: int = 0
    entries: list[str] = field(default_factory=list)


def section_paths_in_order(chunks: list[RetrievedChunk]) -> list[str]:
    """Return unique non-empty section paths, keeping first-occurrence order.

    Two chunks from the same section are one retrieval unit for Recall/MRR:
    returning the same section twice must not count as two hits.
    """
    seen: set[str] = set()
    paths: list[str] = []
    for chunk in chunks:
        path = chunk.section_path
        if not path or path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return paths


def score_retrieval_case(
    case: dict, chunks: list[RetrievedChunk]
) -> RetrievalScore:
    """Score *chunks* against one case's ground-truth section paths.

    Chunks are truncated to the top 10 before deduplication. With no
    ground-truth paths (the D bucket) every metric is zero and
    ``first_relevant_rank`` is ``None``; callers must report D separately
    instead of aggregating these zeros.
    """
    ground_truth = set(
        (case.get("ground_truth") or {}).get("section_paths") or []
    )
    unique_paths = section_paths_in_order(list(chunks)[:TOP_K])

    def recall_at(k: int) -> float:
        if not ground_truth:
            return 0.0
        hits = {path for path in unique_paths[:k] if path in ground_truth}
        return len(hits) / len(ground_truth)

    first_rank = next(
        (
            rank
            for rank, path in enumerate(unique_paths, 1)
            if path in ground_truth
        ),
        None,
    )
    return RetrievalScore(
        recall_at_5=recall_at(5),
        recall_at_10=recall_at(10),
        mrr_at_10=1.0 / first_rank if first_rank else 0.0,
        first_relevant_rank=first_rank,
    )


def citation_validity(answer: str, sources: list[dict]) -> float | None:
    """Fraction of ``[n]`` occurrences whose ``n`` exists in *sources*.

    ``None`` means the answer cited nothing, so there is no validity signal.
    Repeated occurrences are counted, matching how a reader would encounter
    them.
    """
    numbers = [int(number) for number in _CITATION_RE.findall(answer or "")]
    if not numbers:
        return None
    valid_numbers = {source.get("n") for source in sources}
    return sum(1 for number in numbers if number in valid_numbers) / len(numbers)


def _format_evidence(evidence: list[dict]) -> str:
    if not evidence:
        return "(无可用证据)"
    blocks: list[str] = []
    for source in evidence:
        blocks.append(
            f"[{source.get('n')}] 章节:{source.get('section_path') or '未知'}\n"
            f"问:{source.get('question') or '(无)'}\n"
            f"答:{source.get('answer') or '(无)'}"
        )
    return "\n\n".join(blocks)


def build_eval_answer_messages(
    question: str, evidence: list[dict]
) -> list[BaseMessage]:
    """Build the tool-free generation prompt for one evaluation case.

    The system message is exactly :func:`render_answer_system_prompt`; no
    tools are bound and the evidence uses the same numbered ``[n]`` shape as
    ``query_faq`` so citation validity has the same meaning as in chat.
    """
    content = (
        f"用户问题:{question}\n\n"
        f"本轮知识库证据(共 {len(evidence)} 条):\n\n"
        f"{_format_evidence(evidence)}\n\n"
        "请只依据以上证据回答;每个事实后标注对应证据编号 [n];"
        "证据不足时明确说明无法确认,不要使用证据之外的知识。"
    )
    return [
        SystemMessage(content=render_answer_system_prompt()),
        HumanMessage(content=content),
    ]


def _message_text(message) -> str:
    """Extract text from an AIMessage-like object, tolerating list content."""
    content = getattr(message, "content", message)
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content or "")


def _is_model_refusal(answer: str) -> bool:
    return any(marker in (answer or "") for marker in MODEL_REFUSAL_MARKERS)


def _describe_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _safe_analyze(query: str, llm) -> QueryAnalysis:
    """Run query understanding, degrading to the deterministic fallback.

    ``analyze`` already catches its own failures; this guard only protects the
    evaluation run from an unexpected injected error so one case can never
    abort the report.
    """
    try:
        return analyze(query, llm=llm)
    except Exception:  # noqa: BLE001 - an eval run reports, never crashes
        return QueryAnalysis(
            original=query,
            rewritten=query,
            keywords=[],
            rewritten_by="fallback",
        )


def _run_case(
    case: dict,
    strategy: RetrievalStrategy,
    analysis: QueryAnalysis,
    judge_llm,
    *,
    retrieval_only: bool = False,
) -> CaseRun:
    """Run one case under one strategy and record its full outcome."""
    case_id = str(case.get("id") or "")
    bucket = str(case.get("bucket") or "")
    query = str(case.get("query") or "")
    answerable = bool(case.get("answerable", True))
    filters = case.get("filters") or {}

    try:
        result = retrieve_hybrid(
            query,
            strategy=strategy,
            analysis=analysis,
            category=filters.get("category"),
            content_type=filters.get("content_type"),
            top_k=TOP_K,
        )
    except Exception as exc:  # noqa: BLE001 - keep the report complete
        return CaseRun(
            case_id=case_id,
            bucket=bucket,
            query=query,
            answerable=answerable,
            strategy=strategy.value,
            status="retrieval_error",
            analysis_by=analysis.rewritten_by,
            error=_describe_error(exc),
        )

    chunks = list(result.chunks)
    score = score_retrieval_case(case, chunks)
    collector = CitationCollector()
    collector.add(order_for_prompt(chunks))
    citations = collector.payload()

    base = CaseRun(
        case_id=case_id,
        bucket=bucket,
        query=query,
        answerable=answerable,
        strategy=strategy.value,
        status="retrieval_only" if retrieval_only else "pending",
        chunks=chunks,
        score=score,
        rerank_failed=bool(result.rerank_failed),
        analysis_by=analysis.rewritten_by,
        citations=citations,
    )
    if retrieval_only:
        return base

    try:
        assessment = assess_evidence(judge_llm, query, citations)
    except Exception as exc:  # noqa: BLE001 - infra failure is its own status
        return replace(
            base, status="self_check_error", error=_describe_error(exc)
        )
    if not assessment.useful:
        return replace(
            base, status="refusal", refusal_reason=assessment.reason
        )

    try:
        response = judge_llm.invoke(
            build_eval_answer_messages(query, citations)
        )
    except Exception as exc:  # noqa: BLE001 - generation failure is reported
        return replace(
            base, status="generation_error", error=_describe_error(exc)
        )

    answer = _message_text(response)
    validity = citation_validity(answer, citations)
    if _is_model_refusal(answer):
        return replace(
            base,
            status="model_refusal",
            answer=answer,
            citation_validity=validity,
        )

    try:
        verdict = judge_faithfulness(judge_llm, query, answer, citations)
    except Exception as exc:  # noqa: BLE001 - keep it out of Faithfulness
        return replace(
            base,
            status="judge_error",
            answer=answer,
            citation_validity=validity,
            error=_describe_error(exc),
        )

    return replace(
        base,
        status="answered",
        answer=answer,
        citation_validity=validity,
        faithful=verdict.faithful,
        faith_reason=verdict.reason,
        unsupported_claims=list(verdict.unsupported_claims),
    )


def _mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _retrieval_summary(runs: list[CaseRun], strategy: str) -> dict:
    """Aggregate retrieval metrics, excluding D and rerank fallbacks."""
    scored = [
        run
        for run in runs
        if run.strategy == strategy
        and run.answerable
        and not run.rerank_failed
        and run.score is not None
    ]
    with_evidence = sum(
        1 for run in scored if run.score is not None and run.score.recall_at_10 > 0
    )
    return {
        "n": len(scored),
        "recall_at_5": _mean([run.score.recall_at_5 for run in scored]),
        "recall_at_10": _mean([run.score.recall_at_10 for run in scored]),
        "mrr_at_10": _mean([run.score.mrr_at_10 for run in scored]),
        "cases_with_evidence": with_evidence,
        "evidence_case_coverage": (
            with_evidence / len(scored) if scored else None
        ),
    }


def _generation_summary(runs: list[CaseRun], strategy: str) -> dict:
    """Aggregate answer metrics for one strategy.

    ``rerank_failed=True`` runs are excluded from every generation metric as
    well, so a ``hybrid_rerank`` fallback can never be credited with rerank
    quality. Refusal rate is over runs where the evidence gate completed
    (including D). Faithfulness only over successfully judged answers.
    Citation validity only over generated answers that cited at least one
    number.
    """
    relevant = [
        run
        for run in runs
        if run.strategy == strategy and not run.rerank_failed
    ]
    attempts = [
        run for run in relevant if run.status in _GENERATION_ATTEMPT_STATUSES
    ]
    refusals = [run for run in attempts if run.status == "refusal"]
    answerable_attempts = [run for run in attempts if run.answerable]
    answered = [run for run in answerable_attempts if run.status == "answered"]
    return {
        "n": len(attempts),
        "refusal_rate": (
            len(refusals) / len(attempts) if attempts else None
        ),
        "faithfulness": _mean([run.faithful for run in relevant]),
        "citation_validity": _mean(
            [run.citation_validity for run in relevant]
        ),
        "model_refusals": sum(
            1 for run in attempts if run.status == "model_refusal"
        ),
        "answer_coverage": (
            len(answered) / len(answerable_attempts)
            if answerable_attempts
            else None
        ),
    }


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _metric_row(strategy: str, runs: list[CaseRun]) -> str:
    retrieval = _retrieval_summary(runs, strategy)
    generation = _generation_summary(runs, strategy)
    return (
        f"| {strategy} | {_fmt(retrieval['recall_at_5'])} "
        f"| {_fmt(retrieval['recall_at_10'])} "
        f"| {_fmt(retrieval['mrr_at_10'])} "
        f"| {_fmt(generation['faithfulness'])} "
        f"| {_fmt(generation['refusal_rate'])} "
        f"| {_fmt(generation['citation_validity'])} |"
    )


_METRIC_HEADER = (
    "| strategy | Recall@5 | Recall@10 | MRR@10 | Faithfulness "
    "| Refusal rate | Citation validity |\n"
    "| --- | --- | --- | --- | --- | --- | --- |"
)


def _d_accuracy(runs: list[CaseRun], strategy: str) -> tuple[int, int, int, int]:
    """Return ``(evaluable, refusal, model_refusal, other)`` for the D bucket.

    Runs without an outcome (self-check or retrieval errors, generation
    errors) are not evaluable. ``judge_error`` still produced an answer, so it
    counts as "other" unless it was detected as a model refusal first.
    """
    relevant = [
        run
        for run in runs
        if run.strategy == strategy and run.bucket == D_BUCKET
    ]
    outcomes = [
        run
        for run in relevant
        if run.status
        in {"refusal", "model_refusal", "answered", "judge_error"}
    ]
    refusals = sum(1 for run in outcomes if run.status == "refusal")
    model_refusals = sum(
        1 for run in outcomes if run.status == "model_refusal"
    )
    return (
        len(outcomes),
        refusals,
        model_refusals,
        len(outcomes) - refusals - model_refusals,
    )


def _fallback_case_ids(runs: list[CaseRun]) -> list[str]:
    return sorted({run.case_id for run in runs if run.rerank_failed})


def _error_block(runs: list[CaseRun]) -> list[str]:
    lines: list[str] = []
    for status in (
        "retrieval_error",
        "self_check_error",
        "generation_error",
        "judge_error",
    ):
        hits = [run for run in runs if run.status == status]
        if not hits:
            continue
        ids = ", ".join(f"{run.case_id}/{run.strategy}" for run in hits)
        lines.append(f"- {status}: {len(hits)} ({ids})")
    return lines or ["- none"]


def _write_faith_cases(
    runs: list[CaseRun], settings, enabled: bool
) -> FaithStats:
    """Upsert every unfaithful judged answer, one row per ``eval_id``."""
    stats = FaithStats(enabled=enabled)
    if not enabled:
        return stats

    candidates = [
        run
        for run in runs
        if run.status == "answered" and run.faithful is False
    ]
    if not candidates:
        return stats

    session = None
    try:
        try:
            session = get_session_factory()()
        except Exception:  # noqa: BLE001 - report the write failures
            stats.errors = len(candidates)
            return stats
        for run in candidates:
            payload = FaithCasePayload(
                eval_id=run.case_id,
                bucket=run.bucket,
                query=run.query,
                strategy=run.strategy,
                answer=run.answer or "",
                reason=run.faith_reason,
                citations=list(run.citations),
                judge_model=settings.model_name,
            )
            try:
                row = upsert_faith_case(session, payload)
                session.commit()
            except Exception:  # noqa: BLE001 - one bad write must not abort
                session.rollback()
                stats.errors += 1
                continue
            if (row.seen_count or 0) <= 1:
                stats.new += 1
            else:
                stats.recurrence += 1
            stats.entries.append(
                f"{run.case_id} ({run.strategy}, seen_count={row.seen_count})"
            )
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:  # pragma: no cover - defensive
                pass
    return stats


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:  # noqa: BLE001 - reporting must not fail the run
        return "unknown"
    return result.stdout.strip() or "unknown"


def _strategy_summary(runs: list[CaseRun], strategy: str) -> dict:
    """Structured metrics for one strategy, used by the JSON page."""
    retrieval = _retrieval_summary(runs, strategy)
    generation = _generation_summary(runs, strategy)
    evaluable, refusals, model_refusals, other = _d_accuracy(runs, strategy)
    accuracy = (
        (refusals + model_refusals) / evaluable if evaluable else None
    )
    status_counts: dict[str, int] = {}
    for run in runs:
        if run.strategy == strategy and not run.rerank_failed:
            status_counts[run.status] = status_counts.get(run.status, 0) + 1
    return {
        "retrieval": retrieval,
        "generation": generation,
        "d_accuracy": {
            "evaluable": evaluable,
            "refusal": refusals,
            "model_refusal": model_refusals,
            "other": other,
            "accuracy": accuracy,
        },
        "status_counts": status_counts,
    }


def _build_summary(
    *,
    cases: list[dict],
    loaded_cases: int,
    excluded_cases: list[dict],
    runs: list[CaseRun],
    strategies: list[RetrievalStrategy],
    settings,
    args: argparse.Namespace,
    faith: FaithStats,
    commit: str,
) -> dict:
    """Build the machine-readable sidecar consumed by the ``/rag-eval`` page."""
    strategy_names = [strategy.value for strategy in strategies]
    buckets = sorted({run.bucket for run in runs if run.bucket})
    fallback_runs = [run for run in runs if run.rerank_failed]
    rewrite_fallback_ids = {
        run.case_id for run in runs if run.analysis_by == "fallback"
    }
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "commit": commit,
        "cases_loaded": loaded_cases,
        "cases": len(cases),
        "excluded": {
            "count": len(excluded_cases),
            "cases": [
                {
                    "id": str(case.get("id") or ""),
                    "bucket": str(case.get("bucket") or ""),
                    "reason": str(
                        case.get("exclusion_reason")
                        or case.get("coverage")
                        or "excluded"
                    ),
                }
                for case in excluded_cases
            ],
        },
        "strategies": strategy_names,
        "mode": "retrieval-only" if args.retrieval_only else "full",
        "run": {
            "model": settings.model_name,
            "embedding_model": settings.embedding_model,
            "rerank_model": settings.rerank_model,
            "recall_top_k": settings.hybrid_recall_top_k,
            "rrf_k": settings.hybrid_rrf_k,
            "top_k": TOP_K,
        },
        "overall": {
            name: _strategy_summary(runs, name) for name in strategy_names
        },
        "buckets": {
            bucket: {
                name: _strategy_summary(
                    [run for run in runs if run.bucket == bucket], name
                )
                for name in strategy_names
            }
            for bucket in buckets
        },
        "fallbacks": {
            "rerank": {
                "runs": len(fallback_runs),
                "case_ids": sorted({run.case_id for run in fallback_runs}),
            },
            "rewrite": {
                "cases": len(rewrite_fallback_ids),
                "case_ids": sorted(rewrite_fallback_ids),
            },
        },
        "faith_cases": {
            "enabled": faith.enabled,
            "new": faith.new,
            "recurrence": faith.recurrence,
            "errors": faith.errors,
            "entries": faith.entries,
        },
    }


def _render_report(
    *,
    cases: list[dict],
    loaded_cases: int,
    excluded_cases: list[dict],
    runs: list[CaseRun],
    strategies: list[RetrievalStrategy],
    settings,
    args: argparse.Namespace,
    faith: FaithStats,
    commit: str,
) -> str:
    strategy_names = [strategy.value for strategy in strategies]
    buckets = sorted({run.bucket for run in runs if run.bucket})
    fallback_runs = [run for run in runs if run.rerank_failed]
    fallback_ids = _fallback_case_ids(runs)
    case_ids = {run.case_id for run in runs}
    rewrite_fallback_ids = {
        run.case_id for run in runs if run.analysis_by == "fallback"
    }
    mode = "retrieval-only" if args.retrieval_only else "full"

    lines = [
        "# ch04 RAG 四策略评估报告",
        "",
        "## 运行参数",
        "",
        f"- cases: {args.cases or _DEFAULT_CASES} "
        f"({len(cases)} evaluable, {len(excluded_cases)} excluded, "
        f"limit={args.limit})",
        f"- strategies: {', '.join(strategy_names)}",
        f"- mode: {mode}",
        f"- generation/judge/rewrite model: {settings.model_name} "
        "(temperature 0)",
        f"- embedding model: {settings.embedding_model}",
        f"- rerank model: {settings.rerank_model}",
        f"- retrieval: recall_top_k={settings.hybrid_recall_top_k}, "
        f"final Top-K={TOP_K}, RRF k={settings.hybrid_rrf_k}",
        f"- faith_cases: {'disabled' if not faith.enabled else 'enabled'}",
        f"- generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- commit: {commit}",
        "",
        "## 排除用例",
        "",
        f"- excluded: {len(excluded_cases)} / {loaded_cases}",
    ]
    if excluded_cases:
        lines.extend(
            f"- {case.get('id') or '?'} "
            f"({case.get('exclusion_reason') or case.get('coverage') or 'excluded'})"
            for case in excluded_cases
        )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "口径:语料未覆盖或与语料冲突的用例不参与任何指标聚合;",
            "补语料或修正预期后再将其纳入基准。",
            "",
            "## 总体指标",
            "",
            _METRIC_HEADER,
        ]
    )
    for strategy in strategy_names:
        lines.append(_metric_row(strategy, runs))
    lines.extend(
        [
            "",
            "口径:Recall/MRR 只统计可答题(A/B/C/E),并按 section_path 去重;",
            "`hybrid_rerank` 的 rerank 失败样本已从 Recall/MRR、Faithfulness、",
            "Refusal rate、Citation validity 聚合中剔除(D 桶拒答准确率单独统计)。",
            "Faithfulness 只统计裁判成功且未被证据门拒答的回答;",
            "Refusal rate 的分母是证据门完成的运行(含 D 桶预期拒答);",
            "Citation validity 只统计实际带 `[n]` 的生成回答。",
            "",
            "## 分桶指标",
        ]
    )
    for bucket in buckets:
        bucket_runs = [run for run in runs if run.bucket == bucket]
        lines.extend(["", f"### {bucket}", "", _METRIC_HEADER])
        for strategy in strategy_names:
            lines.append(_metric_row(strategy, bucket_runs))

    lines.extend(
        [
            "",
            "## D 桶拒答准确率",
            "",
            "| strategy | evaluable | refusal | model_refusal | answered/other "
            "| accuracy |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for strategy in strategy_names:
        evaluable, refusals, model_refusals, other = _d_accuracy(
            runs, strategy
        )
        accuracy = (
            (refusals + model_refusals) / evaluable if evaluable else None
        )
        lines.append(
            f"| {strategy} | {evaluable} | {refusals} | {model_refusals} "
            f"| {other} | {_fmt(accuracy)} |"
        )
    lines.extend(
        [
            "",
            "口径:证据门拒答(`refusal`)与模型自述无法回答(`model_refusal`)",
            "都算正确拒答;self-check/检索/生成错误不计入分母;",
            "rerank fallback 运行不剔除(本项只判断是否拒答,不评价 rerank 质量)。",
            "",
            "## Rerank fallback",
            "",
            f"- fallback runs: {len(fallback_runs)}",
            f"- case ids: {', '.join(fallback_ids) if fallback_ids else '(none)'}",
            "",
            "`rerank_failed=True` 的运行没有被当作成功的 rerank,"
            "已从 `hybrid_rerank` 的 Recall/MRR、Faithfulness、Refusal rate、"
            "Citation validity 聚合中剔除。",
            "",
            "## Query rewrite fallback",
            "",
            f"- cases with fallback rewrite: {len(rewrite_fallback_ids)} / "
            f"{len(case_ids)}",
            f"- case ids: "
            f"{', '.join(sorted(rewrite_fallback_ids)) if rewrite_fallback_ids else '(none)'}",
            "",
            "## Faith cases 台账",
            "",
        ]
    )
    if args.retrieval_only:
        lines.append("- skipped (--retrieval-only)")
    elif not faith.enabled:
        lines.append("- skipped (--no-faith-cases)")
    else:
        lines.extend(
            [
                f"- new: {faith.new}",
                f"- recurrence: {faith.recurrence}",
                f"- write errors: {faith.errors}",
            ]
        )
        if faith.entries:
            lines.append(f"- cases: {', '.join(faith.entries)}")
        elif not faith.errors:
            lines.append("- no unfaithful answers")
        lines.append(
            "- 口径:台账按 eval_id 一题一行(Task 11);同一次评估中同一题在"
            "多个策略下复发会累加 seen_count 并计入 recurrence。"
        )

    lines.extend(
        [
            "",
            "## 排除与错误",
            "",
            *_error_block(runs),
            "",
            "`self_check_error` 与 `judge_error` 已从 Faithfulness 中排除,",
            "`retrieval_error` 已从检索聚合中排除,以上均单独列明。",
            "",
        ]
    )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the four-strategy RAG evaluation (offline report)."
    )
    parser.add_argument(
        "--cases",
        default=None,
        help=f"path to the cases JSON (default: {_DEFAULT_CASES})",
    )
    parser.add_argument(
        "--strategies",
        default=_DEFAULT_STRATEGIES,
        help="comma-separated retrieval strategies in report order",
    )
    parser.add_argument(
        "--out",
        default=str(_DEFAULT_OUT),
        help=f"markdown report path (default: {_DEFAULT_OUT})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="evaluate only the first N cases",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="skip evidence self-check, generation and faithfulness judging",
    )
    parser.add_argument(
        "--no-faith-cases",
        action="store_true",
        help="do not write unfaithful answers to the faith_cases ledger",
    )
    return parser


def parse_strategies(value: str) -> list[RetrievalStrategy]:
    """Parse and validate a comma-separated strategy list, preserving order."""
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        raise ValueError("no strategies supplied")
    unknown = sorted(
        set(names) - {strategy.value for strategy in RetrievalStrategy}
    )
    if unknown:
        raise ValueError(f"unknown strategy: {', '.join(unknown)}")
    strategies: list[RetrievalStrategy] = []
    for name in names:
        strategy = RetrievalStrategy(name)
        if strategy not in strategies:
            strategies.append(strategy)
    return strategies


def _load_cases(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return list(data)
    return list(data.get("cases") or [])


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation and return a process exit code."""
    args = _build_parser().parse_args(argv)
    path = Path(args.cases) if args.cases else _DEFAULT_CASES
    try:
        cases = _load_cases(path)
    except (OSError, json.JSONDecodeError, TypeError, AttributeError) as exc:
        print(f"eval_rag: cannot read {path}: {exc}")
        return 2
    if args.limit is not None:
        cases = cases[: max(0, args.limit)]
    if not cases:
        print(f"eval_rag: no cases found in {path}")
        return 2

    loaded_cases = len(cases)
    excluded_cases = [case for case in cases if case.get("excluded")]
    cases = [case for case in cases if not case.get("excluded")]
    if not cases:
        print(
            f"eval_rag: no evaluable cases in {path} "
            f"({loaded_cases} excluded)"
        )
        return 2

    try:
        strategies = parse_strategies(args.strategies)
    except ValueError as exc:
        print(f"eval_rag: {exc}")
        return 2

    settings = get_settings()
    judge_llm = build_llm(settings, temperature=0.0)
    print(
        f"eval_rag: cases={len(cases)} evaluable, "
        f"excluded={len(excluded_cases)} "
        f"strategies={','.join(strategy.value for strategy in strategies)} "
        f"mode={'retrieval-only' if args.retrieval_only else 'full'}"
    )

    runs: list[CaseRun] = []
    for number, case in enumerate(cases, 1):
        query = str(case.get("query") or "")
        analysis = _safe_analyze(query, judge_llm)
        print(f"[{number}/{len(cases)}] {case.get('id')} query={query!r}")
        for strategy in strategies:
            runs.append(
                _run_case(
                    case,
                    strategy,
                    analysis,
                    judge_llm,
                    retrieval_only=args.retrieval_only,
                )
            )

    faith = _write_faith_cases(
        runs,
        settings,
        enabled=not args.retrieval_only and not args.no_faith_cases,
    )

    report = _render_report(
        cases=cases,
        loaded_cases=loaded_cases,
        excluded_cases=excluded_cases,
        runs=runs,
        strategies=strategies,
        settings=settings,
        args=args,
        faith=faith,
        commit=_git_commit(),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    summary = _build_summary(
        cases=cases,
        loaded_cases=loaded_cases,
        excluded_cases=excluded_cases,
        runs=runs,
        strategies=strategies,
        settings=settings,
        args=args,
        faith=faith,
        commit=_git_commit(),
    )
    json_path = out_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"eval_rag: report written to {out_path}")
    print(f"eval_rag: summary written to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
