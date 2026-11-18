"""第 6 章 Query 扩写评估:8 条 held-out 样例 + 真实 LLM 验证。

数据集见 ``tests/eval/ch06_expand_cases.json``,每条包含 ``id``、``query``、
``order_data`` 与 ``must_include``。``must_include`` 是订单号 / 商品名 / 金额
等必须在扩写结果中原样保留的实体。

``expand_queries`` 的契约是「正好 3 条、去重、非空」,因此本评估逐条检查:

- 长度必须正好 3;
- 3 条查询去重后仍为 3 条(无重复);
- 每条查询去除空白后非空;
- ``must_include`` 中每个实体至少出现在一条查询里。

:func:`load_cases` / :func:`evaluate_case` / :func:`summarize` 是纯函数;
导入本模块不会构建 LLM,也不会访问网络。只有 :func:`main` 读取 settings、
构建真实模型并调用 :func:`app.rag.query_understanding.expand_queries`。

``expand_queries`` 在异常、数量不符或去重后不足 3 条时会静默回退
``[query]``。因此 :func:`run_cases` 捕获 ``app.rag.query_understanding``
的 warning:回退样例被标记为无效(``valid=False``),不计入准确率,也绝不把
「回退后只有一条查询」当作模型输出。
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from app.config import build_llm, get_settings
from app.rag.query_understanding import expand_queries

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CASES = _ROOT / "tests" / "eval" / "ch06_expand_cases.json"
_REPORT_PATH = (
    _ROOT / "docs" / "superpowers" / "reports" / "2026-11-18-ch06-expand-eval.md"
)

PASS_OVERALL = 0.90

#: 生成报告时写入的 prompt 调整记录;若本轮调整了 prompt,在运行前改这里。
PROMPT_NOTE = (
    "无。本次验证未修改 `app/rag/query_understanding.py` 的 "
    "`QUERY_EXPANSION_SYSTEM_PROMPT`。"
)

_EXPANSION_LOGGER = logging.getLogger("app.rag.query_understanding")


class _WarningCapture(logging.Handler):
    """Collect warning messages emitted by the expansion module during one call."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def load_cases(path: Path = _DEFAULT_CASES) -> list[dict]:
    """Read the annotated expansion cases from *path*."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"expand cases must be a list, got {type(rows).__name__}")
    return rows


def _clean_queries(queries) -> list[str]:
    """Return stripped string queries; non-strings become empty strings."""
    if not isinstance(queries, (list, tuple)):
        return []
    return [item.strip() if isinstance(item, str) else "" for item in queries]


def evaluate_case(case: dict, queries) -> dict:
    """Score one expansion output against *case*.

    Returns ``{"id", "passed", "reason", "shape_ok", "entities_ok"}``.
    ``shape_ok`` covers the hard contract (exactly 3, unique, non-empty);
    ``entities_ok`` requires every ``must_include`` token to appear in at
    least one query. The case passes only when both hold.
    """
    reasons: list[str] = []
    raw = list(queries) if isinstance(queries, (list, tuple)) else []
    cleaned = _clean_queries(queries)

    if not isinstance(queries, (list, tuple)):
        reasons.append(f"queries is not a list: {type(queries).__name__}")

    count_ok = len(raw) == 3
    if not count_ok:
        reasons.append(f"expected 3 queries, got {len(raw)}")

    non_empty_ok = len(cleaned) == len(raw) and all(cleaned)
    if not non_empty_ok:
        reasons.append("query is empty or not a string")

    unique_ok = len(cleaned) == len(set(cleaned))
    if not unique_ok:
        reasons.append("duplicate queries")

    missing = [
        entity
        for entity in case.get("must_include") or []
        if not any(entity in query for query in cleaned)
    ]
    if missing:
        reasons.append("missing: " + ", ".join(repr(entity) for entity in missing))

    shape_ok = count_ok and non_empty_ok and unique_ok
    entities_ok = not missing
    return {
        "id": case.get("id", ""),
        "passed": shape_ok and entities_ok,
        "reason": "; ".join(reasons) or "ok",
        "shape_ok": shape_ok,
        "entities_ok": entities_ok,
    }


def summarize(results: list[dict]) -> dict:
    """Return overall metrics.

    The result shape is::

        {"total": int, "passed": int, "accuracy": float,
         "shape_rate": float, "entity_rate": float, "failures": list}

    Only valid results (``valid`` missing or true) are scored; LLM fallback
    rows produced by :func:`run_cases` are excluded and surfaced separately by
    the report. ``failures`` lists the invalid-scoring rows.
    """
    valid = [result for result in results if result.get("valid", True)]
    total = len(valid)
    passed = sum(1 for result in valid if result.get("passed"))
    shape_passed = sum(
        1 for result in valid if result.get("shape_ok", result.get("passed"))
    )
    entity_passed = sum(
        1 for result in valid if result.get("entities_ok", result.get("passed"))
    )
    failures = [result for result in valid if not result.get("passed")]
    return {
        "total": total,
        "passed": passed,
        "accuracy": passed / total if total else 0.0,
        "shape_rate": shape_passed / total if total else 0.0,
        "entity_rate": entity_passed / total if total else 0.0,
        "failures": failures,
    }


def run_cases(cases: list[dict], llm) -> list[dict]:
    """Run the real expander over *cases* and return one result per case.

    Each result holds ``id``, ``query``, ``queries``, ``passed``, ``reason``,
    ``valid``, ``fallback_reason``, ``shape_ok`` and ``entities_ok``. A case
    whose call triggered the module's fallback warning is marked
    ``valid=False`` and ``passed=False`` so it can neither inflate nor silently
    satisfy the score.
    """
    results: list[dict] = []
    for number, case in enumerate(cases, 1):
        capture = _WarningCapture()
        _EXPANSION_LOGGER.addHandler(capture)
        try:
            queries = expand_queries(llm, case["query"], case.get("order_data"))
        except Exception as exc:  # pragma: no cover - expand_queries catches
            queries = [case["query"]]
            capture.messages.append(f"{type(exc).__name__}: {exc}")
        finally:
            _EXPANSION_LOGGER.removeHandler(capture)

        fallbacks = [
            message
            for message in capture.messages
            if "fallback to original" in message
        ]
        fallback = bool(fallbacks)
        verdict = evaluate_case(case, queries)
        passed = verdict["passed"] and not fallback
        if fallback:
            reason = f"LLM fallback: {fallbacks[0]}"
        else:
            reason = verdict["reason"]

        results.append(
            {
                "id": case["id"],
                "query": case["query"],
                "queries": list(queries),
                "passed": passed,
                "reason": reason,
                "valid": not fallback,
                "fallback_reason": fallbacks[0] if fallback else "",
                "shape_ok": verdict["shape_ok"],
                "entities_ok": verdict["entities_ok"],
            }
        )
        if fallback:
            mark = "INVALID(fallback)"
        else:
            mark = "PASS" if verdict["passed"] else "FAIL"
        print(
            f"[{number}/{len(cases)}] {case['id']} {mark} "
            f"queries={queries!r}"
        )
    return results


def _one_line(text: str, limit: int = 260) -> str:
    """Collapse whitespace, escape table pipes and truncate for the report."""
    collapsed = " ".join(str(text or "").split()).replace("|", "\\|")
    return collapsed if len(collapsed) <= limit else collapsed[:limit] + "..."


def _command_line() -> str:
    module = getattr(__spec__, "name", None) or "app.jobs.eval_expand"
    return f"{Path(sys.executable).name} -m {module}"


def _render_queries(queries: list[str]) -> str:
    return " / ".join(_one_line(query, 120) for query in queries)


def render_report(
    results: list[dict],
    metrics: dict,
    model_name: str,
    cases_path: Path,
    generated_at: datetime | None = None,
    prompt_note: str | None = None,
) -> str:
    """Render the markdown evaluation report (Chinese, repo style)."""
    generated_at = generated_at or datetime.now()
    excluded = [result for result in results if not result["valid"]]
    overall_pass = metrics["accuracy"] >= PASS_OVERALL
    passed = overall_pass and not excluded
    encoding = os.environ.get("PYTHONIOENCODING") or "(unset)"

    lines = [
        "# ch06 Query 扩写评估报告",
        "",
        "## 运行参数",
        "",
        f"- command: `{_command_line()}`",
        f"- env: `PYTHONIOENCODING={encoding}`",
        f"- dataset: {cases_path.name} ({len(results)} cases, "
        f"{metrics['total']} valid)",
        f"- model: {model_name} (temperature 0)",
        f"- generated at: {generated_at:%Y-%m-%d %H:%M:%S}",
        f"- pass bar: overall accuracy >= {PASS_OVERALL:.0%}, "
        "每条正好 3 条 / 去重 / 非空, must_include 全部保留, no LLM fallback",
        "",
        "## 总体指标",
        "",
        "| total | passed | accuracy | 3 条且非空去重 | 实体保留 | "
        "LLM fallback | 结论 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        f"| {metrics['total']} | {metrics['passed']} | "
        f"{metrics['accuracy']:.3f} | {metrics['shape_rate']:.3f} | "
        f"{metrics['entity_rate']:.3f} | {len(excluded)} | "
        f"{'PASS' if passed else 'FAIL'} |",
        "",
        "准确率只统计有效模型输出,且要求长度/去重/非空/实体四项同时通过;"
        "LLM 兜底样例(回退 `[query]`)不参与计分,并直接判 FAIL。",
        "",
        "## 逐条结果",
        "",
        "| id | query | expanded queries | 判定 | 原因 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in results:
        if not result["valid"]:
            verdict = "INVALID(fallback)"
        else:
            verdict = "PASS" if result["passed"] else "FAIL"
        lines.append(
            f"| {result['id']} | {_one_line(result['query'])} | "
            f"{_render_queries(result['queries'])} | {verdict} | "
            f"{_one_line(result['reason'])} |"
        )

    lines += ["", "## 失败样例", ""]
    failures = metrics["failures"]
    if failures:
        lines += [
            "| id | query | expanded queries | 原因 |",
            "| --- | --- | --- | --- |",
        ]
        for result in failures:
            lines.append(
                f"| {result['id']} | {_one_line(result['query'])} | "
                f"{_render_queries(result['queries'])} | "
                f"{_one_line(result['reason'])} |"
            )
    else:
        lines.append("无。")

    lines += ["", "## LLM 兜底样例", ""]
    if excluded:
        lines += [
            "以下样例触发 `expand_queries` 的 `[query]` 回退,不是模型的 3 条"
            "扩写,已从指标中排除:",
            "",
            "| id | query | error |",
            "| --- | --- | --- |",
        ]
        for result in excluded:
            lines.append(
                f"| {result['id']} | {_one_line(result['query'])} | "
                f"{_one_line(result['fallback_reason'])} |"
            )
    else:
        lines.append("无。")

    lines += [
        "",
        "## Prompt 调整",
        "",
        prompt_note or PROMPT_NOTE,
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    """Run the live expansion evaluation and write the markdown report."""
    cases = load_cases(_DEFAULT_CASES)
    settings = get_settings()
    model_name = settings.intent_model or settings.model_name
    llm = build_llm(
        settings,
        temperature=0.0,
        model=settings.intent_model or None,
    )
    print(f"eval_expand: cases={len(cases)} model={model_name} temperature=0")

    results = run_cases(cases, llm)
    excluded = [result for result in results if not result["valid"]]
    if excluded and len(excluded) == len(results):
        print("\neval_expand: all cases fell back to [query]; API/network failure.")
        print(f"first error: {excluded[0]['fallback_reason']}")
        return 2

    metrics = summarize(results)
    report = render_report(
        results,
        metrics,
        model_name,
        _DEFAULT_CASES,
    )
    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text(report, encoding="utf-8")

    passed = metrics["accuracy"] >= PASS_OVERALL and not excluded

    print(
        f"\neval_expand: valid={metrics['total']}/{len(results)} "
        f"passed={metrics['passed']} accuracy={metrics['accuracy']:.3f} "
        f"shape={metrics['shape_rate']:.3f} entities={metrics['entity_rate']:.3f} "
        f"fallback={len(excluded)}"
    )
    if metrics["failures"]:
        print("\neval_expand: failed cases")
        for result in metrics["failures"]:
            print(
                f"  {result['id']} query={result['query']!r} "
                f"queries={result['queries']!r} reason={result['reason']}"
            )
    if excluded:
        print("\neval_expand: invalid fallback cases")
        for result in excluded:
            print(f"  {result['id']}: {result['fallback_reason']}")
    print(f"\neval_expand: report written to {_REPORT_PATH}")
    print(f"eval_expand: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
