"""第 6 章指代消解评估:13 条 held-out 多轮样例 + 真实 LLM 验证。

数据集见 ``tests/eval/ch06_coref_cases.json``,每条包含 ``id``、``history``、
``message``,以及 ``contains`` / ``not_contains`` 或 ``passthrough``:

- ``contains``:改写结果必须包含的实体或标准词(订单号、商品名、型号);
- ``not_contains``:改写结果不得残留的未消解指代词;
- ``passthrough``:问题已完整,``resolved_query`` 必须与原文完全一致。

:func:`evaluate_case` / :func:`summarize` 是纯函数;导入本模块不会构建
LLM,也不会访问网络。只有 :func:`main` 读取 settings、构建真实模型并调用
:func:`app.agent.coref.resolve_query`。

``resolve_query`` 出错时会静默回退原文,因此 :func:`run_cases` 捕获
``app.agent.coref`` 的 warning:回退样例被标记为无效(``valid=False``),
不计入准确率,也绝不把「回退后恰好等于原文」的透传样例算作通过。
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.agent.coref import resolve_query
from app.config import build_llm, get_settings

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CASES = _ROOT / "tests" / "eval" / "ch06_coref_cases.json"
_REPORT_PATH = (
    _ROOT / "docs" / "superpowers" / "reports" / "2026-11-18-ch06-coref-eval.md"
)

PASS_OVERALL = 0.90
PASS_PASSTHROUGH = 1.00

#: 生成报告时写入的 prompt 调整记录;若本轮调整了 prompt,在运行前改这里。
PROMPT_NOTE = (
    "无。本次验证未修改 `app/agent/coref.py` 的 `COREF_SYSTEM_PROMPT`。"
)

_ROLE_MESSAGES = {
    "user": HumanMessage,
    "assistant": AIMessage,
    "system": SystemMessage,
}

_COREF_LOGGER = logging.getLogger("app.agent.coref")


class _WarningCapture(logging.Handler):
    """Collect warning messages emitted by ``app.agent.coref`` during one call."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def load_cases(path: Path = _DEFAULT_CASES) -> list[dict]:
    """Read the annotated coref cases from *path*."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"coref cases must be a list, got {type(rows).__name__}")
    return rows


def _history_messages(history: list[dict]) -> list[BaseMessage]:
    """Convert dataset history rows into LangChain messages."""
    messages: list[BaseMessage] = []
    for row in history:
        message_type = _ROLE_MESSAGES.get(row.get("role"))
        if message_type is None:
            raise ValueError(f"unknown history role: {row.get('role')!r}")
        messages.append(message_type(content=row["content"]))
    return messages


def evaluate_case(case: dict, resolved: str) -> dict:
    """Score one rewritten query against *case*.

    Returns ``{"id", "passed", "reason"}``. ``passthrough`` cases require the
    resolved text to equal ``case["message"]`` exactly; other cases require all
    ``contains`` tokens and no ``not_contains`` tokens.
    """
    text = resolved or ""
    reasons: list[str] = []

    if case.get("passthrough"):
        expected = case.get("message") or ""
        if text != expected:
            reasons.append(f"passthrough changed text: expected {expected!r}, got {text!r}")
    else:
        missing = [token for token in case.get("contains") or [] if token not in text]
        if missing:
            reasons.append("missing: " + ", ".join(repr(token) for token in missing))
        banned = [
            token for token in case.get("not_contains") or [] if token in text
        ]
        if banned:
            reasons.append("banned: " + ", ".join(repr(token) for token in banned))

    return {
        "id": case.get("id", ""),
        "passed": not reasons,
        "reason": "; ".join(reasons) or "ok",
    }


def summarize(results: list[dict]) -> dict:
    """Return ``{"total", "passed", "accuracy", "failures"}``.

    Only valid results (``valid`` missing or true) are scored; LLM fallback
    rows produced by :func:`run_cases` are excluded and surfaced separately by
    the report. ``failures`` lists the invalid-scoring rows.
    """
    valid = [result for result in results if result.get("valid", True)]
    total = len(valid)
    passed = sum(1 for result in valid if result.get("passed"))
    failures = [result for result in valid if not result.get("passed")]
    return {
        "total": total,
        "passed": passed,
        "accuracy": passed / total if total else 0.0,
        "failures": failures,
    }


def run_cases(cases: list[dict], llm) -> list[dict]:
    """Run the real coref rewriter over *cases* and return one result per case.

    Each result holds ``id``, ``message``, ``resolved``, ``passed``,
    ``reason``, ``valid``, ``fallback_reason`` and ``passthrough``. A case whose
    call triggered the module's fallback warning is marked ``valid=False`` and
    ``passed=False`` so it can neither inflate nor silently satisfy the score.
    """
    results: list[dict] = []
    for number, case in enumerate(cases, 1):
        history = _history_messages(case.get("history") or [])
        capture = _WarningCapture()
        _COREF_LOGGER.addHandler(capture)
        try:
            resolved = resolve_query(llm, case["message"], history)
        finally:
            _COREF_LOGGER.removeHandler(capture)

        fallbacks = [
            message
            for message in capture.messages
            if "fallback to original" in message
        ]
        fallback = bool(fallbacks)
        verdict = evaluate_case(case, resolved)
        passed = verdict["passed"] and not fallback
        if fallback:
            reason = f"LLM fallback: {fallbacks[0]}"
        else:
            reason = verdict["reason"]

        results.append(
            {
                "id": case["id"],
                "message": case["message"],
                "resolved": resolved,
                "passed": passed,
                "reason": reason,
                "valid": not fallback,
                "fallback_reason": fallbacks[0] if fallback else "",
                "passthrough": bool(case.get("passthrough")),
            }
        )
        if fallback:
            mark = "INVALID(fallback)"
        else:
            mark = "PASS" if verdict["passed"] else "FAIL"
        print(f"[{number}/{len(cases)}] {case['id']} {mark} resolved={resolved!r}")
    return results


def _passthrough_stats(results: list[dict]) -> tuple[int, int, float]:
    """Return ``(passed, total, rate)`` for the pass-through subset."""
    valid = [result for result in results if result["valid"]]
    cases = [result for result in valid if result["passthrough"]]
    passed = sum(1 for result in cases if result["passed"])
    rate = passed / len(cases) if cases else 1.0
    return passed, len(cases), rate


def _one_line(text: str, limit: int = 200) -> str:
    """Collapse whitespace, escape table pipes and truncate for the report."""
    collapsed = " ".join(str(text or "").split()).replace("|", "\\|")
    return collapsed if len(collapsed) <= limit else collapsed[:limit] + "..."


def _command_line() -> str:
    module = getattr(__spec__, "name", None) or "app.jobs.eval_coref"
    return f"{Path(sys.executable).name} -m {module}"


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
    passed_pt, total_pt, rate_pt = _passthrough_stats(results)
    excluded = [result for result in results if not result["valid"]]
    overall_pass = metrics["accuracy"] >= PASS_OVERALL
    passthrough_pass = rate_pt >= PASS_PASSTHROUGH
    passed = overall_pass and passthrough_pass and not excluded
    encoding = os.environ.get("PYTHONIOENCODING") or "(unset)"

    lines = [
        "# ch06 指代消解评估报告",
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
        f"passthrough = {PASS_PASSTHROUGH:.0%}, no LLM fallback",
        "",
        "## 总体指标",
        "",
        "| total | passed | accuracy | passthrough | LLM fallback | 结论 |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| {metrics['total']} | {metrics['passed']} | "
        f"{metrics['accuracy']:.3f} | "
        f"{passed_pt}/{total_pt} ({rate_pt:.3f}) | "
        f"{len(excluded)} | {'PASS' if passed else 'FAIL'} |",
        "",
        "准确率只统计有效模型输出;LLM 兜底样例(回退原文)不参与计分,"
        "并直接判 FAIL。透传样例必须与原文逐字符一致。",
        "",
        "## 逐条结果",
        "",
        "| id | 类型 | message | resolved | 判定 | 原因 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        if not result["valid"]:
            verdict = "INVALID(fallback)"
        else:
            verdict = "PASS" if result["passed"] else "FAIL"
        kind = "透传" if result["passthrough"] else "改写"
        lines.append(
            f"| {result['id']} | {kind} | {_one_line(result['message'])} | "
            f"{_one_line(result['resolved'])} | {verdict} | "
            f"{_one_line(result['reason'])} |"
        )

    lines += ["", "## 失败样例", ""]
    failures = metrics["failures"]
    if failures:
        lines += [
            "| id | message | resolved | 原因 |",
            "| --- | --- | --- | --- |",
        ]
        for result in failures:
            lines.append(
                f"| {result['id']} | {_one_line(result['message'])} | "
                f"{_one_line(result['resolved'])} | {_one_line(result['reason'])} |"
            )
    else:
        lines.append("无。")

    lines += ["", "## LLM 兜底样例", ""]
    if excluded:
        lines += [
            "以下样例触发 `resolve_query` 的原文回退,不是模型输出,"
            "已从指标中排除:",
            "",
            "| id | message | error |",
            "| --- | --- | --- |",
        ]
        for result in excluded:
            lines.append(
                f"| {result['id']} | {_one_line(result['message'])} | "
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
    """Run the live coref evaluation and write the markdown report."""
    cases = load_cases(_DEFAULT_CASES)
    settings = get_settings()
    model_name = settings.intent_model or settings.model_name
    llm = build_llm(
        settings,
        temperature=0.0,
        model=settings.intent_model or None,
    )
    print(f"eval_coref: cases={len(cases)} model={model_name} temperature=0")

    results = run_cases(cases, llm)
    excluded = [result for result in results if not result["valid"]]
    if excluded and len(excluded) == len(results):
        print("\neval_coref: all cases fell back to original; API/network failure.")
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

    passed_pt, total_pt, rate_pt = _passthrough_stats(results)
    passed = (
        metrics["accuracy"] >= PASS_OVERALL
        and rate_pt >= PASS_PASSTHROUGH
        and not excluded
    )

    print(
        f"\neval_coref: valid={metrics['total']}/{len(results)} "
        f"passed={metrics['passed']} accuracy={metrics['accuracy']:.3f} "
        f"passthrough={passed_pt}/{total_pt} ({rate_pt:.3f})"
    )
    if metrics["failures"]:
        print("\neval_coref: failed cases")
        for result in metrics["failures"]:
            print(
                f"  {result['id']} message={result['message']!r} "
                f"resolved={result['resolved']!r} reason={result['reason']}"
            )
    if excluded:
        print("\neval_coref: invalid fallback cases")
        for result in excluded:
            print(f"  {result['id']}: {result['fallback_reason']}")
    print(f"\neval_coref: report written to {_REPORT_PATH}")
    print(f"eval_coref: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
