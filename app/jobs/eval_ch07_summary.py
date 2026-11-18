"""第 7 章摘要 prompt 标注样例评估。

数据集见 ``tests/eval/ch07_summary_cases.json``,每条包含 ``id``、``history``、
``prior_summary``、``contains`` / ``not_contains``,以及可选的 ``max_chars``
和 ``not_contains_regex``:

- ``contains``:摘要必须保留的事实(订单号、手机号、型号、诉求、拒绝原因);
- ``not_contains``:摘要不得残留的内容(寒暄、更正前的旧订单号);
- ``max_chars``:摘要长度上限(纯寒暄样例要求更短);
- ``not_contains_regex``:禁止出现的内容模式(如无来源的 4 位以上数字);
- ``allow_empty``:纯寒暄样例允许空摘要(运行时按 ``too_short`` 跳过)。

:func:`load_cases` / :func:`build_dialogue` / :func:`evaluate_case` /
:func:`summarize` 是纯函数;导入本模块不会构建 LLM,也不会访问网络。只有
:func:`main` 读取 settings、构建真实模型,并走与
:func:`app.services.summary_job.run_summary` 相同的结构化输出路径
(``SegmentSummary`` + ``method="json_mode"``)。

本脚本评估的是 ``app.services.summary_job.SUMMARY_SYSTEM_PROMPT`` 这一版
prompt:模型必须返回严格 JSON,且摘要满足每条样例的事实/禁用/长度断言。
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from app.config import build_llm, get_settings
from app.services.summary_job import (
    SUMMARY_DEFAULT_MAX_CHARS,
    SUMMARY_SYSTEM_PROMPT,
    SegmentSummary,
    build_summary_prompt,
)

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CASES = _ROOT / "tests" / "eval" / "ch07_summary_cases.json"
_REPORT_PATH = (
    _ROOT / "docs" / "superpowers" / "reports" / "2026-11-18-ch07-summary-eval.md"
)

PASS_OVERALL = 0.90

#: 生成报告时写入的 prompt 调整记录;若本轮调整了 prompt,在运行前改这里。
PROMPT_NOTE = (
    "本次验证为 `app/services/summary_job.py` 的 `SUMMARY_SYSTEM_PROMPT` "
    "新增两条红线:(7) 整段只有问候/寒暄且无事实或明确诉求时 summary 留空;"
    "(8) 同一事实被用户更正后只保留最新值,不重复被更正的旧值。"
    "评估使用 200 字默认模板(`build_summary_prompt` 的默认 `max_chars`)。"
)

_ROLE_LABELS = {"user": "用户", "assistant": "客服", "system": "系统"}


def load_cases(path: Path = _DEFAULT_CASES) -> list[dict]:
    """Read the annotated summary cases from *path*."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"summary cases must be a list, got {type(rows).__name__}")
    return rows


def build_dialogue(history: list[dict]) -> str:
    """Render dataset history rows as ``用户：``/``客服：`` dialogue lines."""
    lines: list[str] = []
    for row in history or []:
        role = row.get("role", "")
        label = _ROLE_LABELS.get(role, role or "未知")
        lines.append(f"{label}：{row.get('content', '')}")
    return "\n".join(lines)


def evaluate_case(case: dict, summary: str | None) -> dict:
    """Score one summary against *case*.

    Returns ``{"id", "passed", "reason", "chars"}``. A case passes only when
    every ``contains`` token is present, no ``not_contains`` token is present,
    the text fits ``max_chars`` (when given) and ``not_contains_regex`` (when
    given) does not match. ``allow_empty`` marks pure-chitchat cases where an
    empty summary is the correct answer; all other cases reject empty output.
    """
    text = summary.strip() if isinstance(summary, str) else ""
    reasons: list[str] = []

    if not text:
        # Pure chitchat segments should yield an empty summary (the runtime
        # skips it as too_short); every other case still needs real content.
        if case.get("allow_empty"):
            return {
                "id": case.get("id", ""),
                "passed": True,
                "reason": "ok (empty allowed for chitchat)",
                "chars": 0,
            }
        return {
            "id": case.get("id", ""),
            "passed": False,
            "reason": "empty summary",
            "chars": 0,
        }

    missing = [token for token in case.get("contains") or [] if token not in text]
    if missing:
        reasons.append("missing: " + ", ".join(repr(token) for token in missing))

    banned = [token for token in case.get("not_contains") or [] if token in text]
    if banned:
        reasons.append("banned: " + ", ".join(repr(token) for token in banned))

    max_chars = case.get("max_chars")
    if max_chars is not None and len(text) > max_chars:
        reasons.append(f"too long: {len(text)} > {max_chars}")

    pattern = case.get("not_contains_regex")
    if pattern:
        try:
            match = re.search(pattern, text)
        except re.error as exc:
            reasons.append(f"invalid regex {pattern!r}: {exc}")
        else:
            if match:
                reasons.append(f"banned regex matched: {match.group(0)!r}")

    return {
        "id": case.get("id", ""),
        "passed": not reasons,
        "reason": "; ".join(reasons) or "ok",
        "chars": len(text),
    }


def summarize(results: list[dict]) -> dict:
    """Return ``{"total", "passed", "accuracy", "failures", "invalid"}``.

    Only valid results (``valid`` missing or true) are scored. Invalid rows
    (strict-JSON parse or model-call failures) are excluded from the accuracy
    denominator and surfaced separately so they can never inflate the score.
    """
    valid = [result for result in results if result.get("valid", True)]
    invalid = [result for result in results if not result.get("valid", True)]
    total = len(valid)
    passed = sum(1 for result in valid if result.get("passed"))
    failures = [result for result in valid if not result.get("passed")]
    return {
        "total": total,
        "passed": passed,
        "accuracy": passed / total if total else 0.0,
        "failures": failures,
        "invalid": invalid,
    }


def _run_one(case: dict, llm) -> dict:
    """Run the structured-output summary path for one case."""
    dialogue = build_dialogue(case.get("history") or [])
    prompt = build_summary_prompt(
        case.get("prior_summary") or "",
        dialogue,
        SUMMARY_DEFAULT_MAX_CHARS,
    )
    try:
        response = llm.with_structured_output(
            SegmentSummary, method="json_mode"
        ).invoke(prompt)
    except OutputParserException as exc:
        return {
            "id": case.get("id", ""),
            "note": case.get("note", ""),
            "summary": "",
            "chars": 0,
            "passed": False,
            "valid": False,
            "reason": f"invalid JSON: {exc}",
        }
    except Exception as exc:  # noqa: BLE001 - eval must record any model failure
        return {
            "id": case.get("id", ""),
            "note": case.get("note", ""),
            "summary": "",
            "chars": 0,
            "passed": False,
            "valid": False,
            "reason": f"error: {type(exc).__name__}: {exc}",
        }

    summary = getattr(response, "summary", None)
    if not isinstance(summary, str):
        summary = ""
    verdict = evaluate_case(case, summary)
    return {
        "id": case.get("id", ""),
        "note": case.get("note", ""),
        "summary": summary.strip(),
        "chars": verdict["chars"],
        "passed": verdict["passed"],
        "valid": True,
        "reason": verdict["reason"],
    }


def run_cases(cases: list[dict], llm) -> list[dict]:
    """Run the real summary model over *cases* and return one result per case."""
    results: list[dict] = []
    for number, case in enumerate(cases, 1):
        result = _run_one(case, llm)
        results.append(result)
        if not result["valid"]:
            mark = "INVALID(" + result["reason"].split(":", 1)[0] + ")"
        else:
            mark = "PASS" if result["passed"] else "FAIL"
        print(
            f"[{number}/{len(cases)}] {result['id']} {mark} "
            f"chars={result['chars']} summary={result['summary']!r}"
        )
    return results


def _one_line(text: str, limit: int = 200) -> str:
    """Collapse whitespace, escape table pipes and truncate for the report."""
    collapsed = " ".join(str(text or "").split()).replace("|", "\\|")
    return collapsed if len(collapsed) <= limit else collapsed[:limit] + "..."


def _command_line() -> str:
    module = getattr(__spec__, "name", None) or "app.jobs.eval_ch07_summary"
    return f"{Path(sys.executable).name} -m {module}"


def render_skip_report(
    reason: str,
    cases_path: Path = _DEFAULT_CASES,
    generated_at: datetime | None = None,
) -> str:
    """Render the report body for a skipped run (no usable API key)."""
    generated_at = generated_at or datetime.now()
    return "\n".join(
        [
            "# ch07 摘要 prompt 评估报告",
            "",
            "## 运行参数",
            "",
            f"- command: `{_command_line()}`",
            f"- dataset: {cases_path.name}",
            f"- model: (not called)",
            f"- generated at: {generated_at:%Y-%m-%d %H:%M:%S}",
            f"- status: skipped: {reason}",
            "",
            "## 说明",
            "",
            "未检测到可用的 API key，本次未调用真实模型，也没有产生准确率。"
            "纯函数的评分逻辑由 `tests/test_eval_ch07_summary.py` 覆盖。",
            "",
        ]
    )


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
    encoding = os.environ.get("PYTHONIOENCODING") or "(unset)"
    passed = metrics["accuracy"] >= PASS_OVERALL and not metrics["invalid"]

    lines = [
        "# ch07 摘要 prompt 评估报告",
        "",
        "## 运行参数",
        "",
        f"- command: `{_command_line()}`",
        f"- env: `PYTHONIOENCODING={encoding}`",
        f"- dataset: {cases_path.name} ({len(results)} cases, "
        f"{metrics['total']} valid)",
        f"- model: {model_name} (temperature 0, structured JSON)",
        f"- generated at: {generated_at:%Y-%m-%d %H:%M:%S}",
        f"- pass bar: overall accuracy >= {PASS_OVERALL:.0%}, "
        "strict JSON on every case",
        "",
        "## 总体指标",
        "",
        "| total | passed | accuracy | invalid | 结论 |",
        "| --- | --- | --- | --- | --- |",
        f"| {metrics['total']} | {metrics['passed']} | "
        f"{metrics['accuracy']:.3f} | {len(metrics['invalid'])} | "
        f"{'PASS' if passed else 'FAIL'} |",
        "",
        "准确率只统计成功返回严格 JSON 的样例;解析失败或模型调用失败的样例"
        "不计入分母,并直接判 FAIL。",
        "",
        "## 逐条结果",
        "",
        "| id | 样例 | 字数 | 摘要 | 判定 | 原因 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        if not result["valid"]:
            verdict = "INVALID"
        else:
            verdict = "PASS" if result["passed"] else "FAIL"
        lines.append(
            f"| {result['id']} | {_one_line(result['note'], 60)} | "
            f"{result['chars']} | {_one_line(result['summary'], 120)} | "
            f"{verdict} | {_one_line(result['reason'])} |"
        )

    lines += ["", "## 失败样例", ""]
    if metrics["failures"]:
        lines += [
            "| id | 摘要 | 原因 |",
            "| --- | --- | --- |",
        ]
        for result in metrics["failures"]:
            lines.append(
                f"| {result['id']} | {_one_line(result['summary'], 120)} | "
                f"{_one_line(result['reason'])} |"
            )
    else:
        lines.append("无。")

    lines += ["", "## 无效输出", ""]
    if metrics["invalid"]:
        lines += [
            "以下样例没有返回严格 JSON(或模型调用失败),不计入准确率:",
            "",
            "| id | 原因 |",
            "| --- | --- |",
        ]
        for result in metrics["invalid"]:
            lines.append(
                f"| {result['id']} | {_one_line(result['reason'])} |"
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


def _write_report(report: str) -> None:
    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text(report, encoding="utf-8")


def _skip(reason: str, cases_path: Path) -> int:
    """Write a skip report, print it, and return exit code 0."""
    _write_report(render_skip_report(reason, cases_path))
    print(f"eval_ch07_summary: skipped: {reason}")
    print(f"eval_ch07_summary: report written to {_REPORT_PATH}")
    return 0


def main() -> int:
    """Run the live summary evaluation and write the markdown report."""
    try:
        settings = get_settings()
    except ValidationError as exc:
        return _skip(f"no API key ({exc})", _DEFAULT_CASES)
    if not (settings.openai_api_key or "").strip():
        return _skip("no API key", _DEFAULT_CASES)

    cases = load_cases(_DEFAULT_CASES)

    # The prompt under test must be the importable default system prompt.
    system_prompt = build_summary_prompt("", "")[0].content
    if system_prompt != SUMMARY_SYSTEM_PROMPT:
        raise RuntimeError(
            "build_summary_prompt default system prompt drifted from "
            "SUMMARY_SYSTEM_PROMPT"
        )

    model_name = settings.summary_model or settings.model_name
    llm = build_llm(
        settings,
        temperature=0.0,
        model=settings.summary_model or None,
    )
    print(
        f"eval_ch07_summary: cases={len(cases)} model={model_name} "
        "temperature=0 structured=json"
    )

    results = run_cases(cases, llm)
    metrics = summarize(results)
    report = render_report(results, metrics, model_name, _DEFAULT_CASES)
    _write_report(report)

    if metrics["invalid"] and len(metrics["invalid"]) == len(results):
        print("\neval_ch07_summary: all cases failed to return strict JSON; "
              "API/network failure.")
        print(f"first error: {metrics['invalid'][0]['reason']}")
        return 2

    passed = metrics["accuracy"] >= PASS_OVERALL and not metrics["invalid"]
    print(
        f"\neval_ch07_summary: valid={metrics['total']}/{len(results)} "
        f"passed={metrics['passed']} accuracy={metrics['accuracy']:.3f} "
        f"invalid={len(metrics['invalid'])}"
    )
    if metrics["failures"]:
        print("\neval_ch07_summary: failed cases")
        for result in metrics["failures"]:
            print(
                f"  {result['id']} summary={result['summary']!r} "
                f"reason={result['reason']}"
            )
    if metrics["invalid"]:
        print("\neval_ch07_summary: invalid cases")
        for result in metrics["invalid"]:
            print(f"  {result['id']}: {result['reason']}")
    print(f"\neval_ch07_summary: report written to {_REPORT_PATH}")
    print(f"eval_ch07_summary: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
