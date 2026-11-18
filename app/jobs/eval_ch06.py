"""第 6 章多轮验收:指代消解 + 意图分类的真实对话回归。

三轮对话共享同一份 history,逐轮验证「先消解、再分类」的链路:

1. 「订单 1001 的物流到哪了」→ 物流,改写结果含 ``1001``;
2. 「它能退吗」→ 退款退货,改写结果含 ``1001`` 且不残留「它」;
3. 「那它现在到哪了」→ 物流,改写结果含 ``1001``。

:func:`run_dialogue` 每轮调用 :func:`app.agent.coref.resolve_query` 与
:func:`app.agent.intent.classify_intent`,随后把本轮原始用户消息与改写后的
问题追加进 history——真实图里 assistant 的答案同样会把订单号带进上下文,
这里用改写后的独立问题代替 assistant 内容,保证下一轮指代可见。

:func:`evaluate_turn` / :func:`summarize` 是纯函数;导入本模块不会构建 LLM,
也不会访问网络。只有 :func:`main` 读取 settings、构建真实模型并写报告。

``resolve_query`` 出错时会静默回退原文,因此 :func:`run_dialogue` 捕获
``app.agent.coref`` 的 warning:回退轮次标记 ``valid=False`` 并直接判失败,
不会因为回退后的原文恰好包含实体而伪造通过。
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.agent.coref import resolve_query
from app.agent.intent import classify_intent
from app.config import build_llm, get_settings

_ROOT = Path(__file__).resolve().parents[2]
_REPORT_PATH = (
    _ROOT / "docs" / "superpowers" / "reports" / "2026-11-18-ch06-acceptance.md"
)

#: 多轮验收集,顺序即真实对话顺序。
TURNS: list[dict] = [
    {
        "id": "turn1",
        "message": "订单 1001 的物流到哪了",
        "expected": "物流",
        "must_include": ["1001"],
        "must_not_include": [],
    },
    {
        "id": "turn2",
        "message": "它能退吗",
        "expected": "退款退货",
        "must_include": ["1001"],
        "must_not_include": ["它"],
    },
    {
        "id": "turn3",
        "message": "那它现在到哪了",
        "expected": "物流",
        "must_include": ["1001"],
        "must_not_include": [],
    },
]

#: 生成报告时写入的 prompt 调整记录;若本轮调整了 prompt,在运行前改这里。
PROMPT_NOTE = (
    "无。本次验收未修改 `app/agent/coref.py` 或 `app/agent/intent.py` 的 prompt。"
)

_COREF_LOGGER = logging.getLogger("app.agent.coref")


class _WarningCapture(logging.Handler):
    """Collect warning messages emitted by ``app.agent.coref`` during one call."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def evaluate_turn(result: dict) -> dict:
    """Score one dialogue turn; return ``{"passed", "reason"}``.

    A turn passes only when the intent matches ``expected``, every
    ``must_include`` token appears in ``resolved_query``, no ``must_not_include``
    token remains, and the turn is neither an LLM fallback (``valid=False``) nor
    a classifier error.
    """
    text = result.get("resolved_query") or ""
    reasons: list[str] = []

    if result.get("error"):
        reasons.append(f"error: {result['error']}")
    if not result.get("valid", True):
        reasons.append(f"LLM fallback: {result.get('fallback_reason', '')}")
    if result.get("intent") != result.get("expected"):
        reasons.append(
            f"intent: expected {result.get('expected')!r}, "
            f"got {result.get('intent')!r}"
        )

    missing = [token for token in result.get("must_include") or [] if token not in text]
    if missing:
        reasons.append("missing: " + ", ".join(repr(token) for token in missing))
    banned = [
        token for token in result.get("must_not_include") or [] if token in text
    ]
    if banned:
        reasons.append("banned: " + ", ".join(repr(token) for token in banned))

    return {"passed": not reasons, "reason": "; ".join(reasons) or "ok"}


def summarize(results: list[dict]) -> dict:
    """Return ``{"total", "passed", "failed", "accuracy", "failures"}``.

    Pass/fail is recomputed from each row's ``expected`` / ``intent`` /
    ``resolved_query`` / ``must_include`` / ``must_not_include``, so callers can
    pass raw model outputs without pre-scoring them.
    """
    failures: list[dict] = []
    for result in results:
        verdict = evaluate_turn(result)
        if not verdict["passed"]:
            failures.append({**result, "reason": verdict["reason"]})

    total = len(results)
    passed = total - len(failures)
    return {
        "total": total,
        "passed": passed,
        "failed": len(failures),
        "accuracy": passed / total if total else 0.0,
        "failures": failures,
    }


def run_dialogue(llm, turns: list[dict] | None = None) -> list[dict]:
    """Run *turns* against the real coref + intent models, accumulating history.

    Each result holds ``id``, ``message``, ``resolved_query``, ``intent``,
    ``confidence``, ``expected``, ``must_include``, ``must_not_include``,
    ``valid``, ``fallback_reason``, ``error``, ``passed`` and ``reason``. A
    coref fallback or classifier exception is recorded and fails the turn.
    """
    turns = TURNS if turns is None else turns
    history: list[BaseMessage] = []
    results: list[dict] = []

    for number, turn in enumerate(turns, 1):
        capture = _WarningCapture()
        _COREF_LOGGER.addHandler(capture)
        try:
            resolved = resolve_query(llm, turn["message"], history)
        finally:
            _COREF_LOGGER.removeHandler(capture)

        fallbacks = [
            message
            for message in capture.messages
            if "fallback to original" in message
        ]

        intent = None
        confidence = None
        error = ""
        try:
            prediction = classify_intent(
                llm, resolved, history, raise_on_error=True
            )
            intent = prediction.intent
            confidence = prediction.confidence
        except Exception as exc:  # noqa: BLE001 - record the exact upstream error
            error = f"{type(exc).__name__}: {exc}"

        result = {
            "id": turn.get("id", f"turn{number}"),
            "message": turn["message"],
            "resolved_query": resolved,
            "intent": intent,
            "confidence": confidence,
            "expected": turn["expected"],
            "must_include": list(turn.get("must_include") or []),
            "must_not_include": list(turn.get("must_not_include") or []),
            "valid": not fallbacks,
            "fallback_reason": fallbacks[0] if fallbacks else "",
            "error": error,
        }
        verdict = evaluate_turn(result)
        result["passed"] = verdict["passed"]
        result["reason"] = verdict["reason"]
        results.append(result)

        # 下一轮可见本轮原始消息与消解后的问题,模拟 assistant 在回答中
        # 重复订单号后带来的上下文。
        history.append(HumanMessage(content=turn["message"]))
        history.append(AIMessage(content=resolved))

        mark = "PASS" if verdict["passed"] else "FAIL"
        confidence_text = f"{confidence:.2f}" if confidence is not None else "-"
        print(
            f"[{number}/{len(turns)}] {result['id']} {mark} "
            f"resolved={resolved!r} intent={intent} conf={confidence_text}"
        )
        if not verdict["passed"]:
            print(f"  reason: {verdict['reason']}")

    return results


def _one_line(text: str, limit: int = 200) -> str:
    """Collapse whitespace, escape table pipes and truncate for the report."""
    collapsed = " ".join(str(text or "").split()).replace("|", "\\|")
    return collapsed if len(collapsed) <= limit else collapsed[:limit] + "..."


def _command_line() -> str:
    module = getattr(__spec__, "name", None) or "app.jobs.eval_ch06"
    return f"{Path(sys.executable).name} -m {module}"


def render_report(
    results: list[dict],
    metrics: dict,
    model_name: str,
    generated_at: datetime | None = None,
    prompt_note: str | None = None,
) -> str:
    """Render the markdown acceptance report (Chinese, repo style)."""
    generated_at = generated_at or datetime.now()
    encoding = os.environ.get("PYTHONIOENCODING") or "(unset)"
    passed = metrics["total"] > 0 and metrics["failed"] == 0

    lines = [
        "# ch06 多轮验收报告",
        "",
        "## 运行参数",
        "",
        f"- command: `{_command_line()}`",
        f"- env: `PYTHONIOENCODING={encoding}`",
        f"- dialogue: {metrics['total']} turns",
        f"- model: {model_name} (temperature 0)",
        f"- generated at: {generated_at:%Y-%m-%d %H:%M:%S}",
        "- pass bar: 每轮 intent 与 expected 一致,"
        "resolved_query 含全部必须实体且不含禁止指代词,无 LLM 兜底",
        "",
        "## 总体结果",
        "",
        "| total | passed | failed | accuracy | 结论 |",
        "| --- | --- | --- | --- | --- |",
        f"| {metrics['total']} | {metrics['passed']} | {metrics['failed']} | "
        f"{metrics['accuracy']:.3f} | {'PASS' if passed else 'FAIL'} |",
        "",
        "history 逐轮累积「用户原句 + 消解后的问题」,因此后两轮的"
        "「它」只有在 coref 正确补出订单 1001 后才能通过。",
        "",
        "## 逐轮结果",
        "",
        "| id | message | resolved_query | intent | confidence | expected | "
        "判定 | 原因 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        confidence = (
            f"{result['confidence']:.2f}"
            if result.get("confidence") is not None
            else "-"
        )
        verdict = "PASS" if result["passed"] else "FAIL"
        lines.append(
            f"| {result['id']} | {_one_line(result['message'])} | "
            f"{_one_line(result['resolved_query'])} | "
            f"{result.get('intent') or '-'} | {confidence} | "
            f"{result['expected']} | {verdict} | "
            f"{_one_line(result['reason'])} |"
        )

    lines += ["", "## 失败轮次", ""]
    if metrics["failures"]:
        lines += [
            "| id | message | resolved_query | intent | expected | 原因 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for result in metrics["failures"]:
            lines.append(
                f"| {result['id']} | {_one_line(result['message'])} | "
                f"{_one_line(result['resolved_query'])} | "
                f"{result.get('intent') or '-'} | {result['expected']} | "
                f"{_one_line(result['reason'])} |"
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
    """Run the live multi-turn acceptance and write the markdown report."""
    settings = get_settings()
    model_name = settings.intent_model or settings.model_name
    llm = build_llm(
        settings,
        temperature=0.0,
        model=settings.intent_model or None,
    )
    print(f"eval_ch06: turns={len(TURNS)} model={model_name} temperature=0")

    results = run_dialogue(llm, TURNS)
    invalid = [result for result in results if result["error"] or not result["valid"]]
    if invalid and len(invalid) == len(results):
        print("\neval_ch06: all turns failed at coref/intent; API/network or JSON failure.")
        first = invalid[0]
        print(f"first error: {first['error'] or first['fallback_reason']}")
        return 2

    metrics = summarize(results)
    report = render_report(results, metrics, model_name)
    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text(report, encoding="utf-8")

    print(
        f"\neval_ch06: passed={metrics['passed']}/{metrics['total']} "
        f"failed={metrics['failed']} accuracy={metrics['accuracy']:.3f}"
    )
    if metrics["failures"]:
        print("\neval_ch06: failed turns")
        for result in metrics["failures"]:
            print(
                f"  {result['id']} message={result['message']!r} "
                f"resolved={result['resolved_query']!r} "
                f"intent={result.get('intent')!r} expected={result['expected']!r} "
                f"reason={result['reason']}"
            )
    print(f"\neval_ch06: report written to {_REPORT_PATH}")
    passed = metrics["total"] > 0 and metrics["failed"] == 0
    print(f"eval_ch06: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
