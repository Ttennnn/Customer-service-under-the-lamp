"""第 6 章意图分类评估:62 条标注样例 + 真实 LLM 验证。

数据集见 ``tests/eval/ch06_intent_cases.json``,每条为
``{"id": str, "text": str, "intent": Intent, "history": list | 无}``;``history``
为可选的多轮上下文,交给 ``classify_intent`` 与当前消息一起判断。样例均为
held-out 文本,不得复用 ``INTENT_SYSTEM_PROMPT`` 的 few-shot 示例(见
``tests/test_intent_cases.py``)。

``intent="其他"`` 是合法预测(真实「其他」),不会被当成兜底;只有
``classify_intent(..., raise_on_error=True)`` 抛出的异常才记为 ``error``,
不计入准确率并在报告单列。

:func:`compute_metrics` 是纯函数;导入本模块不会构建 LLM,也不会访问网络。
只有 :func:`main` 读取 settings、构建真实模型并调用分类器。
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from app.agent.intent import classify_intent
from app.config import build_llm, get_settings

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CASES = _ROOT / "tests" / "eval" / "ch06_intent_cases.json"
_REPORT_PATH = (
    _ROOT / "docs" / "superpowers" / "reports" / "2026-11-18-ch06-intent-eval.md"
)

PASS_OVERALL = 0.90
PASS_PER_CLASS = 0.80

#: 生成报告时写入的 prompt 调整记录;若本轮调整了 prompt,在运行前改这里。
PROMPT_NOTE = (
    "无。本次验证未修改 `app/agent/intent.py` 的 prompt 或 few-shot 示例。"
)

#: 报告展示顺序,与 ``app.agent.state.Intent`` 一致。
_LABEL_ORDER = ["物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊", "其他"]


def compute_metrics(expected: list[str], predicted: list[str]) -> dict:
    """Return overall/per-class accuracy and the confusion matrix.

    The result shape is exactly::

        {"total": int, "correct": int, "accuracy": float,
         "per_class": {label: {"total": int, "correct": int}},
         "confusion": {expected_label: {predicted_label: int}}}

    ``per_class`` counts by *expected* label (recall); ``confusion`` rows are
    expected labels and columns are predicted labels. All labels seen in either
    list appear in both structures so the matrix is rectangular.
    """
    if len(expected) != len(predicted):
        raise ValueError(
            f"expected/predicted length mismatch: {len(expected)} != {len(predicted)}"
        )

    total = len(expected)
    correct = sum(1 for exp, pred in zip(expected, predicted) if exp == pred)
    labels = list(dict.fromkeys([*expected, *predicted]))
    per_class = {label: {"total": 0, "correct": 0} for label in labels}
    confusion = {label: {other: 0 for other in labels} for label in labels}
    for exp, pred in zip(expected, predicted):
        per_class[exp]["total"] += 1
        if exp == pred:
            per_class[exp]["correct"] += 1
        confusion[exp][pred] += 1

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "per_class": per_class,
        "confusion": confusion,
    }


def load_cases(path: Path = _DEFAULT_CASES) -> list[dict]:
    """Read the annotated intent cases from *path*."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"intent cases must be a list, got {type(rows).__name__}")
    return rows


def run_cases(rows: list[dict], llm) -> list[dict]:
    """Run the real classifier over *rows* and return one result per case.

    Each result holds ``id``, ``text``, ``expected``, ``predicted``,
    ``confidence``, ``error`` and ``correct``. ``classify_intent`` is called
    with ``raise_on_error=True``: an exception is recorded in ``error`` (the
    case is excluded from metrics by :func:`summarize`) instead of being
    silently turned into a fabricated ``其他`` prediction.
    """
    results: list[dict] = []
    for number, row in enumerate(rows, 1):
        try:
            prediction = classify_intent(
                llm,
                row["text"],
                row.get("history", []),
                raise_on_error=True,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            results.append(
                {
                    "id": row["id"],
                    "text": row["text"],
                    "expected": row["intent"],
                    "predicted": None,
                    "confidence": None,
                    "error": error,
                    "correct": False,
                }
            )
            print(
                f"[{number}/{len(rows)}] {row['id']} expect={row['intent']} "
                f"ERROR {error}"
            )
            continue

        correct = prediction.intent == row["intent"]
        results.append(
            {
                "id": row["id"],
                "text": row["text"],
                "expected": row["intent"],
                "predicted": prediction.intent,
                "confidence": prediction.confidence,
                "error": "",
                "correct": correct,
            }
        )
        mark = "OK" if correct else "WRONG"
        print(
            f"[{number}/{len(rows)}] {row['id']} expect={row['intent']} "
            f"pred={prediction.intent} conf={prediction.confidence:.2f} {mark}"
        )
    return results


def summarize(results: list[dict]) -> tuple[dict, list[dict], list[dict]]:
    """Return ``(metrics, wrong, errors)``.

    Metrics cover only valid model predictions; exceptions raised by
    ``classify_intent(..., raise_on_error=True)`` are returned separately as
    *errors* so the report can list them and force a FAIL instead of crediting a
    fabricated label.
    """
    valid = [result for result in results if not result["error"]]
    metrics = compute_metrics(
        [result["expected"] for result in valid],
        [result["predicted"] for result in valid],
    )
    wrong = [result for result in valid if not result["correct"]]
    errors = [result for result in results if result["error"]]
    return metrics, wrong, errors


def _per_class_pass(per_class: dict) -> dict[str, bool | None]:
    """Map each scored label to its >= 80% verdict (``None`` if no cases)."""
    verdicts: dict[str, bool | None] = {}
    for label, counts in per_class.items():
        if not counts["total"]:
            verdicts[label] = None
            continue
        verdicts[label] = counts["correct"] / counts["total"] >= PASS_PER_CLASS
    return verdicts


def _ordered_labels(metrics: dict) -> list[str]:
    labels = [label for label in _LABEL_ORDER if label in metrics["per_class"]]
    labels += [label for label in metrics["per_class"] if label not in labels]
    return labels


def _command_line() -> str:
    module = getattr(__spec__, "name", None) or "app.jobs.eval_intent"
    return f"{Path(sys.executable).name} -m {module}"


def _one_line(text: str, limit: int = 200) -> str:
    """Collapse whitespace, escape table pipes and truncate for the report."""
    collapsed = " ".join(str(text or "").split()).replace("|", "\\|")
    return collapsed if len(collapsed) <= limit else collapsed[:limit] + "..."


def render_report(
    results: list[dict],
    metrics: dict,
    wrong: list[dict],
    errors: list[dict],
    model_name: str,
    cases_path: Path,
    generated_at: datetime | None = None,
    prompt_note: str | None = None,
) -> str:
    """Render the markdown evaluation report (Chinese, repo style)."""
    generated_at = generated_at or datetime.now()
    labels = _ordered_labels(metrics)
    verdicts = _per_class_pass(metrics["per_class"])
    overall_pass = metrics["accuracy"] >= PASS_OVERALL
    class_pass = all(verdicts.get(label) is True for label in _LABEL_ORDER)
    passed = overall_pass and class_pass and not errors
    json_rate = metrics["total"] / len(results) if results else 0.0
    encoding = os.environ.get("PYTHONIOENCODING") or "(unset)"

    lines = [
        "# ch06 意图分类评估报告",
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
        f"each class >= {PASS_PER_CLASS:.0%}, no `error`",
        "",
        "## 总体指标",
        "",
        "| total | correct | accuracy | JSON 可解析率 | overall pass | "
        "each class pass | errors | 结论 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        f"| {metrics['total']} | {metrics['correct']} | "
        f"{metrics['accuracy']:.3f} | {json_rate:.3f} | "
        f"{'PASS' if overall_pass else 'FAIL'} | "
        f"{'PASS' if class_pass else 'FAIL'} | "
        f"{len(errors)} | {'PASS' if passed else 'FAIL'} |",
        "",
        "准确率只统计有效模型输出;`intent=\"其他\"` 是合法预测,不当作兜底。"
        "分类器抛异常的样例从 total/correct/per_class/confusion 中排除,"
        "在「异常样例」单列,并直接判 FAIL。",
        "",
    ]

    lines += ["## 每类准确率", "", "| intent | total | correct | accuracy | pass |",
              "| --- | --- | --- | --- | --- |"]
    for label in labels:
        counts = metrics["per_class"][label]
        accuracy = counts["correct"] / counts["total"] if counts["total"] else 0.0
        verdict = verdicts[label]
        verdict_text = "-" if verdict is None else ("PASS" if verdict else "FAIL")
        lines.append(
            f"| {label} | {counts['total']} | {counts['correct']} | "
            f"{accuracy:.3f} | {verdict_text} |"
        )

    lines += [
        "",
        "## 混淆矩阵",
        "",
        "行 = 标注(expected),列 = 预测(predicted)。",
        "",
        "| expected \\ predicted | " + " | ".join(labels) + " |",
        "| --- | " + " | ".join("---" for _ in labels) + " |",
    ]
    for expected_label in labels:
        row = [str(metrics["confusion"][expected_label][pred]) for pred in labels]
        lines.append(f"| {expected_label} | " + " | ".join(row) + " |")

    lines += [
        "",
        "## 异常样例(JSON/网络失败)",
        "",
    ]
    if errors:
        lines += [
            "以下样例调用 `classify_intent(..., raise_on_error=True)` 抛出异常,"
            "不是模型输出,已从指标中排除并直接判 FAIL:",
            "",
            "| id | text | expected | error |",
            "| --- | --- | --- | --- |",
        ]
        for case in errors:
            lines.append(
                f"| {case['id']} | {_one_line(case['text'])} | "
                f"{case['expected']} | {_one_line(case['error'])} |"
            )
    else:
        lines.append("无。")

    lines += [
        "",
        "## 完整逐条预测",
        "",
        "| id | text | expected | predicted | confidence | correct |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for case in results:
        if case["error"]:
            predicted = "-"
            confidence = "-"
            correct = "ERROR"
        else:
            predicted = case["predicted"]
            confidence = f"{case['confidence']:.2f}"
            correct = "PASS" if case["correct"] else "FAIL"
        lines.append(
            f"| {case['id']} | {_one_line(case['text'])} | {case['expected']} | "
            f"{predicted} | {confidence} | {correct} |"
        )

    lines += ["", "## 错误样例", ""]
    if wrong:
        lines += [
            "| id | text | expected | predicted | confidence |",
            "| --- | --- | --- | --- | --- |",
        ]
        for case in wrong:
            lines.append(
                f"| {case['id']} | {_one_line(case['text'])} | {case['expected']} | "
                f"{case['predicted']} | {case['confidence']:.2f} |"
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
    """Run the live intent evaluation and write the markdown report."""
    rows = load_cases(_DEFAULT_CASES)
    settings = get_settings()
    model_name = settings.intent_model or settings.model_name
    llm = build_llm(
        settings,
        temperature=0.0,
        model=settings.intent_model or None,
    )
    print(f"eval_intent: cases={len(rows)} model={model_name} temperature=0")

    results = run_cases(rows, llm)
    errors = [result for result in results if result["error"]]
    if errors and len(errors) == len(results):
        print("\neval_intent: all cases raised; API/network or JSON failure.")
        print(f"first error: {errors[0]['error']}")
        return 2

    metrics, wrong, errors = summarize(results)
    report = render_report(
        results,
        metrics,
        wrong,
        errors,
        model_name,
        _DEFAULT_CASES,
    )
    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text(report, encoding="utf-8")

    verdicts = _per_class_pass(metrics["per_class"])
    class_pass = all(verdicts.get(label) is True for label in _LABEL_ORDER)
    passed = metrics["accuracy"] >= PASS_OVERALL and class_pass and not errors

    print(
        f"\neval_intent: valid={metrics['total']}/{len(results)} "
        f"correct={metrics['correct']} accuracy={metrics['accuracy']:.3f} "
        f"errors={len(errors)}"
    )
    for label in _LABEL_ORDER:
        counts = metrics["per_class"].get(label)
        if not counts or not counts["total"]:
            continue
        accuracy = counts["correct"] / counts["total"]
        print(f"  {label}: {counts['correct']}/{counts['total']} = {accuracy:.3f}")
    if wrong:
        print("\neval_intent: wrong cases")
        for case in wrong:
            print(
                f"  {case['id']} expected={case['expected']} "
                f"predicted={case['predicted']} conf={case['confidence']:.2f} "
                f"text={case['text']}"
            )
    if errors:
        print("\neval_intent: error cases")
        for case in errors:
            print(f"  {case['id']}: {case['error']}")
    print(f"\neval_intent: report written to {_REPORT_PATH}")
    print(f"eval_intent: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
