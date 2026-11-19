"""飞轮标准化/查重 prompt 评估(Task 6)。

用真实 LLM 跑 ``tests/eval/flywheel_samples.json`` 中的 10–15 条样例,
统计命中/不命中/坏 JSON(含模型异常),通过率低于 80% 时退出码为 1。

样例格式(顶层对象含 ``samples`` 列表)::

    {"id": str, "raw_question": str,
     "candidates": [{"id": int, "normalized_question": str,
                     "occurrence_count": int, "review_status": str}],
     "expect_matched_id": int | null,
     "expect_normalized_any": [str, ...]}

评分:``matched_question_id`` 必须与标注完全一致,且标准化问题至少包含
一个期望关键词(空列表视为通配)。坏 JSON 与网络异常记为 error,算作失败
并单独计数。

导入本模块不会创建 LLM,也不会访问网络;只有 :func:`main` 在未注入
*llm* 时读取 settings 并构建真实模型。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from app.config import build_llm, get_settings
from app.services.flywheel import FlywheelOutput, normalize_and_match

_DEFAULT_SAMPLES = (
    Path(__file__).resolve().parents[2] / "tests" / "eval" / "flywheel_samples.json"
)
PASS_THRESHOLD = 0.80


def _candidate(row: dict) -> SimpleNamespace:
    """Build the small duck-typed candidate object the prompt renderer needs."""
    return SimpleNamespace(
        id=int(row["id"]),
        normalized_question=str(row["normalized_question"]),
        occurrence_count=int(row.get("occurrence_count", 1)),
        review_status=str(row.get("review_status", "待审")),
    )


def score_flywheel_sample(sample: dict, output: FlywheelOutput) -> bool:
    """Return whether *output* matches the annotated match id and keywords."""
    if output is None:
        return False
    if getattr(output, "matched_question_id", None) != sample.get("expect_matched_id"):
        return False
    normalized = (getattr(output, "normalized_question", "") or "").strip()
    if not normalized:
        return False
    keywords = sample.get("expect_normalized_any") or []
    if keywords and not any(keyword in normalized for keyword in keywords):
        return False
    return True


def run_cases(samples: list[dict], llm) -> list[dict]:
    """Run the real normalization prompt over *samples*; never aborts the suite."""
    results: list[dict] = []
    for number, sample in enumerate(samples, 1):
        candidates = [_candidate(row) for row in sample.get("candidates") or []]
        try:
            output = normalize_and_match(llm, sample["raw_question"], candidates)
        except Exception as exc:  # noqa: BLE001 - the eval reports, never crashes
            results.append(
                {
                    "id": sample.get("id", ""),
                    "raw_question": sample.get("raw_question", ""),
                    "expected_id": sample.get("expect_matched_id"),
                    "predicted_id": None,
                    "normalized": "",
                    "error": f"{type(exc).__name__}: {exc}",
                    "correct": False,
                }
            )
            print(f"[{number}/{len(samples)}] {sample.get('id', '')} ERROR {exc}")
            continue

        correct = score_flywheel_sample(sample, output)
        results.append(
            {
                "id": sample.get("id", ""),
                "raw_question": sample.get("raw_question", ""),
                "expected_id": sample.get("expect_matched_id"),
                "predicted_id": output.matched_question_id,
                "normalized": output.normalized_question,
                "error": "",
                "correct": correct,
            }
        )
        print(
            f"[{number}/{len(samples)}] {sample.get('id', '')} "
            f"expect={sample.get('expect_matched_id')} "
            f"pred={output.matched_question_id} "
            f"normalized={output.normalized_question!r} "
            f"{'PASS' if correct else 'FAIL'}"
        )
    return results


def summarize(results: list[dict]) -> dict:
    """Return pass/fail/error counts and pass rate (errors count as failures)."""
    total = len(results)
    passed = sum(1 for result in results if result["correct"])
    errors = sum(1 for result in results if result["error"])
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "errors": errors,
        "accuracy": passed / total if total else 0.0,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the flywheel normalization/dedup prompt on samples."
    )
    parser.add_argument(
        "--samples",
        default=None,
        help=f"path to the samples JSON (default: {_DEFAULT_SAMPLES})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=PASS_THRESHOLD,
        help=f"pass rate required (default {PASS_THRESHOLD:.0%})",
    )
    return parser


def _load_samples(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return list(data)
    return list(data.get("samples") or [])


def main(argv: list[str] | None = None, *, llm=None) -> int:
    """Run the sample suite and return 0/1 (2 for unreadable/empty samples)."""
    args = _build_parser().parse_args(argv)
    path = Path(args.samples) if args.samples else _DEFAULT_SAMPLES
    try:
        samples = _load_samples(path)
    except (OSError, json.JSONDecodeError, TypeError, AttributeError) as exc:
        print(f"eval_flywheel_prompts: cannot read {path}: {exc}")
        return 2
    if not samples:
        print(f"eval_flywheel_prompts: no samples found in {path}")
        return 2

    if llm is None:
        settings = get_settings()
        llm = build_llm(
            settings,
            temperature=0.0,
            max_tokens=settings.flywheel_llm_max_tokens,
            request_timeout=settings.flywheel_timeout_s,
        )
        model_name = settings.model_name
    else:
        model_name = type(llm).__name__
    print(
        f"eval_flywheel_prompts: samples={len(samples)} model={model_name} "
        f"threshold={args.threshold:.0%}"
    )

    results = run_cases(samples, llm)
    metrics = summarize(results)
    matched = sum(
        1
        for result in results
        if not result["error"] and result["predicted_id"] is not None
    )
    unmatched = sum(
        1
        for result in results
        if not result["error"] and result["predicted_id"] is None
    )

    print(
        "\neval_flywheel_prompts: "
        f"total={metrics['total']} passed={metrics['passed']} "
        f"failed={metrics['failed']} errors={metrics['errors']} "
        f"accuracy={metrics['accuracy']:.3f} "
        f"matched={matched} unmatched={unmatched}"
    )
    passed = metrics["accuracy"] >= args.threshold
    print(f"eval_flywheel_prompts: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
