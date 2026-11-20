"""飞轮标准化/查重 prompt 评估工具(纯离线测试)。

真实 LLM 只在 :func:`main` 里显式构造;导入模块不建模型、不联网。
"""

import importlib
import json
from pathlib import Path

from app.services.flywheel import FlywheelOutput

SAMPLES = Path(__file__).parent / "eval" / "flywheel_samples.json"


class FakeLLM:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def with_structured_output(self, schema, method=None):
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        human = messages[-1].content
        for raw_question, output in self.outputs.items():
            if raw_question in human:
                return output
        raise AssertionError(f"unexpected sample: {human[:80]}")


def _samples() -> list[dict]:
    return json.loads(SAMPLES.read_text(encoding="utf-8"))["samples"]


def _expected_llm(samples: list[dict]) -> FakeLLM:
    outputs = {}
    for sample in samples:
        keywords = sample.get("expect_normalized_any") or ["问题"]
        outputs[sample["raw_question"]] = FlywheelOutput(
            " ".join(keywords),
            sample.get("expect_matched_id"),
            "",
        )
    return FakeLLM(outputs)


def test_samples_are_valid_json_and_count():
    samples = _samples()
    assert 10 <= len(samples) <= 15
    assert len({sample["id"] for sample in samples}) == len(samples)


def test_samples_schema_and_candidate_ids():
    for sample in _samples():
        assert isinstance(sample["raw_question"], str) and sample["raw_question"]
        candidates = sample["candidates"]
        assert candidates
        ids = [candidate["id"] for candidate in candidates]
        for candidate in candidates:
            assert isinstance(candidate["id"], int)
            assert isinstance(candidate["normalized_question"], str)
            assert candidate["normalized_question"]
            assert isinstance(candidate["occurrence_count"], int)
            assert candidate["review_status"] in ("待审", "通过", "驳回")
        expected = sample.get("expect_matched_id")
        assert expected is None or expected in ids
        assert isinstance(sample.get("expect_normalized_any"), list)


def test_samples_cover_match_and_no_match():
    samples = _samples()
    matched = [sample for sample in samples if sample.get("expect_matched_id") is not None]
    unmatched = [sample for sample in samples if sample.get("expect_matched_id") is None]
    assert len(matched) >= 4
    assert len(unmatched) >= 4


def test_import_does_not_create_llm(monkeypatch):
    import app.config

    calls = []
    monkeypatch.setattr(
        app.config, "build_llm", lambda *args, **kwargs: calls.append("build")
    )
    module = importlib.reload(
        importlib.import_module("app.jobs.eval_flywheel_prompts")
    )
    assert module is not None
    assert calls == []


def test_score_sample_requires_match_and_keyword():
    from app.jobs.eval_flywheel_prompts import score_flywheel_sample

    sample = {
        "expect_matched_id": 7,
        "expect_normalized_any": ["退款", "到账"],
    }
    assert score_flywheel_sample(
        sample, FlywheelOutput("退款多久到账", 7, "")
    )
    assert not score_flywheel_sample(
        sample, FlywheelOutput("退款多久到账", None, "")
    )
    assert not score_flywheel_sample(
        sample, FlywheelOutput("怎么退货", 7, "")
    )


def test_run_cases_counts_errors_separately():
    from app.jobs.eval_flywheel_prompts import run_cases

    samples = _samples()[:2]
    llm = FakeLLM({})  # 每条都抛 AssertionError
    results = run_cases(samples, llm)

    assert len(results) == 2
    assert all(result["error"] for result in results)
    assert all(not result["correct"] for result in results)


def test_summarize_pass_rate_includes_errors_as_failures():
    from app.jobs.eval_flywheel_prompts import summarize

    results = [
        {"id": "a", "correct": True, "error": ""},
        {"id": "b", "correct": False, "error": ""},
        {"id": "c", "correct": False, "error": "bad json"},
    ]
    metrics = summarize(results)
    assert metrics["total"] == 3
    assert metrics["passed"] == 1
    assert metrics["failed"] == 2
    assert metrics["errors"] == 1
    assert metrics["accuracy"] == 1 / 3


def test_main_passes_at_or_above_threshold(capsys):
    from app.jobs.eval_flywheel_prompts import main

    samples = _samples()
    code = main(["--samples", str(SAMPLES)], llm=_expected_llm(samples))

    assert code == 0
    output = capsys.readouterr().out
    assert "passed=" in output
    assert "PASS" in output


def test_main_fails_below_threshold(capsys):
    from app.jobs.eval_flywheel_prompts import main

    samples = _samples()
    outputs = {
        sample["raw_question"]: FlywheelOutput("问题", None, "") for sample in samples
    }
    code = main(["--samples", str(SAMPLES)], llm=FakeLLM(outputs))

    assert code == 1
    assert "FAIL" in capsys.readouterr().out


def test_main_returns_2_for_unreadable_samples(tmp_path):
    from app.jobs.eval_flywheel_prompts import main

    assert main(["--samples", str(tmp_path / "missing.json")], llm=object()) == 2


def test_main_returns_2_for_empty_samples(tmp_path):
    from app.jobs.eval_flywheel_prompts import main

    path = tmp_path / "empty.json"
    path.write_text('{"samples": []}', encoding="utf-8")
    assert main(["--samples", str(path)], llm=object()) == 2
