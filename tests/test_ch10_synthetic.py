from scripts.ch10.corpus_lib import Sample
from scripts.ch10.synthetic import (
    cross_check_labels,
    load_or_generate,
    merge_labeled,
    run_id_for,
    target_gap,
    validate_synthetic,
)


def test_target_gap_fills_missing_only():
    gap = target_gap({"退换货": 88, "物流": 3}, target=100)
    assert gap["退换货"] == 12
    assert gap["物流"] == 97
    assert gap["尺码"] == 100


def test_run_id_is_stable_and_force_changes_it():
    params = {"target": 100, "multi": 0.15}
    assert run_id_for(params) == run_id_for(params)
    assert run_id_for(params, force=True) != run_id_for(params)


def test_validate_synthetic_rejects_unknown_and_empty():
    good = Sample(
        id="a",
        text="买大了想退",
        labels=("退换货", "尺码"),
        origin="synthetic",
    )
    bad = Sample(id="b", text="售后", labels=("售后",), origin="synthetic")
    empty = Sample(id="c", text="", labels=("尺码",), origin="synthetic")
    assert validate_synthetic([good, bad, empty]) == [good]


def test_cross_check_flags_label_disagreement():
    samples = [
        Sample(
            id="a",
            text="买大了想退",
            labels=("尺码", "退换货"),
            origin="synthetic",
        )
    ]
    report = cross_check_labels(samples, lambda text: ["尺码"])
    assert report["agreement_rate"] == 0.0
    assert report["disagreements"][0]["id"] == "a"


def test_merge_labeled_applies_corrections_and_dedupes():
    real = [
        Sample(id="r1", text="买大了想退", labels=("尺码",), origin="real")
    ]
    merged = merge_labeled(real, [], {"r1": ["尺码", "退换货"]})
    assert merged[0].labels == ("退换货", "尺码")
    assert merged[0].meta["audited"] is True


class FakeStructured:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = 0

    def invoke(self, messages):
        out = self.outputs[self.calls]
        self.calls += 1
        return out


class FakeLLM:
    def __init__(self, outputs):
        self.outputs = outputs

    def with_structured_output(self, schema, method="json_mode"):
        return FakeStructured(self.outputs)


def test_load_or_generate_reuses_frozen_file(tmp_path):
    gap = {label: 0 for label in target_gap({}, target=100)}
    gap["尺码"] = 1
    llm = FakeLLM([{"items": [{"text": "猫窝买大了", "labels": ["尺码"]}]}])
    from scripts.ch10.llm_io import CallBudget

    first = load_or_generate(
        llm,
        gap,
        tmp_path,
        ratios={"multi": 0.15},
        budget=CallBudget(5),
        batch_size=10,
    )
    assert len(first) == 1

    class ExplodingLLM:
        def with_structured_output(self, *args, **kwargs):
            raise AssertionError("cached generation must not call the LLM")

    second = load_or_generate(
        ExplodingLLM(),
        gap,
        tmp_path,
        ratios={"multi": 0.15},
        budget=CallBudget(5),
        batch_size=10,
    )
    assert [s.text for s in second] == [s.text for s in first]
