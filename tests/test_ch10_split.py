from scripts.ch10.build_dataset import (
    check_no_leak,
    class_coverage,
    split_samples,
)
from scripts.ch10.corpus_lib import Sample


def _sample(i, labels=("尺码",)):
    return Sample(id=f"s{i}", text=f"问题{i}", labels=labels, origin="synthetic")


def test_split_is_deterministic_and_roughly_80_10_10():
    data = [_sample(i, ("尺码",)) for i in range(100)]
    a = split_samples(data, seed=42)
    b = split_samples(data, seed=42)
    assert [s.id for s in a["train"]] == [s.id for s in b["train"]]
    assert abs(len(a["train"]) / len(data) - 0.8) <= 0.03


def test_no_leak_and_every_class_in_all_splits():
    data = [_sample(i, ("尺码",)) for i in range(60)] + [
        _sample(i + 60, ("物流",)) for i in range(60)
    ]
    splits = split_samples(data, seed=42)
    assert check_no_leak(splits) == {
        "train_val": 0,
        "train_test": 0,
        "val_test": 0,
    }
    assert all(
        class_coverage(splits)[label] == [True, True, True]
        for label in ("尺码", "物流")
    )


def test_rare_combo_uses_first_label_layer():
    data = [_sample(i, ("尺码", "退换货")) for i in range(12)] + [
        _sample(i + 12, ("尺码",)) for i in range(12)
    ]
    splits = split_samples(data, seed=42)
    assert len(splits["val"]) + len(splits["test"]) > 0


def test_near_duplicates_are_merged_before_split():
    data = [
        Sample(
            id="a",
            text="买大了想退货",
            labels=("尺码", "退换货"),
            origin="synthetic",
        ),
        Sample(
            id="b",
            text="买大了想退货",
            labels=("尺码", "退换货"),
            origin="synthetic",
        ),
    ]
    splits = split_samples(data, seed=42)
    assert sum(len(v) for v in splits.values()) == 1
    assert check_no_leak(splits) == {
        "train_val": 0,
        "train_test": 0,
        "val_test": 0,
    }
