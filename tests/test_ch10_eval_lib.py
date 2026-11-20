from scripts.ch10.eval_lib import (
    error_directions,
    red_line_for,
    replay_threshold,
    tier_for,
)


def test_tiers_and_red_lines():
    assert tier_for("退换货") == "严"
    assert red_line_for("退换货") == 0.9
    assert tier_for("运费") == "中"
    assert red_line_for("运费") == 0.8
    assert tier_for("评价") == "宽"
    assert red_line_for("评价") is None


def test_error_directions_miss_extra_misplaced():
    gold = [{"尺码", "退换货"}]
    pred = [{"尺码", "物流"}]
    out = error_directions(gold, pred)
    assert out["miss"] == {"退换货": 1}
    assert out["extra"] == {"物流": 1}
    assert out["misplaced_pairs"] == [
        {"gold": "退换货", "pred": "物流"}
    ]
    assert out["miss_total"] == 1
    assert out["extra_total"] == 1


def test_replay_threshold_returns_scan_and_expected():
    scores = [
        {"probs": [0.9] + [0.0] * 16, "gold": ["退换货"]},
        {"probs": [0.4] + [0.0] * 16, "gold": []},
    ]
    out = replay_threshold(
        scores, candidates=(0.3, 0.5, 0.7), expected=0.5
    )
    assert len(out["scan"]) == 3
    assert out["best_threshold"] in (0.3, 0.5, 0.7)
    assert out["expected"] == 0.5
    assert "matches" in out


def test_check_service_threshold_detects_mismatch_and_non_onnx():
    from scripts.ch10.eval_lib import check_service_threshold

    assert (
        check_service_threshold(
            {"threshold": 0.45}, {"threshold": 0.45, "source": "onnx"}
        )
        is None
    )
    mismatch = check_service_threshold(
        {"threshold": 0.5}, {"threshold": 0.45, "source": "onnx"}
    )
    assert mismatch and "service" in mismatch
    non_onnx = check_service_threshold(
        {"threshold": 0.45}, {"threshold": 0.45, "source": "torch"}
    )
    assert non_onnx and "onnx" in non_onnx
