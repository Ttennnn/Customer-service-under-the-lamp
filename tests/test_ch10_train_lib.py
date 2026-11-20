from scripts.ch10.train_lib import compute_metrics, multi_hot, scan_threshold


def test_multi_hot_uses_taxonomy_order():
    from app.core.taxonomy import LABEL2ID

    vec = multi_hot(["尺码", "退换货"], LABEL2ID)
    assert len(vec) == 17
    assert vec[LABEL2ID["尺码"]] == 1.0
    assert vec[LABEL2ID["退换货"]] == 1.0
    assert sum(vec) == 2.0


def test_compute_metrics_known_values():
    gold = [{"尺码", "退换货"}, {"物流"}]
    pred = [{"尺码"}, {"物流"}]
    out = compute_metrics(gold, pred)
    assert round(out["micro_f1"], 4) == 0.8
    assert 0 <= out["macro_f1"] <= 1


def test_scan_threshold_returns_best_candidate():
    probs = [[0.9] + [0.0] * 16, [0.4] + [0.0] * 16]
    gold = [{"退换货"}, set()]
    out = scan_threshold(probs, gold, candidates=(0.3, 0.5, 0.7))
    assert out["best_threshold"] in (0.3, 0.5, 0.7)
    assert len(out["scan"]) == 3
