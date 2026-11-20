from scripts.ch10.validate_golden import compare_labels


def test_compare_labels_exact_set_match_and_rate():
    golden = [
        {"id": "a", "text": "买大了想退", "labels": ["尺码", "退换货"]},
        {"id": "b", "text": "开发票", "labels": ["发票"]},
    ]
    predicted = {"a": ["退换货", "尺码"], "b": ["支付"]}
    out = compare_labels(golden, predicted)
    assert out["pass_rate"] == 0.5
    assert out["mismatches"] == [
        {"id": "b", "gold": ["发票"], "predicted": ["支付"]}
    ]
