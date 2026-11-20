import pytest

from app.core import paths, taxonomy


def test_taxonomy_has_exactly_17_unique_classes_in_order():
    assert taxonomy.TOPIC_CLASSES == (
        "退换货",
        "物流",
        "尺码",
        "发票",
        "质量问题",
        "运费",
        "优惠活动",
        "价保",
        "支付",
        "订单修改",
        "库存补货",
        "商品信息",
        "保修维修",
        "账号",
        "会员积分",
        "评价",
        "其他",
    )
    assert len(set(taxonomy.TOPIC_CLASSES)) == 17


def test_boundaries_and_examples_cover_every_class():
    assert set(taxonomy.TOPIC_BOUNDARIES) == set(taxonomy.TOPIC_CLASSES)
    assert set(taxonomy.TOPIC_EXAMPLES) == set(taxonomy.TOPIC_CLASSES)
    assert all(
        taxonomy.TOPIC_BOUNDARIES[label].strip()
        for label in taxonomy.TOPIC_CLASSES
    )
    assert all(taxonomy.TOPIC_EXAMPLES[label] for label in taxonomy.TOPIC_CLASSES)


def test_normalize_rejects_unknown_and_orders_by_taxonomy():
    assert taxonomy.normalize_labels(["退换货", "尺码", "退换货"]) == [
        "退换货",
        "尺码",
    ]
    with pytest.raises(ValueError, match="unknown topic label"):
        taxonomy.normalize_labels(["售后"])


def test_labels_from_scores_threshold_and_fallback():
    scores = {label: 0.1 for label in taxonomy.TOPIC_CLASSES}
    scores["尺码"] = 0.91
    scores["退换货"] = 0.88
    scores["其他"] = 0.95
    assert taxonomy.labels_from_scores(scores, 0.5) == ["退换货", "尺码"]

    low = {label: 0.2 for label in taxonomy.TOPIC_CLASSES}
    low["物流"] = 0.49
    assert taxonomy.labels_from_scores(low, 0.5) == ["物流"]


def test_paths_are_under_repo_root():
    assert paths.DATA_DIR == paths.ROOT / "data" / "ch10"
    assert paths.REPORT_DIR == paths.DATA_DIR / "reports"
    assert paths.GOLDEN_PATH == paths.DATA_DIR / "golden_samples.jsonl"


def test_normalize_drops_fallback_when_specific_label_present():
    assert taxonomy.normalize_labels(["其他", "尺码"]) == ["尺码"]
