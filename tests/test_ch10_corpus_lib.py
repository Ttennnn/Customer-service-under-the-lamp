from scripts.ch10.corpus_lib import (
    Sample,
    apply_corrections,
    dedupe_exact,
    dedupe_near,
    jaccard_3gram,
    redact_pii,
    stable_id,
)


def test_redact_phone_order_and_social_ids():
    text, flags = redact_pii(
        "手机号 13800000000，订单 E20241111001，微信 wx_abc123"
    )
    assert "13800000000" not in text
    assert "E20241111001" not in text
    assert "wx_abc123" not in text
    assert "[手机号]" in text
    assert "[订单号]" in text
    assert "[微信号]" in text
    assert set(flags) >= {"phone", "order", "wechat"}

    # product model numbers are not order ids
    clean, flags2 = redact_pii("MH-LP100 废砂盒还有吗")
    assert clean == "MH-LP100 废砂盒还有吗"
    assert flags2 == []


def test_stable_id_is_content_addressed_and_origin_scoped():
    assert stable_id("real", "买大了想退") == stable_id("real", "买大了想退")
    assert stable_id("real", "买大了想退") != stable_id(
        "synthetic", "买大了想退"
    )


def test_dedupe_exact_merges_source_refs():
    a = Sample(
        id="a", text="开发票", origin="real", source_refs=({"question_id": 1},)
    )
    b = Sample(
        id="b", text="开发票", origin="real", source_refs=({"question_id": 2},)
    )
    out = dedupe_exact([a, b])
    assert len(out) == 1
    assert out[0].source_refs == ({"question_id": 1}, {"question_id": 2})


def test_jaccard_and_near_dedupe():
    assert jaccard_3gram("买大了想退", "买大了想退货") > 0.5
    a = Sample(id="a", text="买大了想退货", origin="synthetic")
    b = Sample(id="b", text="买大了想退货", origin="synthetic")
    out, merged = dedupe_near([a, b])
    assert len(out) == 1
    assert merged == 1


def test_apply_corrections_unknown_id_raises():
    s = Sample(id="x", text="退活", origin="real")
    out = apply_corrections([s], {"x": ["退换货"]})
    assert out[0].labels == ("退换货",)
    assert out[0].meta["audited"] is True
