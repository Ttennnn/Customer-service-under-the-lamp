from scripts.ch10.audit import review_rows
from scripts.ch10.corpus_lib import Sample


def test_review_rows_empty_input_is_safe():
    assert review_rows(real=[], synthetic=[]) == []


def test_review_rows_include_all_real_and_three_per_label():
    real = [Sample(id="r", text="真实问题", labels=("尺码",), origin="real")]
    synthetic = [
        Sample(
            id=f"s{i}",
            text=f"模拟问题{i}",
            labels=("尺码",),
            origin="synthetic",
        )
        for i in range(5)
    ]
    rows = review_rows(real, synthetic)
    assert any(row["origin"] == "real" for row in rows)
    assert sum(1 for row in rows if row["origin"] == "synthetic") == 3
