from scripts.ch10.augment_train import accept_variant, augment_train
from scripts.ch10.corpus_lib import Sample


def test_accept_only_when_label_set_unchanged():
    original = Sample(
        id="a",
        text="买大了想退",
        labels=("尺码", "退换货"),
        origin="real",
    )
    assert (
        accept_variant(original, "买大了一码想退货", ["退换货", "尺码"])
        is True
    )
    assert accept_variant(original, "买大了想退", ["尺码"]) is False


class FakeLLM:
    def __init__(self, variants, prelabel_fn):
        self.variants = variants
        self.prelabel_fn = prelabel_fn
        self.index = 0

    def with_structured_output(self, schema, method="json_mode"):
        return self

    def invoke(self, messages):
        human = messages[-1].content
        sample_id = human.split("\t", 1)[0]
        text = self.variants[self.index]
        self.index += 1
        return {"items": [{"id": sample_id, "text": text}]}


def test_augment_keeps_original_and_reports_drop_rate():
    train = [Sample(id="a", text="开发票", labels=("发票",), origin="real")]

    def prelabel(text):
        return ["发票"]

    out, report = augment_train(
        FakeLLM(["开票"], prelabel), train, budget=None
    )
    assert len(out) == 2
    assert report["kept"] == 1
    assert report["dropped"] == 0


def test_assemble_drops_exact_and_near_val_leak():
    from scripts.ch10.augment_train import assemble

    original = Sample(id="a", text="开发票", labels=("发票",), origin="real")
    out, report = assemble(
        [original],
        [(original, "开票")],
        {"开票": ["发票"]},
        leak_texts=["开票"],
    )
    assert len(out) == 1
    assert report["leak_dropped"] == 1

    original2 = Sample(
        id="b", text="猫窝买大了", labels=("尺码",), origin="real"
    )
    variant = "这猫窝买大了想退"
    leak = "这猫窝买大了想退货"
    out2, report2 = assemble(
        [original2],
        [(original2, variant)],
        {variant: ["尺码"]},
        leak_texts=[leak],
    )
    assert len(out2) == 1
    assert report2["leak_dropped"] == 1


def test_assemble_dedupes_variant_text():
    from scripts.ch10.augment_train import assemble

    a = Sample(id="a", text="开发票", labels=("发票",), origin="real")
    b = Sample(id="b", text="要发票", labels=("发票",), origin="real")
    out, report = assemble(
        [a, b], [(a, "开票"), (b, "开票")], {"开票": ["发票"]}
    )
    assert len(out) == 3
    assert report["duplicates"] == 1


def test_augment_raises_on_budget_exceeded():
    import pytest

    from scripts.ch10.llm_io import BudgetExceeded

    class Boom:
        def with_structured_output(self, *args, **kwargs):
            return self

        def invoke(self, messages):
            raise BudgetExceeded("cap")

    with pytest.raises(BudgetExceeded):
        augment_train(
            Boom(),
            [Sample(id="a", text="开发票", labels=("发票",), origin="real")],
            None,
            lambda text: ["发票"],
        )
