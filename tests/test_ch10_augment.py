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
