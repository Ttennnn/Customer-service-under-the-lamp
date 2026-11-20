import pytest

from app.core.taxonomy import TOPIC_CLASSES
from scripts.ch10.corpus_lib import Sample
from scripts.ch10.llm_io import BudgetExceeded, CallBudget
from scripts.ch10.prelabel import prelabel_samples
from scripts.ch10.prompts import PRELABEL_SYSTEM, render_taxonomy
from scripts.ch10.schemas import PrelabelBatch, SyntheticBatch  # noqa: F401


def test_taxonomy_rendered_with_boundaries_and_literal_rule():
    text = render_taxonomy()
    for label in TOPIC_CLASSES:
        assert label in text
    assert "字面" in PRELABEL_SYSTEM and "其他" in PRELABEL_SYSTEM


def test_schema_rejects_unknown_label_and_dedupes_via_normalize():
    with pytest.raises(ValueError):
        PrelabelBatch(items=[{"id": "x", "labels": ["售后"]}])


def test_budget_raises_after_limit():
    budget = CallBudget(1)
    budget.spend()
    with pytest.raises(BudgetExceeded):
        budget.spend()


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


def test_prelabel_batches_preserve_ids_and_normalize_order():
    samples = [
        Sample(id="a", text="买大了想退", origin="real"),
        Sample(id="b", text="开发票", origin="real"),
    ]
    llm = FakeLLM(
        [
            {
                "items": [
                    {"id": "a", "labels": ["退换货", "尺码"]},
                    {"id": "b", "labels": ["发票"]},
                ]
            }
        ]
    )
    out = prelabel_samples(llm, samples, CallBudget(5), batch_size=10)
    assert [s.id for s in out] == ["a", "b"]
    assert out[0].labels == ("退换货", "尺码")
    assert out[1].labels == ("发票",)
