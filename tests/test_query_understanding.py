from app.rag.query_understanding import (
    SYNONYMS,
    QueryAnalysis,
    QueryRewrite,
    analyze,
    expand_synonyms,
)


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    def with_structured_output(self, schema, method=None):
        return self

    def invoke(self, messages):
        return self.payload


def test_analyze_returns_rewrite_and_keywords():
    a = analyze(
        "邮费咋算",
        llm=FakeLLM(QueryRewrite(rewritten="运费怎么计算", keywords=["运费", "配送费"])),
    )
    assert a.rewritten == "运费怎么计算" and a.keywords == ["运费", "配送费"] and a.rewritten_by == "llm"
    assert a.original == "邮费咋算"


def test_analyze_falls_back_to_original_and_static_synonyms():
    class Boom:
        def with_structured_output(self, *a, **k):
            raise RuntimeError("down")

    a = analyze("邮费咋算", llm=Boom())
    assert a.rewritten == "邮费咋算" and "运费" in a.keywords and a.rewritten_by == "fallback"
    assert a.original == "邮费咋算"


def test_expand_synonyms_skips_terms_already_present():
    assert expand_synonyms("邮费 运费") == ["配送费"]


def test_synonyms_cover_required_pairs():
    assert SYNONYMS["邮费"] == ("运费", "配送费")
    assert SYNONYMS["快递费"] == ("运费",)
    assert SYNONYMS["到账"] == ("退款", "原路退回")
    assert SYNONYMS["灯不亮"] == ("故障", "维修")
    assert SYNONYMS["换货"] == ("换新",)


def test_expand_synonyms_dedupes_and_preserves_order():
    assert expand_synonyms("快递费和邮费怎么算") == ["运费", "配送费"]
    assert expand_synonyms("没有命中词") == []


def test_analyze_uses_lazy_default_llm_with_zero_temperature(monkeypatch):
    import app.config as config

    captured = {}

    def fake_build_llm(settings, temperature=0.7):
        captured["temperature"] = temperature
        return FakeLLM(QueryRewrite(rewritten="运费怎么算", keywords=["运费"]))

    monkeypatch.setattr(config, "build_llm", fake_build_llm)

    a = analyze("邮费怎么算")
    assert captured["temperature"] == 0.0
    assert a.rewritten == "运费怎么算" and a.rewritten_by == "llm"


def test_query_analysis_is_a_dataclass_with_expected_fields():
    a = QueryAnalysis(
        original="原问题",
        rewritten="标准问法",
        keywords=["关键词"],
        rewritten_by="fallback",
    )
    assert (a.original, a.rewritten, a.keywords, a.rewritten_by) == (
        "原问题",
        "标准问法",
        ["关键词"],
        "fallback",
    )
