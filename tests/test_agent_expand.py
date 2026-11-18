from app.rag.query_understanding import QueryExpansion, expand_queries


class FakeStructured:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.messages = result, error, []

    def with_structured_output(self, schema, method=None):
        return self

    def invoke(self, messages):
        self.messages.extend(messages)
        if self.error:
            raise self.error
        return self.result


def test_expand_returns_three_unique_queries():
    llm = FakeStructured(QueryExpansion(queries=["a", "b", "c"]))
    assert expand_queries(llm, "订单 1001 能退吗", {"order_id": "1001"}) == ["a", "b", "c"]


def test_expand_falls_back_when_not_three():
    for queries in (["a", "b"], ["a", "b", "c", "d"], ["a", "a", "b"]):
        assert expand_queries(FakeStructured(QueryExpansion(queries=queries)), "q") == ["q"]


def test_expand_falls_back_on_error():
    assert expand_queries(FakeStructured(error=ValueError("x")), "q") == ["q"]


def test_expand_prompt_mentions_queries_and_three():
    from app.rag.query_understanding import QUERY_EXPANSION_SYSTEM_PROMPT
    assert "queries" in QUERY_EXPANSION_SYSTEM_PROMPT
    assert "3" in QUERY_EXPANSION_SYSTEM_PROMPT
    assert "JSON" in QUERY_EXPANSION_SYSTEM_PROMPT


def test_expand_includes_order_data():
    llm = FakeStructured(QueryExpansion(queries=["a", "b", "c"]))
    expand_queries(llm, "能退吗", {"order_id": "1001", "status": "已签收"})
    assert "1001" in llm.messages[-1].content and "已签收" in llm.messages[-1].content
