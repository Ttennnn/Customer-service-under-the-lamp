"""Citation collector and prompt-ordering tests (Task 6)."""

import pytest
from langchain.tools import tool

from app.rag.retrieval import RetrievedChunk
from app.tools.context import CitationCollector, current_citation_collector
from app.tools.executor import execute_tool
from app.tools.faq import order_for_prompt


def make_chunk(chunk_id: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        score=1.0 - chunk_id / 100,
        questions=f"问题{chunk_id}",
        answer=f"答案{chunk_id}",
        section_path=f"章节 > 小节{chunk_id}",
        category="商品",
        content_type="faq",
    )


@pytest.fixture(autouse=True)
def _reset_collector():
    """Keep a collector set by one test from leaking into the next one."""
    yield
    current_citation_collector.set(None)


def test_order_for_prompt_puts_strongest_first_and_last():
    items = [make_chunk(i) for i in range(5)]
    assert [c.chunk_id for c in order_for_prompt(items)] == [0, 2, 4, 3, 1]


def test_order_for_prompt_edge_sizes():
    assert order_for_prompt([]) == []
    assert [c.chunk_id for c in order_for_prompt([make_chunk(0)])] == [0]
    assert [
        c.chunk_id
        for c in order_for_prompt([make_chunk(0), make_chunk(1)])
    ] == [0, 1]
    ten = [make_chunk(i) for i in range(10)]
    assert [c.chunk_id for c in order_for_prompt(ten)] == [
        0, 2, 4, 6, 8, 9, 7, 5, 3, 1,
    ]


def test_collector_dedupes_and_numbers_globally():
    c = CitationCollector()
    assert c.add([make_chunk(1), make_chunk(2)]) == [1, 2]
    assert c.add([make_chunk(2), make_chunk(3)]) == [2, 3]
    assert [s["n"] for s in c.payload()] == [1, 2, 3]
    assert [s["chunk_id"] for s in c.payload()] == [1, 2, 3]


def test_collector_payload_has_exact_keys():
    c = CitationCollector()
    c.add([make_chunk(7)])
    assert set(c.payload()[0]) == {
        "n",
        "chunk_id",
        "section_path",
        "question",
        "answer",
        "category",
        "content_type",
        "score",
        "is_key_clause",
    }


def test_collector_mutation_is_visible_from_execute_tool():
    collector = CitationCollector()
    current_citation_collector.set(collector)

    @tool
    def fake_tool() -> str:
        """Add one source from the worker thread."""
        collector.add([make_chunk(7)])
        return "ok"

    assert execute_tool(fake_tool, {}) == "ok"
    assert collector.payload()[0]["chunk_id"] == 7
