from langchain_core.messages import AIMessageChunk


class ScriptedLLM:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.bound_tool_names = []

    def bind_tools(self, tools, **kwargs):
        self.bound_tool_names = [t.name for t in tools]
        return self

    def stream(self, messages):
        yield self.chunks.pop(0)


def text_chunk(text):
    return AIMessageChunk(content=text)


def tool_chunk(name, args, call_id="c1"):
    return AIMessageChunk(
        content="",
        tool_call_chunks=[{"name": name, "args": args, "id": call_id, "index": 0}],
    )


def make_chunk(score=0.9):
    from app.rag.retrieval import RetrievedChunk
    return RetrievedChunk(
        chunk_id=1, score=score, questions="q", answer="a",
        section_path="s", category="商品", content_type="faq",
    )


def make_retrieval_result(chunks=None, rerank_failed=False):
    from app.rag.query_understanding import QueryAnalysis
    from app.rag.retrieval import RetrievalResult, RetrievalStrategy
    return RetrievalResult(
        chunks=list(chunks or []),
        strategy=RetrievalStrategy.HYBRID_RERANK,
        analysis=QueryAnalysis(
            original="q", rewritten="q", keywords=[], rewritten_by="fallback"
        ),
        rerank_failed=rerank_failed,
    )
