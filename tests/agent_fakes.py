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


class ScriptedStructuredLLM:
    """Schema-aware fake for coref, intent, query-expansion and self-check."""

    def __init__(
        self,
        resolved: str,
        intent,
        queries=None,
        *,
        evidence_assessment=None,
        evidence_error: Exception | None = None,
    ):
        self.resolved = resolved
        self.intent = intent
        self.queries = queries or ["q1", "q2", "q3"]
        self.evidence_assessment = evidence_assessment
        self.evidence_error = evidence_error
        self.schema = None
        self.messages = []

    def with_structured_output(self, schema, method=None):
        self.schema = schema
        return self

    def invoke(self, messages):
        from app.agent.coref import CorefRewrite
        from app.rag.query_understanding import QueryExpansion
        from app.rag.self_check import EvidenceAssessment
        self.messages.extend(messages)
        if self.schema is CorefRewrite:
            return CorefRewrite(resolved_query=self.resolved)
        if self.schema is QueryExpansion:
            return QueryExpansion(queries=self.queries)
        if self.schema is EvidenceAssessment:
            if self.evidence_error is not None:
                raise self.evidence_error
            if self.evidence_assessment is not None:
                return self.evidence_assessment
            return EvidenceAssessment(useful=True)
        return self.intent


def text_chunk(text):
    return AIMessageChunk(content=text)


def tool_chunk(name, args, call_id="c1"):
    return AIMessageChunk(
        content="",
        tool_call_chunks=[{"name": name, "args": args, "id": call_id, "index": 0}],
    )


def make_chunk(score=0.9, is_key_clause=False):
    from app.rag.retrieval import RetrievedChunk
    return RetrievedChunk(
        chunk_id=1, score=score, questions="q", answer="a",
        section_path="s", category="商品", content_type="faq",
        is_key_clause=is_key_clause,
    )


class FakeGraph:
    """Scripted graph that can expose a sequence of state snapshots.

    ``snapshots`` models ``graph.get_state`` calls in order: the service checks
    for a pending interrupt before and after each stream. When the queue is
    exhausted, a plain snapshot without interrupts is returned, so older tests
    that only need ``final`` keep working unchanged.
    """

    def __init__(self, frames, final, snapshots=None):
        self.frames = frames
        self.final = final
        self.snapshots = list(snapshots) if snapshots is not None else []
        self.inputs = []
        self.config = None
        self.stream_calls = []

    def stream(self, inputs, config, stream_mode, subgraphs, version):
        self.inputs.append(inputs)
        self.stream_calls.append(inputs)
        self.config = config
        assert stream_mode == "custom" and subgraphs is True
        for frame in self.frames:
            # Real langgraph 1.2.20 shape for custom + subgraphs=True.
            yield {"type": "custom", "ns": (), "data": frame}

    def get_state(self, config):
        from types import SimpleNamespace
        self.config = config
        if self.snapshots:
            return self.snapshots.pop(0)
        return SimpleNamespace(values=self.final, interrupts=())


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
