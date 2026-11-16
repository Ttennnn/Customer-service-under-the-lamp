"""工具执行的会话/引用上下文。

用 ``contextvars.ContextVar`` 在请求/会话执行链中传递两类上下文:

1. ``current_conversation_id``:当前会话 id,供需要感知会话上下文的工具
   (如 ``create_ticket``)读取。默认 ``None`` 表示没有关联会话。
2. ``current_citation_collector``:本轮的可变引用收集器。``execute_tool`` 在
   子线程中执行工具前会 ``copy_context()``;contextvar 里放的是可变对象引用,
   所以工具在子线程里 ``add(...)`` 的修改对主线程可见。不要用不可变值,也
   不要在工具里 ``set()`` 一个新对象,否则主线程看不到新增引用。
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from app.rag.retrieval import RetrievedChunk

current_conversation_id: ContextVar[int | None] = ContextVar(
    "current_conversation_id", default=None
)


@dataclass
class CitationCollector:
    """Per-turn citation registry, deduped by ``chunk_id``.

    Numbering is global for the turn: the first chunk added gets ``n=1`` and a
    chunk already seen reuses its earlier number.  ``faq_called`` is set by
    ``query_faq`` on hit, miss and infrastructure error so the self-check step
    can tell "knowledge was consulted" from "no knowledge question at all".
    """

    sources: list[dict] = field(default_factory=list)
    _by_chunk: dict[int, int] = field(default_factory=dict)
    faq_called: bool = False

    def add(self, chunks: Sequence[RetrievedChunk]) -> list[int]:
        """Register *chunks* in order and return their citation numbers."""
        numbers: list[int] = []
        for chunk in chunks:
            number = self._by_chunk.get(chunk.chunk_id)
            if number is None:
                number = len(self.sources) + 1
                self._by_chunk[chunk.chunk_id] = number
                self.sources.append(
                    {
                        "n": number,
                        "chunk_id": chunk.chunk_id,
                        "section_path": chunk.section_path,
                        "question": chunk.questions,
                        "answer": chunk.answer,
                        "category": chunk.category,
                        "content_type": chunk.content_type,
                        "score": chunk.score,
                    }
                )
            numbers.append(number)
        return numbers

    def payload(self) -> list[dict]:
        """Return a snapshot copy of the collected sources."""
        return [dict(source) for source in self.sources]


current_citation_collector: ContextVar[CitationCollector | None] = ContextVar(
    "current_citation_collector", default=None
)
