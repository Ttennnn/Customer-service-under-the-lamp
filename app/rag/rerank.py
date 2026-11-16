"""SiliconFlow BGE reranker client.

Synchronous client for SiliconFlow's rerank endpoint::

    POST {base_url}/rerank
    {"model": "BAAI/bge-reranker-v2-m3", "query": "...", "documents": [...],
     "top_n": 10, "return_documents": false}

The response is not guaranteed to be ordered, so results are sorted by
``relevance_score`` descending.  Only 429, 5xx, and network errors are
retried; authentication and other client errors fail immediately.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from app.config import get_settings


class RerankError(RuntimeError):
    """Raised when a rerank request cannot be completed."""


@dataclass(frozen=True)
class RerankHit:
    """A single reranked document, identified by its input index."""

    index: int
    score: float


class SiliconFlowReranker:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 10.0,
        max_retries: int = 2,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._sleep = sleep
        self._client = client if client is not None else httpx.Client(timeout=timeout_s)

    def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[RerankHit]:
        """Rerank *documents* against *query*, best score first."""
        if not documents:
            return []

        url = f"{self.base_url}/rerank"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload: dict[str, object] = {
            "model": self.model,
            "query": query,
            "documents": list(documents),
            "return_documents": False,
        }
        if top_n is not None:
            payload["top_n"] = top_n

        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.post(url, headers=headers, json=payload)
            except httpx.TransportError as exc:
                if attempt >= self.max_retries:
                    raise RerankError(
                        f"rerank request failed after {attempt + 1} attempts: {exc}"
                    ) from exc
                self._sleep(0.5 * 2**attempt)
                continue

            if response.status_code == 200:
                hits = self._parse_response(response, len(documents))
                if top_n is not None:
                    hits = hits[: max(top_n, 0)]
                return hits

            retryable = response.status_code == 429 or 500 <= response.status_code < 600
            if retryable and attempt < self.max_retries:
                self._sleep(0.5 * 2**attempt)
                continue

            raise RerankError(
                f"rerank request failed with status {response.status_code}: "
                f"{response.text}"
            )

        raise RerankError("rerank request failed")  # pragma: no cover

    def close(self) -> None:
        self._client.close()

    def _parse_response(
        self, response: httpx.Response, document_count: int
    ) -> list[RerankHit]:
        try:
            rows = response.json()["results"]
            if not isinstance(rows, list):
                raise TypeError("results must be a list")
            hits: list[RerankHit] = []
            for row in rows:
                index = row["index"]
                score = row["relevance_score"]
                if isinstance(index, bool) or not isinstance(index, int):
                    raise TypeError(f"index must be an integer, got {index!r}")
                if isinstance(score, bool) or not isinstance(score, (int, float)):
                    raise TypeError(
                        f"relevance_score must be numeric, got {score!r}"
                    )
                hits.append(RerankHit(index=index, score=float(score)))
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            raise RerankError(
                f"invalid rerank response: {response.text}"
            ) from exc

        for hit in hits:
            if not 0 <= hit.index < document_count:
                raise RerankError(
                    f"invalid rerank response: index {hit.index} out of range "
                    f"for {document_count} documents: {response.text}"
                )

        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits


_reranker: SiliconFlowReranker | None = None
_reranker_lock = threading.Lock()


def get_reranker() -> SiliconFlowReranker:
    """Return the process-wide reranker, building it from settings once.

    ``rerank_base_url``/``rerank_api_key`` fall back to the embedding settings,
    so the BGE reranker can share SiliconFlow credentials with BGE-M3.
    """
    global _reranker

    with _reranker_lock:
        if _reranker is None:
            settings = get_settings()
            api_key = settings.rerank_api_key or settings.embedding_api_key
            if not api_key:
                raise RerankError(
                    "rerank_api_key is not configured; set RERANK_API_KEY or "
                    "EMBEDDING_API_KEY in .env to call SiliconFlow reranking"
                )
            _reranker = SiliconFlowReranker(
                base_url=settings.rerank_base_url or settings.embedding_base_url,
                api_key=api_key,
                model=settings.rerank_model,
                timeout_s=settings.rerank_timeout_s,
            )
        return _reranker
