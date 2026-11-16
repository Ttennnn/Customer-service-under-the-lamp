"""SiliconFlow BGE-M3 embedding client.

Synchronous client for SiliconFlow's OpenAI-compatible embeddings endpoint::

    POST {base_url}/embeddings
    {"model": "BAAI/bge-m3", "input": [...], "encoding_format": "float"}

Inputs are split into ``batch_size`` chunks.  Each response is ordered by the
``index`` field because the service does not guarantee response order.  Only
429, 5xx, and network errors are retried; authentication and other client
errors fail immediately.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import httpx

from app.config import get_settings


class EmbeddingError(RuntimeError):
    """Raised when an embedding request cannot be completed."""


class SiliconFlowEmbedder:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "BAAI/bge-m3",
        dim: int = 1024,
        batch_size: int = 16,
        client: httpx.Client | None = None,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self.max_retries = max_retries
        self._sleep = sleep
        self._client = client if client is not None else httpx.Client(timeout=30.0)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts* in batches, preserving input order."""
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(texts[start : start + self.batch_size]))
        return vectors

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def close(self) -> None:
        self._client.close()

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        url = f"{self.base_url}/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model, "input": batch, "encoding_format": "float"}

        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.post(url, headers=headers, json=payload)
            except httpx.TransportError as exc:
                if attempt >= self.max_retries:
                    raise EmbeddingError(
                        f"embedding request failed after {attempt + 1} attempts: {exc}"
                    ) from exc
                self._sleep(0.5 * 2**attempt)
                continue

            if response.status_code == 200:
                return self._parse_response(response, len(batch))

            retryable = response.status_code == 429 or 500 <= response.status_code < 600
            if retryable and attempt < self.max_retries:
                self._sleep(0.5 * 2**attempt)
                continue

            raise EmbeddingError(
                f"embedding request failed with status {response.status_code}: "
                f"{response.text}"
            )

        raise EmbeddingError("embedding request failed")  # pragma: no cover

    def _parse_response(
        self, response: httpx.Response, expected_count: int
    ) -> list[list[float]]:
        try:
            rows = sorted(response.json()["data"], key=lambda row: row["index"])
            vectors = [row["embedding"] for row in rows]
        except (ValueError, KeyError, TypeError) as exc:
            raise EmbeddingError(
                f"invalid embedding response: {response.text}"
            ) from exc

        if len(vectors) != expected_count:
            raise EmbeddingError(
                f"embedding response returned {len(vectors)} vectors for "
                f"{expected_count} inputs: {response.text}"
            )
        for vector in vectors:
            if len(vector) != self.dim:
                raise EmbeddingError(
                    f"embedding dimension mismatch: expected {self.dim}, "
                    f"got {len(vector)}"
                )
        return vectors


_embedder: SiliconFlowEmbedder | None = None
_embedder_lock = threading.Lock()


def get_embedder() -> SiliconFlowEmbedder:
    """Return the process-wide embedder, building it from settings once."""
    global _embedder

    with _embedder_lock:
        if _embedder is None:
            settings = get_settings()
            if not settings.embedding_api_key:
                raise EmbeddingError(
                    "embedding_api_key is not configured; set EMBEDDING_API_KEY "
                    "in .env to call SiliconFlow embeddings"
                )
            _embedder = SiliconFlowEmbedder(
                base_url=settings.embedding_base_url,
                api_key=settings.embedding_api_key,
                model=settings.embedding_model,
                dim=settings.embedding_dim,
                batch_size=settings.embedding_batch_size,
            )
        return _embedder
