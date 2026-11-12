"""Thin Milvus wrapper for the ``knowledge`` collection.

The collection stores one vector per knowledge chunk::

    id        VARCHAR(64)  primary key (``chunk_{chunk_id}``)
    chunk_id  INT64        MySQL ``knowledge_chunks.id``
    vector    FLOAT_VECTOR(1024)  BGE-M3 embedding

The index is ``FLAT`` with ``COSINE`` metric and the collection uses strong
consistency, so a write is immediately visible to the next search.  The
wrapper is deliberately small: it does not embed text, own sessions, or
interpret chunk content.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from pymilvus import DataType, MilvusClient

from app.config import get_settings


@dataclass(frozen=True)
class VectorItem:
    vector_id: str
    chunk_id: int
    vector: list[float]


@dataclass(frozen=True)
class SearchHit:
    chunk_id: int
    score: float


class KnowledgeVectorStore:
    def __init__(self, client, collection: str, dim: int) -> None:
        self.client = client
        self.collection = collection
        self.dim = dim
        self._ensured = False

    def ensure_collection(self) -> None:
        """Create the collection if needed and ensure it is loaded.

        An existing collection is still loaded on the first call: Milvus
        unloads collections on restart/release, and every search/upsert fails
        until they are loaded again.  Loading is idempotent, and the result is
        cached so steady-state queries do not pay an extra RPC.
        """
        if self._ensured:
            return

        if not self.client.has_collection(self.collection):
            schema = self.client.create_schema()
            schema.add_field(
                "id", DataType.VARCHAR, is_primary=True, max_length=64
            )
            schema.add_field("chunk_id", DataType.INT64)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.dim)

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="vector", index_type="FLAT", metric_type="COSINE"
            )

            self.client.create_collection(
                self.collection,
                schema=schema,
                index_params=index_params,
                consistency_level="Strong",
            )

        if hasattr(self.client, "load_collection"):
            self.client.load_collection(self.collection)
        self._ensured = True

    def upsert(self, items: list[VectorItem]) -> None:
        self.ensure_collection()
        self.client.upsert(
            self.collection,
            [
                {
                    "id": item.vector_id,
                    "chunk_id": item.chunk_id,
                    "vector": item.vector,
                }
                for item in items
            ],
        )

    def search(self, vector: list[float], top_k: int) -> list[SearchHit]:
        self.ensure_collection()
        results = self.client.search(
            collection_name=self.collection,
            data=[vector],
            anns_field="vector",
            search_params={"metric_type": "COSINE"},
            limit=top_k,
            output_fields=["chunk_id"],
        )
        if not results:
            return []

        hits: list[SearchHit] = []
        for hit in results[0]:
            entity = hit.get("entity") or {}
            chunk_id = entity.get("chunk_id")
            if chunk_id is None:
                chunk_id = hit.get("chunk_id")
            if chunk_id is None:
                continue
            hits.append(SearchHit(chunk_id, float(hit["distance"])))
        return hits

    def delete_ids(self, ids: list[str]) -> None:
        self.ensure_collection()
        if not ids:
            return
        self.client.delete(self.collection, ids)

    def count(self) -> int:
        self.ensure_collection()
        stats = self.client.get_collection_stats(self.collection)
        return int(stats.get("row_count", 0))


_store: KnowledgeVectorStore | None = None
_store_lock = threading.Lock()


def get_vector_store() -> KnowledgeVectorStore:
    """Return the process-wide Milvus store, building it from settings once."""
    global _store

    with _store_lock:
        if _store is None:
            settings = get_settings()
            _store = KnowledgeVectorStore(
                MilvusClient(uri=settings.milvus_uri),
                collection=settings.milvus_collection,
                dim=settings.embedding_dim,
            )
        return _store
