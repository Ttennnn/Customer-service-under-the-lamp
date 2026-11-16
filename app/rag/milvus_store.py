"""Thin Milvus wrapper for the hybrid ``knowledge_hybrid`` collection.

Each record mirrors one MySQL knowledge chunk::

    id            VARCHAR(64)  primary key (``chunk_{chunk_id}``)
    chunk_id      INT64        MySQL ``knowledge_chunks.id``
    vector        FLOAT_VECTOR(1024)  BGE-M3 dense embedding
    text          VARCHAR(8192)       analyzed text (BM25 input)
    sparse        SPARSE_FLOAT_VECTOR BM25 function output
    category      VARCHAR(255)
    content_type  VARCHAR(32)
    section_path  VARCHAR(512)

Dense search uses ``FLAT``/``COSINE``; BM25 uses the native Milvus ``text``
analyzer (Chinese) plus a ``SPARSE_INVERTED_INDEX`` and the ``BM25`` metric.
Hybrid search issues one dense and one BM25 request and fuses them with RRF.
The wrapper is deliberately small: it does not embed text, own sessions, or
interpret chunk content.  The legacy ``knowledge`` collection is untouched.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
)

from app.config import get_settings


@dataclass(frozen=True)
class VectorItem:
    vector_id: str
    chunk_id: int
    vector: list[float]
    text: str = ""
    category: str = ""
    content_type: str | None = None
    section_path: str | None = None


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
            schema.add_field(
                "text",
                DataType.VARCHAR,
                max_length=8192,
                enable_analyzer=True,
                analyzer_params={"type": "chinese"},
            )
            schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field("category", DataType.VARCHAR, max_length=255)
            schema.add_field("content_type", DataType.VARCHAR, max_length=32)
            schema.add_field("section_path", DataType.VARCHAR, max_length=512)
            schema.add_function(
                Function(
                    name="bm25_fn",
                    function_type=FunctionType.BM25,
                    input_field_names=["text"],
                    output_field_names=["sparse"],
                )
            )

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="vector", index_type="FLAT", metric_type="COSINE"
            )
            index_params.add_index(
                field_name="sparse",
                index_name="sparse_inverted",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
                params={"bm25_k1": 1.2, "bm25_b": 0.75},
            )
            index_params.add_index(field_name="category", index_type="INVERTED")
            index_params.add_index(
                field_name="content_type", index_type="INVERTED"
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

    def recreate_collection(self) -> None:
        """Drop and recreate only this (hybrid) collection, then reload.

        The legacy ``knowledge`` collection is never involved: this store is
        bound to a single collection name.  Any cached "ensured" state is
        reset so the next call rebuilds the schema/function/index.
        """
        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)
        self._ensured = False
        self.ensure_collection()

    def upsert(self, items: list[VectorItem]) -> None:
        self.ensure_collection()
        self.client.upsert(
            self.collection,
            [
                {
                    "id": item.vector_id,
                    "chunk_id": item.chunk_id,
                    "vector": item.vector,
                    "text": item.text,
                    "category": item.category,
                    # Milvus VARCHAR fields here are not nullable; empty string
                    # keeps 3-arg VectorItem constructors insertable while the
                    # authoritative value still lives in MySQL.
                    "content_type": item.content_type or "",
                    "section_path": item.section_path or "",
                }
                for item in items
            ],
        )

    def search(self, vector: list[float], top_k: int) -> list[SearchHit]:
        """Legacy dense alias kept for existing ch03 callers/tests."""
        self.ensure_collection()
        results = self.client.search(
            collection_name=self.collection,
            data=[vector],
            anns_field="vector",
            search_params={"metric_type": "COSINE"},
            limit=top_k,
            output_fields=["chunk_id"],
        )
        return self._parse_hits(results)

    def dense_search(
        self, vector: list[float], top_k: int, expr: str | None = None
    ) -> list[SearchHit]:
        self.ensure_collection()
        results = self.client.search(
            collection_name=self.collection,
            data=[vector],
            anns_field="vector",
            search_params={"metric_type": "COSINE"},
            limit=top_k,
            filter=expr or "",
            output_fields=["chunk_id"],
        )
        return self._parse_hits(results)

    def bm25_search(
        self, query_text: str, top_k: int, expr: str | None = None
    ) -> list[SearchHit]:
        self.ensure_collection()
        results = self.client.search(
            collection_name=self.collection,
            # BM25 search takes the raw query string; the analyzer runs server
            # side against the ``text`` field via the BM25 function.
            data=[query_text],
            anns_field="sparse",
            search_params={"metric_type": "BM25"},
            limit=top_k,
            filter=expr or "",
            output_fields=["chunk_id"],
        )
        return self._parse_hits(results)

    def hybrid_search(
        self,
        vector: list[float],
        query_text: str,
        top_k: int,
        expr: str | None = None,
        rrf_k: int = 60,
    ) -> list[SearchHit]:
        self.ensure_collection()
        reqs = [
            AnnSearchRequest(
                [vector],
                "vector",
                {"metric_type": "COSINE"},
                limit=top_k,
                expr=expr,
            ),
            AnnSearchRequest(
                [query_text],
                "sparse",
                {"metric_type": "BM25"},
                limit=top_k,
                expr=expr,
            ),
        ]
        results = self.client.hybrid_search(
            self.collection,
            reqs,
            ranker=RRFRanker(k=rrf_k),
            limit=top_k,
            output_fields=["chunk_id"],
        )
        return self._parse_hits(results)

    @staticmethod
    def _parse_hits(results) -> list[SearchHit]:
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
    """Return the process-wide hybrid Milvus store, built from settings once."""
    global _store

    with _store_lock:
        if _store is None:
            settings = get_settings()
            _store = KnowledgeVectorStore(
                MilvusClient(uri=settings.milvus_uri),
                collection=settings.milvus_hybrid_collection,
                dim=settings.embedding_dim,
            )
        return _store
