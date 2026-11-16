from pymilvus import DataType

from app.rag.milvus_store import KnowledgeVectorStore, VectorItem


class FakeClient:
    def __init__(self):
        self.created = None
        self.upserted = None
        self.deleted = None
        self.search_result = [
            [{"id": "chunk_1", "distance": 0.8, "entity": {"chunk_id": 1}}]
        ]
        self.hybrid_result = [
            [{"id": "chunk_1", "distance": 0.9, "entity": {"chunk_id": 1}}]
        ]
        self.collection_exists = False
        self.has_collection_calls = 0
        self.fields = []
        self.functions = []
        self.indexes = []
        self.create_calls = []
        self.loaded = []
        self.search_calls = []
        self.hybrid_calls = []
        self.dropped = []

    def has_collection(self, name):
        self.has_collection_calls += 1
        return self.collection_exists

    def create_schema(self, **kw):
        fields = self.fields
        functions = self.functions

        class S:
            def add_field(self, name, datatype, **kwargs):
                fields.append({"name": name, "datatype": datatype, **kwargs})

            def add_function(self, function):
                functions.append(function)

        return S()

    def prepare_index_params(self):
        indexes = self.indexes

        class I:
            def add_index(self, **kwargs):
                indexes.append(kwargs)

        return I()

    def create_collection(self, name, schema=None, index_params=None, **kw):
        self.created = (name, kw)
        self.create_calls.append(
            {
                "collection_name": name,
                "schema": schema,
                "index_params": index_params,
                **kw,
            }
        )
        self.collection_exists = True

    def load_collection(self, name):
        self.loaded.append(name)

    def upsert(self, name, data):
        self.upserted = (name, data)

    def delete(self, name, ids=None):
        self.deleted = (name, ids)

    def search(self, **kw):
        self.search_calls.append(kw)
        return self.search_result

    def hybrid_search(
        self, collection_name, reqs, ranker=None, limit=10, output_fields=None, **kw
    ):
        self.hybrid_calls.append(
            {
                "collection_name": collection_name,
                "reqs": reqs,
                "ranker": ranker,
                "limit": limit,
                "output_fields": output_fields,
                **kw,
            }
        )
        return self.hybrid_result

    def drop_collection(self, name):
        self.dropped.append(name)
        self.collection_exists = False

    def get_collection_stats(self, name):
        return {"row_count": 7}


def test_ensure_creates_schema_index_and_loads():
    c = FakeClient()
    s = KnowledgeVectorStore(c, "knowledge", 1024)
    s.ensure_collection()

    assert c.created[0] == "knowledge"
    assert c.create_calls[0]["consistency_level"] == "Strong"
    assert c.fields == [
        {
            "name": "id",
            "datatype": DataType.VARCHAR,
            "is_primary": True,
            "max_length": 64,
        },
        {"name": "chunk_id", "datatype": DataType.INT64},
        {"name": "vector", "datatype": DataType.FLOAT_VECTOR, "dim": 1024},
        {
            "name": "text",
            "datatype": DataType.VARCHAR,
            "max_length": 8192,
            "enable_analyzer": True,
            "analyzer_params": {"type": "chinese"},
        },
        {"name": "sparse", "datatype": DataType.SPARSE_FLOAT_VECTOR},
        {"name": "category", "datatype": DataType.VARCHAR, "max_length": 255},
        {"name": "content_type", "datatype": DataType.VARCHAR, "max_length": 32},
        {"name": "section_path", "datatype": DataType.VARCHAR, "max_length": 512},
    ]
    assert c.indexes == [
        {
            "field_name": "vector",
            "index_type": "FLAT",
            "metric_type": "COSINE",
        },
        {
            "field_name": "sparse",
            "index_name": "sparse_inverted",
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "BM25",
            "params": {"bm25_k1": 1.2, "bm25_b": 0.75},
        },
        {"field_name": "category", "index_type": "INVERTED"},
        {"field_name": "content_type", "index_type": "INVERTED"},
    ]
    assert c.functions and c.functions[0].name == "bm25_fn"
    assert c.loaded == ["knowledge"]


def test_ensure_loads_existing_collection_and_caches_state():
    c = FakeClient()
    c.collection_exists = True
    s = KnowledgeVectorStore(c, "knowledge", 1024)

    s.ensure_collection()
    assert c.create_calls == []
    assert c.loaded == ["knowledge"]

    # A second call must not issue any RPC: the ensured state is cached.
    c.has_collection_calls = 0
    c.loaded.clear()
    s.ensure_collection()
    assert c.has_collection_calls == 0
    assert c.loaded == []


def test_ensure_collection_is_idempotent_when_present():
    c = FakeClient()
    s = KnowledgeVectorStore(c, "knowledge", 1024)
    s.ensure_collection()

    c.has_collection_calls = 0
    c.create_calls.clear()
    c.loaded.clear()
    s.ensure_collection()

    assert c.has_collection_calls == 0
    assert c.create_calls == []
    assert c.loaded == []


def test_ensure_upsert_search_delete():
    c = FakeClient()
    s = KnowledgeVectorStore(c, "knowledge", 1024)
    s.ensure_collection()
    assert c.created[0] == "knowledge"
    s.upsert([VectorItem("chunk_1", 1, [0.0] * 1024)])
    assert c.upserted[1][0] == {
        "id": "chunk_1",
        "chunk_id": 1,
        "vector": [0.0] * 1024,
        "text": "",
        "category": "",
        "content_type": "",
        "section_path": "",
    }
    assert s.search([0.0] * 1024, 3)[0].chunk_id == 1
    s.delete_ids(["chunk_1"])
    assert c.deleted[1] == ["chunk_1"]
    assert s.count() == 7


def test_search_passes_collection_cosine_params_and_output_fields():
    c = FakeClient()
    s = KnowledgeVectorStore(c, "knowledge", 1024)
    s.search([0.25] * 1024, 3)

    assert c.search_calls == [
        {
            "collection_name": "knowledge",
            "data": [[0.25] * 1024],
            "anns_field": "vector",
            "search_params": {"metric_type": "COSINE"},
            "limit": 3,
            "output_fields": ["chunk_id"],
        }
    ]


def test_search_accepts_top_level_chunk_id():
    c = FakeClient()
    c.search_result = [[{"id": "chunk_2", "distance": 0.5, "chunk_id": 2}]]
    assert KnowledgeVectorStore(c, "knowledge", 1024).search([0.0] * 1024, 3)[
        0
    ].chunk_id == 2


def test_search_skips_hits_without_chunk_id():
    c = FakeClient()
    c.search_result = [
        [
            {"id": "chunk_1", "distance": 0.8, "entity": {"chunk_id": 1}},
            {"id": "chunk_bad", "distance": 0.4, "entity": {}},
            {"id": "chunk_no_entity", "distance": 0.2},
        ]
    ]
    hits = KnowledgeVectorStore(c, "knowledge", 1024).search([0.0] * 1024, 3)
    assert [hit.chunk_id for hit in hits] == [1]
    assert hits[0].score == 0.8


def test_every_public_method_ensures_collection():
    calls = [
        lambda s: s.upsert([VectorItem("chunk_1", 1, [0.0] * 1024)]),
        lambda s: s.search([0.0] * 1024, 3),
        lambda s: s.delete_ids(["chunk_1"]),
        lambda s: s.count(),
    ]
    for call in calls:
        c = FakeClient()
        s = KnowledgeVectorStore(c, "knowledge", 1024)
        c.has_collection_calls = 0
        c.collection_exists = False
        call(s)
        assert c.has_collection_calls == 1
        assert len(c.create_calls) == 1


def test_delete_ids_empty_is_noop():
    c = FakeClient()
    KnowledgeVectorStore(c, "knowledge", 1024).delete_ids([])
    assert c.deleted is None


def test_get_vector_store_builds_and_caches(monkeypatch):
    import app.rag.milvus_store as store_module

    created = []

    class FakeSettings:
        milvus_uri = "http://milvus:19530"
        milvus_collection = "knowledge"
        milvus_hybrid_collection = "knowledge_hybrid"
        embedding_dim = 1024

    class FakeMilvusClient:
        def __init__(self, uri):
            created.append(uri)

    monkeypatch.setattr(store_module, "MilvusClient", FakeMilvusClient)
    monkeypatch.setattr(store_module, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(store_module, "_store", None)

    store = store_module.get_vector_store()
    assert store_module.get_vector_store() is store
    assert created == ["http://milvus:19530"]
    assert store.collection == "knowledge_hybrid"
    assert store.dim == 1024


def test_hybrid_schema_has_text_bm25_function_and_scalar_fields():
    c = FakeClient()
    KnowledgeVectorStore(c, "knowledge_hybrid", 1024).ensure_collection()
    names = {f["name"] for f in c.fields}
    assert {
        "id",
        "chunk_id",
        "vector",
        "text",
        "sparse",
        "category",
        "content_type",
        "section_path",
    } <= names
    text = next(f for f in c.fields if f["name"] == "text")
    assert text["enable_analyzer"] is True
    assert text["analyzer_params"] == {"type": "chinese"}
    assert c.functions and c.functions[0].name == "bm25_fn"
    assert any(
        i["field_name"] == "sparse" and i["metric_type"] == "BM25"
        for i in c.indexes
    )


def test_upsert_includes_bm25_text_and_metadata():
    c = FakeClient()
    s = KnowledgeVectorStore(c, "knowledge_hybrid", 1024)
    s.upsert(
        [
            VectorItem(
                "chunk_1",
                1,
                [0.0] * 1024,
                "商品\n型号\nLX-A100",
                "商品",
                "manual",
                "商品型号与参数 > LX-A100 落地灯",
            )
        ]
    )
    assert c.upserted[1][0]["text"] == "商品\n型号\nLX-A100"
    assert c.upserted[1][0]["category"] == "商品"


def test_dense_search_passes_vector_and_filter():
    c = FakeClient()
    s = KnowledgeVectorStore(c, "knowledge_hybrid", 1024)
    hits = s.dense_search([0.5] * 1024, 50, expr='content_type == "manual"')

    call = c.search_calls[-1]
    assert call["data"] == [[0.5] * 1024]
    assert call["anns_field"] == "vector"
    assert call["search_params"] == {"metric_type": "COSINE"}
    assert call["filter"] == 'content_type == "manual"'
    assert hits[0].chunk_id == 1


def test_bm25_search_passes_raw_text_and_filter():
    c = FakeClient()
    s = KnowledgeVectorStore(c, "knowledge_hybrid", 1024)
    s.bm25_search("LX-A100", 50, expr='category == "商品"')
    call = c.search_calls[-1]
    assert call["data"] == ["LX-A100"]
    assert call["anns_field"] == "sparse"
    assert call["search_params"] == {"metric_type": "BM25"}
    assert call["filter"] == 'category == "商品"'


def test_hybrid_search_builds_two_requests_and_rrf():
    c = FakeClient()
    s = KnowledgeVectorStore(c, "knowledge_hybrid", 1024)
    hits = s.hybrid_search([0.1] * 1024, "LX-A100", 50, rrf_k=60)
    reqs = c.hybrid_calls[-1]["reqs"]
    assert [r.anns_field for r in reqs] == ["vector", "sparse"]
    # pymilvus 3.0.3's RRFRanker stores k privately; use its public dict().
    assert c.hybrid_calls[-1]["ranker"].dict()["params"]["k"] == 60
    assert c.hybrid_calls[-1]["limit"] == 50
    assert c.hybrid_calls[-1]["output_fields"] == ["chunk_id"]
    assert hits[0].chunk_id == 1


def test_hybrid_search_passes_filter_to_both_requests():
    c = FakeClient()
    s = KnowledgeVectorStore(c, "knowledge_hybrid", 1024)
    s.hybrid_search(
        [0.1] * 1024, "LX-A100", 50, expr='category == "商品"'
    )
    reqs = c.hybrid_calls[-1]["reqs"]
    assert [r.expr for r in reqs] == ['category == "商品"', 'category == "商品"']


def test_recreate_collection_drops_only_hybrid_and_recreates():
    c = FakeClient()
    s = KnowledgeVectorStore(c, "knowledge_hybrid", 1024)
    s.ensure_collection()
    assert s._ensured is True

    s.recreate_collection()

    assert c.dropped == ["knowledge_hybrid"]
    assert s._ensured is True
    assert len(c.create_calls) == 2
