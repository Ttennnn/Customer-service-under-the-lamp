from app.rag.milvus_store import KnowledgeVectorStore, VectorItem


class FakeClient:
    def __init__(self):
        self.created = None
        self.upserted = None
        self.deleted = None
        self.search_result = [
            [{"id": "chunk_1", "distance": 0.8, "entity": {"chunk_id": 1}}]
        ]

    def has_collection(self, name):
        return False

    def create_schema(self, **kw):
        class S:
            def add_field(self, *a, **k):
                pass

        return S()

    def prepare_index_params(self):
        class I:
            def add_index(self, **k):
                pass

        return I()

    def create_collection(self, name, schema=None, index_params=None, **kw):
        self.created = (name, kw)

    def upsert(self, name, data):
        self.upserted = (name, data)

    def delete(self, name, ids=None):
        self.deleted = (name, ids)

    def search(self, **kw):
        return self.search_result

    def get_collection_stats(self, name):
        return {"row_count": 7}


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
    }
    assert s.search([0.0] * 1024, 3)[0].chunk_id == 1
    s.delete_ids(["chunk_1"])
    assert c.deleted[1] == ["chunk_1"]
    assert s.count() == 7


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
    assert store.collection == "knowledge"
    assert store.dim == 1024
