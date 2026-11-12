from types import SimpleNamespace

from sqlalchemy.orm import sessionmaker

from app.models import KnowledgeChunk
from app.rag.milvus_store import SearchHit
from app.rag.retrieval import RetrievedChunk, retrieve


class FakeEmbedder:
    def embed_one(self, text):
        return [0.1] * 1024


class FakeStore:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, vector, top_k):
        self.calls.append((vector, top_k))
        return self.hits


def _row(session, q, a):
    c = KnowledgeChunk(category="售后", questions=q, answer=a, content_type="faq")
    session.add(c)
    session.commit()
    session.refresh(c)
    return c


class _SpySession:
    def __init__(self, inner):
        self._inner = inner
        self.closed = False

    def query(self, *args, **kwargs):
        return self._inner.query(*args, **kwargs)

    def close(self):
        self.closed = True
        self._inner.close()


def test_retrieve_orders_by_score_and_filters(session):
    a = _row(session, "问A", "答A")
    b = _row(session, "问B", "答B")
    hits = [SearchHit(b.id, 0.9), SearchHit(a.id, 0.2)]
    out = retrieve(
        "邮费",
        session=session,
        embedder=FakeEmbedder(),
        store=FakeStore(hits),
        threshold=0.5,
    )
    assert [(r.chunk_id, r.answer) for r in out] == [(b.id, "答B")]
    assert out[0] == RetrievedChunk(b.id, 0.9, "问B", "答B")


def test_retrieve_empty(session):
    assert (
        retrieve(
            "x", session=session, embedder=FakeEmbedder(), store=FakeStore([])
        )
        == []
    )


def test_retrieve_skips_ids_missing_from_mysql(session):
    a = _row(session, "问A", "答A")
    hits = [SearchHit(a.id, 0.9), SearchHit(99999, 0.8)]
    out = retrieve(
        "x",
        session=session,
        embedder=FakeEmbedder(),
        store=FakeStore(hits),
        threshold=0.1,
    )
    assert [(r.chunk_id, r.answer) for r in out] == [(a.id, "答A")]


def test_retrieve_uses_settings_defaults(session):
    row = _row(session, "问A", "答A")
    store = FakeStore([SearchHit(row.id, 0.5), SearchHit(row.id, 0.34)])
    retrieve(
        "x",
        session=session,
        embedder=FakeEmbedder(),
        store=store,
    )
    assert store.calls[0][1] == 3


def test_retrieve_opens_and_closes_its_own_session(session, monkeypatch):
    row = _row(session, "问A", "答A")
    factory = sessionmaker(
        bind=session.get_bind(), autoflush=False, expire_on_commit=False
    )
    spies = []

    def make():
        spy = _SpySession(factory())
        spies.append(spy)
        return spy

    monkeypatch.setattr(
        "app.rag.retrieval.db_session.get_session_factory", lambda: make
    )
    out = retrieve(
        "x",
        embedder=FakeEmbedder(),
        store=FakeStore([SearchHit(row.id, 0.9)]),
        threshold=0.1,
    )
    assert [r.answer for r in out] == ["答A"]
    assert len(spies) == 1 and spies[0].closed is True


def test_retrieve_does_not_close_injected_session(session):
    row = _row(session, "问A", "答A")
    spy = _SpySession(session)
    out = retrieve(
        "x",
        session=spy,
        embedder=FakeEmbedder(),
        store=FakeStore([SearchHit(row.id, 0.9)]),
        threshold=0.1,
    )
    assert [r.answer for r in out] == ["答A"]
    assert spy.closed is False


def test_retrieve_resolves_embedder_and_store_from_module(monkeypatch, session):
    row = _row(session, "问A", "答A")
    calls = []

    def fake_embedder():
        calls.append("embedder")
        return FakeEmbedder()

    def fake_store():
        calls.append("store")
        return FakeStore([SearchHit(row.id, 0.9)])

    monkeypatch.setattr("app.rag.retrieval.get_embedder", fake_embedder)
    monkeypatch.setattr("app.rag.retrieval.get_vector_store", fake_store)
    monkeypatch.setattr(
        "app.rag.retrieval.get_settings",
        lambda: SimpleNamespace(faq_top_k=3, faq_score_threshold=0.35),
    )
    out = retrieve("x", session=session)
    assert [r.answer for r in out] == ["答A"]
    assert calls == ["embedder", "store"]
