from app.models import KnowledgeChunk
from app.rag.milvus_store import SearchHit
from app.rag.retrieval import retrieve as _retrieve
from app.tools.faq import query_faq


def _force_legacy_dense(monkeypatch):
    """Pin the ch03 dense contract until Task 6 rewrites query_faq.

    ``retrieve()`` now defaults to hybrid_rerank, which needs a hybrid store
    and a real query rewrite/reranker.  These tests only cover query_faq's
    formatting, so keep them on the explicit-threshold dense path.
    """
    monkeypatch.setattr(
        "app.tools.faq.retrieve",
        lambda keyword, **kwargs: _retrieve(keyword, threshold=0.5, **kwargs),
    )


def test_query_faq_returns_vector_hit_text(session, monkeypatch):
    _force_legacy_dense(monkeypatch)
    c = KnowledgeChunk(
        category="物流",
        questions="邮费怎么算?",
        answer="满 99 元包邮,未满收 8 元。",
        content_type="faq",
    )
    session.add(c)
    session.commit()
    session.refresh(c)
    monkeypatch.setattr(
        "app.tools.faq.get_embedder",
        lambda: type("E", (), {"embed_one": lambda self, t: [0.1] * 1024})(),
    )
    monkeypatch.setattr(
        "app.tools.faq.get_vector_store",
        lambda: type(
            "S",
            (),
            {
                "dense_search": lambda self, v, k, expr=None: [
                    SearchHit(c.id, 0.7)
                ]
            },
        )(),
    )
    out = query_faq.invoke({"keyword": "邮费是多少"})
    assert "问:邮费怎么算?" in out and "答:满 99 元包邮" in out


def test_query_faq_miss_text(session, monkeypatch):
    _force_legacy_dense(monkeypatch)
    monkeypatch.setattr(
        "app.tools.faq.get_embedder",
        lambda: type("E", (), {"embed_one": lambda self, t: [0.1] * 1024})(),
    )
    monkeypatch.setattr(
        "app.tools.faq.get_vector_store",
        lambda: type(
            "S", (), {"dense_search": lambda self, v, k, expr=None: []}
        )(),
    )
    assert query_faq.invoke({"keyword": "不存在"}) == "未找到相关问题,请换个关键词再试试。"


def test_query_faq_infrastructure_error_is_safe(session, monkeypatch):
    monkeypatch.setattr(
        "app.tools.faq.get_embedder",
        lambda: (_ for _ in ()).throw(RuntimeError("down")),
    )
    out = query_faq.invoke({"keyword": "邮费"})
    assert "知识库暂时不可用" in out
