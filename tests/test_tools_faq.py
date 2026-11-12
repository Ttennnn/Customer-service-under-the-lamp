from app.models import KnowledgeChunk
from app.rag.milvus_store import SearchHit
from app.tools.faq import query_faq


def test_query_faq_returns_vector_hit_text(session, monkeypatch):
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
        lambda: type("S", (), {"search": lambda self, v, k: [SearchHit(c.id, 0.7)]})(),
    )
    out = query_faq.invoke({"keyword": "邮费是多少"})
    assert "问:邮费怎么算?" in out and "答:满 99 元包邮" in out


def test_query_faq_miss_text(session, monkeypatch):
    monkeypatch.setattr(
        "app.tools.faq.get_embedder",
        lambda: type("E", (), {"embed_one": lambda self, t: [0.1] * 1024})(),
    )
    monkeypatch.setattr(
        "app.tools.faq.get_vector_store",
        lambda: type("S", (), {"search": lambda self, v, k: []})(),
    )
    assert query_faq.invoke({"keyword": "不存在"}) == "未找到相关问题,请换个关键词再试试。"


def test_query_faq_infrastructure_error_is_safe(session, monkeypatch):
    monkeypatch.setattr(
        "app.tools.faq.get_embedder",
        lambda: (_ for _ in ()).throw(RuntimeError("down")),
    )
    out = query_faq.invoke({"keyword": "邮费"})
    assert "知识库暂时不可用" in out
