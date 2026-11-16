# tests/test_knowledge_models.py
from app.models import KnowledgeChunk, QaExtractionStaging


def _cols(model):
    return {c.name: c for c in model.__table__.columns}


def test_knowledge_chunk_columns_and_enums():
    c = _cols(KnowledgeChunk)
    assert list(c) == [
        "id", "category", "questions", "answer", "section_path", "content_type",
        "is_key_clause", "prev_chunk_id", "next_chunk_id", "vector_id",
        "vectorize_status", "created_at", "updated_at",
    ]
    assert c["section_path"].type.length == 512
    assert c["vector_id"].type.length == 64
    assert c["vectorize_status"].type.enums == ["pending", "done"]
    assert c["is_key_clause"].nullable is False
    assert c["section_path"].nullable is True
    assert any(str(fk.target_fullname) == "knowledge_chunks.id"
               for fk in KnowledgeChunk.__table__.foreign_keys)


def test_staging_columns_and_enums():
    c = _cols(QaExtractionStaging)
    assert list(c) == ["id", "batch_no", "source_ref", "question", "answer", "status", "created_at"]
    assert c["status"].type.enums == ["extracted", "kept", "discarded"]
    assert c["batch_no"].index is True
    assert c["status"].index is True
