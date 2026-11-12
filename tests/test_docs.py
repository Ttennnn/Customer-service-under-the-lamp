from pathlib import Path

import pytest

from app.rag.docs import load_document, load_documents


def test_load_document_frontmatter(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("---\ncontent_type: policy\ncategory: 售后\n---\n# 标题\n\n正文。\n", encoding="utf-8")
    d = load_document(p)
    assert d.title == "标题"
    assert d.content_type == "policy" and d.category == "售后"
    assert "正文。" in d.body and "content_type" not in d.body


def test_invalid_content_type_raises(tmp_path):
    p = tmp_path / "a.md"; p.write_text("---\ncontent_type: blog\n---\n# T\n", encoding="utf-8")
    try:
        load_document(p)
    except ValueError as e:
        assert "content_type" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_missing_h1_raises(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("---\ncontent_type: policy\n---\n正文。\n", encoding="utf-8")
    with pytest.raises(ValueError, match="H1"):
        load_document(p)


def test_load_documents_sorted_by_filename(tmp_path):
    (tmp_path / "b.md").write_text("---\ncontent_type: policy\n---\n# 乙\n", encoding="utf-8")
    (tmp_path / "a.md").write_text("---\ncontent_type: faq\n---\n# 甲\n", encoding="utf-8")
    docs = load_documents(tmp_path)
    assert [d.path.name for d in docs] == ["a.md", "b.md"]
    assert [d.title for d in docs] == ["甲", "乙"]
    assert docs[1].category is None


def test_sample_docs_cover_shipping_fee():
    docs = load_documents(Path("knowledge"))
    assert len(docs) >= 4
    text = "\n".join(d.body for d in docs)
    for needle in ["包邮", "99", "8 元", "邮费", "运费", "7 天"]:
        assert needle in text
