from app.rag.docs import load_document
from app.rag.splitter import split_document


def test_manual_frontmatter_category_wins(tmp_path):
    p = tmp_path / "m.md"
    p.write_text(
        "---\ncontent_type: manual\ncategory: 商品\n---\n# 商品型号\n\n## LX-A100 落地灯\n\n色温 3000K\n",
        encoding="utf-8",
    )
    chunk = split_document(load_document(p))[0]
    assert chunk.category == "商品"
    assert chunk.section_path == "商品型号 > LX-A100 落地灯"


def test_manual_without_frontmatter_category_keeps_heading_path(tmp_path):
    p = tmp_path / "m.md"
    p.write_text(
        "---\ncontent_type: manual\n---\n# 商品型号\n\n## LX-A100 落地灯\n\n色温 3000K\n",
        encoding="utf-8",
    )
    chunk = split_document(load_document(p))[0]
    assert chunk.category == "商品型号"
    assert chunk.section_path == "商品型号 > LX-A100 落地灯"
