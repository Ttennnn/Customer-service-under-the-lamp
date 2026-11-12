from app.rag.docs import load_document
from app.rag.splitter import split_document


def _doc(tmp_path, text, content_type="policy", category=None):
    fm = f"---\ncontent_type: {content_type}\n" + (f"category: {category}\n" if category else "") + "---\n"
    p = tmp_path / "d.md"; p.write_text(fm + text, encoding="utf-8")
    return load_document(p)


def test_policy_heading_paths(tmp_path):
    d = _doc(tmp_path, "# 退货政策\n\n## 退货时限\n\n签收后 7 天内。\n\n### 七天无理由\n\n商品完好。\n")
    chunks = split_document(d)
    assert [(c.questions, c.category, c.section_path) for c in chunks] == [
        ("退货时限", "退货政策", "退货政策 > 退货时限"),
        ("七天无理由", "退货政策 > 退货时限", "退货政策 > 退货时限 > 七天无理由"),
    ]
    assert [c.answer for c in chunks] == ["签收后 7 天内。", "商品完好。"]
    assert all(c.content_type == "policy" for c in chunks)


def test_faq_uses_frontmatter_category(tmp_path):
    d = _doc(tmp_path, "# 商品FAQ\n\n## 尺码怎么选?\n\n按脚长选。\n", "faq", "商品")
    (c,) = split_document(d)
    assert c.questions == "尺码怎么选?"
    assert c.category == "商品"
    assert c.section_path == "商品FAQ > 尺码怎么选?"
    assert c.answer == "按脚长选。"
    assert c.content_type == "faq"


def test_code_fence_heading_ignored(tmp_path):
    d = _doc(tmp_path, "# 手册\n\n## 示例\n\n```\n# 不是标题\n```\n\n正文。\n")
    chunks = split_document(d)
    assert len(chunks) == 1
    assert chunks[0].answer == "```\n# 不是标题\n```\n\n正文。"
    assert chunks[0].content_type == "policy"


def test_empty_section_skipped_and_key_clause(tmp_path):
    d = _doc(tmp_path, "# 政策\n\n## 空节\n\n## 关键\n\n【关键条款】必须在 7 天内申请。\n")
    (c,) = split_document(d)
    assert c.questions == "关键" and c.is_key_clause is True
    assert c.answer == "【关键条款】必须在 7 天内申请。"
    assert c.content_type == "policy"


def test_parent_intro_emitted(tmp_path):
    d = _doc(tmp_path, "# 政策\n\n总则介绍。\n\n## 退货时限\n\n签收后 7 天内。\n")
    chunks = split_document(d)
    assert [(c.questions, c.category, c.section_path, c.answer) for c in chunks] == [
        ("政策", "政策", "政策", "总则介绍。"),
        ("退货时限", "政策", "政策 > 退货时限", "签收后 7 天内。"),
    ]
    assert all(c.content_type == "policy" for c in chunks)


def test_policy_regex_key_clause(tmp_path):
    d = _doc(tmp_path, "# 政策\n\n## 条款\n\n商品不得转售。\n")
    (c,) = split_document(d)
    assert c.is_key_clause is True
    assert c.answer == "商品不得转售。"


def test_faq_regex_not_key_clause(tmp_path):
    d = _doc(tmp_path, "# FAQ\n\n## 可以退货吗?\n\n商品不得转售。\n", "faq", "售后")
    (c,) = split_document(d)
    assert c.is_key_clause is False
    assert c.content_type == "faq"
