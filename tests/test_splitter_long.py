from app.rag.splitter import split_document


def _doc(tmp_path, text, content_type="policy"):
    p = tmp_path / "d.md"
    p.write_text(f"---\ncontent_type: {content_type}\n---\n{text}", encoding="utf-8")
    from app.rag.docs import load_document
    return load_document(p)


def test_long_text_splits_on_sentence_boundaries(tmp_path):
    body = "".join(f"第{i}句话说完整。" for i in range(40))
    d = _doc(tmp_path, f"# 政策\n\n## 长节\n\n{body}\n")
    chunks = split_document(d, max_chars=120, overlap_chars=30)
    assert len(chunks) > 1
    for c in chunks:
        assert c.answer.endswith(("。", "!", "?", ";", "!"))
        assert len(c.answer) <= 120 + 30
    assert chunks[0].answer != chunks[1].answer


def test_table_repeats_header_in_every_chunk(tmp_path):
    rows = "\n".join(f"| 商品{i} | {i} 元 |" for i in range(40))
    table = "| 商品 | 价格 |\n|---|---|\n" + rows
    d = _doc(tmp_path, f"# 价目\n\n## 价格表\n\n{table}\n")
    chunks = split_document(d, max_chars=160, overlap_chars=0)
    assert len(chunks) > 1
    for c in chunks:
        assert c.answer.startswith("| 商品 | 价格 |")
        assert "|---|" in c.answer
    joined = "\n".join(c.answer for c in chunks)
    assert "商品39" in joined


def test_no_punctuation_still_splits(tmp_path):
    d = _doc(tmp_path, "# 政策\n\n## 长节\n\n" + "无标点" * 200)
    chunks = split_document(d, max_chars=100, overlap_chars=0)
    assert len(chunks) > 1 and all(c.answer for c in chunks)
