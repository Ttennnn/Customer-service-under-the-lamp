from app.config import Settings, build_llm


def test_settings_defaults(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    s = Settings()
    assert s.openai_base_url == "https://api.deepseek.com"
    assert s.model_name == "deepseek-chat"
    assert s.max_context_tokens == 4096


def test_build_llm_uses_settings():
    s = Settings(openai_api_key="sk-test", openai_base_url="http://localhost:11434/v1", model_name="qwen")
    llm = build_llm(s)
    assert llm.model_name == "qwen"
    assert llm.openai_api_base == "http://localhost:11434/v1"


def test_ch03_settings_defaults():
    s = Settings(_env_file=None, openai_api_key="x")
    assert s.embedding_model == "BAAI/bge-m3"
    assert s.embedding_dim == 1024
    assert s.embedding_batch_size == 16
    assert s.milvus_uri == "http://localhost:19530"
    assert s.milvus_collection == "knowledge"
    assert s.faq_top_k == 3
    assert s.faq_score_threshold == 0.35
    assert s.chunk_max_chars == 500
    assert s.chunk_overlap_chars == 80
    assert s.qa_dedup_threshold == 0.92
