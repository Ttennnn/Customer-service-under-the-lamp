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
    assert s.faq_score_threshold == 0.5
    assert s.chunk_max_chars == 500
    assert s.chunk_overlap_chars == 80
    assert s.qa_dedup_threshold == 0.92


def test_ch04_defaults_exist():
    s = Settings(openai_api_key="x")
    assert s.milvus_hybrid_collection == "knowledge_hybrid"
    assert s.hybrid_recall_top_k == 50 and s.hybrid_rrf_k == 60
    assert s.rerank_top_n == 10
    assert s.rerank_model == "BAAI/bge-reranker-v2-m3"
    assert s.rerank_base_url == ""
    assert s.rerank_api_key == ""
    assert s.rerank_timeout_s == 10.0
    assert s.query_rewrite_enabled is True


def test_build_llm_accepts_temperature():
    s = Settings(_env_file=None, openai_api_key="sk-test")
    assert build_llm(s).temperature == 0.7
    assert build_llm(s, temperature=0.0).temperature == 0.0
