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
