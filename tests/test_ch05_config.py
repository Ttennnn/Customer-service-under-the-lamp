from app.config import Settings


def test_ch05_settings_defaults():
    s = Settings(_env_file=None, openai_api_key="x")
    assert s.agent_max_steps == 6
    assert s.agent_token_budget == 12000
    assert s.agent_tool_result_max_chars == 2000
    assert s.confidence_min_score == 0.1
    assert s.langgraph_checkpoint_db == ".data/ch05-checkpoints.sqlite"


def test_ch05_settings_env_override(monkeypatch):
    monkeypatch.setenv("AGENT_MAX_STEPS", "3")
    monkeypatch.setenv("CONFIDENCE_MIN_SCORE", "0.55")
    s = Settings(_env_file=None, openai_api_key="x")
    assert s.agent_max_steps == 3
    assert s.confidence_min_score == 0.55
