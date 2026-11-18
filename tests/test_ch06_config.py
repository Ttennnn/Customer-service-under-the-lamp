from app.config import Settings, build_llm


def test_ch06_settings_defaults():
    s = Settings(_env_file=None, openai_api_key="x")
    assert s.intent_model == ""
    assert s.intent_small_model == ""
    assert s.intent_mode == "accuracy"
    assert s.intent_conf_threshold == 0.7
    assert s.ch06_checkpoint_db == ".data/ch06-checkpoints.sqlite"


def test_build_llm_model_override():
    s = Settings(_env_file=None, openai_api_key="x", model_name="base")
    llm = build_llm(s, model="intent-model")
    assert llm.model_name == "intent-model"
