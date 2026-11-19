from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_base_url: str = "https://api.deepseek.com"
    openai_api_key: str
    model_name: str = "deepseek-chat"
    host: str = "0.0.0.0"
    port: int = 8000
    max_context_tokens: int = 4096
    database_url: str = "mysql+pymysql://root:root@localhost:3306/cs_customer?charset=utf8mb4"

    embedding_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_api_key: str = ""
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedding_batch_size: int = 16
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "knowledge"
    milvus_hybrid_collection: str = "knowledge_hybrid"
    hybrid_recall_top_k: int = 50
    hybrid_rrf_k: int = 60
    rerank_top_n: int = 10
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_base_url: str = ""
    rerank_api_key: str = ""
    rerank_timeout_s: float = 10.0
    query_rewrite_enabled: bool = True
    faq_top_k: int = 3
    faq_score_threshold: float = 0.5
    knowledge_docs_dir: str = "knowledge"
    chunk_max_chars: int = 500
    chunk_overlap_chars: int = 80
    qa_mining_batch_size: int = 10
    qa_dedup_threshold: float = 0.92

    agent_max_steps: int = 6
    agent_token_budget: int = 12000
    agent_tool_result_max_chars: int = 2000
    confidence_min_score: float = 0.1
    langgraph_checkpoint_db: str = ".data/ch05-checkpoints.sqlite"

    intent_model: str = ""
    intent_small_model: str = ""
    intent_mode: Literal["accuracy", "cost"] = "accuracy"
    intent_conf_threshold: float = 0.7
    ch06_checkpoint_db: str = ".data/ch06-checkpoints.sqlite"

    model_context_window: int = 65536
    max_output_tokens: int = 2000
    max_user_input_tokens: int = 2000
    max_agent_steps: int | None = None
    tool_result_max_tokens: int | None = None
    rerank_top_k: int | None = None
    safety_margin_ratio: float = 0.10
    system_prompt_reserve_tokens: int = 1200
    evidence_tokens_per_chunk: int = 250
    summary_reserve_tokens: int = 500
    history_target_turns: int = 30
    history_tokens_per_turn: int = 300
    layer1_ratio: float = Field(default=0.7, ge=0, le=1)
    layer1_degrade_low_water_ratio: float = 0.8
    assistant_l2_chars: int = 40
    tool_result_marker_chars: int = 200
    summary_model: str = ""
    summary_max_chars: int = 200
    ch07_checkpoint_db: str = ".data/ch07-checkpoints.sqlite"
    context_log_preview_chars: int = 40
    log_to_file: bool = True

    # --- ch08 tool system ---
    mcp_enabled: bool = False
    mcp_logistics_url: str = "http://127.0.0.1:8101/mcp"
    mcp_aftersales_url: str = "http://127.0.0.1:8102/mcp"
    mcp_discovery_timeout_s: float = 2
    mcp_tool_timeout_s: float = 10
    tool_default_timeout_s: float = 5
    tool_max_retries: int = 2
    tool_audit_result_max_chars: int = 500
    demo_ticket_delay_seconds: float = 0

    @property
    def effective_max_agent_steps(self) -> int:
        return self.max_agent_steps if self.max_agent_steps is not None else self.agent_max_steps

    @property
    def effective_tool_result_max_tokens(self) -> int:
        return (
            self.tool_result_max_tokens
            if self.tool_result_max_tokens is not None
            else self.agent_tool_result_max_chars
        )

    @property
    def effective_rerank_top_k(self) -> int:
        return self.rerank_top_k if self.rerank_top_k is not None else self.rerank_top_n

    @field_validator("max_agent_steps", "tool_result_max_tokens", "rerank_top_k", mode="before")
    @classmethod
    def _blank_optional_int_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


def build_llm(
    settings: Settings,
    temperature: float = 0.7,
    model: str | None = None,
    max_tokens: int | None = None,
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or settings.model_name,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        streaming=True,
        temperature=temperature,
        max_tokens=max_tokens or settings.max_output_tokens,
    )


def get_settings() -> Settings:
    return Settings()
