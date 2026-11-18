from typing import Literal

from langchain_openai import ChatOpenAI
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


def build_llm(
    settings: Settings,
    temperature: float = 0.7,
    model: str | None = None,
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or settings.model_name,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        streaming=True,
        temperature=temperature,
    )


def get_settings() -> Settings:
    return Settings()
