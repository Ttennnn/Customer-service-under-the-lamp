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
    faq_top_k: int = 3
    faq_score_threshold: float = 0.35
    knowledge_docs_dir: str = "knowledge"
    chunk_max_chars: int = 500
    chunk_overlap_chars: int = 80
    qa_mining_batch_size: int = 10
    qa_dedup_threshold: float = 0.92


def build_llm(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.model_name,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        streaming=True,
        temperature=0.7,
    )


def get_settings() -> Settings:
    return Settings()
