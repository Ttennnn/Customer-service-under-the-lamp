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
