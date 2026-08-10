from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings, read from environment variables or an optional .env file."""

    app_name: str = "POC Kanini Enterprise AI Assistant"
    environment: str = "development"
    gemini_model: str = "gemini-2.5-flash"
    gemini_api_key: str | None = None
    gemini_temperature: float = 0.2
    gemini_max_output_tokens: int = 2048
    gemini_embedding_model: str = "gemini-embedding-001"
    rag_vector_store_dir: str = "data/chroma"
    rag_top_k: int = 5
    rag_enable_reranking: bool = False
    frontend_build_dir: str = "../frontend/dist"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Return process-wide, validated application settings."""

    return Settings()
