from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration from environment variables."""

    openai_api_key: str
    app_api_key: str = "change-me"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "knowledge_base"
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 3
    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MB

    class Config:
        env_file = ".env"
        case_sensitive = False


_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the application settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
