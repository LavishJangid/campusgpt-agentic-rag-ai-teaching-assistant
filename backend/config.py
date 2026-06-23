"""Backend configuration module using Pydantic Settings."""

import os
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field

BASE_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_DB = BASE_DIR / "data" / "campusgpt.db"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App
    app_name: str = Field(default="CampusGPT", env="APP_NAME")
    app_version: str = Field(default="2.0.0", env="APP_VERSION")
    app_env: str = Field(default="development", env="APP_ENV")
    debug: bool = Field(default=True, env="DEBUG")

    # API Keys
    gemini_api_key: str = Field(default="", env="GEMINI_API_KEY")

    # Backend
    backend_host: str = Field(default="0.0.0.0", env="BACKEND_HOST")
    backend_port: int = Field(default=8000, env="BACKEND_PORT")
    backend_url: str = Field(default="http://localhost:8000", env="BACKEND_URL")

    # Frontend
    frontend_port: int = Field(default=8501, env="FRONTEND_PORT")

    # ChromaDB
    chroma_persist_dir: str = Field(
        default=str(BASE_DIR / "chroma_db"), env="CHROMA_PERSIST_DIR"
    )
    chroma_collection_name: str = Field(
        default="teaching_assistant", env="CHROMA_COLLECTION_NAME"
    )

    # Embedding
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL"
    )

    # LLM
    llm_model: str = Field(default="gemini-2.5-flash", env="LLM_MODEL")
    llm_temperature: float = Field(default=0.3, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=4096, env="LLM_MAX_TOKENS")

    # RAG
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    top_k_results: int = Field(default=5, env="TOP_K_RESULTS")

    # Rate Limiting
    rate_limit: str = Field(default="30/minute", env="RATE_LIMIT")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(
        default=str(BASE_DIR / "artifacts" / "logs" / "app.log"), env="LOG_FILE"
    )

    # Upload
    max_upload_size_mb: int = Field(default=50, env="MAX_UPLOAD_SIZE_MB")
    allowed_extensions: str = Field(
        default="pdf,docx,txt,pptx", env="ALLOWED_EXTENSIONS"
    )

    # Paths
    data_dir: str = Field(default=str(BASE_DIR / "data"))
    upload_dir: str = Field(default=str(BASE_DIR / "data" / "uploads"))
    processed_dir: str = Field(default=str(BASE_DIR / "data" / "processed"))

    # Database
    database_url: str = Field(
        default=f"sqlite:///{_DEFAULT_DB.as_posix()}",
        env="DATABASE_URL",
    )

    # JWT Auth
    jwt_secret_key: str = Field(default="change-me-in-production-use-strong-secret", env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expire_minutes: int = Field(default=1440, env="JWT_EXPIRE_MINUTES")
    auth_disabled: bool = Field(default=False, env="AUTH_DISABLED")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    redis_enabled: bool = Field(default=True, env="REDIS_ENABLED")

    # Reranker
    reranker_model: str = Field(default="BAAI/bge-reranker-base", env="RERANKER_MODEL")
    reranker_disabled: bool = Field(default=False, env="RERANKER_DISABLED")
    retrieval_fetch_k: int = Field(default=10, env="RETRIEVAL_FETCH_K")
    rerank_top_k: int = Field(default=3, env="RERANK_TOP_K")

    # LangGraph Agent
    use_langgraph_agent: bool = Field(default=True, env="USE_LANGGRAPH_AGENT")

    # RAGAS
    ragas_disabled: bool = Field(default=False, env="RAGAS_DISABLED")

    @property
    def allowed_extensions_list(self) -> list[str]:
        return [ext.strip().lower() for ext in self.allowed_extensions.split(",")]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
