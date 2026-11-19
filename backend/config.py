"""Backend configuration module using Pydantic Settings."""

from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

BASE_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_DB = BASE_DIR / "data" / "campusgpt.db"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App
    app_name: str = Field(default="CampusGPT", validation_alias="APP_NAME")
    app_version: str = Field(default="2.0.0", validation_alias="APP_VERSION")
    app_env: str = Field(default="development", validation_alias="APP_ENV")
    debug: bool = Field(default=True, validation_alias="DEBUG")

    # API Keys
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")

    # Backend
    backend_host: str = Field(default="0.0.0.0", validation_alias="BACKEND_HOST")
    backend_port: int = Field(default=8000, validation_alias="BACKEND_PORT")
    backend_url: str = Field(default="http://localhost:8000", validation_alias="BACKEND_URL")

    # Frontend
    frontend_port: int = Field(default=8501, validation_alias="FRONTEND_PORT")

    # ChromaDB
    chroma_persist_dir: str = Field(
        default=str(BASE_DIR / "chroma_db"), validation_alias="CHROMA_PERSIST_DIR"
    )
    chroma_collection_name: str = Field(
        default="teaching_assistant", validation_alias="CHROMA_COLLECTION_NAME"
    )

    # Embedding
    embedding_provider: str = Field(default="local", validation_alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", validation_alias="EMBEDDING_MODEL"
    )
    gemini_embedding_model: str = Field(
        default="models/gemini-embedding-001", validation_alias="GEMINI_EMBEDDING_MODEL"
    )

    # LLM
    llm_model: str = Field(default="gemini-2.5-flash", validation_alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.3, validation_alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=4096, validation_alias="LLM_MAX_TOKENS")

    # RAG
    chunk_size: int = Field(default=1000, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, validation_alias="CHUNK_OVERLAP")
    top_k_results: int = Field(default=5, validation_alias="TOP_K_RESULTS")

    # Rate Limiting
    rate_limit: str = Field(default="30/minute", validation_alias="RATE_LIMIT")

    # Logging
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_file: str = Field(
        default=str(BASE_DIR / "artifacts" / "logs" / "app.log"), validation_alias="LOG_FILE"
    )

    # Upload
    max_upload_size_mb: int = Field(default=50, validation_alias="MAX_UPLOAD_SIZE_MB")
    allowed_extensions: str = Field(
        default="pdf,docx,txt,pptx", validation_alias="ALLOWED_EXTENSIONS"
    )

    # Paths
    data_dir: str = Field(default=str(BASE_DIR / "data"))
    upload_dir: str = Field(default=str(BASE_DIR / "data" / "uploads"))
    processed_dir: str = Field(default=str(BASE_DIR / "data" / "processed"))

    # Database
    database_url: str = Field(
        default=f"sqlite:///{_DEFAULT_DB.as_posix()}",
        validation_alias="DATABASE_URL",
    )

    # JWT Auth
    jwt_secret_key: str = Field(default="change-me-in-production-use-strong-secret", validation_alias="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", validation_alias="JWT_ALGORITHM")
    jwt_expire_minutes: int = Field(default=1440, validation_alias="JWT_EXPIRE_MINUTES")
    auth_disabled: bool = Field(default=False, validation_alias="AUTH_DISABLED")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", validation_alias="REDIS_URL")
    redis_enabled: bool = Field(default=True, validation_alias="REDIS_ENABLED")

    # Reranker
    reranker_model: str = Field(default="BAAI/bge-reranker-base", validation_alias="RERANKER_MODEL")
    reranker_disabled: bool = Field(default=False, validation_alias="RERANKER_DISABLED")
    retrieval_fetch_k: int = Field(default=10, validation_alias="RETRIEVAL_FETCH_K")
    rerank_top_k: int = Field(default=3, validation_alias="RERANK_TOP_K")

    # LangGraph Agent
    use_langgraph_agent: bool = Field(default=True, validation_alias="USE_LANGGRAPH_AGENT")

    # RAGAS
    ragas_disabled: bool = Field(default=False, validation_alias="RAGAS_DISABLED")

    @property
    def allowed_extensions_list(self) -> list[str]:
        return [ext.strip().lower() for ext in self.allowed_extensions.split(",")]

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        populate_by_name=True,
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
