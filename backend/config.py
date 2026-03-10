"""Configuration management using Pydantic Settings."""
import logging
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM Configuration
    llm_provider: Literal["openai", "ollama", "mock"] = Field(default="mock")
    openai_api_key: str = Field(default="")
    ollama_base_url: str = Field(default="http://localhost:11434")
    model_name: str = Field(default="gpt-3.5-turbo")

    # Vector Database Configuration
    vector_db_type: Literal["chroma", "faiss"] = Field(default="chroma")
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    chroma_persist_dir: str = Field(default="./chroma_data")
    collection_name: str = Field(default="email_knowledge_base")

    # RAG Configuration
    chunk_size: int = Field(default=500, ge=100, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=200)
    top_k_results: int = Field(default=5, ge=1, le=20)

    # LLM Generation Parameters
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=500, ge=50, le=4000)

    # Email Configuration
    default_tone: Literal["formal", "friendly", "concise"] = Field(default="formal")

    # API Configuration
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_debug: bool = Field(default=False)


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings."""
    settings = Settings()
    logger.info(
        "Loaded settings: llm_provider=%s, vector_db=%s, embedding_model=%s",
        settings.llm_provider,
        settings.vector_db_type,
        settings.embedding_model,
    )
    return settings
