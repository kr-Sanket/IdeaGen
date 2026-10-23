"""Application configuration using pydantic-settings.

All settings are read from environment variables prefixed with ``IDEAGEN_``.
A ``.env`` file in the backend directory is loaded automatically if present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices, field_validator

# Must match ideagen.opportunity.ranking.get_candidate_ranker.
SUPPORTED_CANDIDATE_RANKING_STRATEGIES = frozenset({"tfidf", "fifo"})

# Resolve the project root (backend/) relative to this file.
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
_PROJECT_ROOT = _BACKEND_DIR.parent


class Settings(BaseSettings):
    """Central configuration for the IdeaGen application.

    Every field can be overridden by setting an environment variable
    with the ``IDEAGEN_`` prefix. For example, ``IDEAGEN_DEBUG=true``.
    """

    model_config = SettingsConfigDict(
        env_prefix="IDEAGEN_",
        env_file=_BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── General ───────────────────────────────────────────────────────
    app_name: str = "IdeaGen"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    environment: Literal["development", "staging", "production"] = "development"

    # ── Database ──────────────────────────────────────────────────────
    database_url: str | None = None

    # ── Providers & Models ────────────────────────────────────────────
    llm_provider: str = Field(default="ollama", validation_alias=AliasChoices("IDEAGEN_LLM_PROVIDER", "LLM_PROVIDER"))
    embedding_provider: str = Field(default="ollama", validation_alias=AliasChoices("IDEAGEN_EMBEDDING_PROVIDER", "EMBEDDING_PROVIDER"))

    # ── Ollama ────────────────────────────────────────────────────────
    ollama_base_url: str = Field(default="http://localhost:11434", validation_alias=AliasChoices("IDEAGEN_OLLAMA_BASE_URL", "OLLAMA_BASE_URL"))
    ollama_model: str = Field(default="qwen3:8b", validation_alias=AliasChoices("IDEAGEN_OLLAMA_MODEL", "OLLAMA_MODEL"))
    ollama_embedding_model: str = Field(default="nomic-embed-text", validation_alias=AliasChoices("IDEAGEN_OLLAMA_EMBEDDING_MODEL", "OLLAMA_EMBEDDING_MODEL"))
    ollama_embedding_dimension: int = Field(default=768, validation_alias=AliasChoices("IDEAGEN_OLLAMA_EMBEDDING_DIMENSION", "OLLAMA_EMBEDDING_DIMENSION"))
    ollama_timeout: float = Field(default=300.0, validation_alias=AliasChoices("IDEAGEN_OLLAMA_TIMEOUT", "OLLAMA_TIMEOUT"))
    ollama_num_ctx: int = Field(default=8192, validation_alias=AliasChoices("IDEAGEN_OLLAMA_NUM_CTX", "OLLAMA_NUM_CTX"))
    ollama_num_predict: int = Field(default=512, validation_alias=AliasChoices("IDEAGEN_OLLAMA_NUM_PREDICT", "OLLAMA_NUM_PREDICT"))
    ollama_auto_manage: bool = Field(default=True, validation_alias=AliasChoices("IDEAGEN_OLLAMA_AUTO_MANAGE", "OLLAMA_AUTO_MANAGE", "IDEAGEN_OLLAMA_AUTO_START"))
    ollama_idle_timeout: float = Field(default=60.0, validation_alias=AliasChoices("IDEAGEN_OLLAMA_IDLE_TIMEOUT", "OLLAMA_IDLE_TIMEOUT"))
    ollama_startup_timeout: float = Field(default=30.0, validation_alias=AliasChoices("IDEAGEN_OLLAMA_STARTUP_TIMEOUT", "OLLAMA_STARTUP_TIMEOUT"))
    ollama_polling_interval: float = Field(default=0.5, validation_alias=AliasChoices("IDEAGEN_OLLAMA_POLLING_INTERVAL", "OLLAMA_POLLING_INTERVAL"))
    ollama_binary_path: str = Field(default="ollama", validation_alias=AliasChoices("IDEAGEN_OLLAMA_BINARY_PATH", "OLLAMA_BINARY_PATH"))

    # ── Gemini (Preserved) ────────────────────────────────────────────
    gemini_api_key: str | None = Field(default=None, validation_alias=AliasChoices("IDEAGEN_GEMINI_API_KEY", "GEMINI_API_KEY"))
    gemini_model_name: str = Field(default="gemini-2.5-flash", validation_alias=AliasChoices("IDEAGEN_GEMINI_MODEL_NAME", "GEMINI_MODEL_NAME"))

    # ── OpenAlex ──────────────────────────────────────────────────────
    openalex_email: str | None = Field(default=None, validation_alias=AliasChoices("IDEAGEN_OPENALEX_EMAIL", "OPENALEX_EMAIL"))

    # ── Evolution ─────────────────────────────────────────────────────
    enable_post_ingestion_evolution: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "IDEAGEN_ENABLE_POST_INGESTION_EVOLUTION",
            "ENABLE_POST_INGESTION_EVOLUTION",
        ),
    )
    # Upper bound on eligible opportunities classified (one LLM call each) per ingested paper.
    post_ingestion_evolution_max_candidates: int = Field(
        default=20,
        ge=1,
        validation_alias=AliasChoices(
            "IDEAGEN_POST_INGESTION_EVOLUTION_MAX_CANDIDATES",
            "POST_INGESTION_EVOLUTION_MAX_CANDIDATES",
        ),
    )
    # Strategy for ranking candidate opportunities during incremental evolution ("tfidf" or "fifo").
    candidate_ranking_strategy: str = Field(
        default="tfidf",
        validation_alias=AliasChoices(
            "IDEAGEN_CANDIDATE_RANKING_STRATEGY",
            "CANDIDATE_RANKING_STRATEGY",
        ),
    )

    @field_validator("candidate_ranking_strategy", mode="before")
    @classmethod
    def _validate_candidate_ranking_strategy(cls, value: object) -> str:
        """Reject unknown strategies when settings load instead of on every ingestion."""
        normalized = str(value).strip().lower() if value is not None else ""
        if normalized not in SUPPORTED_CANDIDATE_RANKING_STRATEGIES:
            raise ValueError(
                f"Unsupported candidate ranking strategy {value!r}; "
                f"expected one of {sorted(SUPPORTED_CANDIDATE_RANKING_STRATEGIES)}."
            )
        return normalized

    # ── Paths ─────────────────────────────────────────────────────────
    project_root: Path = _PROJECT_ROOT
    data_dir: Path = _PROJECT_ROOT / "data"
    raw_data_dir: Path = _PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = _PROJECT_ROOT / "data" / "processed"
    index_dir: Path = _PROJECT_ROOT / "data" / "indexes"


def get_settings() -> Settings:
    """Return a ``Settings`` instance.

    This is a plain factory function. Caching / singleton behaviour
    will be added when the FastAPI dependency-injection layer is
    introduced in a later phase.
    """
    return Settings()
