"""Application configuration using pydantic-settings.

All settings are read from environment variables prefixed with ``IDEAGEN_``.
A ``.env`` file in the backend directory is loaded automatically if present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

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

    # ── LLM ───────────────────────────────────────────────────────────
    gemini_api_key: str | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    gemini_model_name: str = Field(default="gemini-2.5-flash", validation_alias="GEMINI_MODEL_NAME")

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
