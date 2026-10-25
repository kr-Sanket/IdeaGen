"""Tests for ideagen.core.config — Phase 0 backend configuration."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ideagen.core.config import Settings, get_settings


class TestSettingsDefaults:
    """Verify that default settings are sensible without any env vars."""

    def test_app_name(self) -> None:
        settings = Settings()
        assert settings.app_name == "IdeaGen"

    def test_debug_defaults_to_false(self) -> None:
        settings = Settings()
        assert settings.debug is False

    def test_log_level_defaults_to_info(self) -> None:
        settings = Settings()
        assert settings.log_level == "INFO"

    def test_environment_defaults_to_development(self) -> None:
        settings = Settings()
        assert settings.environment == "development"

    def test_data_directories_are_paths(self) -> None:
        settings = Settings()
        assert isinstance(settings.data_dir, Path)
        assert isinstance(settings.raw_data_dir, Path)
        assert isinstance(settings.processed_data_dir, Path)
        assert isinstance(settings.index_dir, Path)

    def test_data_dir_structure(self, monkeypatch) -> None:
        # Defaults only: drop the test-wide index isolation override from conftest.
        monkeypatch.delenv("IDEAGEN_INDEX_DIR", raising=False)
        settings = Settings()
        assert settings.raw_data_dir == settings.data_dir / "raw"
        assert settings.processed_data_dir == settings.data_dir / "processed"
        assert settings.index_dir == settings.data_dir / "indexes"


    def test_provider_defaults(self) -> None:
        settings = Settings()
        assert settings.llm_provider == "ollama"
        assert settings.embedding_provider == "ollama"
        assert settings.ollama_base_url == "http://localhost:11434"
        assert settings.ollama_model == "qwen3:8b"
        assert settings.ollama_embedding_model == "nomic-embed-text"
        assert settings.ollama_embedding_dimension == 768
        assert settings.ollama_timeout == 300.0
        assert settings.ollama_num_ctx == 8192
        assert settings.ollama_num_predict == 1024
        assert settings.ollama_recovery_num_predict == 2048
        assert settings.ollama_auto_manage is True
        assert settings.ollama_idle_timeout == 60.0
        assert settings.ollama_startup_timeout == 30.0
        assert settings.ollama_polling_interval == 0.5
        assert settings.ollama_binary_path == "ollama"


class TestSettingsFromEnv:
    """Verify that IDEAGEN_ environment variables override defaults."""

    def test_debug_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("IDEAGEN_DEBUG", "true")
        settings = Settings()
        assert settings.debug is True

    def test_log_level_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("IDEAGEN_LOG_LEVEL", "DEBUG")
        settings = Settings()
        assert settings.log_level == "DEBUG"

    def test_environment_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("IDEAGEN_ENVIRONMENT", "production")
        settings = Settings()
        assert settings.environment == "production"

    def test_app_name_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("IDEAGEN_APP_NAME", "TestApp")
        settings = Settings()
        assert settings.app_name == "TestApp"

    def test_ollama_settings_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("IDEAGEN_LLM_PROVIDER", "gemini")
        monkeypatch.setenv("IDEAGEN_EMBEDDING_PROVIDER", "sentence_transformers")
        monkeypatch.setenv("IDEAGEN_OLLAMA_BASE_URL", "http://remote:11434")
        monkeypatch.setenv("IDEAGEN_OLLAMA_MODEL", "qwen3:14b")
        monkeypatch.setenv("IDEAGEN_OLLAMA_EMBEDDING_MODEL", "custom-embed")
        monkeypatch.setenv("IDEAGEN_OLLAMA_EMBEDDING_DIMENSION", "1024")
        monkeypatch.setenv("IDEAGEN_OLLAMA_TIMEOUT", "60.0")
        monkeypatch.setenv("IDEAGEN_OLLAMA_NUM_CTX", "16384")
        monkeypatch.setenv("IDEAGEN_OLLAMA_NUM_PREDICT", "512")
        monkeypatch.setenv("IDEAGEN_OLLAMA_RECOVERY_NUM_PREDICT", "1024")
        monkeypatch.setenv("IDEAGEN_OLLAMA_AUTO_MANAGE", "false")
        monkeypatch.setenv("IDEAGEN_OLLAMA_IDLE_TIMEOUT", "120.0")
        monkeypatch.setenv("IDEAGEN_OLLAMA_STARTUP_TIMEOUT", "45.0")
        monkeypatch.setenv("IDEAGEN_OLLAMA_POLLING_INTERVAL", "0.2")
        monkeypatch.setenv("IDEAGEN_OLLAMA_BINARY_PATH", "/usr/local/bin/ollama")

        settings = Settings()
        assert settings.llm_provider == "gemini"
        assert settings.embedding_provider == "sentence_transformers"
        assert settings.ollama_base_url == "http://remote:11434"
        assert settings.ollama_model == "qwen3:14b"
        assert settings.ollama_embedding_model == "custom-embed"
        assert settings.ollama_embedding_dimension == 1024
        assert settings.ollama_timeout == 60.0
        assert settings.ollama_num_ctx == 16384
        assert settings.ollama_num_predict == 512
        assert settings.ollama_recovery_num_predict == 1024
        assert settings.ollama_auto_manage is False
        assert settings.ollama_idle_timeout == 120.0
        assert settings.ollama_startup_timeout == 45.0
        assert settings.ollama_polling_interval == 0.2
        assert settings.ollama_binary_path == "/usr/local/bin/ollama"


class TestGetSettings:
    """Verify the get_settings factory function."""

    def test_returns_settings_instance(self) -> None:
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_returns_new_instance_each_call(self) -> None:
        s1 = get_settings()
        s2 = get_settings()
        # No caching yet — each call returns a fresh instance
        assert s1 is not s2
        assert s1.app_name == s2.app_name


class TestPackageImport:
    """Verify that the ideagen package is importable and has a version."""

    def test_version_exists(self) -> None:
        import ideagen

        assert hasattr(ideagen, "__version__")
        assert isinstance(ideagen.__version__, str)
        assert ideagen.__version__ == "0.1.0"


class TestCandidateRankingStrategySetting:
    """The ranking strategy is validated when settings load, not on each ingestion."""

    def test_defaults_to_tfidf(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("IDEAGEN_CANDIDATE_RANKING_STRATEGY", raising=False)
        monkeypatch.delenv("CANDIDATE_RANKING_STRATEGY", raising=False)
        assert Settings().candidate_ranking_strategy == "tfidf"

    @pytest.mark.parametrize("raw,expected", [("fifo", "fifo"), (" TFIDF ", "tfidf"), ("FIFO", "fifo")])
    def test_accepts_and_normalizes_supported_values(
        self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: str
    ) -> None:
        monkeypatch.setenv("IDEAGEN_CANDIDATE_RANKING_STRATEGY", raw)
        assert Settings().candidate_ranking_strategy == expected

    @pytest.mark.parametrize("raw", ["bm25", "", "tf-idf"])
    def test_rejects_unsupported_values_at_load(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        from pydantic import ValidationError

        monkeypatch.setenv("IDEAGEN_CANDIDATE_RANKING_STRATEGY", raw)
        with pytest.raises(ValidationError, match="Unsupported candidate ranking strategy"):
            Settings()

    def test_supported_values_match_ranker_factory(self) -> None:
        from ideagen.core.config import SUPPORTED_CANDIDATE_RANKING_STRATEGIES
        from ideagen.opportunity.ranking import FIFOCandidateRanker, TFIDFCandidateRanker, get_candidate_ranker

        assert isinstance(get_candidate_ranker("tfidf"), TFIDFCandidateRanker)
        assert isinstance(get_candidate_ranker("fifo"), FIFOCandidateRanker)
        assert SUPPORTED_CANDIDATE_RANKING_STRATEGIES == {"tfidf", "fifo"}
