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

    def test_data_dir_structure(self) -> None:
        settings = Settings()
        assert settings.raw_data_dir == settings.data_dir / "raw"
        assert settings.processed_data_dir == settings.data_dir / "processed"
        assert settings.index_dir == settings.data_dir / "indexes"


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
