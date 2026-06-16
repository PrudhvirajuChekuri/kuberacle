"""Tests for env-driven observability settings."""

from kuberacle.observability.settings import load_observability_settings


def test_defaults_disabled(monkeypatch):
    """Observability is disabled by default with no env set."""
    for var in (
        "OBSERVABILITY_ENABLED",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
        "OBSERVABILITY_ENVIRONMENT",
        "SERVICE_VERSION",
    ):
        monkeypatch.delenv(var, raising=False)
    settings = load_observability_settings()
    assert settings.enabled is False
    assert settings.langfuse_enabled is False
    assert settings.environment == "production"
    assert settings.langfuse_host.startswith("https://")


def test_langfuse_enabled_requires_keys(monkeypatch):
    """Langfuse export needs the master switch plus both keys."""
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "true")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    settings = load_observability_settings()
    assert settings.enabled is True
    assert settings.langfuse_enabled is False

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    settings = load_observability_settings()
    assert settings.langfuse_enabled is True


def test_overrides_from_env(monkeypatch):
    """Host, environment, and version come from the environment."""
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "1")
    monkeypatch.setenv("LANGFUSE_HOST", "https://eu.cloud.langfuse.com")
    monkeypatch.setenv("OBSERVABILITY_ENVIRONMENT", "staging")
    monkeypatch.setenv("SERVICE_VERSION", "abc1234")
    settings = load_observability_settings()
    assert settings.langfuse_host == "https://eu.cloud.langfuse.com"
    assert settings.environment == "staging"
    assert settings.service_version == "abc1234"
