"""Tests for env-driven guardrail and cache settings."""

import pytest

from kuberacle.api.settings import load_cache_settings, load_guardrail_settings


def test_turnstile_hostnames_parsed_from_env(monkeypatch):
    """A comma-separated hostname list is parsed and trimmed into a tuple."""
    monkeypatch.setenv("TURNSTILE_HOSTNAMES", " kuberacle.dev , localhost ,")
    settings = load_guardrail_settings()
    assert settings.turnstile_hostnames == ("kuberacle.dev", "localhost")


def test_turnstile_hostnames_default_empty(monkeypatch):
    """Unset hostnames yield an empty tuple (hostname check disabled)."""
    monkeypatch.delenv("TURNSTILE_HOSTNAMES", raising=False)
    settings = load_guardrail_settings()
    assert settings.turnstile_hostnames == ()


def test_enabled_requires_secrets(monkeypatch):
    """Enabling guardrails without the required secrets raises."""
    monkeypatch.setenv("GUARDRAILS_ENABLED", "true")
    monkeypatch.delenv("TURNSTILE_SECRET", raising=False)
    monkeypatch.delenv("IP_HASH_SALT", raising=False)
    monkeypatch.delenv("GCP_PROJECT", raising=False)
    with pytest.raises(RuntimeError):
        load_guardrail_settings()


def test_enabled_requires_turnstile_hostnames(monkeypatch):
    """Enabling guardrails without TURNSTILE_HOSTNAMES raises (fail-closed)."""
    monkeypatch.setenv("GUARDRAILS_ENABLED", "true")
    monkeypatch.setenv("TURNSTILE_SECRET", "secret")
    monkeypatch.setenv("IP_HASH_SALT", "salt")
    monkeypatch.setenv("GCP_PROJECT", "p")
    monkeypatch.delenv("TURNSTILE_HOSTNAMES", raising=False)
    with pytest.raises(RuntimeError, match="TURNSTILE_HOSTNAMES"):
        load_guardrail_settings()


def test_enabled_passes_with_all_required(monkeypatch):
    """All required vars present yields settings without raising."""
    monkeypatch.setenv("GUARDRAILS_ENABLED", "true")
    monkeypatch.setenv("TURNSTILE_SECRET", "secret")
    monkeypatch.setenv("IP_HASH_SALT", "salt")
    monkeypatch.setenv("GCP_PROJECT", "p")
    monkeypatch.setenv("TURNSTILE_HOSTNAMES", "kuberacle.dev")
    settings = load_guardrail_settings()
    assert settings.enabled is True
    assert settings.turnstile_hostnames == ("kuberacle.dev",)


def test_cache_disabled_by_default(monkeypatch):
    """The answer cache defaults to disabled and needs no GCP project."""
    monkeypatch.delenv("ANSWER_CACHE_ENABLED", raising=False)
    monkeypatch.delenv("GCP_PROJECT", raising=False)
    settings = load_cache_settings()
    assert settings.enabled is False
    assert settings.collection == "answer_cache"
    assert settings.ttl_days == 14


def test_cache_reads_overrides(monkeypatch):
    """Collection, TTL, and database are read from the environment."""
    monkeypatch.setenv("ANSWER_CACHE_ENABLED", "true")
    monkeypatch.setenv("GCP_PROJECT", "p")
    monkeypatch.setenv("ANSWER_CACHE_COLLECTION", "answers_v2")
    monkeypatch.setenv("ANSWER_CACHE_TTL_DAYS", "30")
    monkeypatch.setenv("FIRESTORE_DATABASE", "demo")
    settings = load_cache_settings()
    assert settings.enabled is True
    assert settings.collection == "answers_v2"
    assert settings.ttl_days == 30
    assert settings.firestore_database == "demo"


def test_cache_enabled_requires_gcp_project(monkeypatch):
    """Enabling the cache without GCP_PROJECT raises (cannot reach Firestore)."""
    monkeypatch.setenv("ANSWER_CACHE_ENABLED", "true")
    monkeypatch.delenv("GCP_PROJECT", raising=False)
    with pytest.raises(RuntimeError, match="GCP_PROJECT"):
        load_cache_settings()
