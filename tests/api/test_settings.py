"""Tests for env-driven guardrail settings."""

import pytest

from kuberacle.api.settings import load_guardrail_settings


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
