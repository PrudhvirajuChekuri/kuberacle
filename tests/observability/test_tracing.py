"""Tests for tracing setup (disabled path; no network)."""

from kuberacle.config import ObservabilityConfig
from kuberacle.observability import tracing
from kuberacle.observability.settings import ObservabilitySettings

OBS_CONFIG = ObservabilityConfig("kuberacle-api", "INFO", "json", 1.0)


def _disabled_settings():
    return ObservabilitySettings(
        enabled=False,
        langfuse_public_key="",
        langfuse_secret_key="",
        langfuse_host="https://us.cloud.langfuse.com",
        environment="production",
        service_version="dev",
    )


def test_disabled_returns_empty_handles():
    """When disabled, configure_tracing creates no provider or client."""
    handles = tracing.configure_tracing(
        _disabled_settings(), OBS_CONFIG, "test-project"
    )
    assert handles.enabled is False
    assert handles.langfuse is None
    assert tracing.get_langfuse() is None


def test_flush_and_shutdown_are_safe_when_disabled():
    """Flush and shutdown never raise on empty handles."""
    handles = tracing.configure_tracing(
        _disabled_settings(), OBS_CONFIG, "test-project"
    )
    handles.force_flush()
    handles.shutdown()
