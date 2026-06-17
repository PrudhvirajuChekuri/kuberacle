"""Env-driven settings for the observability layer.

Secrets (Langfuse keys) and the deployment on/off switch are runtime concerns,
so they come from the environment rather than ``configs/rag.yaml`` (which stays
the single source of truth for the RAG pipeline and the non-secret logging and
tracing knobs). Observability defaults to disabled so local development, the
test suite, and the CLI need no Langfuse account and emit no traces.
"""

import os
from dataclasses import dataclass

# Default Langfuse Cloud ingestion host (US region).
_DEFAULT_LANGFUSE_HOST = "https://us.cloud.langfuse.com"


@dataclass(frozen=True)
class ObservabilitySettings:
    """Runtime settings for tracing and Langfuse export.

    Attributes:
        enabled: Master switch for tracing and Langfuse export. When false the
            service still configures structured logging but creates no spans.
        langfuse_public_key: Langfuse project public key.
        langfuse_secret_key: Langfuse project secret key.
        langfuse_host: Langfuse ingestion host URL.
        environment: Deployment environment tag attached to traces.
        service_version: Release identifier (e.g. git SHA) attached to traces.
    """

    enabled: bool
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_host: str
    environment: str
    service_version: str

    @property
    def langfuse_enabled(self) -> bool:
        """Whether Langfuse export is both enabled and fully configured."""
        return bool(
            self.enabled and self.langfuse_public_key and self.langfuse_secret_key
        )


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean from the environment.

    Args:
        name: Environment variable name.
        default: Value to use when the variable is unset.

    Returns:
        True when the value is one of 1/true/yes/on (case-insensitive).
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_observability_settings() -> ObservabilitySettings:
    """Load observability settings from the environment.

    Returns:
        Parsed ObservabilitySettings. Missing Langfuse keys disable only the
        Langfuse plane (Cloud Trace still works); they are not fatal, since
        observability failures must never take down request serving.
    """
    return ObservabilitySettings(
        enabled=_env_bool("OBSERVABILITY_ENABLED", False),
        langfuse_public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
        langfuse_secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
        langfuse_host=os.environ.get("LANGFUSE_HOST", _DEFAULT_LANGFUSE_HOST),
        environment=os.environ.get("OBSERVABILITY_ENVIRONMENT", "production"),
        service_version=os.environ.get("SERVICE_VERSION", "unknown"),
    )
