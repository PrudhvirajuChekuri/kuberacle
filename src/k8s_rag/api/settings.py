"""Env-driven settings for API request guardrails.

These knobs are deployment and runtime concerns (secrets, rate caps), so they
come from the environment rather than ``configs/rag.yaml`` (which stays the
single source of truth for the RAG pipeline). Guardrails default to disabled so
local development and the test suite need neither Cloudflare Turnstile nor
Firestore.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class GuardrailSettings:
    """Runtime configuration for the API guardrails.

    Attributes:
        enabled: Whether guardrails are enforced at all.
        turnstile_secret: Cloudflare Turnstile secret key for siteverify.
        rate_limit_per_ip: Max queries per client IP per UTC day.
        global_daily_cap: Max queries across all clients per UTC day.
        ip_hash_salt: Salt mixed into client IPs before hashing for storage.
        gcp_project: GCP project ID hosting the Firestore database.
        firestore_database: Firestore database name (``(default)`` normally).
        turnstile_hostnames: Hostnames a Turnstile token must have been solved
            on. Empty disables the check (local dev); set in prod to reject
            tokens farmed on another allowed host with the public site key.
    """

    enabled: bool
    turnstile_secret: str
    rate_limit_per_ip: int
    global_daily_cap: int
    ip_hash_salt: str
    gcp_project: str
    firestore_database: str
    turnstile_hostnames: tuple[str, ...] = ()


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


def load_guardrail_settings() -> GuardrailSettings:
    """Load guardrail settings from the environment.

    Returns:
        Parsed GuardrailSettings.

    Raises:
        RuntimeError: When guardrails are enabled but a required secret or
            identifier (Turnstile secret, IP hash salt, GCP project) is missing.
    """
    enabled = _env_bool("GUARDRAILS_ENABLED", False)
    turnstile_secret = os.environ.get("TURNSTILE_SECRET", "")
    ip_hash_salt = os.environ.get("IP_HASH_SALT", "")
    gcp_project = os.environ.get("GCP_PROJECT", "")

    if enabled:
        missing = [
            name
            for name, value in (
                ("TURNSTILE_SECRET", turnstile_secret),
                ("IP_HASH_SALT", ip_hash_salt),
                ("GCP_PROJECT", gcp_project),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                "Guardrails are enabled but required environment variables are "
                f"missing: {', '.join(missing)}."
            )

    hostnames = tuple(
        h.strip()
        for h in os.environ.get("TURNSTILE_HOSTNAMES", "").split(",")
        if h.strip()
    )

    return GuardrailSettings(
        enabled=enabled,
        turnstile_secret=turnstile_secret,
        rate_limit_per_ip=int(os.environ.get("RATE_LIMIT_PER_IP", "10")),
        global_daily_cap=int(os.environ.get("GLOBAL_DAILY_CAP", "300")),
        ip_hash_salt=ip_hash_salt,
        gcp_project=gcp_project,
        firestore_database=os.environ.get("FIRESTORE_DATABASE", "(default)"),
        turnstile_hostnames=hostnames,
    )
