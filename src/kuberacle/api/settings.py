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


@dataclass(frozen=True)
class CacheSettings:
    """Runtime configuration for the Firestore answer cache.

    Attributes:
        enabled: Whether the answer cache is consulted and written.
        collection: Firestore collection holding cached answers.
        ttl_days: Days before a cached answer expires (drives the Firestore TTL
            policy field and the read-time expiry check).
        gcp_project: GCP project ID hosting the Firestore database.
        firestore_database: Firestore database name (``(default)`` normally).
    """

    enabled: bool
    collection: str
    ttl_days: int
    gcp_project: str
    firestore_database: str


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
    hostnames = tuple(
        h.strip()
        for h in os.environ.get("TURNSTILE_HOSTNAMES", "").split(",")
        if h.strip()
    )

    if enabled:
        # TURNSTILE_HOSTNAMES is required so an enabled deploy cannot silently
        # run with hostname checking disabled (which would allow tokens farmed
        # on another allowed host using the public site key).
        missing = [
            name
            for name, value in (
                ("TURNSTILE_SECRET", turnstile_secret),
                ("IP_HASH_SALT", ip_hash_salt),
                ("GCP_PROJECT", gcp_project),
                ("TURNSTILE_HOSTNAMES", hostnames),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                "Guardrails are enabled but required environment variables are "
                f"missing: {', '.join(missing)}."
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


def load_cache_settings() -> CacheSettings:
    """Load answer-cache settings from the environment.

    Returns:
        Parsed CacheSettings. Defaults to disabled so local development, tests,
        and the CLI need no Firestore.

    Raises:
        RuntimeError: When the cache is enabled but ``GCP_PROJECT`` (required to
            reach Firestore) is missing.
    """
    enabled = _env_bool("ANSWER_CACHE_ENABLED", False)
    gcp_project = os.environ.get("GCP_PROJECT", "")
    if enabled and not gcp_project:
        raise RuntimeError(
            "ANSWER_CACHE_ENABLED is set but GCP_PROJECT is missing."
        )
    return CacheSettings(
        enabled=enabled,
        collection=os.environ.get("ANSWER_CACHE_COLLECTION", "answer_cache").strip()
        or "answer_cache",
        ttl_days=int(os.environ.get("ANSWER_CACHE_TTL_DAYS", "14")),
        gcp_project=gcp_project,
        firestore_database=os.environ.get("FIRESTORE_DATABASE", "(default)"),
    )
