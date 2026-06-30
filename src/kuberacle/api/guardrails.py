"""Request guardrails: Turnstile verification then Firestore daily caps.

Runs before any model call so abusive or unverified traffic never reaches the
(billable) RAG pipeline. A failed guardrail raises ``GuardrailError`` carrying
the HTTP status and a user-facing message the API surfaces over SSE.
"""

import hashlib
import logging
from collections.abc import Callable

from kuberacle.api.counters import FirestoreCounters
from kuberacle.api.settings import GuardrailSettings
from kuberacle.api.turnstile import verify_turnstile

logger = logging.getLogger(__name__)


class GuardrailError(Exception):
    """Raised when a request fails a guardrail.

    Attributes:
        status_code: HTTP status to return (403 bot check, 429 rate limited).
        message: User-facing message safe to show in the UI.
    """

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def hash_ip(ip: str, salt: str) -> str:
    """Hash a client IP with a salt for privacy-preserving storage.

    Args:
        ip: Client IP address.
        salt: Secret salt mixed in before hashing.

    Returns:
        Hex-encoded SHA-256 digest of ``salt:ip``.
    """
    return hashlib.sha256(f"{salt}:{ip}".encode("utf-8")).hexdigest()


class Guardrails:
    """Enforces Turnstile verification and per-IP / global daily caps."""

    def __init__(
        self,
        settings: GuardrailSettings,
        counters: FirestoreCounters,
        verifier: Callable[..., bool] = verify_turnstile,
    ):
        """Initialize the guardrails.

        Args:
            settings: Guardrail settings (secret, caps, salt).
            counters: Firestore-backed daily counter store.
            verifier: Turnstile verification callable (injectable for tests).
        """
        self._settings = settings
        self._counters = counters
        self._verify = verifier

    _PER_IP_MESSAGE = (
        "You have reached your daily query limit. Please try again tomorrow."
    )
    _GLOBAL_MESSAGE = (
        "The daily query limit for this demo has been reached. "
        "Please try again tomorrow."
    )

    def _ip_hash(self, client_ip: str) -> str:
        """Return the salted hash of a client IP (bucketing missing IPs)."""
        return hash_ip(client_ip or "unknown", self._settings.ip_hash_salt)

    def verify_turnstile(self, client_ip: str, turnstile_token: str) -> None:
        """Verify the Turnstile token, the first gate before any app work.

        Args:
            client_ip: Client IP forwarded by the web service.
            turnstile_token: Turnstile token from the browser widget.

        Raises:
            GuardrailError: 403 when the Turnstile check fails.
        """
        settings = self._settings
        if not self._verify(
            turnstile_token,
            settings.turnstile_secret,
            client_ip,
            settings.turnstile_hostnames,
        ):
            raise GuardrailError(403, "Bot check failed. Please reload and try again.")

    def check_ip_rate_limit(self, client_ip: str) -> None:
        """Reject an already over-cap client before the cache is consulted.

        A read-only check that charges nothing, so an over-cap client never
        reaches the cache lookup. The per-IP counter is charged later, once the
        request is actually served, by :meth:`charge_ip` (hit) or
        :meth:`charge_ip_and_global` (miss).

        Args:
            client_ip: Client IP forwarded by the web service.

        Raises:
            GuardrailError: 429 when the per-IP daily cap is already reached.
        """
        if not self._counters.ip_under_cap(
            self._ip_hash(client_ip), self._settings.rate_limit_per_ip
        ):
            raise GuardrailError(429, self._PER_IP_MESSAGE)

    def charge_ip(self, client_ip: str) -> None:
        """Charge one unit of per-IP budget for a served cache hit.

        A hit bypasses the global cap (it costs nothing) but still consumes
        per-IP budget so a cached answer cannot be hammered past the cap.

        Args:
            client_ip: Client IP forwarded by the web service.

        Raises:
            GuardrailError: 429 when the per-IP daily cap is reached (a race
                since the pre-cache check, atomically rejected here).
        """
        if not self._counters.check_and_increment_ip(
            self._ip_hash(client_ip), self._settings.rate_limit_per_ip
        ):
            raise GuardrailError(429, self._PER_IP_MESSAGE)

    def charge_ip_and_global(self, client_ip: str) -> None:
        """Charge per-IP and global budget atomically for a cache miss.

        A miss can trigger the paid pipeline, so it draws down both caps in one
        transaction: neither counter is charged unless both are below their cap,
        and an exhausted global cap is reported as a demo-wide outage.

        Args:
            client_ip: Client IP forwarded by the web service.

        Raises:
            GuardrailError: 429 when either the global or per-IP daily cap is
                reached.
        """
        settings = self._settings
        decision = self._counters.check_and_increment(
            self._ip_hash(client_ip),
            settings.rate_limit_per_ip,
            settings.global_daily_cap,
        )
        if decision.allowed:
            return
        if decision.reason == "global":
            raise GuardrailError(429, self._GLOBAL_MESSAGE)
        raise GuardrailError(429, self._PER_IP_MESSAGE)
