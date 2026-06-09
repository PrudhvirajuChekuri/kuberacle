"""Request guardrails: Turnstile verification then Firestore daily caps.

Runs before any model call so abusive or unverified traffic never reaches the
(billable) RAG pipeline. A failed guardrail raises ``GuardrailError`` carrying
the HTTP status and a user-facing message the API surfaces over SSE.
"""

import hashlib
import logging
from collections.abc import Callable

from k8s_rag.api.counters import FirestoreCounters
from k8s_rag.api.settings import GuardrailSettings
from k8s_rag.api.turnstile import verify_turnstile

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

    def enforce(self, client_ip: str, turnstile_token: str) -> None:
        """Verify the request and consume one unit of rate-limit budget.

        Args:
            client_ip: Client IP forwarded by the web service.
            turnstile_token: Turnstile token from the browser widget.

        Raises:
            GuardrailError: 403 when the Turnstile check fails; 429 when a
                per-IP or global daily cap is reached.
        """
        settings = self._settings
        if not self._verify(turnstile_token, settings.turnstile_secret, client_ip):
            raise GuardrailError(403, "Bot check failed. Please reload and try again.")

        ip_hash = hash_ip(client_ip or "unknown", settings.ip_hash_salt)
        decision = self._counters.check_and_increment(
            ip_hash, settings.rate_limit_per_ip, settings.global_daily_cap
        )
        if decision.allowed:
            return

        if decision.reason == "global":
            raise GuardrailError(
                429,
                "The daily query limit for this demo has been reached. "
                "Please try again tomorrow.",
            )
        raise GuardrailError(
            429,
            "You have reached your daily query limit. Please try again tomorrow.",
        )
