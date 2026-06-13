"""Cloudflare Turnstile token verification.

Verifies a client-supplied Turnstile token against Cloudflare's siteverify
endpoint. Reuses ``requests`` (already a runtime dependency) rather than adding
a new HTTP client. Any network or parsing failure is treated as a failed
verification so a broken siteverify call never silently lets traffic through.
"""

import logging
from collections.abc import Iterable

import requests

logger = logging.getLogger(__name__)

_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"


def verify_turnstile(
    token: str,
    secret: str,
    remote_ip: str | None = None,
    expected_hostnames: Iterable[str] | None = None,
    timeout: float = 5.0,
) -> bool:
    """Verify a Turnstile token with Cloudflare siteverify.

    Args:
        token: Turnstile response token minted by the browser widget.
        secret: Turnstile secret key.
        remote_ip: Optional client IP to include in the verification.
        expected_hostnames: Optional allowlist of hostnames the token must have
            been solved on. When non-empty, a token whose ``hostname`` is not in
            the set is rejected, closing token-farming from another allowed host
            (e.g. localhost) using the public site key.
        timeout: HTTP timeout in seconds.

    Returns:
        True only when Cloudflare reports the token as successful (and, when
        ``expected_hostnames`` is given, the solved hostname is allowed); False
        on an empty token, a non-success result, a hostname mismatch, or any
        request/parse failure.
    """
    if not token:
        return False

    data = {"secret": secret, "response": token}
    if remote_ip:
        data["remoteip"] = remote_ip

    try:
        response = requests.post(_VERIFY_URL, data=data, timeout=timeout)
        response.raise_for_status()
        result = response.json()
    except (requests.RequestException, ValueError):
        logger.exception("Turnstile verification request failed")
        return False

    if not result.get("success", False):
        return False

    allowed = {h for h in (expected_hostnames or []) if h}
    if allowed and result.get("hostname") not in allowed:
        logger.warning(
            "Turnstile token solved on unexpected hostname %r", result.get("hostname")
        )
        return False

    return True
