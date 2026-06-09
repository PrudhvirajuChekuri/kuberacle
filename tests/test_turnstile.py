"""Tests for Cloudflare Turnstile token verification.

The siteverify HTTP call is stubbed; no network requests are made.
"""

import requests

from k8s_rag.api import turnstile
from k8s_rag.api.turnstile import verify_turnstile


class FakeResponse:
    """Minimal stand-in for a requests.Response."""

    def __init__(self, payload, raise_exc=None):
        self._payload = payload
        self._raise_exc = raise_exc

    def raise_for_status(self):
        if self._raise_exc is not None:
            raise self._raise_exc

    def json(self):
        return self._payload


def test_empty_token_short_circuits(monkeypatch):
    """An empty token returns False without calling siteverify."""
    called = False

    def fake_post(*args, **kwargs):
        nonlocal called
        called = True
        return FakeResponse({"success": True})

    monkeypatch.setattr(turnstile.requests, "post", fake_post)
    assert verify_turnstile("", "secret") is False
    assert called is False


def test_success_result_is_true(monkeypatch):
    """A success payload yields True and forwards the optional remote IP."""
    captured = {}

    def fake_post(url, data, timeout):
        captured.update(data)
        return FakeResponse({"success": True})

    monkeypatch.setattr(turnstile.requests, "post", fake_post)
    assert verify_turnstile("tok", "secret", remote_ip="1.2.3.4") is True
    assert captured == {"secret": "secret", "response": "tok", "remoteip": "1.2.3.4"}


def test_unsuccessful_result_is_false(monkeypatch):
    """A non-success payload yields False."""
    monkeypatch.setattr(
        turnstile.requests,
        "post",
        lambda *a, **k: FakeResponse({"success": False, "error-codes": ["x"]}),
    )
    assert verify_turnstile("tok", "secret") is False


def test_request_failure_is_false(monkeypatch):
    """A network error is treated as a failed verification."""

    def fake_post(*args, **kwargs):
        raise requests.ConnectionError("boom")

    monkeypatch.setattr(turnstile.requests, "post", fake_post)
    assert verify_turnstile("tok", "secret") is False
