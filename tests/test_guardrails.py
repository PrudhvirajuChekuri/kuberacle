"""Tests for the request guardrails orchestration."""

import pytest

from k8s_rag.api.counters import Decision
from k8s_rag.api.guardrails import GuardrailError, Guardrails, hash_ip
from k8s_rag.api.settings import GuardrailSettings


def _settings() -> GuardrailSettings:
    return GuardrailSettings(
        enabled=True,
        turnstile_secret="secret",
        rate_limit_per_ip=10,
        global_daily_cap=300,
        ip_hash_salt="salt",
        gcp_project="p",
        firestore_database="(default)",
    )


class FakeCounters:
    """Records calls and returns a preset Decision."""

    def __init__(self, decision):
        self._decision = decision
        self.calls = []

    def check_and_increment(self, ip_hash, per_ip_cap, global_cap):
        self.calls.append((ip_hash, per_ip_cap, global_cap))
        return self._decision


def test_hash_ip_is_deterministic_and_salted():
    assert hash_ip("1.2.3.4", "salt") == hash_ip("1.2.3.4", "salt")
    assert hash_ip("1.2.3.4", "salt") != hash_ip("1.2.3.4", "other")
    assert hash_ip("1.2.3.4", "salt") != hash_ip("5.6.7.8", "salt")


def test_bad_token_raises_403_and_skips_counters():
    counters = FakeCounters(Decision(True, None))
    guardrails = Guardrails(_settings(), counters, verifier=lambda *a: False)

    with pytest.raises(GuardrailError) as exc:
        guardrails.enforce("1.2.3.4", "bad")

    assert exc.value.status_code == 403
    assert counters.calls == []


def test_allowed_request_consumes_budget_with_hashed_ip():
    counters = FakeCounters(Decision(True, None))
    guardrails = Guardrails(_settings(), counters, verifier=lambda *a: True)

    guardrails.enforce("1.2.3.4", "good")

    assert counters.calls == [(hash_ip("1.2.3.4", "salt"), 10, 300)]


def test_global_cap_raises_429_with_demo_message():
    counters = FakeCounters(Decision(False, "global"))
    guardrails = Guardrails(_settings(), counters, verifier=lambda *a: True)

    with pytest.raises(GuardrailError) as exc:
        guardrails.enforce("1.2.3.4", "good")

    assert exc.value.status_code == 429
    assert "demo" in exc.value.message


def test_per_ip_cap_raises_429():
    counters = FakeCounters(Decision(False, "per_ip"))
    guardrails = Guardrails(_settings(), counters, verifier=lambda *a: True)

    with pytest.raises(GuardrailError) as exc:
        guardrails.enforce("1.2.3.4", "good")

    assert exc.value.status_code == 429


def test_verifier_receives_ip_and_expected_hostnames():
    counters = FakeCounters(Decision(True, None))
    calls = []

    def recording_verifier(token, secret, client_ip, hostnames):
        calls.append((token, secret, client_ip, hostnames))
        return True

    settings = GuardrailSettings(
        enabled=True,
        turnstile_secret="secret",
        rate_limit_per_ip=10,
        global_daily_cap=300,
        ip_hash_salt="salt",
        gcp_project="p",
        firestore_database="(default)",
        turnstile_hostnames=("kuberacle.dev",),
    )
    guardrails = Guardrails(settings, counters, verifier=recording_verifier)

    guardrails.enforce("1.2.3.4", "tok")

    assert calls == [("tok", "secret", "1.2.3.4", ("kuberacle.dev",))]


def test_missing_ip_uses_unknown_bucket():
    counters = FakeCounters(Decision(True, None))
    guardrails = Guardrails(_settings(), counters, verifier=lambda *a: True)

    guardrails.enforce("", "good")

    assert counters.calls == [(hash_ip("unknown", "salt"), 10, 300)]
