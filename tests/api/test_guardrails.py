"""Tests for the request guardrails orchestration."""

import pytest

from kuberacle.api.counters import Decision
from kuberacle.api.guardrails import GuardrailError, Guardrails, hash_ip
from kuberacle.api.settings import GuardrailSettings


def _settings(**overrides) -> GuardrailSettings:
    base = dict(
        enabled=True,
        turnstile_secret="secret",
        rate_limit_per_ip=10,
        global_daily_cap=300,
        ip_hash_salt="salt",
        gcp_project="p",
        firestore_database="(default)",
    )
    base.update(overrides)
    return GuardrailSettings(**base)


class FakeCounters:
    """Records counter calls and returns preset results."""

    def __init__(self, under_cap=True, ip_allowed=True, decision=Decision(True, None)):
        self._under_cap = under_cap
        self._ip_allowed = ip_allowed
        self._decision = decision
        self.peeks = []
        self.ip_charges = []
        self.combined = []

    def ip_under_cap(self, ip_hash, per_ip_cap):
        self.peeks.append((ip_hash, per_ip_cap))
        return self._under_cap

    def check_and_increment_ip(self, ip_hash, per_ip_cap):
        self.ip_charges.append((ip_hash, per_ip_cap))
        return self._ip_allowed

    def check_and_increment(self, ip_hash, per_ip_cap, global_cap):
        self.combined.append((ip_hash, per_ip_cap, global_cap))
        return self._decision


def test_hash_ip_is_deterministic_and_salted():
    assert hash_ip("1.2.3.4", "salt") == hash_ip("1.2.3.4", "salt")
    assert hash_ip("1.2.3.4", "salt") != hash_ip("1.2.3.4", "other")
    assert hash_ip("1.2.3.4", "salt") != hash_ip("5.6.7.8", "salt")


def test_verify_turnstile_passes_ip_and_hostnames():
    counters = FakeCounters()
    calls = []

    def recording_verifier(token, secret, client_ip, hostnames):
        calls.append((token, secret, client_ip, hostnames))
        return True

    settings = _settings(turnstile_hostnames=("kuberacle.dev",))
    guardrails = Guardrails(settings, counters, verifier=recording_verifier)

    guardrails.verify_turnstile("1.2.3.4", "tok")

    assert calls == [("tok", "secret", "1.2.3.4", ("kuberacle.dev",))]
    assert counters.peeks == []  # Turnstile never touches counters.


def test_bad_token_raises_403():
    counters = FakeCounters()
    guardrails = Guardrails(_settings(), counters, verifier=lambda *a: False)

    with pytest.raises(GuardrailError) as exc:
        guardrails.verify_turnstile("1.2.3.4", "bad")

    assert exc.value.status_code == 403


def test_check_ip_rate_limit_is_read_only():
    counters = FakeCounters(under_cap=True)
    guardrails = Guardrails(_settings(), counters, verifier=lambda *a: True)

    guardrails.check_ip_rate_limit("1.2.3.4")

    assert counters.peeks == [(hash_ip("1.2.3.4", "salt"), 10)]
    # A peek never charges either counter.
    assert counters.ip_charges == [] and counters.combined == []


def test_over_cap_ip_rejected_before_cache():
    counters = FakeCounters(under_cap=False)
    guardrails = Guardrails(_settings(), counters, verifier=lambda *a: True)

    with pytest.raises(GuardrailError) as exc:
        guardrails.check_ip_rate_limit("1.2.3.4")

    assert exc.value.status_code == 429
    assert "demo" not in exc.value.message


def test_missing_ip_uses_unknown_bucket():
    counters = FakeCounters()
    guardrails = Guardrails(_settings(), counters, verifier=lambda *a: True)

    guardrails.check_ip_rate_limit("")

    assert counters.peeks == [(hash_ip("unknown", "salt"), 10)]


def test_charge_ip_charges_per_ip_only():
    counters = FakeCounters(ip_allowed=True)
    guardrails = Guardrails(_settings(), counters, verifier=lambda *a: True)

    guardrails.charge_ip("1.2.3.4")

    assert counters.ip_charges == [(hash_ip("1.2.3.4", "salt"), 10)]
    assert counters.combined == []  # Hits bypass the global cap.


def test_charge_ip_race_loss_raises_429():
    counters = FakeCounters(ip_allowed=False)
    guardrails = Guardrails(_settings(), counters, verifier=lambda *a: True)

    with pytest.raises(GuardrailError) as exc:
        guardrails.charge_ip("1.2.3.4")

    assert exc.value.status_code == 429


def test_charge_ip_and_global_charges_both():
    counters = FakeCounters(decision=Decision(True, None))
    guardrails = Guardrails(_settings(), counters, verifier=lambda *a: True)

    guardrails.charge_ip_and_global("1.2.3.4")

    assert counters.combined == [(hash_ip("1.2.3.4", "salt"), 10, 300)]


def test_global_rejection_uses_demo_message():
    counters = FakeCounters(decision=Decision(False, "global"))
    guardrails = Guardrails(_settings(), counters, verifier=lambda *a: True)

    with pytest.raises(GuardrailError) as exc:
        guardrails.charge_ip_and_global("1.2.3.4")

    assert exc.value.status_code == 429
    assert "demo" in exc.value.message


def test_per_ip_rejection_on_miss_uses_personal_message():
    counters = FakeCounters(decision=Decision(False, "per_ip"))
    guardrails = Guardrails(_settings(), counters, verifier=lambda *a: True)

    with pytest.raises(GuardrailError) as exc:
        guardrails.charge_ip_and_global("1.2.3.4")

    assert exc.value.status_code == 429
    assert "demo" not in exc.value.message
