"""Tests for the API request-summary event and SSE-contract preservation."""

import logging

from fastapi.testclient import TestClient

from kuberacle.api.app import create_app
from kuberacle.api.guardrails import GuardrailError
from kuberacle.config import PricingConfig
from kuberacle.observability import context as ctx
from kuberacle.qa import AnswerDelta, QAResult

PRICING = PricingConfig(0.10, 0.40, 0.15, 1.00)


class FakeQA:
    """Stub QA system that records an outcome via the active request metrics."""

    def __init__(self, events, outcome=ctx.OUTCOME_ANSWERED):
        self._events = events
        self._outcome = outcome

    def ask_stream(self, question, top_k=None):
        del question, top_k
        # Prove the request metrics binding is visible inside the stream.
        ctx.update_metrics(outcome=self._outcome, chunks_retrieved=2)
        yield from self._events


class FakeGuardrails:
    def __init__(self, error=None):
        self._error = error

    def enforce(self, client_ip, turnstile_token):
        del client_ip, turnstile_token
        if self._error is not None:
            raise self._error


def _client(events, *, pricing=PRICING, guardrails=None):
    app = create_app()
    app.state.qa_system = FakeQA(events)
    app.state.pricing = pricing
    if guardrails is not None:
        app.state.guardrails = guardrails
    return TestClient(app)


def _summaries(caplog):
    return [r for r in caplog.records if getattr(r, "event", None) == "request_summary"]


def test_summary_emitted_once_on_answered(caplog):
    """One request-summary event is emitted, carrying RED and outcome fields."""
    events = [AnswerDelta("Pods run."), QAResult("Pods run.", [], [])]
    with caplog.at_level(logging.INFO, logger="kuberacle.observability.request"):
        resp = _client(events).post("/query", json={"question": "hello there"})

    assert resp.status_code == 200
    assert "event: final" in resp.text  # SSE contract preserved
    summaries = _summaries(caplog)
    assert len(summaries) == 1
    record = summaries[0]
    assert record.outcome == ctx.OUTCOME_ANSWERED
    assert record.question_length == len("hello there")
    assert getattr(record, "http.status") == 200
    assert record.chunks_retrieved == 2
    # The context is cleaned up after the request.
    assert ctx.get_metrics() is None


def test_no_summary_without_pricing(caplog):
    """With no pricing wired (lifespan skipped), no summary is emitted."""
    events = [AnswerDelta("x"), QAResult("x", [], [])]
    with caplog.at_level(logging.INFO, logger="kuberacle.observability.request"):
        resp = _client(events, pricing=None).post("/query", json={"question": "q"})

    assert resp.status_code == 200
    assert _summaries(caplog) == []


def test_summary_on_guardrail_rejection(caplog):
    """A rejected request emits a summary with the guardrail outcome and status."""
    guardrails = FakeGuardrails(
        error=GuardrailError(429, "You have reached your daily query limit.")
    )
    with caplog.at_level(logging.INFO, logger="kuberacle.observability.request"):
        resp = _client([], guardrails=guardrails).post(
            "/query", json={"question": "q"}
        )

    assert resp.status_code == 429
    summaries = _summaries(caplog)
    assert len(summaries) == 1
    assert summaries[0].outcome == ctx.OUTCOME_GUARDRAIL_REJECTED
    assert summaries[0].guardrail == "rate_limit"
    assert getattr(summaries[0], "http.status") == 429
