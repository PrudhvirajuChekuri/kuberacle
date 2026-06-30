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

    def verify_turnstile(self, client_ip, turnstile_token):
        del client_ip, turnstile_token
        if self._error is not None and self._error.status_code == 403:
            raise self._error

    def check_ip_rate_limit(self, client_ip):
        del client_ip
        if self._error is not None and self._error.status_code != 403:
            raise self._error

    def charge_ip(self, client_ip):
        return None

    def charge_ip_and_global(self, client_ip):
        return None


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


class StreamingFakeQA:
    """Stub that records outcome AFTER yielding a token, like the real pipeline.

    This guards the metrics-binding contract: the real generator records token
    usage and the final outcome mid/post-stream, which is only captured if the
    metrics context survives across the streaming generator's chunk boundaries.
    """

    def ask_stream(self, question, top_k=None):
        del question, top_k
        yield AnswerDelta("first ")
        yield AnswerDelta("token")
        # Recorded only after the first token has been streamed.
        ctx.update_metrics(outcome=ctx.OUTCOME_ANSWERED, citations_count=3)
        yield QAResult("first token", [], [])


def test_metrics_recorded_after_first_token_are_captured(caplog):
    """Outcome recorded mid/post-stream survives the streaming context boundary."""
    app = create_app()
    app.state.qa_system = StreamingFakeQA()
    app.state.pricing = PRICING
    with caplog.at_level(logging.INFO, logger="kuberacle.observability.request"):
        resp = TestClient(app).post("/query", json={"question": "q"})

    assert resp.status_code == 200
    summaries = _summaries(caplog)
    assert len(summaries) == 1
    assert summaries[0].outcome == ctx.OUTCOME_ANSWERED
    assert summaries[0].citations_count == 3


def test_no_summary_without_pricing(caplog):
    """With no pricing wired (lifespan skipped), no summary is emitted."""
    events = [AnswerDelta("x"), QAResult("x", [], [])]
    with caplog.at_level(logging.INFO, logger="kuberacle.observability.request"):
        resp = _client(events, pricing=None).post("/query", json={"question": "q"})

    assert resp.status_code == 200
    assert _summaries(caplog) == []


def test_traceparent_visible_to_handler_under_fastapi_instrumentation():
    """FastAPIInstrumentor must not hide the inbound traceparent from handlers.

    This is the prod-specific link: the live API runs with FastAPIInstrumentor
    active. The fix reads ``request.headers.get("traceparent")`` in the handler,
    so prove the instrumentation (which parses that header for its own span)
    leaves it readable by the route.
    """
    from fastapi import FastAPI, Request
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    app = FastAPI()

    @app.get("/h")
    async def handler(request: Request):
        return {"tp": request.headers.get("traceparent")}

    FastAPIInstrumentor.instrument_app(app)
    try:
        tp = "00-0123456789abcdef0123456789abcdef-fedcba9876543210-01"
        resp = TestClient(app).get("/h", headers={"traceparent": tp})
        assert resp.json()["tp"] == tp
    finally:
        FastAPIInstrumentor.uninstrument_app(app)


class _FakeRoot:
    id = "root-span"

    def update(self, **kwargs):
        pass

    def end(self):
        pass


class _FakeLangfuse:
    """Records the trace_context the per-request root observation is opened with."""

    def __init__(self):
        self.root_trace_context = None

    def create_trace_id(self):
        return "minted-trace-id"

    def start_observation(self, *, name, as_type, input, trace_context):
        if name == "query":
            self.root_trace_context = trace_context
        return _FakeRoot()

    def flush(self):
        pass


def test_traceparent_header_unifies_request_trace(monkeypatch):
    """End-to-end: the inbound traceparent becomes the pipeline root's trace id.

    Drives the real app via TestClient with a Langfuse client wired in, and
    asserts the per-request root observation is opened on the header's trace id
    (and parented to its span) rather than a freshly minted trace.
    """
    from kuberacle.observability import tracing

    fake = _FakeLangfuse()
    monkeypatch.setattr(tracing, "_HANDLES", tracing.TracingHandles(langfuse=fake))

    events = [AnswerDelta("Pods run."), QAResult("Pods run.", [], [])]
    trace_id = "0123456789abcdef0123456789abcdef"
    span_id = "fedcba9876543210"
    resp = _client(events).post(
        "/query",
        json={"question": "q"},
        headers={"traceparent": f"00-{trace_id}-{span_id}-01"},
    )

    assert resp.status_code == 200
    assert fake.root_trace_context == {
        "trace_id": trace_id,
        "parent_span_id": span_id,
    }


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
