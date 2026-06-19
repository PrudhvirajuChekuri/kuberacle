"""Tests for stage instrumentation helpers (tracing-disabled paths)."""

from kuberacle.config import PricingConfig
from kuberacle.observability import context as ctx
from kuberacle.observability import instrumentation as instr

PRICING = PricingConfig(0.10, 0.40, 0.15, 1.00)


def test_observe_stage_times_without_tracing():
    """With no Langfuse client, observe_stage still times the stage."""
    metrics = ctx.RequestMetrics(pricing=PRICING)
    token = ctx.set_metrics(metrics)
    try:
        with instr.observe_stage("generation", as_type="generation") as obs:
            obs.update(output="ignored")  # no-op handle, must not raise
        assert "generation" in metrics.stage_ms
    finally:
        ctx.reset_metrics(token)


def test_observe_stage_no_metrics_no_error():
    """observe_stage is safe when neither tracing nor metrics are active."""
    with instr.observe_stage("gate") as obs:
        obs.update(output="x")
    assert ctx.get_metrics() is None


def test_enrich_llm_observation_no_op_without_metrics():
    """Enrichment without an active context does nothing and does not raise."""
    instr.enrich_llm_observation(instr._NoopObservation(), "generation", output="x")
    assert ctx.get_metrics() is None


def test_capture_http_trace_context_no_op_without_metrics():
    """Capturing trace context is safe with no active request metrics."""
    instr.capture_http_trace_context()  # must not raise
    assert ctx.get_metrics() is None


def test_capture_http_trace_context_no_span_leaves_ids_unset():
    """With no traceparent and no current OTel span, the HTTP ids stay unset."""
    metrics = ctx.RequestMetrics(pricing=PRICING)
    token = ctx.set_metrics(metrics)
    try:
        instr.capture_http_trace_context()
        assert metrics.http_trace_id is None
        assert metrics.http_span_id is None
    finally:
        ctx.reset_metrics(token)


def test_capture_http_trace_context_parses_traceparent():
    """A valid inbound traceparent populates the HTTP trace and span ids."""
    metrics = ctx.RequestMetrics(pricing=PRICING)
    token = ctx.set_metrics(metrics)
    try:
        instr.capture_http_trace_context(
            "00-0123456789abcdef0123456789abcdef-fedcba9876543210-01"
        )
        assert metrics.http_trace_id == "0123456789abcdef0123456789abcdef"
        assert metrics.http_span_id == "fedcba9876543210"
    finally:
        ctx.reset_metrics(token)


def test_capture_http_trace_context_ignores_malformed_traceparent():
    """A malformed traceparent is ignored and leaves the ids unset."""
    metrics = ctx.RequestMetrics(pricing=PRICING)
    token = ctx.set_metrics(metrics)
    try:
        instr.capture_http_trace_context("not-a-valid-traceparent")
        assert metrics.http_trace_id is None
    finally:
        ctx.reset_metrics(token)


def test_parse_traceparent_rejects_invalid():
    """The traceparent parser rejects absent, malformed, and all-zero ids."""
    assert instr._parse_traceparent(None) is None
    assert instr._parse_traceparent("") is None
    assert instr._parse_traceparent("00-tooshort-tooshort-01") is None
    assert (
        instr._parse_traceparent(f"00-{'0' * 32}-{'0' * 16}-01") is None
    )
    assert instr._parse_traceparent(
        "00-0123456789abcdef0123456789abcdef-fedcba9876543210-01"
    ) == ("0123456789abcdef0123456789abcdef", "fedcba9876543210")


class _FakeObservation:
    def __init__(self, trace_context):
        self.id = "root-span-id"
        self.trace_context = trace_context

    def update(self, **kwargs):
        pass

    def end(self):
        pass


class _FakeLangfuse:
    def __init__(self):
        self.last_trace_context = None

    def create_trace_id(self):
        return "minted-trace-id"

    def start_observation(self, *, name, as_type, input, trace_context):
        self.last_trace_context = trace_context
        return _FakeObservation(trace_context)


def test_start_request_root_reuses_captured_http_trace(monkeypatch):
    """The per-request root reuses the captured HTTP trace id and span as parent."""
    fake = _FakeLangfuse()
    monkeypatch.setattr(instr, "get_langfuse", lambda: fake)
    metrics = ctx.RequestMetrics(pricing=PRICING)
    metrics.http_trace_id = "http-trace-id"
    metrics.http_span_id = "http-span-id"
    token = ctx.set_metrics(metrics)
    try:
        instr.start_request_root("query", user_input="q")
    finally:
        ctx.reset_metrics(token)
    assert fake.last_trace_context == {
        "trace_id": "http-trace-id",
        "parent_span_id": "http-span-id",
    }
    assert metrics.trace_id == "http-trace-id"
    assert metrics.root_span_id == "root-span-id"


def test_start_request_root_mints_trace_when_no_http_context(monkeypatch):
    """Without a captured HTTP trace, the root mints a fresh trace id."""
    fake = _FakeLangfuse()
    monkeypatch.setattr(instr, "get_langfuse", lambda: fake)
    metrics = ctx.RequestMetrics(pricing=PRICING)
    token = ctx.set_metrics(metrics)
    try:
        instr.start_request_root("query")
    finally:
        ctx.reset_metrics(token)
    assert fake.last_trace_context == {"trace_id": "minted-trace-id"}
    assert metrics.trace_id == "minted-trace-id"
