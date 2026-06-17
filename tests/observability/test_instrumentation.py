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
