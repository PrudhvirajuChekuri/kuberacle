"""Tests for the per-request metrics accumulator."""

from kuberacle.config import PricingConfig
from kuberacle.observability import context as ctx

PRICING = PricingConfig(0.10, 0.40, 0.15, 1.00)


def _bind():
    """Bind a fresh metrics object and return (metrics, token)."""
    metrics = ctx.RequestMetrics(pricing=PRICING)
    return metrics, ctx.set_metrics(metrics)


def test_record_helpers_no_op_without_context():
    """Recording without a bound context must not raise."""
    assert ctx.get_metrics() is None
    ctx.record_model_usage("generation", 100, 50)
    ctx.record_embedding_usage(20)
    ctx.record_rerank()
    with ctx.stage_timer("gate"):
        pass
    assert ctx.get_metrics() is None


def test_stage_timer_records_duration():
    """stage_timer records a non-negative duration into metrics."""
    metrics, token = _bind()
    try:
        with ctx.stage_timer("rerank"):
            pass
        assert "rerank" in metrics.stage_ms
        assert metrics.stage_ms["rerank"] >= 0.0
    finally:
        ctx.reset_metrics(token)


def test_record_model_usage_sets_tokens_and_cost():
    """Generation usage records token counts and the matching cost."""
    metrics, token = _bind()
    try:
        ctx.record_model_usage("generation", 1_000_000, 1_000_000)
        assert metrics.tokens["generation_in"] == 1_000_000
        assert metrics.tokens["generation_out"] == 1_000_000
        assert metrics.cost_usd["generation"] == 0.10 + 0.40
    finally:
        ctx.reset_metrics(token)


def test_record_embedding_and_rerank_costs():
    """Embedding and rerank record their own cost lines."""
    metrics, token = _bind()
    try:
        ctx.record_embedding_usage(1_000_000)
        ctx.record_rerank()
        assert metrics.cost_usd["embed"] == 0.15
        assert metrics.cost_usd["rerank"] == 0.001
        assert metrics.rerank_queries == 1
    finally:
        ctx.reset_metrics(token)


def test_total_cost_sums_all_stages():
    """Total cost sums every recorded stage."""
    metrics, token = _bind()
    try:
        ctx.record_model_usage("gate", 1_000_000, 0)
        ctx.record_model_usage("generation", 0, 1_000_000)
        ctx.record_embedding_usage(1_000_000)
        ctx.record_rerank()
        expected = 0.10 + 0.40 + 0.15 + 0.001
        assert round(metrics.total_cost_usd(), 6) == round(expected, 6)
    finally:
        ctx.reset_metrics(token)
