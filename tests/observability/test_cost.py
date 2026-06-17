"""Tests for per-query cost estimation."""

from kuberacle.config import PricingConfig
from kuberacle.observability import cost

PRICING = PricingConfig(
    generation_input_per_1m_usd=0.10,
    generation_output_per_1m_usd=0.40,
    embedding_input_per_1m_usd=0.15,
    reranker_per_1k_queries_usd=1.00,
)


def test_model_token_cost_combines_input_and_output():
    """Generation cost sums input and output token prices."""
    result = cost.model_token_cost(PRICING, input_tokens=1_000_000, output_tokens=1_000_000)
    assert result == 0.10 + 0.40


def test_model_token_cost_zero_tokens():
    """No tokens means no cost."""
    assert cost.model_token_cost(PRICING, 0, 0) == 0.0


def test_embedding_cost_input_only():
    """Embedding cost prices input tokens only."""
    assert cost.embedding_cost(PRICING, 1_000_000) == 0.15


def test_rerank_cost_per_query():
    """One rerank is one billable query at the per-1k rate."""
    assert cost.rerank_cost(PRICING, 1) == 0.001
    assert cost.rerank_cost(PRICING, 1000) == 1.00
