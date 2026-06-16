"""Per-query cost estimation for LLM and reranker spend.

Pure functions over a :class:`~kuberacle.config.PricingConfig`. Prices live in
``configs/rag.yaml`` (never hard-coded) so estimates stay auditable and can be
updated without touching code.
"""

from kuberacle.config import PricingConfig


def model_token_cost(
    pricing: PricingConfig, input_tokens: int, output_tokens: int
) -> float:
    """Estimate generation-model cost for a single call.

    Used for both the answer generator and the relevance gate, which share the
    generation model. Thinking/reasoning tokens are billed as output tokens and
    are already included in the model's reported output token count.

    Args:
        pricing: Pricing table.
        input_tokens: Prompt (input) token count.
        output_tokens: Completion (output) token count.

    Returns:
        Estimated cost in USD.
    """
    return (
        input_tokens / 1_000_000 * pricing.generation_input_per_1m_usd
        + output_tokens / 1_000_000 * pricing.generation_output_per_1m_usd
    )


def embedding_cost(pricing: PricingConfig, input_tokens: int) -> float:
    """Estimate embedding cost for a single embed call.

    Args:
        pricing: Pricing table.
        input_tokens: Embedded input token count.

    Returns:
        Estimated cost in USD (embeddings have no output tokens).
    """
    return input_tokens / 1_000_000 * pricing.embedding_input_per_1m_usd


def rerank_cost(pricing: PricingConfig, num_queries: int = 1) -> float:
    """Estimate reranker cost for one or more ranking queries.

    The Discovery Engine ranking API bills per query, where one query reranks up
    to 100 records; our candidate pool stays well under that, so a single rerank
    is one billable query.

    Args:
        pricing: Pricing table.
        num_queries: Number of billable ranking queries.

    Returns:
        Estimated cost in USD.
    """
    return num_queries / 1000 * pricing.reranker_per_1k_queries_usd
