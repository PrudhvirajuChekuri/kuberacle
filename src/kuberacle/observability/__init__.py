"""Observability layer: structured logging, tracing, cost, and request metrics.

This package is the single owner of the serving layer's observability. It is
additive and fail-safe: when ``OBSERVABILITY_ENABLED`` is unset (the default for
local development, tests, and the CLI) tracing and Langfuse export are skipped,
structured logging falls back to plain text, and the per-request recording hooks
become no-ops. Nothing here changes the SSE contract or the answer pipeline's
behavior.
"""
