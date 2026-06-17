"""The per-request summary event.

One structured log line is emitted per request after the response completes. It
carries RED signals, per-stage latency, token usage, estimated cost (with the
reranker as its own line item), RAG-quality outcome, and guardrail signal. Every
log-based metric, dashboard, and alert derives from this single event. It records
metadata only: question length and outcomes, never question or answer text.
"""

import logging

from kuberacle.observability.context import RequestMetrics

logger = logging.getLogger("kuberacle.observability.request")


def build_request_summary(
    metrics: RequestMetrics,
    method: str,
    route: str,
    status: int,
    duration_ms: float,
) -> dict:
    """Build the request-summary payload from accumulated metrics.

    Args:
        metrics: The request's accumulated metrics.
        method: HTTP method.
        route: Request route/path.
        status: HTTP status code.
        duration_ms: Total request duration in milliseconds.

    Returns:
        A flat, JSON-serializable summary dict (no user content).
    """
    cost = dict(metrics.cost_usd)
    cost["total"] = round(metrics.total_cost_usd(), 6)
    return {
        "event": "request_summary",
        "http.method": method,
        "http.route": route,
        "http.status": status,
        "duration_ms": round(duration_ms, 2),
        "outcome": metrics.outcome,
        "guardrail": metrics.guardrail,
        "gate_decision": metrics.gate_decision,
        "question_length": metrics.question_length,
        "cold_start": metrics.cold_start,
        "chunks_retrieved": metrics.chunks_retrieved,
        "citations_count": metrics.citations_count,
        "insufficient_evidence": metrics.insufficient_evidence,
        "rerank_queries": metrics.rerank_queries,
        "stage_ms": metrics.stage_ms,
        "tokens": metrics.tokens,
        "cost_usd": cost,
    }


def emit_request_summary(
    metrics: RequestMetrics,
    method: str,
    route: str,
    status: int,
    duration_ms: float,
) -> None:
    """Emit the request-summary event as a single structured log line.

    Args:
        metrics: The request's accumulated metrics.
        method: HTTP method.
        route: Request route/path.
        status: HTTP status code.
        duration_ms: Total request duration in milliseconds.
    """
    summary = build_request_summary(metrics, method, route, status, duration_ms)
    logger.info("request_summary", extra=summary)
