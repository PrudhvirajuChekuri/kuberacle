"""Tests for the request-summary event."""

from kuberacle.config import PricingConfig
from kuberacle.observability import context as ctx
from kuberacle.observability.events import build_request_summary

PRICING = PricingConfig(0.10, 0.40, 0.15, 1.00)


def _metrics_answered():
    metrics = ctx.RequestMetrics(pricing=PRICING)
    metrics.question_length = 42
    metrics.outcome = ctx.OUTCOME_ANSWERED
    metrics.gate_decision = ctx.GATE_IN_SCOPE
    metrics.chunks_retrieved = 5
    metrics.citations_count = 3
    metrics.stage_ms = {"gate": 50.0, "rerank": 80.0, "generation": 900.0}
    metrics.tokens = {"generation_in": 3000, "generation_out": 600}
    metrics.cost_usd = {"generation": 0.00054, "rerank": 0.001}
    metrics.rerank_queries = 1
    return metrics


def test_summary_contains_red_and_rag_fields():
    """Summary carries RED, stage latency, cost, and RAG-quality fields."""
    summary = build_request_summary(
        _metrics_answered(), "POST", "/query", 200, 1050.0
    )
    assert summary["event"] == "request_summary"
    assert summary["http.method"] == "POST"
    assert summary["http.status"] == 200
    assert summary["duration_ms"] == 1050.0
    assert summary["outcome"] == "answered"
    assert summary["gate_decision"] == "in_scope"
    assert summary["question_length"] == 42
    assert summary["citations_count"] == 3
    assert summary["chunks_retrieved"] == 5
    assert summary["stage_ms"]["generation"] == 900.0
    assert summary["rerank_queries"] == 1


def test_summary_cost_includes_total():
    """Cost block adds a summed total across stage lines."""
    summary = build_request_summary(
        _metrics_answered(), "POST", "/query", 200, 1050.0
    )
    assert summary["cost_usd"]["total"] == round(0.00054 + 0.001, 6)


def test_summary_omits_user_content():
    """The summary schema is a fixed metadata-only key set (no content fields)."""
    summary = build_request_summary(
        _metrics_answered(), "POST", "/query", 200, 1050.0
    )
    assert set(summary.keys()) == {
        "event",
        "http.method",
        "http.route",
        "http.status",
        "duration_ms",
        "outcome",
        "guardrail",
        "gate_decision",
        "question_length",
        "cold_start",
        "chunks_retrieved",
        "citations_count",
        "insufficient_evidence",
        "cache_hit",
        "cache_backend",
        "saved_cost_estimate",
        "rerank_queries",
        "stage_ms",
        "tokens",
        "cost_usd",
    }
    assert "question_text" not in summary
    assert "answer_text" not in summary
