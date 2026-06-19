"""Tests for the structured logging owner."""

import json
import logging

from kuberacle.config import ObservabilityConfig, PricingConfig
from kuberacle.observability import context as ctx
from kuberacle.observability.logging import configure_logging


def _json_config():
    return ObservabilityConfig(
        service_name="kuberacle-api", log_level="INFO", log_format="json",
        trace_sample_ratio=1.0,
    )


def test_json_logging_emits_structured_line(capsys):
    """JSON format emits a single parseable line with promoted extra fields."""
    configure_logging(_json_config())
    logging.getLogger("kuberacle.test").info(
        "request_summary", extra={"event": "request_summary", "duration_ms": 12.5}
    )
    out = capsys.readouterr().out.strip()
    record = json.loads(out)
    assert record["severity"] == "INFO"
    assert record["message"] == "request_summary"
    assert record["event"] == "request_summary"
    assert record["duration_ms"] == 12.5
    assert "logger" in record and "timestamp" in record


def test_json_logging_includes_exception(capsys):
    """Exceptions are serialized into the structured line."""
    configure_logging(_json_config())
    try:
        raise ValueError("boom")
    except ValueError:
        logging.getLogger("kuberacle.test").exception("failed")
    record = json.loads(capsys.readouterr().out.strip())
    assert record["severity"] == "ERROR"
    assert "boom" in record["exception"]


def test_text_format_is_plain(capsys):
    """Text format produces a non-JSON readable line."""
    configure_logging(
        ObservabilityConfig("kuberacle-api", "INFO", "text", 1.0)
    )
    logging.getLogger("kuberacle.test").info("hello")
    out = capsys.readouterr().out.strip()
    assert "hello" in out
    assert not out.startswith("{")


def test_trace_correlation_falls_back_to_metrics(capsys):
    """A log line with no current span correlates via the captured HTTP trace id.

    This is the request-summary case: it is emitted from the streaming threadpool
    context where the FastAPI span is not current, so the formatter must fall back
    to the trace id captured on the request metrics.
    """
    configure_logging(_json_config(), gcp_project="proj-x")
    metrics = ctx.RequestMetrics(pricing=PricingConfig(0.10, 0.40, 0.15, 1.00))
    metrics.http_trace_id = "0123456789abcdef0123456789abcdef"
    metrics.http_span_id = "fedcba9876543210"
    token = ctx.set_metrics(metrics)
    try:
        logging.getLogger("kuberacle.test").info("request_summary")
    finally:
        ctx.reset_metrics(token)
    record = json.loads(capsys.readouterr().out.strip())
    assert record["logging.googleapis.com/trace"] == (
        "projects/proj-x/traces/0123456789abcdef0123456789abcdef"
    )
    assert record["logging.googleapis.com/spanId"] == "fedcba9876543210"


def test_no_trace_fields_without_span_or_metrics(capsys):
    """With neither a current span nor metrics, no trace fields are emitted."""
    configure_logging(_json_config(), gcp_project="proj-x")
    logging.getLogger("kuberacle.test").info("startup")
    record = json.loads(capsys.readouterr().out.strip())
    assert "logging.googleapis.com/trace" not in record


def test_configure_logging_is_idempotent(capsys):
    """Reconfiguring does not stack handlers (one line per log call)."""
    configure_logging(_json_config())
    configure_logging(_json_config())
    logging.getLogger("kuberacle.test").info("once")
    lines = [ln for ln in capsys.readouterr().out.strip().splitlines() if ln]
    assert len(lines) == 1
