"""Single owner of serving-layer logging configuration.

Configures the root logger once at API startup: JSON to stdout in production
(parsed by Cloud Logging into structured fields, with trace correlation so logs
link to their Cloud Trace span) or plain text for local readability. The API
process never configured logging before this module existed; the CLI keeps its
own ``basicConfig`` and is unaffected.

Content policy: this formatter emits log metadata only. Call sites must never log
question or answer text; the request-summary event records lengths and outcomes,
not user content.
"""

import json
import logging
import sys

from kuberacle.config import ObservabilityConfig

# Standard LogRecord attributes, used to detect caller-supplied extra fields.
_RESERVED = set(
    logging.makeLogRecord({}).__dict__.keys()
) | {"message", "asctime", "taskName"}


def _trace_fields(gcp_project: str) -> dict:
    """Return Cloud Logging trace-correlation fields for the active span.

    Args:
        gcp_project: GCP project id used to qualify the trace resource name.

    Returns:
        A dict with ``logging.googleapis.com/trace`` and ``.../spanId`` when a
        valid span is in context, otherwise empty.
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context()
        if not ctx.is_valid:
            return {}
        fields = {"logging.googleapis.com/spanId": format(ctx.span_id, "016x")}
        if gcp_project:
            trace_hex = format(ctx.trace_id, "032x")
            fields["logging.googleapis.com/trace"] = (
                f"projects/{gcp_project}/traces/{trace_hex}"
            )
        return fields
    except Exception:
        return {}


class _JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON for Cloud Logging.

    Args:
        gcp_project: GCP project id for trace-resource qualification.
    """

    def __init__(self, gcp_project: str = "") -> None:
        super().__init__()
        self._gcp_project = gcp_project

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a record to a JSON line.

        Args:
            record: Log record to format.

        Returns:
            JSON-encoded log line.
        """
        payload: dict = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # Promote caller-supplied structured fields (logger.info(..., extra={...})).
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        payload.update(_trace_fields(self._gcp_project))
        return json.dumps(payload, default=str)


def configure_logging(config: ObservabilityConfig, gcp_project: str = "") -> None:
    """Configure root logging for the serving layer (idempotent).

    Args:
        config: Observability config carrying the level and format.
        gcp_project: GCP project id for trace-correlation fields.
    """
    root = logging.getLogger()
    root.setLevel(config.log_level.upper())

    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    if config.log_format.lower() == "json":
        handler.setFormatter(_JsonFormatter(gcp_project))
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
    root.addHandler(handler)

    # Keep access-log and HTTP-client noise out of the structured stream.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
