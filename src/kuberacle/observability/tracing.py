"""OpenTelemetry tracing setup for the two observability planes.

A single global ``TracerProvider`` feeds both planes:

* the operational plane, via the Cloud Trace span exporter plus FastAPI and
  outbound-``requests`` auto-instrumentation (HTTP and downstream-call spans);
* the LLM/product plane, via the Langfuse client, which attaches its own span
  processor to the same provider so pipeline-stage observations land in Langfuse
  while still appearing in the unified Cloud Trace.

Everything here is gated by ``OBSERVABILITY_ENABLED`` and is fail-safe: any setup
error degrades to logging-only rather than breaking startup, and export happens
on a per-request ``force_flush`` because Cloud Run freezes instances between
requests (a background batch flush could otherwise lose spans).
"""

import logging
from dataclasses import dataclass
from typing import Any

from kuberacle.config import ObservabilityConfig
from kuberacle.observability.settings import ObservabilitySettings

logger = logging.getLogger(__name__)


@dataclass
class TracingHandles:
    """Handles to the active tracing backends.

    Attributes:
        provider: The global tracer provider, or None when tracing is disabled.
        langfuse: The Langfuse client, or None when Langfuse export is off.
    """

    provider: Any = None
    langfuse: Any = None

    @property
    def enabled(self) -> bool:
        """Whether any tracing backend is active."""
        return self.provider is not None

    def force_flush(self) -> None:
        """Flush pending spans to both backends. Never raises."""
        if self.provider is not None:
            try:
                self.provider.force_flush()
            except Exception:
                logger.warning("Cloud Trace force_flush failed", exc_info=True)
        if self.langfuse is not None:
            try:
                self.langfuse.flush()
            except Exception:
                logger.warning("Langfuse flush failed", exc_info=True)

    def shutdown(self) -> None:
        """Flush and shut down both backends. Never raises."""
        if self.langfuse is not None:
            try:
                self.langfuse.shutdown()
            except Exception:
                logger.warning("Langfuse shutdown failed", exc_info=True)
        if self.provider is not None:
            try:
                self.provider.shutdown()
            except Exception:
                logger.warning("Tracer provider shutdown failed", exc_info=True)


# Module-level handles, set by configure_tracing and read by the instrumentation
# helpers so call sites need not thread the client through every signature.
_HANDLES = TracingHandles()


def get_langfuse() -> Any:
    """Return the active Langfuse client, or None when Langfuse export is off."""
    return _HANDLES.langfuse


def configure_tracing(
    settings: ObservabilitySettings,
    config: ObservabilityConfig,
    gcp_project: str,
    app: Any = None,
) -> TracingHandles:
    """Configure the global tracer provider and backends (idempotent per process).

    Args:
        settings: Env-driven observability settings (enable flag, Langfuse keys).
        config: Non-secret observability config (service name, sample ratio).
        gcp_project: GCP project id for the Cloud Trace exporter.
        app: FastAPI app to auto-instrument, when provided.

    Returns:
        The active :class:`TracingHandles` (empty when disabled or on failure).
    """
    global _HANDLES
    if not settings.enabled:
        return _HANDLES

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import (
            ParentBased,
            TraceIdRatioBased,
        )

        resource = Resource.create(
            {
                "service.name": config.service_name,
                "service.version": settings.service_version,
                "deployment.environment": settings.environment,
            }
        )
        provider = TracerProvider(
            resource=resource,
            sampler=ParentBased(TraceIdRatioBased(config.trace_sample_ratio)),
        )
        provider.add_span_processor(
            BatchSpanProcessor(CloudTraceSpanExporter(project_id=gcp_project))
        )
        trace.set_tracer_provider(provider)
        _instrument_libraries(provider, app)

        langfuse = _build_langfuse(settings, provider)
        _HANDLES = TracingHandles(provider=provider, langfuse=langfuse)
        logger.info(
            "Tracing enabled (cloud_trace=on, langfuse=%s)",
            "on" if langfuse is not None else "off",
        )
    except Exception:
        logger.warning(
            "Tracing setup failed; continuing with logging only", exc_info=True
        )
        _HANDLES = TracingHandles()
    return _HANDLES


def _instrument_libraries(provider: Any, app: Any) -> None:
    """Auto-instrument FastAPI and outbound requests on the given provider."""
    from opentelemetry.instrumentation.requests import RequestsInstrumentor

    RequestsInstrumentor().instrument(tracer_provider=provider)
    if app is not None:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)


def _build_langfuse(settings: ObservabilitySettings, provider: Any) -> Any:
    """Build a Langfuse client bound to the shared provider, or None.

    Args:
        settings: Observability settings carrying the Langfuse keys and host.
        provider: The shared tracer provider Langfuse should export through.

    Returns:
        A configured Langfuse client, or None when keys are absent or setup
        fails (Cloud Trace still works in that case).
    """
    if not settings.langfuse_enabled:
        return None
    try:
        from langfuse import Langfuse

        return Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            release=settings.service_version,
            environment=settings.environment,
            tracer_provider=provider,
        )
    except Exception:
        logger.warning(
            "Langfuse setup failed; continuing with Cloud Trace only",
            exc_info=True,
        )
        return None
