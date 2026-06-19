"""Stage-level tracing helpers used by the orchestration layer.

The deep pipeline call sites (generator, gate, reranker, embedder) record token
usage and cost only into the pure :mod:`~kuberacle.observability.context`
accumulator, so they stay decoupled from tracing and trivially testable. This
module is used at the orchestration layer (``qa`` and the hybrid retriever) to
open a Langfuse observation per stage (a real OpenTelemetry span on the shared
provider, so it also appears in Cloud Trace) and to enrich it from the recorded
metrics. All helpers are no-ops when tracing is disabled.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from kuberacle.observability import context as ctx
from kuberacle.observability.tracing import get_langfuse

logger = logging.getLogger(__name__)


class _NoopObservation:
    """Stand-in observation handle used when tracing is disabled."""

    def update(self, **kwargs: Any) -> None:
        """Ignore enrichment when no tracing backend is active."""


def _root_trace_context() -> dict | None:
    """Trace context pointing top-level stages at the per-request root span.

    Returns:
        A ``{"trace_id", "parent_span_id"}`` dict when a request root exists,
        else None (so the observation falls back to the current OTel context).
    """
    metrics = ctx.get_metrics()
    if metrics is None or metrics.trace_id is None:
        return None
    trace_context: dict = {"trace_id": metrics.trace_id}
    if metrics.root_span_id is not None:
        trace_context["parent_span_id"] = metrics.root_span_id
    return trace_context


def _parse_traceparent(traceparent: str | None) -> tuple[str, str] | None:
    """Parse a W3C ``traceparent`` header into its trace id and span id.

    Args:
        traceparent: Header value of the form ``version-traceid-spanid-flags``.

    Returns:
        A ``(trace_id, span_id)`` hex tuple, or None when absent or malformed
        (including the all-zero invalid ids).
    """
    if not traceparent:
        return None
    parts = traceparent.split("-")
    if len(parts) != 4:
        return None
    trace_id, span_id = parts[1], parts[2]
    if len(trace_id) != 32 or len(span_id) != 16:
        return None
    if trace_id == "0" * 32 or span_id == "0" * 16:
        return None
    return trace_id, span_id


def capture_http_trace_context(traceparent: str | None = None) -> None:
    """Record the inbound request's trace context on the active metrics.

    The per-request root reuses this trace id so the Cloud Trace HTTP span and
    the pipeline trace stay unified, and the log formatter uses it to correlate
    the request-summary line (emitted from the streaming threadpool copy). The
    FastAPI server-span context is not propagated into the route coroutine here,
    so the inbound W3C ``traceparent`` header (sent by the web proxy) is the
    primary source; the current OTel span is a fallback for direct callers that
    set one. No-op without metrics or any trace context.

    Args:
        traceparent: The inbound ``traceparent`` header value, when present.
    """
    metrics = ctx.get_metrics()
    if metrics is None:
        return
    parsed = _parse_traceparent(traceparent)
    if parsed is not None:
        metrics.http_trace_id, metrics.http_span_id = parsed
        return
    try:
        from opentelemetry import trace

        span_ctx = trace.get_current_span().get_span_context()
        if span_ctx.is_valid:
            metrics.http_trace_id = format(span_ctx.trace_id, "032x")
            metrics.http_span_id = format(span_ctx.span_id, "016x")
    except Exception:
        logger.debug("Capturing HTTP trace context failed", exc_info=True)


def start_request_root(name: str, user_input: Any = None) -> Any:
    """Open the single root observation that all stages of a request nest under.

    Creates one Langfuse trace per request and records its id and span id in the
    active metrics so the top-level stages (gate, retrieval, generation) attach
    to it via :func:`_root_trace_context`. Returns None when tracing is off.

    Reuses the inbound HTTP trace id captured by
    :func:`capture_http_trace_context` (so the Langfuse trace and the Cloud Trace
    HTTP span stay unified); mints a fresh trace only when none was captured.

    Args:
        name: Root observation name (e.g. ``query``).
        user_input: Request input to record on the trace (e.g. the question).

    Returns:
        The root observation handle, or None.
    """
    client = get_langfuse()
    metrics = ctx.get_metrics()
    if client is None or metrics is None:
        return None
    try:
        if metrics.http_trace_id is not None:
            trace_id = metrics.http_trace_id
            trace_context = {"trace_id": trace_id}
            if metrics.http_span_id is not None:
                trace_context["parent_span_id"] = metrics.http_span_id
        else:
            trace_id = client.create_trace_id()
            trace_context = {"trace_id": trace_id}
        root = client.start_observation(
            name=name,
            as_type="span",
            input=user_input,
            trace_context=trace_context,
        )
        metrics.trace_id = trace_id
        metrics.root_span_id = root.id
        return root
    except Exception:
        logger.debug("Langfuse request root failed", exc_info=True)
        return None


def finalize_request_root(observation: Any, output: Any = None) -> None:
    """Record the request output and cost breakdown on the root, then end it.

    The full cost breakdown (including the reranker's Discovery Engine cost) is
    attached as trace metadata, not as Langfuse ``cost_details``: metadata is
    visible and searchable on the trace without polluting Langfuse's LLM-only
    cost views. The authoritative cost record remains the ``request_summary`` log.

    Args:
        observation: Handle from :func:`start_request_root`, or None.
        output: Request output to record on the trace (e.g. the answer).
    """
    if observation is None:
        return
    metadata = None
    metrics = ctx.get_metrics()
    if metrics is not None:
        cost = dict(metrics.cost_usd)
        cost["total"] = round(metrics.total_cost_usd(), 6)
        metadata = {"cost_usd": cost, "rerank_queries": metrics.rerank_queries}
    try:
        observation.update(output=output, metadata=metadata)
        observation.end()
    except Exception:
        logger.debug("Finalizing request root failed", exc_info=True)


@contextmanager
def observe_stage(
    name: str,
    as_type: str = "span",
    model: str | None = None,
    user_input: Any = None,
    root: bool = False,
) -> Iterator[Any]:
    """Time a pipeline stage and, when tracing is on, open a Langfuse observation.

    The stage is always timed into the active request metrics. When a Langfuse
    client is configured, the stage is also opened as an observation. Top-level
    stages pass ``root=True`` so they attach to the per-request root trace;
    sub-stages omit it and nest under the current observation. Observation errors
    never break the pipeline.

    Args:
        name: Stage name (e.g. ``gate``, ``semantic``, ``rerank``, ``generation``).
        as_type: Langfuse observation type (``span``, ``generation``,
            ``embedding``, ``retriever``, ...).
        model: Model id, for generation/embedding observations.
        user_input: Optional input payload to record on the observation.
        root: Whether this is a top-level stage that attaches to the request root.

    Yields:
        An observation handle exposing ``update(**kwargs)``; a no-op when tracing
        is disabled.
    """
    client = get_langfuse()
    with ctx.stage_timer(name):
        if client is None:
            yield _NoopObservation()
            return
        try:
            with client.start_as_current_observation(
                name=name,
                as_type=as_type,
                model=model,
                input=user_input,
                trace_context=_root_trace_context() if root else None,
            ) as observation:
                yield observation
        except Exception:
            logger.debug("Langfuse observation %r failed", name, exc_info=True)
            yield _NoopObservation()


def link_prompt(prompt_ref: Any) -> None:
    """Link a managed Langfuse prompt to the current generation observation.

    No-op when there is no prompt object or no active Langfuse client.

    Args:
        prompt_ref: A Langfuse prompt object, or None.
    """
    if prompt_ref is None:
        return
    client = get_langfuse()
    if client is None:
        return
    try:
        client.update_current_generation(prompt=prompt_ref)
    except Exception:
        logger.debug("Linking managed prompt failed", exc_info=True)


def _usage_and_cost(stage: str) -> tuple[dict | None, dict | None]:
    """Read recorded token usage and cost for a stage from the active metrics.

    Args:
        stage: Stage key used when recording usage (e.g. ``generation``, ``gate``).

    Returns:
        A ``(usage_details, cost_details)`` tuple, each None when unavailable.
    """
    metrics = ctx.get_metrics()
    if metrics is None:
        return None, None
    usage: dict[str, int] = {}
    if f"{stage}_in" in metrics.tokens:
        usage["input"] = metrics.tokens[f"{stage}_in"]
    if f"{stage}_out" in metrics.tokens:
        usage["output"] = metrics.tokens[f"{stage}_out"]
    if usage:
        usage["total"] = usage.get("input", 0) + usage.get("output", 0)
    cost = {"total": metrics.cost_usd[stage]} if stage in metrics.cost_usd else None
    return (usage or None), cost


def enrich_llm_observation(
    observation: Any, stage: str, output: Any = None
) -> None:
    """Set recorded token usage, cost, and output on an LLM observation.

    Reads the values the call site recorded into the active request metrics and
    attaches them to the (current) Langfuse observation. No-op without metrics.

    Args:
        observation: Observation handle returned by :func:`observe_stage`.
        stage: Stage key used when recording usage (e.g. ``generation``, ``gate``,
            ``embed``).
        output: Optional output payload to record (e.g. the answer text).
    """
    usage, cost_details = _usage_and_cost(stage)
    try:
        observation.update(
            output=output, usage_details=usage, cost_details=cost_details
        )
    except Exception:
        logger.debug("Observation enrichment for %r failed", stage, exc_info=True)


def start_generation(
    name: str, model: str | None = None, user_input: Any = None
) -> Any:
    """Start a non-current generation observation for a streamed stage.

    A streamed stage yields across Starlette's per-chunk context boundaries, so
    a current-context span (``start_as_current_observation``) cannot be detached
    safely (it is entered and exited in different contexts). This starts a manual
    observation that attaches to the per-request root trace but is not the current
    OpenTelemetry context, avoiding the detach error. Finalize it with
    :func:`finalize_generation`. Returns None when tracing is disabled.

    Args:
        name: Stage name.
        model: Model id for the generation observation.
        user_input: Optional input payload to record (e.g. the question).

    Returns:
        A Langfuse observation handle, or None.
    """
    client = get_langfuse()
    if client is None:
        return None
    try:
        return client.start_observation(
            name=name,
            as_type="generation",
            model=model,
            input=user_input,
            trace_context=_root_trace_context(),
        )
    except Exception:
        logger.debug("Langfuse generation %r failed", name, exc_info=True)
        return None


def finalize_generation(
    observation: Any, stage: str, output: Any = None, prompt: Any = None
) -> None:
    """Enrich a manual generation observation from metrics and end it.

    Args:
        observation: Handle from :func:`start_generation`, or None.
        stage: Stage key used when recording usage (e.g. ``generation``).
        output: Output payload to record (e.g. the answer text).
        prompt: Optional managed prompt object to link.
    """
    if observation is None:
        return
    usage, cost_details = _usage_and_cost(stage)
    try:
        observation.update(
            output=output,
            usage_details=usage,
            cost_details=cost_details,
            prompt=prompt,
        )
        observation.end()
    except Exception:
        logger.debug("Finalizing generation observation failed", exc_info=True)
