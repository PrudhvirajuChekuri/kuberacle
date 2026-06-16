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


@contextmanager
def observe_stage(
    name: str,
    as_type: str = "span",
    model: str | None = None,
    user_input: Any = None,
) -> Iterator[Any]:
    """Time a pipeline stage and, when tracing is on, open a Langfuse observation.

    The stage is always timed into the active request metrics. When a Langfuse
    client is configured, the stage is also opened as an observation (an OTel
    span on the shared provider). Observation errors never break the pipeline.

    Args:
        name: Stage name (e.g. ``gate``, ``semantic``, ``rerank``, ``generation``).
        as_type: Langfuse observation type (``span``, ``generation``,
            ``embedding``, ``retriever``, ...).
        model: Model id, for generation/embedding observations.
        user_input: Optional input payload to record on the observation.

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
                name=name, as_type=as_type, model=model, input=user_input
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


def start_generation(name: str, model: str | None = None) -> Any:
    """Start a non-current generation observation for a streamed stage.

    A streamed stage yields across Starlette's per-chunk context boundaries, so
    a current-context span (``start_as_current_observation``) cannot be detached
    safely (it is entered and exited in different contexts). This starts a manual
    observation that still nests in the trace but is not attached as the current
    OpenTelemetry context, avoiding the detach error. Finalize it with
    :func:`finalize_generation`. Returns None when tracing is disabled.

    Args:
        name: Stage name.
        model: Model id for the generation observation.

    Returns:
        A Langfuse observation handle, or None.
    """
    client = get_langfuse()
    if client is None:
        return None
    try:
        return client.start_observation(name=name, as_type="generation", model=model)
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
