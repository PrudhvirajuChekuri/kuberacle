"""Per-request metrics accumulator shared across the pipeline.

A single mutable :class:`RequestMetrics` is created per request and stored in a
``contextvars.ContextVar``. Deep call sites (generator, gate, reranker, retrieval
stages) record timings, token usage, and cost through the module-level helpers,
which are no-ops when no request context is active (CLI, tests, direct library
use). The API request handler reads the accumulated metrics to emit the single
request-summary event.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from kuberacle.config import PricingConfig
from kuberacle.observability import cost

# Outcome labels for the request-summary event.
OUTCOME_ANSWERED = "answered"
OUTCOME_GATE_ABSTAINED = "gate_abstained"
OUTCOME_NO_RETRIEVAL = "no_retrieval"
OUTCOME_UNVERIFIED = "unverified"
OUTCOME_GUARDRAIL_REJECTED = "guardrail_rejected"
OUTCOME_ERROR = "error"

# Gate decision labels.
GATE_IN_SCOPE = "in_scope"
GATE_OUT_OF_SCOPE = "out_of_scope"
GATE_SKIPPED = "skipped"


@dataclass
class RequestMetrics:
    """Mutable per-request observability accumulator.

    Attributes:
        pricing: Price table used to convert token usage into cost.
        question_length: Character length of the question (never the text).
        cold_start: Whether this request was served by a freshly started worker.
        outcome: Terminal outcome label (one of the ``OUTCOME_*`` constants).
        guardrail: Guardrail rejection reason, or ``none``.
        gate_decision: Relevance gate decision (one of the ``GATE_*`` constants).
        chunks_retrieved: Number of chunks returned by retrieval.
        citations_count: Number of validated citations on the answer.
        insufficient_evidence: Whether the answer failed citation validation.
        stage_ms: Wall-clock duration per pipeline stage, in milliseconds.
        tokens: Token counts keyed by ``<stage>_in`` / ``<stage>_out``.
        cost_usd: Estimated cost in USD keyed by stage.
        rerank_queries: Number of billable reranking queries issued.
        trace_id: Langfuse/OTel trace id for this request, when tracing is on.
        root_span_id: Span id of the per-request root observation, used to nest
            the top-level pipeline stages under a single trace.
    """

    pricing: PricingConfig
    question_length: int = 0
    cold_start: bool = False
    outcome: str = OUTCOME_ERROR
    guardrail: str = "none"
    gate_decision: str = GATE_SKIPPED
    chunks_retrieved: int = 0
    citations_count: int = 0
    insufficient_evidence: bool = False
    stage_ms: dict[str, float] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)
    cost_usd: dict[str, float] = field(default_factory=dict)
    rerank_queries: int = 0
    trace_id: str | None = None
    root_span_id: str | None = None

    def total_cost_usd(self) -> float:
        """Return the summed estimated cost across all stages."""
        return sum(self.cost_usd.values())


_CURRENT: ContextVar[RequestMetrics | None] = ContextVar(
    "kuberacle_request_metrics", default=None
)


def set_metrics(metrics: RequestMetrics | None) -> object:
    """Bind ``metrics`` as the current request context.

    Args:
        metrics: Metrics accumulator to bind, or None to clear.

    Returns:
        A token that can be passed to :func:`reset_metrics`.
    """
    return _CURRENT.set(metrics)


def reset_metrics(token: object) -> None:
    """Restore the request context to a previous state.

    Args:
        token: Token returned by :func:`set_metrics`.
    """
    _CURRENT.reset(token)  # type: ignore[arg-type]


def get_metrics() -> RequestMetrics | None:
    """Return the active request metrics, or None when none is bound."""
    return _CURRENT.get()


def update_metrics(**fields: object) -> None:
    """Set fields on the active request metrics. No-op without a context.

    Args:
        **fields: Attribute names and values to assign on the metrics object
            (e.g. ``outcome``, ``gate_decision``, ``chunks_retrieved``).
    """
    metrics = get_metrics()
    if metrics is None:
        return
    for name, value in fields.items():
        setattr(metrics, name, value)


@contextmanager
def stage_timer(name: str) -> Iterator[None]:
    """Time a pipeline stage and record its duration in the active metrics.

    A no-op timer still runs the wrapped block when no request context is
    active, so call sites stay unconditional.

    Args:
        name: Stage name (e.g. ``gate``, ``semantic``, ``rerank``).
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        metrics = get_metrics()
        if metrics is not None:
            elapsed_ms = (time.perf_counter() - start) * 1000
            metrics.stage_ms[name] = round(elapsed_ms, 2)


def record_model_usage(
    stage: str, input_tokens: int, output_tokens: int
) -> None:
    """Record generation-model token usage and cost for a stage.

    Used for the gate and the answer generator (both use the generation model).
    No-op when no request context is active.

    Args:
        stage: Stage name (e.g. ``gate``, ``generation``).
        input_tokens: Prompt token count.
        output_tokens: Completion token count.
    """
    metrics = get_metrics()
    if metrics is None:
        return
    metrics.tokens[f"{stage}_in"] = input_tokens
    metrics.tokens[f"{stage}_out"] = output_tokens
    metrics.cost_usd[stage] = cost.model_token_cost(
        metrics.pricing, input_tokens, output_tokens
    )


def record_embedding_usage(input_tokens: int) -> None:
    """Record query-embedding token usage and cost. No-op without a context.

    Args:
        input_tokens: Embedded input token count.
    """
    metrics = get_metrics()
    if metrics is None:
        return
    metrics.tokens["embed_in"] = input_tokens
    metrics.cost_usd["embed"] = cost.embedding_cost(metrics.pricing, input_tokens)


def record_rerank(num_queries: int = 1) -> None:
    """Record a billable reranking query and its cost. No-op without a context.

    Args:
        num_queries: Number of billable ranking queries issued.
    """
    metrics = get_metrics()
    if metrics is None:
        return
    metrics.rerank_queries += num_queries
    metrics.cost_usd["rerank"] = cost.rerank_cost(
        metrics.pricing, metrics.rerank_queries
    )
