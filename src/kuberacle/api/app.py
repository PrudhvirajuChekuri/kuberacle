"""FastAPI application exposing a streaming RAG query endpoint.

The RAG pipeline is built once during application startup (lifespan) and reused
across requests. Answers stream to the client over Server-Sent Events:

    event: token   data: {"text": "..."}                     (zero or more)
    event: final   data: {"citations": [...], "insufficient_evidence": bool,
                          "abstained": bool}
    event: error   data: {"message": "..."}

The ``final`` event always terminates a successful stream. ``insufficient_evidence``
is true when no citations could be validated for the streamed answer; ``abstained``
is true when the answer is an explicit abstention (the model or pipeline declined
to answer), which the client renders as a friendly note instead of the raw text.
"""

import json
import logging
import os
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from kuberacle.api.cache import (
    AnswerCache,
    CachedAnswer,
    answer_cache_key,
    answer_config_version,
    normalize_question,
)
from kuberacle.api.counters import FirestoreCounters
from kuberacle.api.guardrails import GuardrailError, Guardrails
from kuberacle.api.schemas import CitationModel, QueryRequest
from kuberacle.api.settings import load_cache_settings, load_guardrail_settings
from kuberacle.config import PricingConfig, load_rag_config
from kuberacle.constants import is_abstention
from kuberacle.factory import build_qa_system
from kuberacle.index_sync import load_index_settings, resolve_index
from kuberacle.observability import context as obs
from kuberacle.observability.events import emit_request_summary
from kuberacle.observability.instrumentation import (
    capture_http_trace_context,
    finalize_request_root,
    start_request_root,
)
from kuberacle.observability.logging import configure_logging
from kuberacle.observability.settings import load_observability_settings
from kuberacle.observability.tracing import TracingHandles, configure_tracing
from kuberacle.qa import AnswerDelta

logger = logging.getLogger(__name__)

# SSE responses must not be buffered or cached by intermediaries.
_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

# Outcomes worth caching: a clean answer and the two deterministic abstentions.
# The ungrounded ``unverified`` outcome is excluded so a re-roll can improve it.
_CACHEABLE_OUTCOMES = frozenset(
    {obs.OUTCOME_ANSWERED, obs.OUTCOME_GATE_ABSTAINED, obs.OUTCOME_NO_RETRIEVAL}
)

# First-request flag for cold-start detection (approximate under concurrency).
_served = False


def _consume_cold_start() -> bool:
    """Return whether this is the first request served by this worker."""
    global _served
    cold = not _served
    _served = True
    return cold


def _guardrail_label(status_code: int) -> str:
    """Map a guardrail rejection status to a summary-event label."""
    if status_code == 403:
        return "turnstile"
    if status_code == 429:
        return "rate_limit"
    return "other"


def _new_metrics(
    pricing: PricingConfig, question: str, cold_start: bool
) -> obs.RequestMetrics:
    """Create a per-request metrics accumulator."""
    metrics = obs.RequestMetrics(pricing=pricing)
    metrics.question_length = len(question)
    metrics.cold_start = cold_start
    return metrics


def _emit_query_summary(
    metrics: obs.RequestMetrics, status: int, start: float
) -> None:
    """Emit the request-summary event for the ``/query`` route."""
    emit_request_summary(
        metrics, "POST", "/query", status, (time.perf_counter() - start) * 1000
    )


def _guardrail_rejection_response(
    exc: GuardrailError,
    pricing: PricingConfig | None,
    question: str,
    cold_start: bool,
    start: float,
    tracing,
) -> Response:
    """Build the SSE error response for a rejected request and record it.

    Args:
        exc: The guardrail rejection carrying the HTTP status and message.
        pricing: Price table, or None when observability is not wired.
        question: User question (only its length is recorded, never the text).
        cold_start: Whether a freshly started worker served this request.
        start: ``time.perf_counter`` value captured at request entry.
        tracing: Tracing handles, flushed after the summary is emitted.

    Returns:
        A buffered SSE ``error`` Response carrying the rejection status.
    """
    if pricing is not None:
        metrics = _new_metrics(pricing, question, cold_start)
        metrics.outcome = obs.OUTCOME_GUARDRAIL_REJECTED
        metrics.guardrail = _guardrail_label(exc.status_code)
        _emit_query_summary(metrics, exc.status_code, start)
        tracing.force_flush()
    return Response(
        content=_sse("error", {"message": exc.message}),
        media_type="text/event-stream",
        status_code=exc.status_code,
        headers=_SSE_HEADERS,
    )


def _cached_response(
    cached: CachedAnswer,
    pricing: PricingConfig | None,
    question: str,
    cold_start: bool,
    start: float,
    tracing,
) -> Response:
    """Replay a cached answer over SSE and record it as a cache hit.

    Args:
        cached: The cached answer to replay.
        pricing: Price table, or None when observability is not wired.
        question: User question (only its length is recorded, never the text).
        cold_start: Whether a freshly started worker served this request.
        start: ``time.perf_counter`` value captured at request entry.
        tracing: Tracing handles, flushed after the summary is emitted.

    Returns:
        A buffered SSE Response replaying the cached ``token`` and ``final``
        frames with a 200 status.
    """
    body = _sse("token", {"text": cached.answer}) + _sse(
        "final",
        {
            "citations": cached.citations,
            "insufficient_evidence": cached.insufficient_evidence,
            "abstained": cached.abstained,
        },
    )
    if pricing is not None:
        metrics = _new_metrics(pricing, question, cold_start)
        metrics.outcome = cached.outcome or obs.OUTCOME_ANSWERED
        metrics.cache_hit = True
        metrics.cache_backend = "firestore"
        metrics.saved_cost_estimate = cached.cost_usd
        metrics.citations_count = len(cached.citations)
        metrics.insufficient_evidence = cached.insufficient_evidence
        _emit_query_summary(metrics, 200, start)
        tracing.force_flush()
    return Response(
        content=body,
        media_type="text/event-stream",
        status_code=200,
        headers=_SSE_HEADERS,
    )


def _cache_outcome(
    metrics: obs.RequestMetrics | None, has_citations: bool
) -> str | None:
    """Classify a completed response for caching, or None to skip it.

    The pipeline's own outcome is authoritative: a clean answer and the two
    deterministic abstentions (relevance gate, empty retrieval) are cached; the
    non-deterministic ``unverified`` case (ungrounded text, or a model-emitted
    ``INSUFFICIENT_EVIDENCE``) is never cached so a re-roll can improve it.

    Without metrics (tests / local, no observability) the outcome is unknown,
    and a deterministic abstention cannot be told apart from a model-emitted
    sentinel by surface flags alone. So the fallback is conservative: only cited
    answers are cached, never an abstention.

    Args:
        metrics: Active request metrics, if observability is wired.
        has_citations: Whether the answer carries validated citations.

    Returns:
        The outcome label to store, or None when the response is not cacheable.
    """
    if metrics is not None:
        return metrics.outcome if metrics.outcome in _CACHEABLE_OUTCOMES else None
    return obs.OUTCOME_ANSWERED if has_citations else None


def _write_cache(
    cache: AnswerCache,
    key: str,
    metrics: obs.RequestMetrics | None,
    answer: str,
    citations: list[dict],
    insufficient_evidence: bool,
    abstained: bool,
) -> None:
    """Persist a completed answer to the cache (best-effort).

    Only cacheable responses are written, and any Firestore failure is logged
    and swallowed: the answer has already streamed to the client, so a failed
    write must never surface as an error.

    Args:
        cache: The answer cache store.
        key: Cache document ID for this request.
        metrics: Active request metrics, if observability is wired.
        answer: Final answer text to cache.
        citations: Wire-shaped citation payloads.
        insufficient_evidence: Whether the answer was ungrounded.
        abstained: Whether the answer is an explicit abstention.

    Returns:
        None.
    """
    outcome = _cache_outcome(metrics, bool(citations))
    if outcome is None:
        return
    cost = metrics.total_cost_usd() if metrics is not None else 0.0
    try:
        cache.put(
            key,
            CachedAnswer(
                answer=answer,
                citations=citations,
                insufficient_evidence=insufficient_evidence,
                abstained=abstained,
                outcome=outcome,
                cost_usd=round(cost, 6),
            ),
        )
    except Exception:
        logger.exception("Failed to write answer to cache")


def _resolved_prompts(qa_system) -> dict[str, dict]:
    """Collect the actual prompt bundles the built pipeline will use.

    These are the prompts resolved at startup (managed copies from Langfuse, or
    the committed file fallback), so the cache key reflects the text actually in
    use rather than the mutable version label.

    Args:
        qa_system: The assembled QA system.

    Returns:
        Resolved prompt bundles keyed by role (``answer`` and, when the gate is
        wired, ``gate``).
    """
    prompts = {"answer": dict(qa_system.generator.prompt_bundle)}
    if qa_system.relevance_gate is not None:
        prompts["gate"] = dict(qa_system.relevance_gate.prompt_bundle)
    return prompts


def _project_root() -> Path:
    """Resolve the project root used to locate config and data directories."""
    return Path(os.environ.get("RAG_PROJECT_ROOT", Path.cwd()))


def _cors_origins() -> list[str]:
    """Parse allowed CORS origins from the environment."""
    raw = os.environ.get("RAG_CORS_ORIGINS", "http://localhost:3000")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _docs_enabled() -> bool:
    """Whether to expose the interactive API docs and OpenAPI schema.

    Off by default so the private production service does not serve ``/docs``,
    ``/redoc``, or ``/openapi.json``; set ``RAG_DOCS_ENABLED=true`` for local dev.
    """
    return os.environ.get("RAG_DOCS_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _sse(event: str, data: dict) -> str:
    """Format a Server-Sent Events frame.

    Args:
        event: SSE event name.
        data: JSON-serializable payload.

    Returns:
        Encoded SSE frame string.
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the RAG pipeline once and attach it to application state."""
    root = _project_root()
    load_dotenv(root / ".env")
    config = load_rag_config(root / "configs" / "rag.yaml")

    # Configure logging before anything else so startup logs are structured.
    configure_logging(config.observability, config.gcp_project)

    # Configure tracing (and the Langfuse client) before building the pipeline
    # so the factory can fetch managed prompts and link them in traces.
    obs_settings = load_observability_settings()
    app.state.tracing = configure_tracing(
        obs_settings, config.observability, config.gcp_project, app
    )

    # Resolve the index location: a local on-disk directory, or a pinned
    # version pulled from GCS at startup (decoupled from the image). Pulling
    # before building means an incompatible artifact fails the boot fast.
    index_settings = load_index_settings()
    resolved_index = resolve_index(config, index_settings, root)
    app.state.k8s_version = resolved_index.k8s_version
    app.state.qa_system = build_qa_system(
        config, root, index_dir=resolved_index.persist_directory
    )
    app.state.pricing = config.pricing

    settings = load_guardrail_settings()
    if settings.enabled:
        counters = FirestoreCounters(
            settings.gcp_project, settings.firestore_database
        )
        app.state.guardrails = Guardrails(settings, counters)
        logger.info("Guardrails enabled")

    cache_settings = load_cache_settings()
    if cache_settings.enabled:
        app.state.answer_cache = AnswerCache(
            cache_settings.gcp_project,
            cache_settings.firestore_database,
            cache_settings.collection,
            cache_settings.ttl_days,
        )
        # The cache key pins answers to the exact served index and the
        # answer-affecting config, so an index roll or a config change makes
        # stale entries unreachable. Prefer the manifest's version (correct even
        # under INDEX_VERSION=latest), falling back to the env pin then "local".
        app.state.index_version = (
            (resolved_index.manifest or {}).get("index_version")
            or index_settings.version
            or "local"
        )
        app.state.answer_config_version = answer_config_version(
            config, _resolved_prompts(app.state.qa_system)
        )
        logger.info(
            "Answer cache enabled (collection=%s, ttl_days=%d, index_version=%s)",
            cache_settings.collection,
            cache_settings.ttl_days,
            app.state.index_version,
        )

    logger.info("RAG QA system ready")
    yield

    app.state.tracing.shutdown()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI app with CORS, health, and streaming query routes.
    """
    docs = _docs_enabled()
    app = FastAPI(
        title="kuberacle API",
        lifespan=lifespan,
        docs_url="/docs" if docs else None,
        redoc_url="/redoc" if docs else None,
        openapi_url="/openapi.json" if docs else None,
    )
    # Default to no guardrails so local dev and tests (which do not run the
    # lifespan hook) skip them; the lifespan enables them when configured.
    app.state.guardrails = None
    # Default to no answer cache (same rationale); the lifespan enables it and
    # sets the version keys that scope cache entries when configured.
    app.state.answer_cache = None
    app.state.index_version = None
    app.state.answer_config_version = None
    # Observability defaults: disabled tracing and no pricing until the lifespan
    # wires them, so requests served without lifespan emit no metrics.
    app.state.tracing = TracingHandles()
    app.state.pricing = None
    # Set by the lifespan from the served index's manifest; the web UI reads it
    # at runtime so a docs-version bump needs no web rebuild.
    app.state.k8s_version = None
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict:
        """Liveness probe that does not invoke the model."""
        return {"status": "ok"}

    @app.get("/meta")
    async def meta(request: Request) -> dict:
        """Serving metadata, including the docs version of the served index."""
        return {"k8s_version": request.app.state.k8s_version}

    @app.post("/query")
    async def query(payload: QueryRequest, request: Request) -> Response:
        """Stream a grounded answer for a question over SSE.

        Guardrails run in the order Turnstile -> per-IP cap -> answer cache ->
        pipeline. The per-IP counter is checked read-only before the cache (so
        an over-cap client never reaches it) and charged once the request is
        served: alone on a cache hit (which bypasses the global cap, being
        free), or together with the global cap in one atomic transaction on a
        miss (so a global rejection charges neither counter).
        """
        qa_system = request.app.state.qa_system
        tracing = request.app.state.tracing
        pricing = request.app.state.pricing
        cache = request.app.state.answer_cache
        cold_start = _consume_cold_start()
        start = time.perf_counter()

        guardrails = request.app.state.guardrails
        client_ip = request.headers.get("X-Client-IP", "")

        def reject(exc: GuardrailError) -> Response:
            return _guardrail_rejection_response(
                exc, pricing, payload.question, cold_start, start, tracing
            )

        # 1. Turnstile, then a read-only per-IP cap check, both before the cache.
        if guardrails is not None:
            try:
                guardrails.verify_turnstile(
                    client_ip, request.headers.get("X-Turnstile-Token", "")
                )
                guardrails.check_ip_rate_limit(client_ip)
            except GuardrailError as exc:
                return reject(exc)

        # 2. Answer cache: an exact repeat short-circuits the whole pipeline. A
        # hit charges per-IP only; the global cap is bypassed. Empty normalized
        # questions are not cached (they would all collide on one key).
        cache_key: str | None = None
        if cache is not None and normalize_question(payload.question):
            cache_key = answer_cache_key(
                payload.question,
                request.app.state.index_version or "",
                request.app.state.answer_config_version or "",
            )
            try:
                cached = cache.get(cache_key)
            except Exception:
                logger.exception("Answer cache lookup failed")
                cached = None
            if cached is not None:
                if guardrails is not None:
                    try:
                        guardrails.charge_ip(client_ip)
                    except GuardrailError as exc:
                        return reject(exc)
                return _cached_response(
                    cached, pricing, payload.question, cold_start, start, tracing
                )

        # 3. Cache miss: charge per-IP and the global cap atomically before the
        # paid pipeline runs.
        if guardrails is not None:
            try:
                guardrails.charge_ip_and_global(client_ip)
            except GuardrailError as exc:
                return reject(exc)

        # Bind the metrics in this request coroutine's context, not inside the
        # streaming generator: Starlette iterates a sync streaming body in a
        # fresh threadpool context copy per chunk, so a binding made inside it
        # would not survive past the first token. Binding here means every
        # per-chunk copy inherits it, so usage/outcome recorded mid- and
        # post-stream is captured. The binding is scoped to this request's task.
        metrics = (
            _new_metrics(pricing, payload.question, cold_start)
            if pricing is not None
            else None
        )
        if metrics is not None:
            obs.set_metrics(metrics)
            # Reuse the inbound trace (the web proxy sends a W3C traceparent) so
            # the pipeline trace and the Cloud Trace HTTP span share one id. Read
            # the header here, in the request coroutine; the streaming body below
            # runs in detached threadpool copies with no request context.
            capture_http_trace_context(request.headers.get("traceparent"))

        def event_stream() -> Iterator[str]:
            # One root observation per request so the pipeline stages nest under
            # a single Langfuse trace (no-op when tracing is disabled).
            root = start_request_root("query", user_input=payload.question)
            final_answer = ""
            final_citations: list[dict] = []
            final_insufficient = False
            final_abstained = False
            completed = False
            try:
                for event in qa_system.ask_stream(payload.question):
                    if isinstance(event, AnswerDelta):
                        yield _sse("token", {"text": event.text})
                    else:
                        final_answer = event.answer
                        final_citations = [
                            CitationModel(
                                index=citation.index,
                                source_url=citation.source_url,
                                chunk_id=citation.chunk_id,
                                score=citation.score,
                                title=citation.title,
                                snippet=citation.snippet,
                            ).model_dump()
                            for citation in event.citations
                        ]
                        final_insufficient = not event.citations
                        final_abstained = is_abstention(event.answer)
                        yield _sse(
                            "final",
                            {
                                "citations": final_citations,
                                "insufficient_evidence": final_insufficient,
                                "abstained": final_abstained,
                            },
                        )
                        completed = True
            except Exception:
                logger.exception("Error while streaming query response")
                if metrics is not None:
                    metrics.outcome = obs.OUTCOME_ERROR
                yield _sse("error", {"message": "Internal error generating answer."})
            finally:
                finalize_request_root(root, output=final_answer or None)
                # Cache only a clean, server-side completion: a partial or failed
                # generation is never persisted.
                if completed and cache is not None and cache_key is not None:
                    _write_cache(
                        cache,
                        cache_key,
                        metrics,
                        final_answer,
                        final_citations,
                        final_insufficient,
                        final_abstained,
                    )
                if metrics is not None:
                    _emit_query_summary(metrics, 200, start)
                    obs.set_metrics(None)
                tracing.force_flush()

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    return app


# Load .env before create_app() so env-driven config (e.g. RAG_CORS_ORIGINS,
# read at app-construction time) reflects the .env file, not just shell env.
load_dotenv(_project_root() / ".env")
app = create_app()
