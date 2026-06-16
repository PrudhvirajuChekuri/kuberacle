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

from kuberacle.api.counters import FirestoreCounters
from kuberacle.api.guardrails import GuardrailError, Guardrails
from kuberacle.api.schemas import CitationModel, QueryRequest
from kuberacle.api.settings import load_guardrail_settings
from kuberacle.config import PricingConfig, load_rag_config
from kuberacle.constants import is_abstention
from kuberacle.factory import build_qa_system
from kuberacle.observability import context as obs
from kuberacle.observability.events import emit_request_summary
from kuberacle.observability.instrumentation import (
    finalize_request_root,
    start_request_root,
)
from kuberacle.observability.logging import configure_logging
from kuberacle.observability.settings import load_observability_settings
from kuberacle.observability.tracing import TracingHandles, configure_tracing
from kuberacle.qa import AnswerDelta

logger = logging.getLogger(__name__)

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

    app.state.qa_system = build_qa_system(config, root)
    app.state.pricing = config.pricing

    settings = load_guardrail_settings()
    if settings.enabled:
        counters = FirestoreCounters(
            settings.gcp_project, settings.firestore_database
        )
        app.state.guardrails = Guardrails(settings, counters)
        logger.info("Guardrails enabled")

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
    # Observability defaults: disabled tracing and no pricing until the lifespan
    # wires them, so requests served without lifespan emit no metrics.
    app.state.tracing = TracingHandles()
    app.state.pricing = None
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

    @app.post("/query")
    async def query(payload: QueryRequest, request: Request) -> StreamingResponse:
        """Stream a grounded answer for a question over SSE."""
        qa_system = request.app.state.qa_system
        tracing = request.app.state.tracing
        pricing = request.app.state.pricing
        cold_start = _consume_cold_start()
        start = time.perf_counter()

        guardrails = request.app.state.guardrails
        if guardrails is not None:
            try:
                guardrails.enforce(
                    request.headers.get("X-Client-IP", ""),
                    request.headers.get("X-Turnstile-Token", ""),
                )
            except GuardrailError as exc:
                if pricing is not None:
                    metrics = _new_metrics(pricing, payload.question, cold_start)
                    metrics.outcome = obs.OUTCOME_GUARDRAIL_REJECTED
                    metrics.guardrail = _guardrail_label(exc.status_code)
                    emit_request_summary(
                        metrics,
                        "POST",
                        "/query",
                        exc.status_code,
                        (time.perf_counter() - start) * 1000,
                    )
                    tracing.force_flush()
                return Response(
                    content=_sse("error", {"message": exc.message}),
                    media_type="text/event-stream",
                    status_code=exc.status_code,
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )

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

        def event_stream() -> Iterator[str]:
            # One root observation per request so the pipeline stages nest under
            # a single Langfuse trace (no-op when tracing is disabled).
            root = start_request_root("query", user_input=payload.question)
            final_answer = ""
            try:
                for event in qa_system.ask_stream(payload.question):
                    if isinstance(event, AnswerDelta):
                        yield _sse("token", {"text": event.text})
                    else:
                        final_answer = event.answer
                        citations = [
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
                        yield _sse(
                            "final",
                            {
                                "citations": citations,
                                "insufficient_evidence": not event.citations,
                                "abstained": is_abstention(event.answer),
                            },
                        )
            except Exception:
                logger.exception("Error while streaming query response")
                if metrics is not None:
                    metrics.outcome = obs.OUTCOME_ERROR
                yield _sse("error", {"message": "Internal error generating answer."})
            finally:
                finalize_request_root(root, output=final_answer or None)
                if metrics is not None:
                    emit_request_summary(
                        metrics,
                        "POST",
                        "/query",
                        200,
                        (time.perf_counter() - start) * 1000,
                    )
                    obs.set_metrics(None)
                tracing.force_flush()

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


# Load .env before create_app() so env-driven config (e.g. RAG_CORS_ORIGINS,
# read at app-construction time) reflects the .env file, not just shell env.
load_dotenv(_project_root() / ".env")
app = create_app()
