"""FastAPI application exposing a streaming RAG query endpoint.

The RAG pipeline is built once during application startup (lifespan) and reused
across requests. Answers stream to the client over Server-Sent Events:

    event: token   data: {"text": "..."}                     (zero or more)
    event: final   data: {"citations": [...], "insufficient_evidence": bool}
    event: error   data: {"message": "..."}

The ``final`` event always terminates a successful stream; ``insufficient_evidence``
is true when no citations could be validated for the streamed answer.
"""

import json
import logging
import os
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from k8s_rag.api.counters import FirestoreCounters
from k8s_rag.api.guardrails import GuardrailError, Guardrails
from k8s_rag.api.schemas import CitationModel, QueryRequest
from k8s_rag.api.settings import load_guardrail_settings
from k8s_rag.ingestion.config import load_rag_config
from k8s_rag.retrieval.factory import build_qa_system
from k8s_rag.retrieval.qa import AnswerDelta

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    """Resolve the project root used to locate config and data directories."""
    return Path(os.environ.get("RAG_PROJECT_ROOT", Path.cwd()))


def _cors_origins() -> list[str]:
    """Parse allowed CORS origins from the environment."""
    raw = os.environ.get("RAG_CORS_ORIGINS", "http://localhost:3000")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


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
    app.state.qa_system = build_qa_system(config, root)

    settings = load_guardrail_settings()
    if settings.enabled:
        counters = FirestoreCounters(
            settings.gcp_project, settings.firestore_database
        )
        app.state.guardrails = Guardrails(settings, counters)
        logger.info("Guardrails enabled")

    logger.info("RAG QA system ready")
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI app with CORS, health, and streaming query routes.
    """
    app = FastAPI(title="k8s-docs-rag API", lifespan=lifespan)
    # Default to no guardrails so local dev and tests (which do not run the
    # lifespan hook) skip them; the lifespan enables them when configured.
    app.state.guardrails = None
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

        guardrails = request.app.state.guardrails
        if guardrails is not None:
            try:
                guardrails.enforce(
                    request.headers.get("X-Client-IP", ""),
                    request.headers.get("X-Turnstile-Token", ""),
                )
            except GuardrailError as exc:
                return Response(
                    content=_sse("error", {"message": exc.message}),
                    media_type="text/event-stream",
                    status_code=exc.status_code,
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )

        def event_stream() -> Iterator[str]:
            try:
                for event in qa_system.ask_stream(payload.question):
                    if isinstance(event, AnswerDelta):
                        yield _sse("token", {"text": event.text})
                    else:
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
                            },
                        )
            except Exception:
                logger.exception("Error while streaming query response")
                yield _sse("error", {"message": "Internal error generating answer."})

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
