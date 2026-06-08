# Backend image for the k8s-docs-rag FastAPI service.
# Multi-stage: install dependencies into a venv, then copy into a slim runtime.
# The Chroma index is baked in; GCP credentials are provided at runtime, never here.

FROM python:3.12-slim AS builder

# build-essential covers any dependency that lacks a prebuilt wheel.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Isolated virtualenv so the runtime stage receives a clean, copyable tree.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Install the package with the API extra (fastapi + uvicorn).
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[api]"

FROM python:3.12-slim AS runtime

# Reuse the virtualenv built above; no build tools in the final image.
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    RAG_PROJECT_ROOT=/app

WORKDIR /app

# Runtime assets: config + prompts and the prebuilt Chroma index (baked in).
COPY configs ./configs
COPY data/vector/chroma_gemini ./data/vector/chroma_gemini

EXPOSE 8000

# Bind 0.0.0.0 so the service is reachable from outside the container.
CMD ["uvicorn", "k8s_rag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
