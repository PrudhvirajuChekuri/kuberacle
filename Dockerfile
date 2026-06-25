# Backend image for the kuberacle FastAPI service.
# Multi-stage: install dependencies into a venv, then copy into a slim runtime.
# The Chroma index is NOT baked in: at startup the API pulls a pinned version
# from GCS (INDEX_SOURCE=gcs, INDEX_BUCKET, INDEX_VERSION) into a writable cache
# under /tmp. GCP credentials are provided at runtime, never here.

FROM python:3.12-slim AS builder

# build-essential covers any dependency that lacks a prebuilt wheel.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Isolated virtualenv so the runtime stage receives a clean, copyable tree.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Install the package with the API extra (fastapi + uvicorn), pinned to the
# lockfile via constraints so the image is reproducible. Constraints only pin
# versions of what [api] actually pulls in; the dev/eval lines in the lock are
# ignored, so the image stays slim.
COPY pyproject.toml requirements.lock ./
COPY src ./src
RUN pip install --no-cache-dir -c requirements.lock ".[api]"

FROM python:3.12-slim AS runtime

# Reuse the virtualenv built above; no build tools in the final image.
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    RAG_PROJECT_ROOT=/app

WORKDIR /app

# Run as a non-root user. The index is pulled at startup into a writable cache
# (default /tmp/kuberacle-index, world-writable), so no app-owned data dir is
# baked or needed here.
RUN useradd --create-home --uid 10001 appuser

# Runtime assets: config + prompts. The index is fetched from GCS at startup.
COPY configs ./configs

USER appuser

EXPOSE 8000

# Bind 0.0.0.0 so the service is reachable from outside the container.
CMD ["uvicorn", "kuberacle.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
