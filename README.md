# kuberacle

A production-grade Retrieval Augmented Generation (RAG) system for Kubernetes documentation.

Ask questions about Kubernetes and get grounded answers with citations to the official docs.

## Project Status

**Live** at **[kuberacle.dev](https://kuberacle.dev)** on GCP Cloud Run.

The full stack is deployed: the hybrid-retrieval RAG pipeline (preprocessing, ingestion, reranking, cited answer generation, abstention), deterministic and RAGAS evaluation gates with CI on pull requests, a streaming FastAPI service, a Next.js web UI, and two-plane production observability. See [Run the API](#run-the-api), [Web UI](#web-ui), [Observability](#observability), and [Run with Docker](#run-with-docker).

## Architecture

The system processes the official [Kubernetes documentation](https://kubernetes.io/docs/) (v1.36) through a multi-stage pipeline:

1. **Preprocessing** - Parse frontmatter, resolve Hugo shortcodes, smart chunking
2. **Ingestion** - Embed chunks with `gemini-embedding-001` and store in ChromaDB
3. **Retrieval** - Hybrid search (BM25 + semantic) with Discovery Engine reranking
4. **Generation** - Grounded answers with inline citations using `gemini-2.5-flash-lite`
5. **Evaluation** - Deterministic and RAGAS-based quality gates in CI

## Data Source

Raw markdown from the [`kubernetes/website`](https://github.com/kubernetes/website) repository (`main` branch, v1.36), covering:

- **Concepts** - How Kubernetes works (pods, deployments, services, networking, storage)
- **Tasks** - Step-by-step operational guides (debugging, configuration, networking)
- **Tutorials** - End-to-end walkthroughs (deploying applications, stateful workloads)

One dataset config is used for all runs:
- **Full** (`configs/datasets/full.yaml`) - the full corpus used for ingestion and evaluation

## Stack

| Component | Technology |
|---|---|
| Embeddings | `gemini-embedding-001` via Vertex AI (768-dim) |
| Generation | `gemini-2.5-flash-lite` via Vertex AI |
| Reranking | Discovery Engine Ranking API (`semantic-ranker-default@latest`) |
| Vector store | ChromaDB (local) |
| Lexical retrieval | BM25 |
| Auth | Application Default Credentials (ADC) |

Configuration lives in `configs/rag.yaml`; versioned prompts live under `configs/prompts/`.

## Setup

### Prerequisites

- Python 3.10+
- A GCP project with `aiplatform.googleapis.com` and `discoveryengine.googleapis.com` enabled
- `gcloud` CLI installed and authenticated

### Install

```bash
pip install -e ".[dev]"
```

`pyproject.toml` is the source of truth for dependencies (with tested-against
version floors). For a byte-for-byte reproducible environment, install the
pinned lock instead:

```bash
pip install -r requirements.lock
```

Regenerate the lock after changing dependencies, in a clean Python 3.12 venv:

```bash
pip install -e ".[dev]" && pip freeze --exclude-editable > requirements.lock
```

### Configure credentials

```bash
gcloud auth application-default login
```

Create a `.env` file in the project root:

```
GCP_PROJECT=your-project-id
GCP_LOCATION=us-central1
```

## Run RAG Pipeline

The steps below use the unified dispatcher `python -m kuberacle <name>`, run from
the project root. After `pip install`, each is also a console command
(`kuberacle-<name>`); the two are equivalent (e.g. `python -m kuberacle ingest`
is the same as `kuberacle-ingest`). Run `python -m kuberacle` to list commands.

1. Ensure preprocessing output exists:
   ```bash
   python -m kuberacle download-data
   python -m kuberacle preprocess
   ```

2. Ingest chunks into ChromaDB:
   ```bash
   python -m kuberacle ingest
   ```

3. Ask a question:
   ```bash
   python -m kuberacle query "What is a Pod?"
   ```

4. Run offline evaluation:
   ```bash
   # Smoke eval, deterministic gates only (same as CI)
   python -m kuberacle evaluate --dataset evals/golden/smoke.jsonl --mode deterministic

   # Fast local run, deterministic gates only, skips RAGAS (~20s)
   python -m kuberacle evaluate --dataset evals/golden/v2.jsonl --mode deterministic

   # Full benchmark with RAGAS gates
   python -m kuberacle evaluate --dataset evals/golden/v2.jsonl
   ```

The query command prints the grounded answer and a citation list with `source_url` and `chunk_id`.

The evaluate command writes JSON and markdown artifacts under `artifacts/evals/` and returns a non-zero exit code if any quality gate fails.

## Run the API

A FastAPI service streams grounded answers over Server-Sent Events. The pipeline is built once at startup and reused across requests.

```bash
pip install -e ".[api]"
python -m kuberacle serve            # http://127.0.0.1:8000  (add --reload for dev)
```

Query it (stream tokens then a final citations event):

```bash
curl -N -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a Pod?"}'
```

The stream emits these SSE events:

```
event: token   data: {"text": "..."}                              # repeated, live
event: final   data: {"citations": [...], "insufficient_evidence": false, "abstained": false}
event: error   data: {"message": "..."}
```

`insufficient_evidence` is `true` when no citations could be validated for the streamed answer; `abstained` is `true` when the answer is an explicit abstention (the model or pre-RAG gate declined to answer). A `GET /health` endpoint returns `{"status": "ok"}` without invoking the model.

## Web UI

Kuberacle, a Next.js chat interface (`web/`), streams answers token-by-token from the API and renders citations as clickable source cards. Run both processes locally:

```bash
# Terminal 1 - backend
python -m kuberacle serve

# Terminal 2 - frontend
cd web
npm install        # first time only
npm run dev        # http://localhost:3000
```

The frontend proxies requests to the backend via `RAG_API_URL` (defaults to `http://127.0.0.1:8000`, set in `web/.env.local`).

**Stack:** Next.js (App Router) + TypeScript, Tailwind CSS + shadcn/ui, `react-markdown`. A custom hook consumes the SSE stream; clicking an inline `[n]` marker scrolls to and highlights its citation card, hovering one shows a source preview, and an "ungrounded" notice is shown when the answer could not be verified.

## Observability

Production observability spans two planes on one OpenTelemetry instrumentation spine:

- **Operational plane (GCP-native):** structured JSON logs to Cloud Logging (trace-correlated), app and downstream spans to Cloud Trace, log-based metrics, a Cloud Monitoring dashboard, alerts, and Error Reporting. The API emits one `request_summary` event per request carrying RED signals, per-stage latency (gate, semantic, bm25, merge, rerank, generation), token usage, estimated cost (with the reranker as its own line item), the RAG outcome (answered / abstained / unverified / no-retrieval), and guardrail signal. It records metadata only, never question or answer text.
- **LLM/product plane (Langfuse):** the per-query trace (retrieval -> rerank -> generation) with token cost, plus prompt management. Prompts stay versioned in `configs/prompts/` (the source of truth) and are pushed to Langfuse with `python -m kuberacle sync-prompts`; the running service serves the managed copy with the files as fallback.

Observability is off by default (local dev, tests, CLI). Enable it in deployment with `OBSERVABILITY_ENABLED=true` and the `LANGFUSE_*` env vars; non-secret knobs (service name, log level/format, trace sample ratio) and the cost prices live in `configs/rag.yaml`. The dashboards, log-based metrics, alerts, and uptime check are committed as reproducible IaC under `deploy/observability/` (see its README).

## Run with Docker

The full stack (API + web UI) runs in two containers via Docker Compose. The image carries no index: the API pulls the published index from GCS at startup (decoupled from the image) and calls GCP at runtime (embeddings, generation, reranking), so your local ADC is mounted into the container read-only. The web UI reads the corpus version from the API at runtime.

### Prerequisites

- Docker with Compose v2 (on WSL, enable Docker Desktop's WSL integration)
- A GCP project with `aiplatform.googleapis.com` and `discoveryengine.googleapis.com` enabled (queries call these at runtime and consume credits)
- ADC configured: `gcloud auth application-default login` (with read access to the index bucket)
- A `.env` file in the project root with `GCP_PROJECT`, `GCP_LOCATION`, and `GCS_INDEX_BUCKET`
- A published index in that bucket (run the `workflow_dispatch` build, or `python -m kuberacle push-index` after building locally)

Compose pulls the `latest` index version (set explicitly in `docker-compose.yml`); production pins an exact version via `INDEX_VERSION` (required for GCS mode - there is no default, so a deployment can never silently follow the moving pointer). To run fully offline instead, build the index locally with the [Run RAG Pipeline](#run-rag-pipeline) steps and set `INDEX_SOURCE=local`.

### Run

```bash
docker compose up        # add -d to run detached
```

- Web UI: http://localhost:3000
- API: http://localhost:8000 (`GET /health` returns `{"status": "ok"}`)

Stop and remove the containers:

```bash
docker compose down
```

The web container reaches the API over the Compose network via `RAG_API_URL=http://api:8000`. Credentials are provided only at runtime through a read-only volume mount and are never copied into the image. If the single-file ADC mount misbehaves, `docker-compose.yml` documents a whole-directory fallback.

## Evaluation Gates

The smoke eval runs on every pull request. The full deterministic benchmark runs on a manual `workflow_dispatch` and on the weekly `docs-check` rebuild, which publishes a new versioned index only when the gate passes.

| Metric | Threshold | Mode |
|---|---|---|
| `retrieval_recall_at_k` | 0.845925 | deterministic + full |
| `mrr` | 0.90 | deterministic + full |
| `abstention_accuracy` | 0.90 | deterministic + full |
| `non_empty_answer_rate` | 0.90 | deterministic + full |
| `faithfulness` | 0.90 | full only |
| `context_precision` | 0.85 | full only |
| `answer_relevancy` | 0.80 | full only |

The `retrieval_recall_at_k` gate is a ratchet floor at the current full-v2 baseline (~0.846): rebuilds must not regress retrieval, and the threshold is raised as recall improves. Closing the remaining recall gap is tracked as a separate retrieval-tuning effort.
