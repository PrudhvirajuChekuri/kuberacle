<div align="center">

# Kuberacle

### Production-Grade RAG for Kubernetes Documentation · Grounded Answers · Cited Sources

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-SSE%20streaming-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-App%20Router-black?logo=nextdotjs)](https://nextjs.org/)
[![Gemini](https://img.shields.io/badge/Gemini-Vertex%20AI-4285F4?logo=googlecloud&logoColor=white)](https://cloud.google.com/vertex-ai)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-vector%20store-FF6B6B)](https://www.trychroma.com/)
[![Langfuse](https://img.shields.io/badge/Langfuse-LLM%20observability-7C3AED)](https://langfuse.com/)
[![Cloud Run](https://img.shields.io/badge/Cloud%20Run-deployed-4285F4?logo=googlecloud&logoColor=white)](https://cloud.google.com/run)
[![CI](https://img.shields.io/github/actions/workflow/status/PrudhvirajuChekuri/kuberacle/tests.yml?branch=main&label=CI&logo=github)](https://github.com/PrudhvirajuChekuri/kuberacle/actions/workflows/tests.yml)
[![CD](https://img.shields.io/github/actions/workflow/status/PrudhvirajuChekuri/kuberacle/deploy.yml?branch=main&label=CD&logo=github)](https://github.com/PrudhvirajuChekuri/kuberacle/actions/workflows/deploy.yml)

**Live at [kuberacle.dev](https://kuberacle.dev)**

</div>

---

> Every answer is backed by the official Kubernetes documentation, cited inline, and flagged when its citations don't check out. If the docs don't support an answer, Kuberacle tells you that instead of making one up.

https://github.com/user-attachments/assets/dfdae900-d97b-4637-ad60-15e723aba2ce

---

## Table of Contents

1. [Architecture](#1-architecture)
2. [Design Decisions](#2-design-decisions)
3. [Repository Structure](#3-repository-structure)
4. [Getting Started](#4-getting-started)
5. [Run the API](#5-run-the-api)
6. [Web UI](#6-web-ui)
7. [Run with Docker](#7-run-with-docker)
8. [Evaluation](#8-evaluation)
9. [Observability](#9-observability)
10. [Production](#10-production)
11. [Stack](#11-stack)
12. [Known Limitations](#12-known-limitations)
13. [License](#13-license)

---

## 1. Architecture

Two loosely coupled halves: a **serving path** that answers questions, and an **index pipeline** that builds what it serves. Images carry no index; the API pulls a pinned, validated index version from GCS at startup.

<p align="center">
  <img src=".github/assets/architecture.png" alt="Kuberacle architecture. Serving path, top to bottom: a user question enters the Next.js Web UI on Cloud Run, passes through an auth proxy to guardrails (Turnstile plus per-IP rate limits), then an exact-match Firestore answer cache, a relevance gate, hybrid retrieval (semantic plus BM25, fused then reranked), answer generation over SSE, and citation validation before returning answer plus citations. Two abstention paths cover out-of-scope questions and empty retrieval. A weekly index pipeline downloads and preprocesses the Kubernetes docs, ingests them into ChromaDB, gates on a deterministic benchmark, and publishes an immutable versioned artifact that the API pulls at startup. Per-request observability records a request_summary event and distributed trace, feeding a GCP ops plane (Cloud Logging, Cloud Trace, Error Reporting) and Langfuse." width="100%">
</p>

<!-- Editable source for this diagram: .github/assets/architecture.mmd -->


**Serving flow.** A question enters through the Next.js Web UI on Cloud Run and reaches the API only through an auth proxy that keeps the FastAPI service private. Guardrails run first: Turnstile verification, then a read-only per-IP cap check, so an over-cap client never reaches the cache. The answer cache is exact-match: a hit replays the stored answer and citations for free (charging only per-IP budget), while a miss charges both the per-IP and global daily caps in one atomic Firestore transaction before any paid work runs. Past the cache, the relevance gate classifies the question with constrained enum decoding (failing open on any error) and abstains on out-of-scope questions without touching retrieval. In-scope questions run hybrid retrieval with the numbers shown: 10 semantic and 10 lexical candidates, fused to 20, then reranked to a top-5 context; empty retrieval is the second abstention trigger. Generation streams tokens live, and only afterwards are the answer's inline citation markers validated, so an answer that fails validation stays on screen but is flagged as ungrounded. The three cacheable outcomes, a verified answer and both abstentions, are written back to the cache; ungrounded answers never are, so a retry can improve them. Throughout, every request emits a single `request_summary` event and distributed trace, fanned to a GCP ops plane and to Langfuse; the [Observability](#9-observability) section covers both planes. Any mid-stream failure surfaces as an SSE `error` event.

**Index flow.** A weekly job fingerprints the upstream docs (git blob SHAs) and the index-affecting build code; when either drifts, it rebuilds, gates the result on the full deterministic benchmark, and publishes an immutable versioned artifact to GCS. Production pins one exact `INDEX_VERSION`: the API pulls it at startup, verifies the tarball digest and the manifest's compatibility with the running config, and refuses to boot on a mismatch. Rolling a new version into production is a deliberate human step.

---

## 2. Design Decisions

### Abstention as a first-class outcome

Kubernetes docs are operational, not just informational: a plausible but unsupported answer can lead someone to run the wrong command. Kuberacle treats grounding as a contract. Grounded answers must cite retrieved documentation, and each `[n]` marker is validated against its source chunk. Output that fails citation validation is flagged as unverified and never cached. Off-topic questions never reach that machinery: a pre-retrieval relevance gate stops them before retrieval, reranking, or answer generation.

### Hybrid retrieval, then rerank

Kubernetes questions mix natural language ("why is my pod stuck") with exact identifiers (`revisionHistoryLimit`). Semantic search handles the former, BM25 the latter; a weighted merge (0.6/0.4) fuses both candidate sets and a semantic reranker orders the merged pool before the top 5 reach the prompt.

### Exact-match answer cache, never semantic

Repeated questions replay from a Firestore cache keyed by normalized question text plus the served index version plus a fingerprint of every answer-affecting config value and the resolved prompt text. A hit is provably the same question under the same configuration, so the cache can never return a confidently wrong answer for a merely similar question. An index roll or prompt edit auto-invalidates.

### The index is an artifact, not part of the image

Docs change weekly; app code does not. Decoupling them means an index rebuild publishes without rebuilding the app image and a code deploy never silently changes retrieval. Each published index carries a manifest with content and build fingerprints plus a contract version; the API validates compatibility at startup and refuses to serve a mismatched artifact.

### Fail open for observability, fail closed for money

Tracing, metrics, and cache writes can never break serving: every observability failure degrades to logging. Guardrails run the other way: Turnstile verification failures reject the request, and rate caps are charged in atomic Firestore transactions so concurrent requests cannot overshoot the daily budget.

---

## 3. Repository Structure

```
kuberacle/
├── src/kuberacle/           # Python package (pip install -e .)
│   ├── api/                 #   FastAPI app, SSE streaming, guardrails, answer cache
│   ├── cli/                 #   one module per command (download-data ... serve)
│   ├── preprocessing/       #   k8s docs cleanup: frontmatter, Hugo shortcodes, chunking
│   ├── ingestion/           #   Vertex AI embedder, Chroma vector store, pipeline
│   ├── retrieval/           #   semantic + BM25 + hybrid merge + reranker
│   ├── evaluation/          #   golden dataset loader, deterministic + RAGAS metrics
│   ├── observability/       #   structured logging, tracing, per-request cost metrics
│   ├── config.py            #   typed RAGConfig loaded from configs/rag.yaml
│   ├── factory.py           #   build_qa_system(): single source of pipeline wiring
│   ├── gate.py              #   pre-retrieval relevance gate (constrained decoding, fails open)
│   ├── qa.py                #   orchestration: gate → retrieve → generate → cite
│   └── generator.py         #   Gemini answer generation with grounded citations
├── web/                     # Next.js chat UI (App Router, Tailwind, shadcn/ui)
├── configs/                 # rag.yaml (single config source) + versioned prompts
├── evals/golden/            # versioned golden datasets (smoke, v1, v2)
├── tests/                   # pytest suite mirroring the package layout
├── Dockerfile               # API image build (multi-stage, .[api] via requirements.lock)
├── docker-compose.yml       # local api + web stack
├── deploy/observability/    # dashboards, log metrics, alerts as reproducible IaC
└── .github/workflows/       # tests, PR smoke eval, weekly index rebuild, CD
```

---

## 4. Getting Started

### Prerequisites

- Python 3.12
- A GCP project with `aiplatform.googleapis.com` and `discoveryengine.googleapis.com` enabled
- `gcloud` CLI installed and authenticated

### Install

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

`pyproject.toml` is the source of truth for dependencies (with tested-against
version floors). For a reproducible dependency set, install with the
lock applied as constraints instead:

```bash
pip install -c requirements.lock -e ".[dev]"
```

Regenerate the lock after changing dependencies, in a clean Python 3.12 venv (the CI and Docker build version):

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

### Build the index and query it

The steps below use the unified dispatcher `python -m kuberacle <name>`, run from
the project root. After `pip install`, each is also a console command
(`kuberacle-<name>`); the two are equivalent. Run `python -m kuberacle` to list commands.

```bash
# 1. Download and preprocess the Kubernetes docs
python -m kuberacle download-data
python -m kuberacle preprocess

# 2. Embed and ingest chunks into ChromaDB
python -m kuberacle ingest

# 3. Ask a question
python -m kuberacle query "What is a Pod?"
```

The query command prints the grounded answer and a citation list with `source_url` and `chunk_id`.

**Skip the build.** Steps 1-2 embed the whole corpus via Vertex AI, which takes a
while and spends credits. To go straight to querying, download the prebuilt index
instead and extract it at the project root:

```bash
curl -L -o chroma-index.tar.gz \
  https://github.com/PrudhvirajuChekuri/kuberacle/releases/latest/download/chroma-index.tar.gz
tar -xzf chroma-index.tar.gz   # -> data/vector/chroma_gemini + data/k8s_version.txt
```

This is the v1.36 corpus snapshot and drops into the default local index path, so you
can run step 3 directly. Live queries still call GCP at runtime (embedding, generation,
reranking), so ADC from [Configure credentials](#configure-credentials) is still required.

---

## 5. Run the API

A FastAPI service streams answer tokens over Server-Sent Events, then closes with a final event carrying validated citations and grounding flags. The pipeline is built once at startup and reused across requests.

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

---

## 6. Web UI

Kuberacle's chat interface (`web/`) streams answers token-by-token from the API and renders citations as clickable source cards. Run both processes locally:

```bash
# Terminal 1 - backend
python -m kuberacle serve

# Terminal 2 - frontend
cd web
npm install        # first time only
npm run dev        # http://localhost:3000
```

The frontend proxies requests to the backend via `RAG_API_URL` (defaults to `http://127.0.0.1:8000`, set in `web/.env.local`).

**Stack:** Next.js (App Router) + TypeScript, Tailwind CSS + shadcn/ui, `react-markdown` + `remark-gfm`. A custom hook consumes the SSE stream; clicking an inline `[n]` marker scrolls to and highlights its citation card, hovering one shows a source preview, and an "ungrounded" notice is shown when the answer could not be verified.

---

## 7. Run with Docker

The full stack (API + web UI) runs in two containers via Docker Compose. The image carries no index: the API pulls the published index from GCS at startup (decoupled from the image) and calls GCP at runtime (embeddings, generation, reranking), so your local ADC is mounted into the container read-only. The web UI reads the corpus version from the API at runtime.

### Prerequisites

- Docker with Compose v2
- A GCP project with `aiplatform.googleapis.com` and `discoveryengine.googleapis.com` enabled (queries call these at runtime and consume credits)
- ADC configured: `gcloud auth application-default login` (with access to Vertex AI, Discovery Engine, and the index bucket)
- A `.env` file in the project root with `GCP_PROJECT`, `GCP_LOCATION`, and `GCS_INDEX_BUCKET`
- A published index in that bucket (trigger the `rag-evaluation` workflow via `workflow_dispatch`, which runs a full rebuild and eval-gated publish, or `python -m kuberacle push-index` after building locally)

Compose sets `INDEX_VERSION=latest` explicitly (there is no default, so nothing silently follows the moving pointer); production pins an exact version instead (see [Production](#10-production)). To use a locally built index instead of the GCS pull, run the API on the host rather than in Compose: build the index with the [Getting Started](#4-getting-started) steps, where `INDEX_SOURCE` defaults to `local`.

### Run

```bash
docker compose up        # add -d to run detached
```

- Web UI: http://localhost:3000
- API: http://localhost:8000 (`GET /health` returns `{"status": "ok"}`)

The web container reaches the API over the Compose network via `RAG_API_URL=http://api:8000`. Credentials are provided only at runtime through a read-only volume mount and are never copied into the image.

---

## 8. Evaluation

Versioned golden datasets (`evals/golden/`, see its [README](evals/golden/README.md)) drive two layers of gates: deterministic retrieval/abstention metrics on every run, and RAGAS LLM-judged metrics only in full mode. The smoke eval runs on every pull request that touches pipeline code; the deterministic benchmark over the full v2 set runs on a manual `workflow_dispatch` and on the weekly `docs-check` job, which rebuilds and publishes a new versioned index only when it detects docs drift and the eval gate passes.

```bash
# Smoke eval, deterministic gates only (same as CI)
python -m kuberacle evaluate --dataset evals/golden/smoke.jsonl --mode deterministic

# Fast local run over the full golden set, skips RAGAS
python -m kuberacle evaluate --dataset evals/golden/v2.jsonl --mode deterministic

# Full benchmark with RAGAS gates
python -m kuberacle evaluate --dataset evals/golden/v2.jsonl
```

The evaluate command writes JSON and markdown artifacts under `artifacts/evals/` and returns a non-zero exit code if any quality gate fails.

| Metric | Threshold | Mode |
|---|---|---|
| `retrieval_recall_at_k` | 0.845925 | deterministic + full |
| `mrr` | 0.90 | deterministic + full |
| `abstention_accuracy` | 0.90 | deterministic + full |
| `non_empty_answer_rate` | 0.90 | deterministic + full |
| `faithfulness` | 0.90 | full only |
| `context_precision` | 0.85 | full only |
| `answer_relevancy` | 0.80 | full only |

The `retrieval_recall_at_k` gate is a ratchet floor at the current full-v2 baseline (~0.846): rebuilds must not regress retrieval, and the threshold is raised as recall improves.

---

## 9. Observability

Production observability spans two planes on one OpenTelemetry instrumentation spine:

- **Operational plane (GCP-native):** structured JSON logs to Cloud Logging (trace-correlated), app and downstream spans to Cloud Trace, log-based metrics, a Cloud Monitoring dashboard, alerts, and Error Reporting. The API emits one `request_summary` event per request carrying RED signals, per-stage latency (gate, semantic, bm25, merge, rerank, generation), token usage, estimated cost (with the reranker as its own line item), the RAG outcome (answered / abstained / unverified / no-retrieval), cache-hit and saved-cost signals, and the guardrail decision. The `request_summary` event and the log-based metrics and dashboard derived from it record metadata only, never question or answer text.
- **LLM/product plane (Langfuse):** the per-query trace (gate -> retrieval -> rerank -> generation) with the question, retrieved-evidence previews, answer, and token cost, plus prompt management. Prompts stay versioned in `configs/prompts/` (the source of truth) and are pushed to Langfuse with `python -m kuberacle sync-prompts`; the running service serves the managed copy with the files as fallback.

Observability is off by default (local dev, tests, CLI). Enable it in deployment with `OBSERVABILITY_ENABLED=true` and the `LANGFUSE_*` env vars; non-secret knobs and the cost prices live in `configs/rag.yaml`. The dashboards, log-based metrics, alerts, and uptime check are committed as reproducible IaC under [`deploy/observability/`](deploy/observability/README.md).

---

## 10. Production

The live deployment on GCP Cloud Run adds the layers a public LLM endpoint needs:

- **Abuse guardrails:** every query requires a Cloudflare Turnstile token (verified server-side with hostname pinning), and per-IP plus global daily caps are enforced through atomic Firestore transactions, so paid pipeline spend has a hard ceiling.
- **Answer cache:** repeated questions replay from Firestore (14-day TTL) without touching the paid pipeline; hits still charge per-IP budget so a cached answer cannot be hammered.
- **Versioned index serving:** production pins an exact `INDEX_VERSION`; the API validates the artifact's manifest (embedding model, dimension, collection, contract version, tarball digest) before serving and fails the boot on any mismatch.
- **CD:** merges to `main` run the pytest suite, then build and roll out only the services whose code changed (api image, web image, or neither for docs-only changes); the rollout is gated by a `production` environment approval.

---

## 11. Stack

| Layer | Technology |
|---|---|
| Language & packaging | Python 3.12 · `pip install -e .` · typed config |
| API | FastAPI · uvicorn · Server-Sent Events |
| Frontend | Next.js (App Router) · React · TypeScript · Tailwind CSS · shadcn/ui · react-markdown · remark-gfm |
| LLM & embeddings | Google Gemini on Vertex AI (`gemini-2.5-flash-lite` · `gemini-embedding-001`) |
| Retrieval | ChromaDB · rank-bm25 · Discovery Engine semantic ranker |
| State | Firestore (answer cache · rate-limit counters) |
| Evaluation & testing | RAGAS · custom metrics · langchain-google-genai judges · pytest |
| Observability | OpenTelemetry · Cloud Logging / Trace / Monitoring · Langfuse |
| Infra & delivery | GCP Cloud Run · Docker · GitHub Actions CI/CD · Cloudflare Turnstile |

---

## 12. Known Limitations

- **Retrieval recall:** full-v2 `retrieval_recall_at_k` sits at ~0.846 against a 0.90 target. The gate is a ratchet floor so the number cannot regress silently; raising it is the next planned retrieval effort.
- **Corpus scope:** the index is built from ~780 source files across concepts, tasks, tutorials, examples, and the glossary of the English `kubernetes/website` docs (`main`, currently v1.36). The reference section (API and `kubectl` reference) is not ingested, so questions whose only support lives in that section may abstain, and there is no per-version corpus selection.
- **No incremental indexing:** a drift-triggered rebuild re-embeds and republishes the entire corpus; there is no delta update of only the changed pages.
- **No model fallback:** generation, embeddings, and reranking run on managed GCP models (Vertex AI + Discovery Engine) with no cross-provider or self-hosted/open-source fallback. The reranker degrades to fused order on failure, but a generation or embedding outage fails the request.
- **Stateless single-turn:** each question is answered independently, with no conversational memory or multi-turn follow-up.

---

## 13. License

[MIT](LICENSE)
