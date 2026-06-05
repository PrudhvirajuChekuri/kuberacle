# k8s-docs-rag

A production-grade Retrieval Augmented Generation (RAG) system for Kubernetes documentation.

Ask questions about Kubernetes and get grounded answers with citations to the official docs.

## Project Status

🚧 **In Development**

The full RAG pipeline is implemented and running on GCP. Preprocessing, ingestion, hybrid retrieval, reranking, cited answer generation, and deterministic evaluation gates are all in place with CI on pull requests.

## Architecture

The system processes the official [Kubernetes documentation](https://kubernetes.io/docs/) (v1.36) through a multi-stage pipeline:

1. **Preprocessing** — Parse frontmatter, resolve Hugo shortcodes, smart chunking
2. **Ingestion** — Embed chunks with `gemini-embedding-001` and store in ChromaDB
3. **Retrieval** — Hybrid search (BM25 + semantic) with Discovery Engine reranking
4. **Generation** — Grounded answers with inline citations using `gemini-2.5-flash-lite`
5. **Evaluation** — Deterministic retrieval/citation quality gates in CI

## Data Source

Raw markdown from the [`kubernetes/website`](https://github.com/kubernetes/website) repository (`main` branch, v1.36), covering:

- **Concepts** — How Kubernetes works (pods, deployments, services, networking, storage)
- **Tasks** — Step-by-step operational guides (debugging, configuration, networking)
- **Tutorials** — End-to-end walkthroughs (deploying applications, stateful workloads)

One dataset config is used for all runs:
- **Full** (`configs/datasets/full.yaml`) — the full corpus used for ingestion and evaluation

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

1. Ensure preprocessing output exists:
   ```bash
   python scripts/download_data.py
   python scripts/preprocess.py
   ```

2. Ingest chunks into ChromaDB:
   ```bash
   python scripts/ingest.py
   ```

3. Ask a question:
   ```bash
   python scripts/query.py "What is a Pod?"
   ```

4. Run offline evaluation:
   ```bash
   # Smoke eval (fast CI gate)
   python scripts/evaluate.py --dataset evals/golden/smoke.jsonl

   # Full benchmark
   python scripts/evaluate.py --dataset evals/golden/v2.jsonl
   ```

The query script prints the grounded answer and a citation list with `source_url` and `chunk_id`.

The evaluation script writes JSON and markdown artifacts under `artifacts/evals/` and returns a non-zero exit code if any quality gate fails.

## Evaluation Gates

The smoke eval runs on every pull request. The full benchmark can be triggered manually via `workflow_dispatch` in GitHub Actions.

| Metric | Threshold |
|---|---|
| `retrieval_recall_at_k` | 0.75 |
| `precision_at_1` | 0.65 |
| `abstention_accuracy` | 0.90 |
| `non_empty_answer_rate` | 0.90 |
