# k8s-docs-rag

A production-grade Retrieval Augmented Generation (RAG) system for Kubernetes documentation.

Ask questions about Kubernetes and get grounded answers with citations to the official docs.

## Project Status

🚧 **In Development**

Preprocessing is complete, and the baseline RAG pipeline (ingestion + hybrid retrieval + cited answers) is implemented with AWS Bedrock and ChromaDB.

## Architecture

The system processes the official [Kubernetes documentation](https://kubernetes.io/docs/) (v1.36) through a multi-stage pipeline:

1. **Preprocessing** — Parse frontmatter, resolve Hugo shortcodes, smart chunking
2. **Ingestion** — Embed chunks and store in a vector database
3. **Retrieval** — Hybrid search (BM25 + semantic) with cross-encoder reranking
4. **Evaluation** — Automated faithfulness and retrieval quality metrics in CI

## Data Source

Raw markdown from the `[kubernetes/website](https://github.com/kubernetes/website)` repository (`main` branch, v1.36), covering:

- **Concepts** — How Kubernetes works (pods, deployments, services, networking, storage)
- **Tasks** — Step-by-step operational guides (debugging, configuration, networking)
- **Tutorials** — End-to-end walkthroughs (deploying applications, stateful workloads)

## RAG Baseline (Implemented)

Current baseline includes:
- embedding chunked docs with Amazon Bedrock (`amazon.titan-embed-text-v2:0`)
- storing vectors in ChromaDB
- hybrid retrieval (semantic + BM25) with rerank fallback
- answer generation with citations using Bedrock generation models

Default generation model in config is `amazon.nova-lite-v1:0`. You can switch
to Claude models if your account has the required Bedrock throughput mode
(on-demand or inference profile access, depending on model/version).

Configuration lives in `configs/rag.yaml`; versioned prompts live under `configs/prompts/`.

## Run RAG Pipeline

1. Ensure preprocessing output exists:
   - `python scripts/download_data.py`
   - `python scripts/preprocess.py`
2. Configure AWS credentials and Bedrock access (region/model access).
3. Ingest chunks into Chroma:
   - `python scripts/ingest.py`
4. Ask a question:
   - `python scripts/query.py "What is a Pod?"`
5. Run offline evaluation:
   - `python scripts/evaluate.py`

The query script prints:
- grounded answer text
- citation list with `source_url` and `chunk_id`

The evaluation script writes JSON and markdown artifacts under
`artifacts/evals/` and returns a non-zero exit code if deterministic
quality thresholds fail.

