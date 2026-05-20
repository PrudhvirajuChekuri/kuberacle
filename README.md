# k8s-docs-rag

A production-grade Retrieval Augmented Generation (RAG) system for Kubernetes documentation.

Ask questions about Kubernetes and get grounded answers with citations to the official docs.

## Project Status

🚧 **In Development**

Currently building the preprocessing pipeline to transform raw Kubernetes markdown documentation into retrieval-ready chunks.

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

