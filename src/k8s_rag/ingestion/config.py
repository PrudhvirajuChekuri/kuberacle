"""RAG runtime configuration loader."""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RAGConfig:
    """Runtime config for ingestion and retrieval.

    Args:
        gcp_project: GCP project ID (from GCP_PROJECT env var).
        gcp_location: GCP region (from GCP_LOCATION env var).
        embedding_model_id: Vertex AI embedding model id.
        generation_model_id: Vertex AI generation model id.
        embedding_output_dimensionality: Embedding vector dimension.
        collection_name: Chroma collection name.
        persist_directory: Chroma persist path.
        semantic_top_k: Vector retrieval top-k.
        lexical_top_k: BM25 retrieval top-k.
        merged_top_k: Candidate pool size before rerank.
        final_top_k: Final context size for generation.
        hybrid_weight_semantic: Weight for semantic score.
        hybrid_weight_lexical: Weight for lexical score.
        min_evidence_score: Minimum relevance score for citations.
        min_supporting_chunks: Minimum number of supporting chunks.
        reranker_enabled: Whether reranking is enabled.
        reranker_ranking_config: Discovery Engine ranking config name.
        reranker_model: Discovery Engine ranker model string.
        reranker_top_k: Number of reranked chunks to keep.
        citation_strict_used_only: Whether to filter citations by used refs.
        citation_deduplicate: Whether to deduplicate citations by chunk_id.
        prompt_version: Prompt config version.
        prompt_directory: Prompt root directory.
        temperature: Generation temperature.
        max_tokens: Max generated tokens.
        evaluation_dataset_path: Default golden eval dataset path.
        eval_retrieval_recall_at_k_threshold: Min retrieval recall gate.
        eval_precision_at_1_threshold: Min precision@1 post-rerank gate.
        eval_abstention_accuracy_threshold: Min abstention accuracy gate.
        eval_non_empty_answer_rate_threshold: Min non-empty answer gate.
    """

    gcp_project: str
    gcp_location: str
    embedding_model_id: str
    generation_model_id: str
    embedding_output_dimensionality: int
    reranker_ranking_config: str
    reranker_model: str
    collection_name: str
    persist_directory: str
    semantic_top_k: int
    lexical_top_k: int
    merged_top_k: int
    final_top_k: int
    hybrid_weight_semantic: float
    hybrid_weight_lexical: float
    min_evidence_score: float
    min_supporting_chunks: int
    reranker_enabled: bool
    reranker_top_k: int
    citation_strict_used_only: bool
    citation_deduplicate: bool
    prompt_version: str
    prompt_directory: str
    temperature: float
    max_tokens: int
    evaluation_dataset_path: str
    eval_retrieval_recall_at_k_threshold: float
    eval_precision_at_1_threshold: float
    eval_abstention_accuracy_threshold: float
    eval_non_empty_answer_rate_threshold: float


def load_rag_config(config_path: str | Path) -> RAGConfig:
    """Load RAG YAML config into a typed object.

    GCP project and location are read from the GCP_PROJECT and GCP_LOCATION
    environment variables. All other values come from the YAML file.

    Args:
        config_path: Path to ``configs/rag.yaml``.

    Returns:
        Parsed ``RAGConfig``.

    Raises:
        RuntimeError: If GCP_PROJECT or GCP_LOCATION env vars are not set.
    """
    gcp_project = os.environ.get("GCP_PROJECT", "")
    gcp_location = os.environ.get("GCP_LOCATION", "")
    if not gcp_project:
        raise RuntimeError("GCP_PROJECT environment variable is not set.")
    if not gcp_location:
        raise RuntimeError("GCP_LOCATION environment variable is not set.")

    with open(config_path, "r") as file:
        data = yaml.safe_load(file)

    embedding = data.get("embedding", {})
    retrieval = data.get("retrieval", {})
    reranker = data.get("reranker", {})
    citation = data.get("citation", {})
    prompts = data.get("prompts", {})
    evaluation = data.get("evaluation", {})

    return RAGConfig(
        gcp_project=gcp_project,
        gcp_location=gcp_location,
        embedding_model_id=data["models"]["embedding"],
        generation_model_id=data["models"]["generation"],
        embedding_output_dimensionality=int(
            embedding.get("output_dimensionality", 768)
        ),
        reranker_ranking_config=reranker.get(
            "ranking_config", "default_ranking_config"
        ),
        reranker_model=reranker.get("model", "semantic-ranker-default@latest"),
        collection_name=data["vector_store"]["collection_name"],
        persist_directory=data["vector_store"]["persist_directory"],
        semantic_top_k=int(retrieval.get("semantic_top_k", 5)),
        lexical_top_k=int(retrieval.get("lexical_top_k", 5)),
        merged_top_k=int(retrieval.get("merged_top_k", 10)),
        final_top_k=int(retrieval.get("final_top_k", 5)),
        hybrid_weight_semantic=float(retrieval.get("hybrid_weight_semantic", 0.6)),
        hybrid_weight_lexical=float(retrieval.get("hybrid_weight_lexical", 0.4)),
        min_evidence_score=float(retrieval.get("min_evidence_score", 0.0)),
        min_supporting_chunks=int(retrieval.get("min_supporting_chunks", 1)),
        reranker_enabled=bool(reranker.get("enabled", False)),
        reranker_top_k=int(reranker.get("top_k", 5)),
        citation_strict_used_only=bool(citation.get("strict_used_only", True)),
        citation_deduplicate=bool(citation.get("deduplicate", True)),
        prompt_version=str(prompts.get("version", "v1")),
        prompt_directory=str(prompts.get("directory", "configs/prompts")),
        temperature=float(data["generation"]["temperature"]),
        max_tokens=int(data["generation"]["max_tokens"]),
        evaluation_dataset_path=str(
            evaluation.get("dataset_path", "evals/golden/v1.jsonl")
        ),
        eval_retrieval_recall_at_k_threshold=float(
            evaluation.get("retrieval_recall_at_k_threshold", 0.70)
        ),
        eval_precision_at_1_threshold=float(
            evaluation.get("precision_at_1_threshold", 0.70)
        ),
        eval_abstention_accuracy_threshold=float(
            evaluation.get("abstention_accuracy_threshold", 0.90)
        ),
        eval_non_empty_answer_rate_threshold=float(
            evaluation.get("non_empty_answer_rate_threshold", 0.90)
        ),
    )
