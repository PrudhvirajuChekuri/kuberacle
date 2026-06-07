"""RAG runtime configuration loader."""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RAGConfig:
    """Runtime config for ingestion and retrieval.

    Attributes:
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
        citation_strict_used_only: Whether to filter citations by used refs.
        citation_deduplicate: Whether to deduplicate citations by chunk_id.
        prompt_version: Prompt config version.
        prompt_directory: Prompt root directory.
        temperature: Generation temperature.
        max_tokens: Max generated tokens.
        evaluation_dataset_path: Default golden eval dataset path.
        eval_retrieval_recall_at_k_threshold: Min retrieval recall gate.
        eval_mrr_threshold: Min mean reciprocal rank gate.
        eval_abstention_accuracy_threshold: Min abstention accuracy gate.
        eval_non_empty_answer_rate_threshold: Min non-empty answer gate.
        eval_faithfulness_threshold: Min RAGAS faithfulness gate.
        eval_faithfulness_judge_model: Vertex AI model ID used as faithfulness judge.
        eval_faithfulness_min_parsed: Min cases that must score successfully for the gate to be valid.
        eval_context_precision_threshold: Min RAGAS context precision gate.
        eval_context_precision_judge_model: Vertex AI model ID used as context precision judge.
        eval_context_precision_min_parsed: Min cases that must score successfully for the gate to be valid.
        eval_answer_relevancy_threshold: Min RAGAS answer relevancy gate.
        eval_answer_relevancy_judge_model: Vertex AI model ID used as answer relevancy judge.
        eval_answer_relevancy_embedding_model: Embedding model ID used for answer relevancy similarity scoring.
        eval_answer_relevancy_min_parsed: Min cases that must score successfully for the gate to be valid.
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
    citation_strict_used_only: bool
    citation_deduplicate: bool
    prompt_version: str
    prompt_directory: str
    temperature: float
    max_tokens: int
    evaluation_dataset_path: str
    eval_retrieval_recall_at_k_threshold: float
    eval_mrr_threshold: float
    eval_abstention_accuracy_threshold: float
    eval_non_empty_answer_rate_threshold: float
    eval_faithfulness_threshold: float
    eval_faithfulness_judge_model: str
    eval_faithfulness_min_parsed: int
    eval_context_precision_threshold: float
    eval_context_precision_judge_model: str
    eval_context_precision_min_parsed: int
    eval_answer_relevancy_threshold: float
    eval_answer_relevancy_judge_model: str
    eval_answer_relevancy_embedding_model: str
    eval_answer_relevancy_min_parsed: int


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
        RuntimeError: If a required YAML key is missing.
        RuntimeError: If hybrid_weight_semantic and hybrid_weight_lexical do not sum to 1.0.
    """
    gcp_project = os.environ.get("GCP_PROJECT", "")
    gcp_location = os.environ.get("GCP_LOCATION", "")
    if not gcp_project:
        raise RuntimeError("GCP_PROJECT environment variable is not set.")
    if not gcp_location:
        raise RuntimeError("GCP_LOCATION environment variable is not set.")

    with open(config_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    embedding = data.get("embedding", {})
    retrieval = data.get("retrieval", {})
    reranker = data.get("reranker", {})
    citation = data.get("citation", {})
    prompts = data.get("prompts", {})
    evaluation = data.get("evaluation", {})

    try:
        embedding_model_id = data["models"]["embedding"]
        generation_model_id = data["models"]["generation"]
    except KeyError as exc:
        raise RuntimeError(f"Missing required config key under 'models': {exc}") from exc

    try:
        collection_name = data["vector_store"]["collection_name"]
        persist_directory = data["vector_store"]["persist_directory"]
    except KeyError as exc:
        raise RuntimeError(f"Missing required config key under 'vector_store': {exc}") from exc

    try:
        temperature = float(data["generation"]["temperature"])
        max_tokens = int(data["generation"]["max_tokens"])
    except KeyError as exc:
        raise RuntimeError(f"Missing required config key under 'generation': {exc}") from exc

    hybrid_weight_semantic = float(retrieval.get("hybrid_weight_semantic", 0.6))
    hybrid_weight_lexical = float(retrieval.get("hybrid_weight_lexical", 0.4))
    if abs(hybrid_weight_semantic + hybrid_weight_lexical - 1.0) > 1e-9:
        raise RuntimeError(
            f"hybrid_weight_semantic ({hybrid_weight_semantic}) + "
            f"hybrid_weight_lexical ({hybrid_weight_lexical}) must sum to 1.0."
        )

    return RAGConfig(
        gcp_project=gcp_project,
        gcp_location=gcp_location,
        embedding_model_id=embedding_model_id,
        generation_model_id=generation_model_id,
        embedding_output_dimensionality=int(
            embedding.get("output_dimensionality", 768)
        ),
        reranker_ranking_config=reranker.get(
            "ranking_config", "default_ranking_config"
        ),
        reranker_model=reranker.get("model", "semantic-ranker-default@latest"),
        collection_name=collection_name,
        persist_directory=persist_directory,
        semantic_top_k=int(retrieval.get("semantic_top_k", 5)),
        lexical_top_k=int(retrieval.get("lexical_top_k", 5)),
        merged_top_k=int(retrieval.get("merged_top_k", 10)),
        final_top_k=int(retrieval.get("final_top_k", 5)),
        hybrid_weight_semantic=hybrid_weight_semantic,
        hybrid_weight_lexical=hybrid_weight_lexical,
        min_evidence_score=float(retrieval.get("min_evidence_score", 0.0)),
        min_supporting_chunks=int(retrieval.get("min_supporting_chunks", 1)),
        reranker_enabled=reranker.get("enabled", False),
        citation_strict_used_only=citation.get("strict_used_only", True),
        citation_deduplicate=citation.get("deduplicate", True),
        prompt_version=prompts.get("version", "v1"),
        prompt_directory=prompts.get("directory", "configs/prompts"),
        temperature=temperature,
        max_tokens=max_tokens,
        evaluation_dataset_path=evaluation.get("dataset_path", "evals/golden/v2.jsonl"),
        eval_retrieval_recall_at_k_threshold=float(
            evaluation.get("retrieval_recall_at_k_threshold", 0.70)
        ),
        eval_mrr_threshold=float(
            evaluation.get("mrr_threshold", 0.70)
        ),
        eval_abstention_accuracy_threshold=float(
            evaluation.get("abstention_accuracy_threshold", 0.90)
        ),
        eval_non_empty_answer_rate_threshold=float(
            evaluation.get("non_empty_answer_rate_threshold", 0.90)
        ),
        eval_faithfulness_threshold=float(
            evaluation.get("faithfulness_threshold", 0.90)
        ),
        eval_faithfulness_judge_model=evaluation.get(
            "faithfulness_judge_model", "gemini-2.5-flash"
        ),
        eval_faithfulness_min_parsed=int(
            evaluation.get("faithfulness_min_parsed", 10)
        ),
        eval_context_precision_threshold=float(
            evaluation.get("context_precision_threshold", 0.85)
        ),
        eval_context_precision_judge_model=evaluation.get(
            "context_precision_judge_model", "gemini-2.5-flash"
        ),
        eval_context_precision_min_parsed=int(
            evaluation.get("context_precision_min_parsed", 10)
        ),
        eval_answer_relevancy_threshold=float(
            evaluation.get("answer_relevancy_threshold", 0.80)
        ),
        eval_answer_relevancy_judge_model=evaluation.get(
            "answer_relevancy_judge_model", "gemini-2.5-flash"
        ),
        eval_answer_relevancy_embedding_model=evaluation.get(
            "answer_relevancy_embedding_model", "gemini-embedding-001"
        ),
        eval_answer_relevancy_min_parsed=int(
            evaluation.get("answer_relevancy_min_parsed", 10)
        ),
    )
