"""RAG runtime configuration loader."""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding model settings.

    Attributes:
        model_id: Vertex AI embedding model id.
        output_dimensionality: Embedding vector dimension.
    """

    model_id: str
    output_dimensionality: int


@dataclass(frozen=True)
class GenerationConfig:
    """Answer generation model settings.

    Attributes:
        model_id: Vertex AI generation model id.
        temperature: Generation temperature.
        max_tokens: Max generated tokens.
    """

    model_id: str
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class VectorStoreConfig:
    """Vector store location.

    Attributes:
        collection_name: Chroma collection name.
        persist_directory: Chroma persist path (relative to project root).
    """

    collection_name: str
    persist_directory: str


@dataclass(frozen=True)
class RetrievalConfig:
    """Hybrid retrieval and abstention thresholds.

    Attributes:
        semantic_top_k: Vector retrieval top-k.
        lexical_top_k: BM25 retrieval top-k.
        merged_top_k: Candidate pool size before rerank.
        final_top_k: Final context size for generation.
        hybrid_weight_semantic: Weight for semantic score.
        hybrid_weight_lexical: Weight for lexical score.
        min_evidence_score: Minimum relevance score for citations.
        min_supporting_chunks: Minimum number of supporting chunks.
    """

    semantic_top_k: int
    lexical_top_k: int
    merged_top_k: int
    final_top_k: int
    hybrid_weight_semantic: float
    hybrid_weight_lexical: float
    min_evidence_score: float
    min_supporting_chunks: int


@dataclass(frozen=True)
class RerankerConfig:
    """Discovery Engine reranker settings.

    Attributes:
        enabled: Whether reranking is enabled.
        ranking_config: Discovery Engine ranking config name.
        model: Discovery Engine ranker model string.
    """

    enabled: bool
    ranking_config: str
    model: str


@dataclass(frozen=True)
class CitationConfig:
    """Citation selection policy.

    Attributes:
        strict_used_only: Whether to filter citations by used refs.
        deduplicate: Whether to deduplicate citations by chunk_id.
    """

    strict_used_only: bool
    deduplicate: bool


@dataclass(frozen=True)
class GateConfig:
    """Pre-retrieval relevance gate settings.

    Attributes:
        enabled: Whether the pre-retrieval relevance gate is enabled.
        model_id: Vertex AI model id used by the relevance gate.
    """

    enabled: bool
    model_id: str


@dataclass(frozen=True)
class PromptConfig:
    """Prompt selection.

    Attributes:
        version: Prompt config version.
        directory: Prompt root directory.
    """

    version: str
    directory: str


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation dataset and quality-gate thresholds.

    Attributes:
        dataset_path: Default golden eval dataset path.
        retrieval_recall_at_k_threshold: Min retrieval recall gate.
        mrr_threshold: Min mean reciprocal rank gate.
        abstention_accuracy_threshold: Min abstention accuracy gate.
        non_empty_answer_rate_threshold: Min non-empty answer gate.
        faithfulness_threshold: Min RAGAS faithfulness gate.
        faithfulness_judge_model: Vertex AI model id used as faithfulness judge.
        faithfulness_min_parsed: Min cases that must score for a valid gate.
        context_precision_threshold: Min RAGAS context precision gate.
        context_precision_judge_model: Vertex AI model id used as precision judge.
        context_precision_min_parsed: Min cases that must score for a valid gate.
        answer_relevancy_threshold: Min RAGAS answer relevancy gate.
        answer_relevancy_judge_model: Vertex AI model id used as relevancy judge.
        answer_relevancy_embedding_model: Embedding model id for relevancy scoring.
        answer_relevancy_min_parsed: Min cases that must score for a valid gate.
    """

    dataset_path: str
    retrieval_recall_at_k_threshold: float
    mrr_threshold: float
    abstention_accuracy_threshold: float
    non_empty_answer_rate_threshold: float
    faithfulness_threshold: float
    faithfulness_judge_model: str
    faithfulness_min_parsed: int
    context_precision_threshold: float
    context_precision_judge_model: str
    context_precision_min_parsed: int
    answer_relevancy_threshold: float
    answer_relevancy_judge_model: str
    answer_relevancy_embedding_model: str
    answer_relevancy_min_parsed: int


@dataclass(frozen=True)
class RAGConfig:
    """Runtime config for ingestion and retrieval, grouped by concern.

    Attributes:
        gcp_project: GCP project ID (from GCP_PROJECT env var).
        gcp_location: GCP region (from GCP_LOCATION env var).
        embedding: Embedding model settings.
        generation: Answer generation settings.
        vector_store: Vector store location.
        retrieval: Hybrid retrieval and abstention thresholds.
        reranker: Discovery Engine reranker settings.
        citation: Citation selection policy.
        gate: Pre-retrieval relevance gate settings.
        prompts: Prompt selection.
        evaluation: Evaluation dataset and gate thresholds.
    """

    gcp_project: str
    gcp_location: str
    embedding: EmbeddingConfig
    generation: GenerationConfig
    vector_store: VectorStoreConfig
    retrieval: RetrievalConfig
    reranker: RerankerConfig
    citation: CitationConfig
    gate: GateConfig
    prompts: PromptConfig
    evaluation: EvalConfig


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
    gate = data.get("gate", {})
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
        embedding=EmbeddingConfig(
            model_id=embedding_model_id,
            output_dimensionality=int(embedding.get("output_dimensionality", 768)),
        ),
        generation=GenerationConfig(
            model_id=generation_model_id,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        vector_store=VectorStoreConfig(
            collection_name=collection_name,
            persist_directory=persist_directory,
        ),
        retrieval=RetrievalConfig(
            semantic_top_k=int(retrieval.get("semantic_top_k", 5)),
            lexical_top_k=int(retrieval.get("lexical_top_k", 5)),
            merged_top_k=int(retrieval.get("merged_top_k", 10)),
            final_top_k=int(retrieval.get("final_top_k", 5)),
            hybrid_weight_semantic=hybrid_weight_semantic,
            hybrid_weight_lexical=hybrid_weight_lexical,
            min_evidence_score=float(retrieval.get("min_evidence_score", 0.0)),
            min_supporting_chunks=int(retrieval.get("min_supporting_chunks", 1)),
        ),
        reranker=RerankerConfig(
            enabled=reranker.get("enabled", False),
            ranking_config=reranker.get("ranking_config", "default_ranking_config"),
            model=reranker.get("model", "semantic-ranker-default@latest"),
        ),
        citation=CitationConfig(
            strict_used_only=citation.get("strict_used_only", True),
            deduplicate=citation.get("deduplicate", True),
        ),
        gate=GateConfig(
            enabled=gate.get("enabled", False),
            model_id=gate.get("model", generation_model_id),
        ),
        prompts=PromptConfig(
            version=prompts.get("version", "v1"),
            directory=prompts.get("directory", "configs/prompts"),
        ),
        evaluation=EvalConfig(
            dataset_path=evaluation.get("dataset_path", "evals/golden/v2.jsonl"),
            retrieval_recall_at_k_threshold=float(
                evaluation.get("retrieval_recall_at_k_threshold", 0.70)
            ),
            mrr_threshold=float(evaluation.get("mrr_threshold", 0.70)),
            abstention_accuracy_threshold=float(
                evaluation.get("abstention_accuracy_threshold", 0.90)
            ),
            non_empty_answer_rate_threshold=float(
                evaluation.get("non_empty_answer_rate_threshold", 0.90)
            ),
            faithfulness_threshold=float(
                evaluation.get("faithfulness_threshold", 0.90)
            ),
            faithfulness_judge_model=evaluation.get(
                "faithfulness_judge_model", "gemini-2.5-flash"
            ),
            faithfulness_min_parsed=int(
                evaluation.get("faithfulness_min_parsed", 10)
            ),
            context_precision_threshold=float(
                evaluation.get("context_precision_threshold", 0.85)
            ),
            context_precision_judge_model=evaluation.get(
                "context_precision_judge_model", "gemini-2.5-flash"
            ),
            context_precision_min_parsed=int(
                evaluation.get("context_precision_min_parsed", 10)
            ),
            answer_relevancy_threshold=float(
                evaluation.get("answer_relevancy_threshold", 0.80)
            ),
            answer_relevancy_judge_model=evaluation.get(
                "answer_relevancy_judge_model", "gemini-2.5-flash"
            ),
            answer_relevancy_embedding_model=evaluation.get(
                "answer_relevancy_embedding_model", "gemini-embedding-001"
            ),
            answer_relevancy_min_parsed=int(
                evaluation.get("answer_relevancy_min_parsed", 10)
            ),
        ),
    )
