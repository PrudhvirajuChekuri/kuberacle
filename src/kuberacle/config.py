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
class PricingConfig:
    """Per-query prices used to estimate spend (USD).

    Attributes:
        generation_input_per_1m_usd: Generation input price per 1M tokens.
        generation_output_per_1m_usd: Generation output price per 1M tokens.
        embedding_input_per_1m_usd: Embedding input price per 1M tokens.
        reranker_per_1k_queries_usd: Reranker price per 1000 ranking queries.
    """

    generation_input_per_1m_usd: float
    generation_output_per_1m_usd: float
    embedding_input_per_1m_usd: float
    reranker_per_1k_queries_usd: float


@dataclass(frozen=True)
class ObservabilityConfig:
    """Non-secret logging and tracing knobs for the serving layer.

    Attributes:
        service_name: Service name attached to logs and trace spans.
        log_level: Root log level name (e.g. ``INFO``).
        log_format: ``json`` for structured production logs, ``text`` for local.
        trace_sample_ratio: Fraction of traces to sample (0.0-1.0).
    """

    service_name: str
    log_level: str
    log_format: str
    trace_sample_ratio: float


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
        pricing: Per-query prices for cost estimation.
        observability: Logging and tracing knobs for the serving layer.
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
    pricing: PricingConfig
    observability: ObservabilityConfig


def _require(data: dict, path: str):
    """Return the value at a dotted key path, failing loudly when absent.

    Every key in ``configs/rag.yaml`` is required: a silent fallback default
    would create a second source of config truth and let a typo'd key quietly
    change runtime behavior.

    Args:
        data: Parsed YAML mapping.
        path: Dotted key path (e.g. ``retrieval.semantic_top_k``).

    Returns:
        The value stored at the path.

    Raises:
        RuntimeError: When any segment of the path is missing.
    """
    node = data
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise RuntimeError(f"Missing required config key '{path}' in rag.yaml.")
        node = node[part]
    return node


def load_rag_config(config_path: str | Path) -> RAGConfig:
    """Load RAG YAML config into a typed object.

    GCP project and location are read from the GCP_PROJECT and GCP_LOCATION
    environment variables. All other values come from the YAML file, and every
    key is required: the YAML is the single source of config truth, so there
    are no in-code fallback defaults.

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

    hybrid_weight_semantic = float(_require(data, "retrieval.hybrid_weight_semantic"))
    hybrid_weight_lexical = float(_require(data, "retrieval.hybrid_weight_lexical"))
    if abs(hybrid_weight_semantic + hybrid_weight_lexical - 1.0) > 1e-9:
        raise RuntimeError(
            f"hybrid_weight_semantic ({hybrid_weight_semantic}) + "
            f"hybrid_weight_lexical ({hybrid_weight_lexical}) must sum to 1.0."
        )

    return RAGConfig(
        gcp_project=gcp_project,
        gcp_location=gcp_location,
        embedding=EmbeddingConfig(
            model_id=_require(data, "models.embedding"),
            output_dimensionality=int(_require(data, "embedding.output_dimensionality")),
        ),
        generation=GenerationConfig(
            model_id=_require(data, "models.generation"),
            temperature=float(_require(data, "generation.temperature")),
            max_tokens=int(_require(data, "generation.max_tokens")),
        ),
        vector_store=VectorStoreConfig(
            collection_name=_require(data, "vector_store.collection_name"),
            persist_directory=_require(data, "vector_store.persist_directory"),
        ),
        retrieval=RetrievalConfig(
            semantic_top_k=int(_require(data, "retrieval.semantic_top_k")),
            lexical_top_k=int(_require(data, "retrieval.lexical_top_k")),
            merged_top_k=int(_require(data, "retrieval.merged_top_k")),
            final_top_k=int(_require(data, "retrieval.final_top_k")),
            hybrid_weight_semantic=hybrid_weight_semantic,
            hybrid_weight_lexical=hybrid_weight_lexical,
            min_evidence_score=float(_require(data, "retrieval.min_evidence_score")),
            min_supporting_chunks=int(_require(data, "retrieval.min_supporting_chunks")),
        ),
        reranker=RerankerConfig(
            enabled=bool(_require(data, "reranker.enabled")),
            ranking_config=_require(data, "reranker.ranking_config"),
            model=_require(data, "reranker.model"),
        ),
        citation=CitationConfig(
            strict_used_only=bool(_require(data, "citation.strict_used_only")),
            deduplicate=bool(_require(data, "citation.deduplicate")),
        ),
        gate=GateConfig(
            enabled=bool(_require(data, "gate.enabled")),
            model_id=_require(data, "gate.model"),
        ),
        prompts=PromptConfig(
            version=_require(data, "prompts.version"),
            directory=_require(data, "prompts.directory"),
        ),
        evaluation=EvalConfig(
            dataset_path=_require(data, "evaluation.dataset_path"),
            retrieval_recall_at_k_threshold=float(
                _require(data, "evaluation.retrieval_recall_at_k_threshold")
            ),
            mrr_threshold=float(_require(data, "evaluation.mrr_threshold")),
            abstention_accuracy_threshold=float(
                _require(data, "evaluation.abstention_accuracy_threshold")
            ),
            non_empty_answer_rate_threshold=float(
                _require(data, "evaluation.non_empty_answer_rate_threshold")
            ),
            faithfulness_threshold=float(
                _require(data, "evaluation.faithfulness_threshold")
            ),
            faithfulness_judge_model=_require(
                data, "evaluation.faithfulness_judge_model"
            ),
            faithfulness_min_parsed=int(
                _require(data, "evaluation.faithfulness_min_parsed")
            ),
            context_precision_threshold=float(
                _require(data, "evaluation.context_precision_threshold")
            ),
            context_precision_judge_model=_require(
                data, "evaluation.context_precision_judge_model"
            ),
            context_precision_min_parsed=int(
                _require(data, "evaluation.context_precision_min_parsed")
            ),
            answer_relevancy_threshold=float(
                _require(data, "evaluation.answer_relevancy_threshold")
            ),
            answer_relevancy_judge_model=_require(
                data, "evaluation.answer_relevancy_judge_model"
            ),
            answer_relevancy_embedding_model=_require(
                data, "evaluation.answer_relevancy_embedding_model"
            ),
            answer_relevancy_min_parsed=int(
                _require(data, "evaluation.answer_relevancy_min_parsed")
            ),
        ),
        pricing=PricingConfig(
            generation_input_per_1m_usd=float(
                _require(data, "pricing.generation_input_per_1m_usd")
            ),
            generation_output_per_1m_usd=float(
                _require(data, "pricing.generation_output_per_1m_usd")
            ),
            embedding_input_per_1m_usd=float(
                _require(data, "pricing.embedding_input_per_1m_usd")
            ),
            reranker_per_1k_queries_usd=float(
                _require(data, "pricing.reranker_per_1k_queries_usd")
            ),
        ),
        observability=ObservabilityConfig(
            service_name=str(_require(data, "observability.service_name")),
            log_level=str(_require(data, "observability.logging.level")),
            log_format=str(_require(data, "observability.logging.format")),
            trace_sample_ratio=float(_require(data, "observability.tracing.sample_ratio")),
        ),
    )
