"""RAG runtime configuration loader."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RAGConfig:
    """Runtime config for ingestion and retrieval.

    Args:
        aws_region: AWS region for Bedrock runtime calls.
        embedding_model_id: Bedrock embedding model id.
        generation_model_id: Bedrock generation model id.
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
        reranker_model_id: Bedrock reranker model id.
        reranker_top_k: Number of reranked chunks to keep.
        citation_strict_used_only: Whether to filter citations by used refs.
        citation_deduplicate: Whether to deduplicate citations by chunk_id.
        prompt_version: Prompt config version.
        prompt_directory: Prompt root directory.
        temperature: Generation temperature.
        max_tokens: Max generated tokens.
    """

    aws_region: str
    embedding_model_id: str
    generation_model_id: str
    reranker_model_id: str
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


def load_rag_config(config_path: str | Path) -> RAGConfig:
    """Load RAG YAML config into a typed object.

    Args:
        config_path: Path to `configs/rag.yaml`.

    Returns:
        Parsed ``RAGConfig``.
    """

    with open(config_path, "r") as file:
        data = yaml.safe_load(file)

    retrieval = data.get("retrieval", {})
    reranker = data.get("reranker", {})
    citation = data.get("citation", {})
    prompts = data.get("prompts", {})

    return RAGConfig(
        aws_region=data["aws"]["region"],
        embedding_model_id=data["models"]["embedding"],
        generation_model_id=data["models"]["generation"],
        reranker_model_id=data["models"].get("reranker", "cohere.rerank-v3-5:0"),
        collection_name=data["vector_store"]["collection_name"],
        persist_directory=data["vector_store"]["persist_directory"],
        semantic_top_k=int(retrieval.get("semantic_top_k", retrieval.get("top_k", 5))),
        lexical_top_k=int(retrieval.get("lexical_top_k", 5)),
        merged_top_k=int(retrieval.get("merged_top_k", 10)),
        final_top_k=int(retrieval.get("final_top_k", retrieval.get("top_k", 5))),
        hybrid_weight_semantic=float(retrieval.get("hybrid_weight_semantic", 0.6)),
        hybrid_weight_lexical=float(retrieval.get("hybrid_weight_lexical", 0.4)),
        min_evidence_score=float(retrieval.get("min_evidence_score", 0.0)),
        min_supporting_chunks=int(retrieval.get("min_supporting_chunks", 1)),
        reranker_enabled=bool(reranker.get("enabled", False)),
        reranker_top_k=int(reranker.get("top_k", retrieval.get("top_k", 5))),
        citation_strict_used_only=bool(citation.get("strict_used_only", True)),
        citation_deduplicate=bool(citation.get("deduplicate", True)),
        prompt_version=str(prompts.get("version", "v1")),
        prompt_directory=str(prompts.get("directory", "configs/prompts")),
        temperature=float(data["generation"]["temperature"]),
        max_tokens=int(data["generation"]["max_tokens"]),
    )
