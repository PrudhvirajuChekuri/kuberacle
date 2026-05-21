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
        top_k: Default retrieval top-k.
        temperature: Generation temperature.
        max_tokens: Max generated tokens.
    """

    aws_region: str
    embedding_model_id: str
    generation_model_id: str
    collection_name: str
    persist_directory: str
    top_k: int
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

    return RAGConfig(
        aws_region=data["aws"]["region"],
        embedding_model_id=data["models"]["embedding"],
        generation_model_id=data["models"]["generation"],
        collection_name=data["vector_store"]["collection_name"],
        persist_directory=data["vector_store"]["persist_directory"],
        top_k=int(data["retrieval"]["top_k"]),
        temperature=float(data["generation"]["temperature"]),
        max_tokens=int(data["generation"]["max_tokens"]),
    )
