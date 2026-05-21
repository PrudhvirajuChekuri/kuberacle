"""Run ingestion into ChromaDB with Bedrock embeddings.

Usage:
    python scripts/ingest.py
"""

from pathlib import Path

from k8s_rag.ingestion.config import load_rag_config
from k8s_rag.ingestion.embedder import BedrockEmbedder
from k8s_rag.ingestion.pipeline import IngestionPipeline
from k8s_rag.ingestion.vector_store import ChromaVectorStore


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "rag.yaml"
INPUT_JSONL = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"


def main() -> None:
    """Load config and run ingestion pipeline."""
    config = load_rag_config(CONFIG_PATH)
    print(f"Loading config from {CONFIG_PATH}")
    print(f"Embedding model: {config.embedding_model_id}")
    print(f"Collection: {config.collection_name}\n")

    embedder = BedrockEmbedder(
        model_id=config.embedding_model_id,
        region_name=config.aws_region,
    )
    vector_store = ChromaVectorStore(
        collection_name=config.collection_name,
        persist_directory=str(PROJECT_ROOT / config.persist_directory),
    )
    pipeline = IngestionPipeline(embedder=embedder, vector_store=vector_store)
    stats = pipeline.run(INPUT_JSONL)
    print(f"Ingested chunks: {stats['ingested_chunks']}")


if __name__ == "__main__":
    main()
