"""Run ingestion into ChromaDB with Vertex AI embeddings.

Usage:
    python scripts/ingest.py [--input PATH]
"""

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from k8s_rag.ingestion.config import load_rag_config
from k8s_rag.ingestion.embedder import VertexAIEmbedder
from k8s_rag.ingestion.pipeline import IngestionPipeline
from k8s_rag.ingestion.vector_store import ChromaVectorStore


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "rag.yaml"
DEFAULT_INPUT_JSONL = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest chunks into ChromaDB")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_JSONL),
        help="Path to preprocessed chunks JSONL.",
    )
    return parser.parse_args()


def main() -> None:
    """Load config and run ingestion pipeline."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)

    args = parse_args()
    logger.info("Loading config from %s", CONFIG_PATH)
    config = load_rag_config(CONFIG_PATH)
    logger.info("Embedding model: %s", config.embedding_model_id)
    logger.info("Collection: %s", config.collection_name)

    embedder = VertexAIEmbedder(
        model_id=config.embedding_model_id,
        gcp_project=config.gcp_project,
        gcp_location=config.gcp_location,
        output_dimensionality=config.embedding_output_dimensionality,
    )
    vector_store = ChromaVectorStore(
        collection_name=config.collection_name,
        persist_directory=str(PROJECT_ROOT / config.persist_directory),
    )
    vector_store.reset_collection()

    pipeline = IngestionPipeline(embedder=embedder, vector_store=vector_store)
    stats = pipeline.run(Path(args.input))
    logger.info("Upserted chunks: %d", stats["upserted_chunks"])


if __name__ == "__main__":
    main()
