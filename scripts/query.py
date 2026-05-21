"""Query the semantic RAG pipeline.

Usage:
    python scripts/query.py "What is a Pod?"
"""

import argparse
from pathlib import Path

from k8s_rag.ingestion.config import load_rag_config
from k8s_rag.ingestion.embedder import BedrockEmbedder
from k8s_rag.ingestion.vector_store import ChromaVectorStore
from k8s_rag.retrieval.generator import BedrockAnswerGenerator
from k8s_rag.retrieval.qa import RAGQASystem
from k8s_rag.retrieval.retriever import SemanticRetriever


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "rag.yaml"


def parse_args() -> argparse.Namespace:
    """Parse query CLI arguments."""
    parser = argparse.ArgumentParser(description="Ask k8s-docs RAG a question")
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--top-k", type=int, default=None, help="Retrieval depth")
    return parser.parse_args()


def main() -> None:
    """Run retrieval and answer generation for one question."""
    args = parse_args()
    config = load_rag_config(CONFIG_PATH)

    embedder = BedrockEmbedder(
        model_id=config.embedding_model_id,
        region_name=config.aws_region,
    )
    vector_store = ChromaVectorStore(
        collection_name=config.collection_name,
        persist_directory=str(PROJECT_ROOT / config.persist_directory),
    )
    retriever = SemanticRetriever(embedder=embedder, vector_store=vector_store, top_k=config.top_k)
    generator = BedrockAnswerGenerator(
        model_id=config.generation_model_id,
        region_name=config.aws_region,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    qa = RAGQASystem(retriever=retriever, generator=generator)
    result = qa.ask(args.question, top_k=args.top_k)

    print("\nAnswer:\n")
    print(result.answer)
    print("\nCitations:")
    for citation in result.citations:
        print(f"- {citation.source_url} ({citation.chunk_id})")


if __name__ == "__main__":
    main()
