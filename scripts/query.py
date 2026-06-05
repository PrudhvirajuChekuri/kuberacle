"""Query the semantic RAG pipeline.

Usage:
    python scripts/query.py "What is a Pod?"
"""

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from k8s_rag.ingestion.config import load_rag_config
from k8s_rag.ingestion.embedder import VertexAIEmbedder
from k8s_rag.ingestion.vector_store import ChromaVectorStore
from k8s_rag.retrieval.generator import VertexAIAnswerGenerator
from k8s_rag.retrieval.prompts import load_prompt_bundle
from k8s_rag.retrieval.qa import RAGQASystem
from k8s_rag.retrieval.reranker import DiscoveryEngineReranker
from k8s_rag.retrieval.retriever import HybridRetriever, SemanticRetriever
from k8s_rag.retrieval.bm25 import BM25Retriever


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "rag.yaml"


def parse_args() -> argparse.Namespace:
    """Parse query CLI arguments."""
    parser = argparse.ArgumentParser(description="Ask k8s-docs RAG a question")
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--top-k", type=int, default=None, help="Retrieval depth")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print retrieval/prompt runtime metadata",
    )
    return parser.parse_args()


def main() -> None:
    """Run retrieval and answer generation for one question."""
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    config = load_rag_config(CONFIG_PATH)

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
    semantic = SemanticRetriever(
        embedder=embedder,
        vector_store=vector_store,
        top_k=config.semantic_top_k,
    )
    all_chunks = vector_store.fetch_all_chunks()
    lexical = BM25Retriever(chunks=all_chunks, top_k=config.lexical_top_k)
    reranker = DiscoveryEngineReranker(
        gcp_project=config.gcp_project,
        ranking_config=config.reranker_ranking_config,
        model=config.reranker_model,
        enabled=config.reranker_enabled,
    )
    retriever = HybridRetriever(
        semantic_retriever=semantic,
        bm25_retriever=lexical,
        reranker=reranker,
        semantic_top_k=config.semantic_top_k,
        lexical_top_k=config.lexical_top_k,
        merged_top_k=config.merged_top_k,
        final_top_k=config.final_top_k,
        semantic_weight=config.hybrid_weight_semantic,
        lexical_weight=config.hybrid_weight_lexical,
    )
    prompt_bundle = load_prompt_bundle(
        base_dir=str(PROJECT_ROOT / config.prompt_directory),
        version=config.prompt_version,
    )
    generator = VertexAIAnswerGenerator(
        model_id=config.generation_model_id,
        gcp_project=config.gcp_project,
        gcp_location=config.gcp_location,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        prompt_bundle=prompt_bundle,
    )
    qa = RAGQASystem(
        retriever=retriever,
        generator=generator,
        min_evidence_score=config.min_evidence_score,
        min_supporting_chunks=config.min_supporting_chunks,
        strict_used_only=config.citation_strict_used_only,
        deduplicate_citations=config.citation_deduplicate,
    )
    result = qa.ask(args.question, top_k=args.top_k)

    print("\nAnswer:\n")
    print(result.answer)
    print("\nCitations:")
    for citation in result.citations:
        print(f"- {citation.source_url} ({citation.chunk_id})")
    if args.verbose:
        print("\nRuntime:")
        print(f"- prompt_version: {config.prompt_version}")
        print("- retrieval_mode: semantic+bm25+rerank")


if __name__ == "__main__":
    main()
