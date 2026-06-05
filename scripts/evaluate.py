"""Run offline evaluation for the RAG pipeline.

Usage:
    python scripts/evaluate.py
"""

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from k8s_rag.evaluation.dataset import load_golden_dataset
from k8s_rag.evaluation.ragas_metrics import run_optional_ragas_metrics
from k8s_rag.evaluation.report import (
    build_markdown_summary,
    write_json_report,
    write_markdown_summary,
)
from k8s_rag.evaluation.runner import EvaluationThresholds, evaluate_dataset
from k8s_rag.ingestion.config import load_rag_config
from k8s_rag.ingestion.embedder import VertexAIEmbedder
from k8s_rag.ingestion.vector_store import ChromaVectorStore
from k8s_rag.retrieval.bm25 import BM25Retriever
from k8s_rag.retrieval.generator import VertexAIAnswerGenerator
from k8s_rag.retrieval.prompts import load_prompt_bundle
from k8s_rag.retrieval.qa import RAGQASystem
from k8s_rag.retrieval.reranker import DiscoveryEngineReranker
from k8s_rag.retrieval.retriever import HybridRetriever, SemanticRetriever


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "rag.yaml"


def parse_args() -> argparse.Namespace:
    """Parse evaluation CLI arguments."""
    parser = argparse.ArgumentParser(description="Run offline RAG evaluation")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to golden dataset JSONL (defaults to config value).",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Retrieval depth override.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of evaluation cases.",
    )
    parser.add_argument(
        "--tags",
        default=None,
        help="Optional comma-separated tag filter (matches if case has any tag).",
    )
    parser.add_argument(
        "--mode",
        choices=["deterministic", "full"],
        default="deterministic",
        help="deterministic: hard-gate metrics only; full: include optional RAGAS report.",
    )
    parser.add_argument(
        "--json-out",
        default="artifacts/evals/latest-eval.json",
        help="JSON summary output path relative to project root.",
    )
    parser.add_argument(
        "--md-out",
        default="artifacts/evals/latest-eval.md",
        help="Markdown summary output path relative to project root.",
    )
    return parser.parse_args()


def resolve_path(path_like: str) -> Path:
    """Resolve project-relative or absolute path."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def build_qa_system(config):
    """Build hybrid QA system using runtime config."""
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
    return RAGQASystem(
        retriever=retriever,
        generator=generator,
        min_evidence_score=config.min_evidence_score,
        min_supporting_chunks=config.min_supporting_chunks,
        strict_used_only=config.citation_strict_used_only,
        deduplicate_citations=config.citation_deduplicate,
    )


def determine_exit_code(pass_gate: bool) -> int:
    """Return process exit code for CI usage."""
    return 0 if pass_gate else 1


def main() -> None:
    """Execute offline evaluation and write artifacts."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    args = parse_args()
    config = load_rag_config(CONFIG_PATH)

    dataset_path = resolve_path(args.dataset or config.evaluation_dataset_path)
    dataset = load_golden_dataset(dataset_path)
    if args.tags:
        selected_tags = {tag.strip() for tag in args.tags.split(",") if tag.strip()}
        dataset = [
            row for row in dataset
            if any(tag in selected_tags for tag in row.tags)
        ]
    if args.limit is not None:
        dataset = dataset[: args.limit]
    if not dataset:
        raise SystemExit("No evaluation cases selected after applying filters.")
    qa_system = build_qa_system(config)
    thresholds = EvaluationThresholds(
        retrieval_recall_at_k=config.eval_retrieval_recall_at_k_threshold,
        precision_at_1=config.eval_precision_at_1_threshold,
        abstention_accuracy=config.eval_abstention_accuracy_threshold,
        non_empty_answer_rate=config.eval_non_empty_answer_rate_threshold,
    )
    summary = evaluate_dataset(
        qa_system=qa_system,
        dataset=dataset,
        thresholds=thresholds,
        top_k=args.top_k,
    )

    json_path = resolve_path(args.json_out)
    md_path = resolve_path(args.md_out)
    write_json_report(summary, json_path)
    write_markdown_summary(summary, md_path)

    print(build_markdown_summary(summary))
    logger.info("JSON report: %s", json_path)
    logger.info("Markdown report: %s", md_path)

    if args.mode == "full":
        ragas_result = run_optional_ragas_metrics(summary.case_results)
        ragas_path = md_path.parent / "latest-ragas.json"
        ragas_path.write_text(json.dumps(ragas_result, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("RAGAS report: %s", ragas_path)
        if ragas_result.get("status") != "ok":
            logger.warning("RAGAS status: %s — %s", ragas_result.get("status"), ragas_result.get("reason"))

    raise SystemExit(determine_exit_code(summary.pass_gate))


if __name__ == "__main__":
    main()
