"""Run offline evaluation for the RAG pipeline.

Usage:
    python -m kuberacle evaluate
"""

import argparse
import json
import logging
from pathlib import Path
from kuberacle.cli._root import project_root

from dotenv import load_dotenv

load_dotenv(project_root() / ".env")

from kuberacle.evaluation.dataset import load_golden_dataset
from kuberacle.evaluation.ragas_metrics import (
    compute_answer_relevancy,
    compute_context_precision,
    compute_faithfulness,
)
from kuberacle.evaluation.report import (
    build_markdown_summary,
    write_json_report,
    write_markdown_summary,
)
from kuberacle.evaluation.runner import EvaluationThresholds, evaluate_dataset
from kuberacle.config import load_rag_config
from kuberacle.factory import build_qa_system


PROJECT_ROOT = project_root()
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
        default="full",
        help="deterministic: retrieval/generation gates only; full: also runs RAGAS gates (default).",
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


def determine_exit_code(pass_gate: bool) -> int:
    """Return process exit code for CI usage."""
    return 0 if pass_gate else 1


def main() -> None:
    """Execute offline evaluation and write artifacts."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.WARNING,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)

    args = parse_args()
    config = load_rag_config(CONFIG_PATH)

    dataset_path = resolve_path(args.dataset or config.evaluation.dataset_path)
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
    qa_system = build_qa_system(config, PROJECT_ROOT)
    thresholds = EvaluationThresholds(
        retrieval_recall_at_k=config.evaluation.retrieval_recall_at_k_threshold,
        mrr=config.evaluation.mrr_threshold,
        abstention_accuracy=config.evaluation.abstention_accuracy_threshold,
        non_empty_answer_rate=config.evaluation.non_empty_answer_rate_threshold,
    )
    summary = evaluate_dataset(
        qa_system=qa_system,
        dataset=dataset,
        thresholds=thresholds,
        top_k=args.top_k,
    )

    faithfulness_result = None
    context_precision_result = None
    answer_relevancy_result = None
    ragas_passed = True

    if args.mode == "full":
        faithfulness_result = compute_faithfulness(
            case_results=summary.case_results,
            gcp_project=config.gcp_project,
            gcp_location=config.gcp_location,
            judge_model=config.evaluation.faithfulness_judge_model,
        )
        faithfulness_passed = (
            faithfulness_result.parsed_count >= config.evaluation.faithfulness_min_parsed
            and faithfulness_result.mean >= config.evaluation.faithfulness_threshold
        )
        if not faithfulness_passed:
            logger.warning(
                "Faithfulness gate FAILED: mean=%.3f (threshold=%.3f), parsed=%d/%d (min=%d)",
                faithfulness_result.mean,
                config.evaluation.faithfulness_threshold,
                faithfulness_result.parsed_count,
                faithfulness_result.total_count,
                config.evaluation.faithfulness_min_parsed,
            )

        context_precision_result = compute_context_precision(
            case_results=summary.case_results,
            gcp_project=config.gcp_project,
            gcp_location=config.gcp_location,
            judge_model=config.evaluation.context_precision_judge_model,
        )
        context_precision_passed = (
            context_precision_result.parsed_count >= config.evaluation.context_precision_min_parsed
            and context_precision_result.mean >= config.evaluation.context_precision_threshold
        )
        if not context_precision_passed:
            logger.warning(
                "Context precision gate FAILED: mean=%.3f (threshold=%.3f), parsed=%d/%d (min=%d)",
                context_precision_result.mean,
                config.evaluation.context_precision_threshold,
                context_precision_result.parsed_count,
                context_precision_result.total_count,
                config.evaluation.context_precision_min_parsed,
            )

        answer_relevancy_result = compute_answer_relevancy(
            case_results=summary.case_results,
            gcp_project=config.gcp_project,
            gcp_location=config.gcp_location,
            judge_model=config.evaluation.answer_relevancy_judge_model,
            embedding_model=config.evaluation.answer_relevancy_embedding_model,
        )
        answer_relevancy_passed = (
            answer_relevancy_result.parsed_count >= config.evaluation.answer_relevancy_min_parsed
            and answer_relevancy_result.mean >= config.evaluation.answer_relevancy_threshold
        )
        if not answer_relevancy_passed:
            logger.warning(
                "Answer relevancy gate FAILED: mean=%.3f (threshold=%.3f), parsed=%d/%d (min=%d)",
                answer_relevancy_result.mean,
                config.evaluation.answer_relevancy_threshold,
                answer_relevancy_result.parsed_count,
                answer_relevancy_result.total_count,
                config.evaluation.answer_relevancy_min_parsed,
            )

        ragas_passed = faithfulness_passed and context_precision_passed and answer_relevancy_passed

    json_path = resolve_path(args.json_out)
    md_path = resolve_path(args.md_out)
    write_json_report(summary, json_path)
    write_markdown_summary(
        summary, md_path, faithfulness_result, context_precision_result, answer_relevancy_result, ragas_passed
    )

    print(build_markdown_summary(
        summary, faithfulness_result, context_precision_result, answer_relevancy_result, ragas_passed
    ))
    logger.info("JSON report: %s", json_path)
    logger.info("Markdown report: %s", md_path)

    raise SystemExit(determine_exit_code(summary.pass_gate and ragas_passed))


if __name__ == "__main__":
    main()
