"""Reporting helpers for evaluation artifacts."""

import json
from dataclasses import asdict
from pathlib import Path

from kuberacle.evaluation.ragas_metrics import (
    AnswerRelevancyResult,
    ContextPrecisionResult,
    FaithfulnessResult,
)
from kuberacle.evaluation.runner import EvaluationSummary


def write_json_report(summary: EvaluationSummary, output_path: str | Path) -> None:
    """Write full evaluation summary as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(asdict(summary), file, indent=2, ensure_ascii=False)


def build_markdown_summary(
    summary: EvaluationSummary,
    faithfulness: FaithfulnessResult | None = None,
    context_precision: ContextPrecisionResult | None = None,
    answer_relevancy: AnswerRelevancyResult | None = None,
    ragas_passed: bool = True,
) -> str:
    """Build concise markdown summary for CI annotations.

    Args:
        summary: Aggregated deterministic evaluation metrics.
        faithfulness: Optional RAGAS faithfulness result to include in the report.
        context_precision: Optional RAGAS context precision result to include in the report.
        answer_relevancy: Optional RAGAS answer relevancy result to include in the report.
        ragas_passed: Whether all RAGAS gates passed.

    Returns:
        Markdown string.
    """
    status = "PASS" if (summary.pass_gate and ragas_passed) else "FAIL"
    lines = [
        "# RAG Evaluation",
        "",
        f"- Gate status: **{status}**",
        f"- Cases: {summary.total_cases} "
        f"(answerable={summary.answerable_cases}, "
        f"unanswerable={summary.unanswerable_cases})",
        f"- retrieval_recall_at_k: {summary.retrieval_recall_at_k:.3f}",
        f"- mrr: {summary.mrr:.3f}",
        f"- abstention_accuracy: {summary.abstention_accuracy:.3f}",
        f"- non_empty_answer_rate: {summary.non_empty_answer_rate:.3f}",
    ]
    if faithfulness is not None:
        lines.append(
            f"- faithfulness: {faithfulness.mean:.3f} "
            f"({faithfulness.parsed_count}/{faithfulness.total_count} parsed)"
        )
    if context_precision is not None:
        lines.append(
            f"- context_precision: {context_precision.mean:.3f} "
            f"({context_precision.parsed_count}/{context_precision.total_count} parsed)"
        )
    if answer_relevancy is not None:
        lines.append(
            f"- answer_relevancy: {answer_relevancy.mean:.3f} "
            f"({answer_relevancy.parsed_count}/{answer_relevancy.total_count} parsed)"
        )
    if summary.failed_thresholds:
        lines.extend(["", "## Failed thresholds"])
        for metric, (actual, expected) in summary.failed_thresholds.items():
            lines.append(f"- {metric}: actual={actual:.3f}, threshold={expected:.3f}")
    return "\n".join(lines) + "\n"


def write_markdown_summary(
    summary: EvaluationSummary,
    output_path: str | Path,
    faithfulness: FaithfulnessResult | None = None,
    context_precision: ContextPrecisionResult | None = None,
    answer_relevancy: AnswerRelevancyResult | None = None,
    ragas_passed: bool = True,
) -> None:
    """Write markdown summary file to disk.

    Args:
        summary: Aggregated deterministic evaluation metrics.
        output_path: Destination file path.
        faithfulness: Optional RAGAS faithfulness result to include in the report.
        context_precision: Optional RAGAS context precision result to include in the report.
        answer_relevancy: Optional RAGAS answer relevancy result to include in the report.
        ragas_passed: Whether all RAGAS gates passed.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_markdown_summary(summary, faithfulness, context_precision, answer_relevancy, ragas_passed),
        encoding="utf-8",
    )
