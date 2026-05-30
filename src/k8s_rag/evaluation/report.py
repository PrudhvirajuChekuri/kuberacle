"""Reporting helpers for evaluation artifacts."""

import json
from dataclasses import asdict
from pathlib import Path

from k8s_rag.evaluation.runner import EvaluationSummary


def write_json_report(summary: EvaluationSummary, output_path: str | Path) -> None:
    """Write full evaluation summary as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(asdict(summary), file, indent=2, ensure_ascii=False)


def build_markdown_summary(summary: EvaluationSummary) -> str:
    """Build concise markdown summary for CI annotations."""
    status = "PASS" if summary.pass_gate else "FAIL"
    lines = [
        "# RAG Evaluation",
        "",
        f"- Gate status: **{status}**",
        f"- Cases: {summary.total_cases} "
        f"(answerable={summary.answerable_cases}, "
        f"unanswerable={summary.unanswerable_cases})",
        f"- retrieval_recall_at_k: {summary.retrieval_recall_at_k:.3f}",
        f"- precision_at_1: {summary.precision_at_1:.3f}",
        f"- abstention_accuracy: {summary.abstention_accuracy:.3f}",
        f"- non_empty_answer_rate: {summary.non_empty_answer_rate:.3f}",
    ]
    if summary.failed_thresholds:
        lines.extend(["", "## Failed thresholds"])
        for metric, (actual, expected) in summary.failed_thresholds.items():
            lines.append(f"- {metric}: actual={actual:.3f}, threshold={expected:.3f}")
    return "\n".join(lines) + "\n"


def write_markdown_summary(summary: EvaluationSummary, output_path: str | Path) -> None:
    """Write markdown summary file to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_markdown_summary(summary))
