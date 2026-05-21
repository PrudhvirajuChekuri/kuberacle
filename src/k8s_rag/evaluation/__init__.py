"""Evaluation utilities for offline RAG quality checks."""

from k8s_rag.evaluation.dataset import GoldenExample, load_golden_dataset
from k8s_rag.evaluation.runner import (
    EvaluationSummary,
    EvaluationThresholds,
    evaluate_dataset,
)

__all__ = [
    "GoldenExample",
    "EvaluationSummary",
    "EvaluationThresholds",
    "evaluate_dataset",
    "load_golden_dataset",
]
