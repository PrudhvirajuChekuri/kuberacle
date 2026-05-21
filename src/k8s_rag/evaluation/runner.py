"""Offline evaluation runner for dataset-level metrics and gating."""

from dataclasses import dataclass

from k8s_rag.evaluation.dataset import GoldenExample
from k8s_rag.evaluation.metrics import (
    citation_precision,
    is_insufficient_evidence,
    non_empty_answer,
    retrieval_recall_at_k,
)


@dataclass(frozen=True)
class EvaluationThresholds:
    """Thresholds for deterministic quality gates."""

    retrieval_recall_at_k: float
    citation_precision: float
    abstention_accuracy: float
    non_empty_answer_rate: float


@dataclass(frozen=True)
class EvaluationCaseResult:
    """Per-case evaluation outputs."""

    case_id: str
    answerable: bool
    question: str
    answer: str
    expected_answer: str
    retrieved_chunk_ids: list[str]
    retrieved_contexts: list[str]
    citation_chunk_ids: list[str]
    reference_chunk_ids: list[str]
    retrieval_recall_at_k: float | None
    citation_precision: float | None
    abstained: bool
    non_empty_answer: bool
    tags: list[str]


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregated metrics and gate status."""

    total_cases: int
    answerable_cases: int
    unanswerable_cases: int
    retrieval_recall_at_k: float
    citation_precision: float
    abstention_accuracy: float
    non_empty_answer_rate: float
    pass_gate: bool
    failed_thresholds: dict[str, tuple[float, float]]
    case_results: list[EvaluationCaseResult]


def _mean(values: list[float]) -> float:
    """Compute arithmetic mean; return 0 for empty input."""
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def evaluate_dataset(
    qa_system,
    dataset: list[GoldenExample],
    thresholds: EvaluationThresholds,
    top_k: int | None = None,
) -> EvaluationSummary:
    """Evaluate RAG QA system over a golden dataset.

    Args:
        qa_system: Configured QA system exposing ``ask(question, top_k=None)``.
        dataset: Golden evaluation rows.
        thresholds: Gate thresholds.
        top_k: Optional retrieval depth override.

    Returns:
        Dataset-level summary and per-case outputs.
    """
    case_results: list[EvaluationCaseResult] = []
    answerable_count = 0
    unanswerable_count = 0
    retrieval_scores: list[float] = []
    citation_scores: list[float] = []
    abstention_hits: list[float] = []
    non_empty_hits: list[float] = []

    for row in dataset:
        result = qa_system.ask(row.question, top_k=top_k)
        retrieved_chunk_ids = [chunk.chunk_id for chunk in result.retrieved_chunks]
        retrieved_contexts = [chunk.content for chunk in result.retrieved_chunks]
        citation_chunk_ids = [citation.chunk_id for citation in result.citations]
        abstained = is_insufficient_evidence(result.answer)
        answered_non_empty = non_empty_answer(result.answer)

        if row.answerable:
            answerable_count += 1
            recall = retrieval_recall_at_k(retrieved_chunk_ids, row.reference_chunk_ids)
            precision = citation_precision(citation_chunk_ids, row.reference_chunk_ids)
            retrieval_scores.append(recall)
            citation_scores.append(precision)
            non_empty_hits.append(1.0 if answered_non_empty else 0.0)
        else:
            unanswerable_count += 1
            recall = None
            precision = None
            abstention_hits.append(1.0 if abstained else 0.0)

        case_results.append(
            EvaluationCaseResult(
                case_id=row.case_id,
                answerable=row.answerable,
                question=row.question,
                answer=result.answer,
                expected_answer=row.expected_answer,
                retrieved_chunk_ids=retrieved_chunk_ids,
                retrieved_contexts=retrieved_contexts,
                citation_chunk_ids=citation_chunk_ids,
                reference_chunk_ids=row.reference_chunk_ids,
                retrieval_recall_at_k=recall,
                citation_precision=precision,
                abstained=abstained,
                non_empty_answer=answered_non_empty,
                tags=row.tags,
            )
        )

    retrieval_metric = _mean(retrieval_scores)
    citation_metric = _mean(citation_scores)
    abstention_metric = 1.0 if unanswerable_count == 0 else _mean(abstention_hits)
    non_empty_metric = _mean(non_empty_hits)

    checks = {
        "retrieval_recall_at_k": (
            retrieval_metric,
            thresholds.retrieval_recall_at_k,
        ),
        "citation_precision": (
            citation_metric,
            thresholds.citation_precision,
        ),
        "abstention_accuracy": (
            abstention_metric,
            thresholds.abstention_accuracy,
        ),
        "non_empty_answer_rate": (
            non_empty_metric,
            thresholds.non_empty_answer_rate,
        ),
    }
    failed_thresholds = {
        name: (actual, expected)
        for name, (actual, expected) in checks.items()
        if actual < expected
    }

    return EvaluationSummary(
        total_cases=len(dataset),
        answerable_cases=answerable_count,
        unanswerable_cases=unanswerable_count,
        retrieval_recall_at_k=retrieval_metric,
        citation_precision=citation_metric,
        abstention_accuracy=abstention_metric,
        non_empty_answer_rate=non_empty_metric,
        pass_gate=not failed_thresholds,
        failed_thresholds=failed_thresholds,
        case_results=case_results,
    )
