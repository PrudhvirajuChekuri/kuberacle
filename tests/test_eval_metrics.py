"""Tests for deterministic evaluation metrics."""

from k8s_rag.evaluation.metrics import (
    citation_precision,
    is_insufficient_evidence,
    non_empty_answer,
    retrieval_recall_at_k,
)


def test_retrieval_recall_at_k_uses_reference_overlap():
    """Recall should reflect overlap with expected chunk ids."""
    score = retrieval_recall_at_k(
        retrieved_chunk_ids=["a", "b", "d"],
        reference_chunk_ids=["a", "c"],
    )
    assert score == 0.5


def test_citation_precision_returns_zero_without_citations():
    """Precision should be zero when answer has no citations."""
    assert citation_precision([], ["a"]) == 0.0


def test_answer_flags_detect_abstention_and_non_empty():
    """Helper flags should classify abstention answers correctly."""
    abstention = "INSUFFICIENT_EVIDENCE. Could not verify support."
    assert is_insufficient_evidence(abstention) is True
    assert non_empty_answer(abstention) is False
    assert non_empty_answer("Pods are small units [1].") is True
