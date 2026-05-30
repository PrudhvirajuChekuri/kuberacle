"""Tests for deterministic evaluation metrics."""

from k8s_rag.evaluation.metrics import (
    is_insufficient_evidence,
    non_empty_answer,
    precision_at_1,
    retrieval_recall_at_k,
)


def test_retrieval_recall_at_k_uses_reference_overlap():
    """Recall should reflect overlap with expected chunk ids."""
    score = retrieval_recall_at_k(
        retrieved_chunk_ids=["a", "b", "d"],
        reference_chunk_ids=["a", "c"],
    )
    assert score == 0.5


def test_precision_at_1_returns_one_when_top_chunk_is_reference():
    """Precision@1 should be 1.0 when the top-ranked chunk is a reference."""
    assert precision_at_1(["a", "b", "c"], ["a"]) == 1.0


def test_precision_at_1_returns_zero_when_top_chunk_is_not_reference():
    """Precision@1 should be 0.0 when the top-ranked chunk is not a reference."""
    assert precision_at_1(["b", "a", "c"], ["a"]) == 0.0


def test_precision_at_1_returns_zero_for_empty_retrieved():
    """Precision@1 should be 0.0 when no chunks were retrieved."""
    assert precision_at_1([], ["a"]) == 0.0


def test_answer_flags_detect_abstention_and_non_empty():
    """Helper flags should classify abstention answers correctly."""
    abstention = "INSUFFICIENT_EVIDENCE. Could not verify support."
    assert is_insufficient_evidence(abstention) is True
    assert non_empty_answer(abstention) is False
    assert non_empty_answer("Pods are small units [1].") is True
