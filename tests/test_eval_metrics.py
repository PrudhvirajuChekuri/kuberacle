"""Tests for deterministic evaluation metrics."""

from k8s_rag.evaluation.metrics import (
    is_insufficient_evidence,
    mean_reciprocal_rank,
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


def test_mean_reciprocal_rank_returns_one_when_top_chunk_is_reference():
    """MRR should be 1.0 when the first relevant chunk is at rank 1."""
    assert mean_reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0


def test_mean_reciprocal_rank_returns_half_when_relevant_at_rank_2():
    """MRR should be 0.5 when the first relevant chunk is at rank 2."""
    assert mean_reciprocal_rank(["b", "a", "c"], ["a"]) == 0.5


def test_mean_reciprocal_rank_returns_third_when_relevant_at_rank_3():
    """MRR should be 1/3 when the first relevant chunk is at rank 3."""
    assert abs(mean_reciprocal_rank(["b", "c", "a"], ["a"]) - 1 / 3) < 1e-9


def test_mean_reciprocal_rank_returns_zero_for_empty_retrieved():
    """MRR should be 0.0 when no chunks were retrieved."""
    assert mean_reciprocal_rank([], ["a"]) == 0.0


def test_mean_reciprocal_rank_returns_zero_when_no_relevant_chunk_found():
    """MRR should be 0.0 when none of the retrieved chunks are relevant."""
    assert mean_reciprocal_rank(["b", "c"], ["a"]) == 0.0


def test_answer_flags_detect_abstention_and_non_empty():
    """Helper flags should classify abstention answers correctly."""
    abstention = "INSUFFICIENT_EVIDENCE. Could not verify support."
    assert is_insufficient_evidence(abstention) is True
    assert non_empty_answer(abstention) is False
    assert non_empty_answer("Pods are small units [1].") is True
