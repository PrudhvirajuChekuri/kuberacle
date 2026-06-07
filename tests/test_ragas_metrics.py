"""Tests for RAGAS faithfulness and context precision metrics."""

from k8s_rag.evaluation.ragas_metrics import (
    AnswerRelevancyResult,
    ContextPrecisionResult,
    FaithfulnessResult,
)


def test_faithfulness_result_full_parse():
    """FaithfulnessResult should hold mean and counts correctly when all cases parse."""
    result = FaithfulnessResult(mean=0.960, parsed_count=12, total_count=12)
    assert result.mean == 0.960
    assert result.parsed_count == 12
    assert result.total_count == 12


def test_faithfulness_result_partial_parse():
    """FaithfulnessResult should reflect partial parse correctly."""
    result = FaithfulnessResult(mean=1.0, parsed_count=9, total_count=12)
    assert result.parsed_count == 9
    assert result.total_count == 12


def test_faithfulness_result_is_immutable():
    """FaithfulnessResult should be frozen."""
    result = FaithfulnessResult(mean=0.9, parsed_count=10, total_count=10)
    try:
        result.mean = 0.5  # type: ignore[misc]
        assert False, "Should have raised"
    except Exception:
        pass


def test_context_precision_result_full_parse():
    """ContextPrecisionResult should hold mean and counts correctly when all cases parse."""
    result = ContextPrecisionResult(mean=0.902, parsed_count=12, total_count=12)
    assert result.mean == 0.902
    assert result.parsed_count == 12
    assert result.total_count == 12


def test_context_precision_result_partial_parse():
    """ContextPrecisionResult should reflect partial parse correctly."""
    result = ContextPrecisionResult(mean=0.85, parsed_count=10, total_count=12)
    assert result.parsed_count == 10
    assert result.total_count == 12


def test_context_precision_result_is_immutable():
    """ContextPrecisionResult should be frozen."""
    result = ContextPrecisionResult(mean=0.9, parsed_count=10, total_count=10)
    try:
        result.mean = 0.5  # type: ignore[misc]
        assert False, "Should have raised"
    except Exception:
        pass


def test_answer_relevancy_result_full_parse():
    """AnswerRelevancyResult should hold mean and counts correctly when all cases parse."""
    result = AnswerRelevancyResult(mean=0.830, parsed_count=12, total_count=12)
    assert result.mean == 0.830
    assert result.parsed_count == 12
    assert result.total_count == 12


def test_answer_relevancy_result_partial_parse():
    """AnswerRelevancyResult should reflect partial parse correctly."""
    result = AnswerRelevancyResult(mean=0.82, parsed_count=10, total_count=12)
    assert result.parsed_count == 10
    assert result.total_count == 12


def test_answer_relevancy_result_is_immutable():
    """AnswerRelevancyResult should be frozen."""
    result = AnswerRelevancyResult(mean=0.8, parsed_count=10, total_count=10)
    try:
        result.mean = 0.5  # type: ignore[misc]
        assert False, "Should have raised"
    except Exception:
        pass
