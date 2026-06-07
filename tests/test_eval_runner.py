"""Tests for offline evaluation runner and quality gates."""

from k8s_rag.evaluation.dataset import GoldenExample
from k8s_rag.evaluation.runner import EvaluationThresholds, evaluate_dataset
from k8s_rag.ingestion.schemas import RetrievedChunk
from k8s_rag.retrieval.qa import Citation, QAResult


class StubQASystem:
    """Simple QA test double keyed by question."""

    def __init__(self, outputs):
        self.outputs = outputs

    def ask(self, question, top_k=None):
        del top_k
        return self.outputs[question]


def _chunk(chunk_id: str) -> RetrievedChunk:
    """Build a retrieved chunk test fixture."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=f"content for {chunk_id}",
        metadata={"source_url": f"https://example/{chunk_id}"},
        score=0.8,
    )


def test_evaluate_dataset_passes_gate_for_good_outputs():
    """Runner should pass when metrics meet thresholds."""
    dataset = [
        GoldenExample(
            case_id="a1",
            question="What is a Pod?",
            expected_answer="A Pod is a Kubernetes unit.",
            reference_chunk_ids=["c1"],
            answerable=True,
            tags=["concept"],
        ),
        GoldenExample(
            case_id="a2",
            question="Unknown case?",
            expected_answer="No answer.",
            reference_chunk_ids=[],
            answerable=False,
            tags=["abstention"],
        ),
    ]
    outputs = {
        "What is a Pod?": QAResult(
            answer="A Pod is a Kubernetes unit [1].",
            citations=[Citation(chunk_id="c1", source_url="https://example/c1", score=0.9)],
            retrieved_chunks=[_chunk("c1")],
        ),
        "Unknown case?": QAResult(
            answer="INSUFFICIENT_EVIDENCE. Missing support.",
            citations=[],
            retrieved_chunks=[_chunk("x1")],
        ),
    }
    summary = evaluate_dataset(
        qa_system=StubQASystem(outputs),
        dataset=dataset,
        thresholds=EvaluationThresholds(
            retrieval_recall_at_k=0.5,
            mrr=0.5,
            abstention_accuracy=1.0,
            non_empty_answer_rate=1.0,
        ),
    )
    assert summary.pass_gate is True
    assert summary.failed_thresholds == {}


def test_evaluate_dataset_fails_gate_on_threshold_regression():
    """Runner should fail when a required metric dips below threshold."""
    dataset = [
        GoldenExample(
            case_id="a1",
            question="What is a Pod?",
            expected_answer="A Pod is a Kubernetes unit.",
            reference_chunk_ids=["c1"],
            answerable=True,
            tags=["concept"],
        )
    ]
    outputs = {
        "What is a Pod?": QAResult(
            answer="INSUFFICIENT_EVIDENCE. Missing support.",
            citations=[],
            retrieved_chunks=[_chunk("other")],
        )
    }
    summary = evaluate_dataset(
        qa_system=StubQASystem(outputs),
        dataset=dataset,
        thresholds=EvaluationThresholds(
            retrieval_recall_at_k=0.9,
            mrr=0.9,
            abstention_accuracy=0.0,
            non_empty_answer_rate=1.0,
        ),
    )
    assert summary.pass_gate is False
    assert "retrieval_recall_at_k" in summary.failed_thresholds
    assert "mrr" in summary.failed_thresholds
    assert "non_empty_answer_rate" in summary.failed_thresholds
