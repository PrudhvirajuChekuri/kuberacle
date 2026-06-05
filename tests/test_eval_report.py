"""Tests for evaluation report generation."""

import json

from k8s_rag.evaluation.report import (
    build_markdown_summary,
    write_json_report,
    write_markdown_summary,
)
from k8s_rag.evaluation.runner import EvaluationCaseResult, EvaluationSummary


def _make_summary(pass_gate: bool = True, failed_thresholds=None) -> EvaluationSummary:
    case = EvaluationCaseResult(
        case_id="q1",
        answerable=True,
        question="What is a Pod?",
        answer="A Pod runs containers [1].",
        expected_answer="A Pod is the smallest unit.",
        retrieved_chunk_ids=["c1"],
        retrieved_contexts=["A Pod runs containers."],
        citation_chunk_ids=["c1"],
        reference_chunk_ids=["c1"],
        retrieval_recall_at_k=1.0,
        precision_at_1=1.0,
        abstained=False,
        non_empty_answer=True,
        tags=["concept"],
    )
    return EvaluationSummary(
        total_cases=1,
        answerable_cases=1,
        unanswerable_cases=0,
        retrieval_recall_at_k=1.0,
        precision_at_1=1.0,
        abstention_accuracy=1.0,
        non_empty_answer_rate=1.0,
        pass_gate=pass_gate,
        failed_thresholds=failed_thresholds or {},
        case_results=[case],
    )


def test_build_markdown_summary_pass():
    """Markdown summary should show PASS and all four metrics."""
    summary = _make_summary(pass_gate=True)
    md = build_markdown_summary(summary)
    assert "**PASS**" in md
    assert "retrieval_recall_at_k: 1.000" in md
    assert "precision_at_1: 1.000" in md
    assert "abstention_accuracy: 1.000" in md
    assert "non_empty_answer_rate: 1.000" in md
    assert "Failed thresholds" not in md


def test_build_markdown_summary_fail_shows_failed_thresholds():
    """Markdown summary should list failed thresholds when gate fails."""
    summary = _make_summary(
        pass_gate=False,
        failed_thresholds={"retrieval_recall_at_k": (0.4, 0.75)},
    )
    md = build_markdown_summary(summary)
    assert "**FAIL**" in md
    assert "## Failed thresholds" in md
    assert "retrieval_recall_at_k: actual=0.400, threshold=0.750" in md


def test_write_json_report_roundtrips(tmp_path):
    """JSON report should be valid JSON containing summary fields."""
    summary = _make_summary()
    out = tmp_path / "report.json"
    write_json_report(summary, out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["total_cases"] == 1
    assert data["pass_gate"] is True
    assert len(data["case_results"]) == 1


def test_write_json_report_creates_parent_dirs(tmp_path):
    """JSON report writer should create missing parent directories."""
    out = tmp_path / "nested" / "dir" / "report.json"
    write_json_report(_make_summary(), out)
    assert out.exists()


def test_write_markdown_summary_writes_file(tmp_path):
    """Markdown report should be written to disk with correct content."""
    out = tmp_path / "report.md"
    write_markdown_summary(_make_summary(), out)
    content = out.read_text(encoding="utf-8")
    assert "**PASS**" in content


def test_write_markdown_summary_creates_parent_dirs(tmp_path):
    """Markdown report writer should create missing parent directories."""
    out = tmp_path / "nested" / "report.md"
    write_markdown_summary(_make_summary(), out)
    assert out.exists()
