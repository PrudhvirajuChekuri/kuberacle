"""Tests for evaluate CLI helper contracts."""

from kuberacle.cli import evaluate


def test_determine_exit_code_contract():
    """CLI helper should map gate pass/fail to process exit code."""
    assert evaluate.determine_exit_code(True) == 0
    assert evaluate.determine_exit_code(False) == 1
