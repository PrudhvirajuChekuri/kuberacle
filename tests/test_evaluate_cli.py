"""Tests for evaluate CLI helper contracts."""

import importlib.util
from pathlib import Path


def _load_evaluate_module():
    """Load scripts/evaluate.py as a module for unit tests."""
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("evaluate_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_determine_exit_code_contract():
    """CLI helper should map gate pass/fail to process exit code."""
    module = _load_evaluate_module()
    assert module.determine_exit_code(True) == 0
    assert module.determine_exit_code(False) == 1
