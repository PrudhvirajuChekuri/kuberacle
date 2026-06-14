"""Tests for golden dataset loading and validation."""

import pytest

from kuberacle.evaluation.dataset import load_golden_dataset

_VALID_ROW = '{"id":"q1","question":"What is a Pod?","expected_answer":"A Pod is the smallest deployable unit.","reference_chunk_ids":["c1"],"answerable":true,"tags":["concept"]}\n'
_UNANSWERABLE_ROW = '{"id":"q2","question":"Unknown?","expected_answer":"No support.","reference_chunk_ids":[],"answerable":false,"tags":["abstention"]}\n'


def test_load_golden_dataset_parses_valid_rows(tmp_path):
    """Loader should parse JSONL rows into typed examples."""
    dataset_path = tmp_path / "golden.jsonl"
    dataset_path.write_text(_VALID_ROW + _UNANSWERABLE_ROW, encoding="utf-8")

    rows = load_golden_dataset(dataset_path)
    assert len(rows) == 2
    assert rows[0].case_id == "q1"
    assert rows[1].answerable is False


def test_load_golden_dataset_rejects_duplicate_ids(tmp_path):
    """Loader should fail when case ids repeat."""
    dup_row = '{"id":"q1","question":"B?","expected_answer":"B","reference_chunk_ids":["c2"],"answerable":true,"tags":["y"]}\n'
    dataset_path = tmp_path / "dup.jsonl"
    dataset_path.write_text(_VALID_ROW + dup_row, encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate case id"):
        load_golden_dataset(dataset_path)


def test_load_golden_dataset_raises_on_invalid_json(tmp_path):
    """Loader should raise ValueError with line number for malformed JSON."""
    dataset_path = tmp_path / "bad.jsonl"
    dataset_path.write_text(_VALID_ROW + "{bad json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON at line 2"):
        load_golden_dataset(dataset_path)


def test_load_golden_dataset_raises_on_missing_field(tmp_path):
    """Loader should raise ValueError listing missing fields."""
    dataset_path = tmp_path / "missing.jsonl"
    dataset_path.write_text(
        '{"id":"q1","question":"What?","expected_answer":"A","answerable":true,"tags":[]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing required fields"):
        load_golden_dataset(dataset_path)


def test_load_golden_dataset_raises_on_wrong_field_type(tmp_path):
    """Loader should raise ValueError when a field has the wrong type."""
    dataset_path = tmp_path / "badtype.jsonl"
    dataset_path.write_text(
        '{"id":"q1","question":"What?","expected_answer":"A","reference_chunk_ids":["c1"],"answerable":"yes","tags":[]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid field type"):
        load_golden_dataset(dataset_path)
