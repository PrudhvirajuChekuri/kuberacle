"""Tests for golden dataset loading and validation."""

from k8s_rag.evaluation.dataset import load_golden_dataset


def test_load_golden_dataset_parses_valid_rows(tmp_path):
    """Loader should parse JSONL rows into typed examples."""
    dataset_path = tmp_path / "golden.jsonl"
    dataset_path.write_text(
        '{"id":"q1","question":"What is a Pod?","expected_answer":"A Pod is the smallest deployable unit.","reference_chunk_ids":["c1"],"answerable":true,"tags":["concept"]}\n'
        '{"id":"q2","question":"Unknown?","expected_answer":"No support.","reference_chunk_ids":[],"answerable":false,"tags":["abstention"]}\n'
    )

    rows = load_golden_dataset(dataset_path)
    assert len(rows) == 2
    assert rows[0].case_id == "q1"
    assert rows[1].answerable is False


def test_load_golden_dataset_rejects_duplicate_ids(tmp_path):
    """Loader should fail when case ids repeat."""
    dataset_path = tmp_path / "dup.jsonl"
    dataset_path.write_text(
        '{"id":"q1","question":"A?","expected_answer":"A","reference_chunk_ids":["c1"],"answerable":true,"tags":["x"]}\n'
        '{"id":"q1","question":"B?","expected_answer":"B","reference_chunk_ids":["c2"],"answerable":true,"tags":["y"]}\n'
    )

    try:
        load_golden_dataset(dataset_path)
        assert False, "Expected duplicate id validation to fail."
    except ValueError as exc:
        assert "Duplicate case id" in str(exc)
