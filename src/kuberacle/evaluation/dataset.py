"""Golden dataset schemas and loading helpers."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GoldenExample:
    """Single golden evaluation row.

    Args:
        case_id: Stable test case identifier.
        question: User question for the RAG system.
        expected_answer: Human reference answer.
        reference_chunk_ids: Supporting chunk ids expected in retrieval context.
        answerable: Whether the system should answer or abstain.
        tags: Slice labels for segmented reporting.
    """

    case_id: str
    question: str
    expected_answer: str
    reference_chunk_ids: list[str]
    answerable: bool
    tags: list[str]


def _require_type(value, expected_type: type, field_name: str, line_number: int) -> None:
    """Validate field type and raise a readable error on mismatch."""
    if not isinstance(value, expected_type):
        raise ValueError(
            f"Invalid field type at line {line_number}: "
            f"{field_name!r} must be {expected_type.__name__}."
        )


def _parse_example(line: str, line_number: int) -> GoldenExample:
    """Parse and validate one JSONL example row."""
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON at line {line_number}: {exc}"
        ) from exc
    required_fields = {
        "id",
        "question",
        "expected_answer",
        "reference_chunk_ids",
        "answerable",
        "tags",
    }
    missing = required_fields - payload.keys()
    if missing:
        raise ValueError(
            f"Missing required fields at line {line_number}: {sorted(missing)}."
        )

    case_id = payload["id"]
    question = payload["question"]
    expected_answer = payload["expected_answer"]
    reference_chunk_ids = payload["reference_chunk_ids"]
    answerable = payload["answerable"]
    tags = payload["tags"]

    _require_type(case_id, str, "id", line_number)
    _require_type(question, str, "question", line_number)
    _require_type(expected_answer, str, "expected_answer", line_number)
    _require_type(reference_chunk_ids, list, "reference_chunk_ids", line_number)
    _require_type(answerable, bool, "answerable", line_number)
    _require_type(tags, list, "tags", line_number)

    if not case_id.strip():
        raise ValueError(f"Invalid id at line {line_number}: id cannot be empty.")
    if not question.strip():
        raise ValueError(
            f"Invalid question at line {line_number}: question cannot be empty."
        )
    if any(not isinstance(item, str) or not item for item in reference_chunk_ids):
        raise ValueError(
            f"Invalid reference_chunk_ids at line {line_number}: "
            "all ids must be non-empty strings."
        )
    if any(not isinstance(item, str) or not item for item in tags):
        raise ValueError(
            f"Invalid tags at line {line_number}: all tags must be non-empty strings."
        )

    return GoldenExample(
        case_id=case_id,
        question=question,
        expected_answer=expected_answer,
        reference_chunk_ids=reference_chunk_ids,
        answerable=answerable,
        tags=tags,
    )


def load_golden_dataset(dataset_path: str | Path) -> list[GoldenExample]:
    """Load a JSONL golden dataset from disk.

    Args:
        dataset_path: Path to dataset JSONL file.

    Returns:
        Parsed golden examples.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    rows: list[GoldenExample] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = _parse_example(stripped, line_number)
            if row.case_id in seen_ids:
                raise ValueError(f"Duplicate case id at line {line_number}: {row.case_id}")
            seen_ids.add(row.case_id)
            rows.append(row)

    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    return rows
