# Golden Evaluation Dataset

This directory contains versioned golden datasets for offline RAG evaluation.

## File Format

Datasets use JSONL (one JSON object per line).

Required fields per row:

- `id` (string): Stable unique test case id.
- `question` (string): User question to send through the RAG pipeline.
- `expected_answer` (string): Human reference answer used for regression checks and evaluator context.
- `reference_chunk_ids` (list[string]): Chunk ids that contain supporting evidence.
- `answerable` (bool): Whether the system is expected to answer (`true`) or abstain (`false`).
- `tags` (list[string]): Slice labels used for segmented analysis (for example `concept`, `task`, `tutorial`, `abstention`).

## Dataset Lifecycle

- Start with `v1.jsonl` and treat it as immutable once used in CI.
- Add new versions (`v2.jsonl`, `v3.jsonl`) when expanding scope.
- Keep a balanced mix of:
  - concept explanations
  - operational tasks and commands
  - troubleshooting behaviors
  - unanswerable prompts for abstention validation
