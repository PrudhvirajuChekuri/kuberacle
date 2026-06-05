# Golden Evaluation Dataset

This directory contains versioned golden datasets for offline RAG evaluation.
The datasets are designed as regression suites: they check whether retrieval
finds the right evidence and whether generation answers only when the selected
Kubernetes documentation corpus supports an answer.

## File Format

Datasets use JSONL (one JSON object per line).

Required fields per row:

- `id` (string): Stable unique test case id.
- `question` (string): User question to send through the RAG pipeline.
- `expected_answer` (string): Human reference answer used for regression checks and evaluator context.
- `reference_chunk_ids` (list[string]): Chunk ids that contain supporting evidence.
- `answerable` (bool): Whether the system is expected to answer (`true`) or abstain (`false`).
- `tags` (list[string]): Slice labels used for segmented analysis (for example `concept`, `task`, `tutorial`, `abstention`).

## Design Principles

Golden examples should be grounded in the processed documentation chunks, not in
general Kubernetes knowledge. Each answerable row must point to one or more
`reference_chunk_ids` that contain the evidence needed to answer the question.
Unanswerable rows must have an empty `reference_chunk_ids` list and should ask
about plausible adjacent topics that are outside the selected corpus.

The expected answer should be short, factual, and specific enough to support
regression checks. It does not need to copy the documentation wording, but it
must not add claims that are missing from the referenced chunks.

Questions should be phrased like realistic user questions, not only copied
headings. A useful dataset includes both direct questions and cases that require
retrieving from task, tutorial, or concept pages with similar terminology.

Tags are used for slice-level analysis. They should capture both the source
area (`concept`, `task`, `tutorial`, `abstention`) and the behavior or topic
being tested (`troubleshooting`, `yaml`, `comparison`, `multi-hop`, `pods`,
`networking`, `storage`, `security`, and similar labels).

## Dataset Versions

- `smoke.jsonl`: Small curated fast-check dataset for pull request evaluation.
- `v1.jsonl`: Initial full golden dataset.
- `v2.jsonl`: Balanced 50-case full golden dataset for the full Kubernetes
  docs corpus.

`smoke.jsonl` is the active CI smoke gate. It contains a hand-curated set of
cases designed to cover the critical retrieval and generation paths: basic
factual questions, multi-chunk questions, unanswerable abstention cases, and
near-miss precision cases. All cases reference chunks from the full corpus
index. CI runs evaluate against the full index downloaded from GCS — no
separate smoke corpus or ingestion step is needed.

`v2.jsonl` is built around the full dataset configuration, which discovers
Kubernetes docs from:

- `concepts`
- `tasks`
- `tutorials`

The target shape for `v2.jsonl` is:

- 20 concept-grounded cases
- 20 task-grounded cases
- 5 tutorial-grounded cases
- 5 unanswerable abstention cases

The dataset also includes coverage for common RAG failure modes:

- definitions and conceptual explanations
- operational how-to questions
- YAML and API-field questions
- troubleshooting and status interpretation
- comparison questions
- multi-hop workflow questions
- out-of-corpus abstention

## Dataset Lifecycle

- Treat a dataset version as stable once it is used as a baseline.
- Add a new version (`v3.jsonl`, `v4.jsonl`, and so on) when the dataset is
  materially expanded or rebalanced.
- Fix obvious typos or broken chunk ids before a version becomes the accepted
  baseline.
- Do not silently change the meaning of an accepted version, because changing
  the dataset changes the meaning of the evaluation score.
- Keep older versions available for comparison and auditability.

## Review Checklist

Before promoting a dataset version as the active full golden set:

- Every row is valid JSONL and loads through the dataset loader.
- Every answerable row has at least one valid supporting chunk id.
- Every unanswerable row has no supporting chunk ids.
- The expected answer is fully supported by the referenced chunks.
- The question is realistic and not just a copied heading.
- Tags are useful for analyzing failures by source area and behavior.
- The dataset has enough negative cases to verify abstention behavior.
