"""Deterministic evaluation metrics for RAG quality gates."""


def is_insufficient_evidence(answer: str) -> bool:
    """Return whether answer indicates explicit abstention."""
    return answer.strip().startswith("INSUFFICIENT_EVIDENCE")


def retrieval_recall_at_k(
    retrieved_chunk_ids: list[str],
    reference_chunk_ids: list[str],
) -> float:
    """Compute recall over expected supporting chunks."""
    if not reference_chunk_ids:
        return 1.0
    retrieved = set(retrieved_chunk_ids)
    reference = set(reference_chunk_ids)
    return len(retrieved.intersection(reference)) / float(len(reference))


def precision_at_1(
    retrieved_chunk_ids: list[str],
    reference_chunk_ids: list[str],
) -> float:
    """Return 1.0 if the top-ranked retrieved chunk is a reference chunk."""
    if not retrieved_chunk_ids or not reference_chunk_ids:
        return 0.0
    return 1.0 if retrieved_chunk_ids[0] in set(reference_chunk_ids) else 0.0


def non_empty_answer(answer: str) -> bool:
    """Check whether model produced a non-empty non-abstention answer."""
    text = answer.strip()
    return bool(text) and not is_insufficient_evidence(text)
