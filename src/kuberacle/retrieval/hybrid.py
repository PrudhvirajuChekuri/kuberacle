"""Hybrid retrieval candidate merging utilities."""

from kuberacle.domain import RetrievedChunk


def _normalize_scores(chunks: list[RetrievedChunk]) -> dict[str, float]:
    """Min-max normalize chunk scores to [0, 1] keyed by chunk_id.

    Args:
        chunks: Chunks with raw scores to normalize.

    Returns:
        Dict mapping chunk_id to normalized score. Returns empty dict for
        empty input; maps all chunks to 1.0 when all scores are equal.
    """
    if not chunks:
        return {}
    values = [chunk.score for chunk in chunks]
    min_v = min(values)
    max_v = max(values)
    if max_v == min_v:
        return {chunk.chunk_id: 1.0 for chunk in chunks}
    return {
        chunk.chunk_id: (chunk.score - min_v) / (max_v - min_v)
        for chunk in chunks
    }


def merge_hybrid_candidates(
    semantic_chunks: list[RetrievedChunk],
    lexical_chunks: list[RetrievedChunk],
    semantic_weight: float,
    lexical_weight: float,
    top_k: int,
) -> list[RetrievedChunk]:
    """Merge semantic and lexical candidates using weighted score fusion.

    Normalizes each list independently to [0, 1], then combines scores as
    ``semantic_weight * semantic_score + lexical_weight * lexical_score``.
    Chunks found by only one retriever receive 0.0 from the other side.

    Args:
        semantic_chunks: Results from vector similarity search.
        lexical_chunks: Results from BM25 lexical search.
        semantic_weight: Weight applied to normalized semantic scores.
        lexical_weight: Weight applied to normalized lexical scores.
        top_k: Maximum number of merged candidates to return.

    Returns:
        Deduplicated chunks sorted by descending hybrid score, capped at top_k.
    """
    semantic_norm = _normalize_scores(semantic_chunks)
    lexical_norm = _normalize_scores(lexical_chunks)

    by_id: dict[str, RetrievedChunk] = {}
    all_chunks = semantic_chunks + lexical_chunks
    for chunk in all_chunks:
        if chunk.chunk_id not in by_id:
            by_id[chunk.chunk_id] = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                metadata=chunk.metadata,
                score=0.0,
            )

    merged: list[RetrievedChunk] = []
    for chunk_id, chunk in by_id.items():
        score = (
            semantic_weight * semantic_norm.get(chunk_id, 0.0)
            + lexical_weight * lexical_norm.get(chunk_id, 0.0)
        )
        merged.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                metadata=chunk.metadata,
                score=score,
            )
        )

    merged.sort(key=lambda item: item.score, reverse=True)
    return merged[:top_k]
