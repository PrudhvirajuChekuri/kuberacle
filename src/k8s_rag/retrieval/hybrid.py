"""Hybrid retrieval candidate merging utilities."""

from k8s_rag.ingestion.schemas import RetrievedChunk


def _normalize_scores(chunks: list[RetrievedChunk]) -> dict[str, float]:
    """Min-max normalize chunk scores by chunk id."""
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
    """Merge semantic and lexical candidates with weighted scores."""
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
