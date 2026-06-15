"""Shared data models for ingestion and retrieval."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChunkRecord:
    """Chunk payload read from the preprocessing JSONL output.

    Attributes:
        chunk_id: Stable unique identifier for chunk storage.
        content: Chunk text content used for embedding.
        metadata: Chunk metadata map (title, source_url, etc.).
    """

    chunk_id: str
    content: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class RetrievedChunk:
    """Retrieved chunk with relevance score.

    Attributes:
        chunk_id: Stored chunk id.
        content: Chunk text content.
        metadata: Chunk metadata map.
        score: Similarity score (higher is better).
    """

    chunk_id: str
    content: str
    metadata: Mapping[str, Any]
    score: float
