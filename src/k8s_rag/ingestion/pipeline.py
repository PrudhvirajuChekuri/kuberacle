"""Ingestion pipeline from JSONL chunks to vector store."""

import json
from pathlib import Path
from typing import Any

from k8s_rag.ingestion.schemas import ChunkRecord


class IngestionPipeline:
    """Embed and upsert preprocessed chunks into a vector store.

    Args:
        embedder: Object that exposes ``embed_texts(list[str])``.
        vector_store: Object that exposes ``upsert_chunks(chunks, embeddings)``.
        batch_size: Batch size for embedding calls.
    """

    def __init__(
        self,
        embedder: Any,
        vector_store: Any,
        batch_size: int = 32,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.batch_size = batch_size

    def load_chunks(self, jsonl_path: str | Path) -> list[ChunkRecord]:
        """Load chunk records from preprocessing JSONL output.

        Args:
            jsonl_path: Path to chunk JSONL file.

        Returns:
            Parsed chunk list.
        """
        chunks: list[ChunkRecord] = []
        with open(jsonl_path, "r", encoding="utf-8") as file:
            for lineno, line in enumerate(file, start=1):
                row = json.loads(line)
                try:
                    chunk_id = row["chunk_id"]
                    content = row["content"]
                except KeyError as exc:
                    raise ValueError(
                        f"{jsonl_path}:{lineno}: missing required field {exc}"
                    ) from exc
                metadata = {
                    k: v for k, v in row.items() if k not in {"chunk_id", "content"}
                }
                chunks.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        content=content,
                        metadata=metadata,
                    )
                )
        return chunks

    def run(self, jsonl_path: str | Path) -> dict[str, int]:
        """Execute full ingestion flow.

        Args:
            jsonl_path: Path to processed chunks JSONL.

        Returns:
            Ingestion summary stats.
        """
        chunks = self.load_chunks(jsonl_path)
        upserted = 0

        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start:start + self.batch_size]
            texts = [chunk.content for chunk in batch]
            embeddings = self.embedder.embed_texts(texts)
            self.vector_store.upsert_chunks(batch, embeddings)
            upserted += len(batch)

        return {"upserted_chunks": upserted}
