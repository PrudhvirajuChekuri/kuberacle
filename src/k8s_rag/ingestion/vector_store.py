"""Vector store adapters used by ingestion and retrieval."""

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from k8s_rag.ingestion.schemas import ChunkRecord, RetrievedChunk

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """ChromaDB-backed vector store.

    Args:
        collection_name: Logical collection name.
        persist_directory: On-disk persistence path.
        collection: Optional injected collection for testing.
    """

    def __init__(
        self,
        collection_name: str,
        persist_directory: str,
        collection: Any = None,
    ) -> None:
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._collection = collection

    @property
    def collection(self) -> Any:
        """Get or initialize Chroma collection."""
        if self._collection is None:
            import chromadb

            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=self.persist_directory)
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
            )
        return self._collection

    def reset_collection(self) -> None:
        """Drop and recreate the collection for a clean ingest."""
        import chromadb

        logger.info("Resetting collection %r", self.collection_name)
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=self.persist_directory)
        try:
            client.delete_collection(self.collection_name)
        except Exception:
            logger.debug("Collection %r did not exist, creating fresh", self.collection_name)
        self._collection = client.create_collection(name=self.collection_name)
        logger.info("Collection %r ready", self.collection_name)

    def upsert_chunks(
        self,
        chunks: list[ChunkRecord],
        embeddings: list[list[float]],
    ) -> None:
        """Upsert chunk vectors into collection.

        Args:
            chunks: Chunk records to store.
            embeddings: Embedding list aligned to chunks.
        """
        logger.debug("Upserting %d chunks into collection %r", len(chunks), self.collection_name)
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [self._normalize_metadata(chunk.metadata) for chunk in chunks]
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def _normalize_metadata(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        """Convert metadata into Chroma-compatible primitive fields.

        Chroma supports scalar primitives and non-empty primitive lists.
        This normalizer converts empty lists, nested objects, and None values
        into deterministic JSON/string forms to avoid upsert validation errors.

        Args:
            metadata: Original metadata from chunk records.

        Returns:
            Sanitized metadata safe for Chroma upsert.
        """
        normalized: dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                normalized[key] = value
                continue

            if value is None:
                normalized[key] = ""
                continue

            if isinstance(value, list):
                if not value:
                    normalized[key] = ""
                elif all(isinstance(item, (str, int, float, bool)) for item in value):
                    normalized[key] = value
                else:
                    normalized[key] = json.dumps(value, ensure_ascii=False)
                continue

            if isinstance(value, dict):
                normalized[key] = json.dumps(value, ensure_ascii=False)
                continue

            normalized[key] = str(value)
        return normalized

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Run vector similarity query and normalize response.

        Args:
            query_embedding: Embedded query vector.
            top_k: Number of results to return.

        Returns:
            Ordered list of retrieved chunks.
        """
        logger.debug("Querying collection %r top_k=%d", self.collection_name, top_k)
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        rows: list[RetrievedChunk] = []
        for chunk_id, content, metadata, distance in zip(ids, docs, metas, dists):
            rows.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    content=content,
                    metadata=metadata or {},
                    score=1.0 / (1.0 + float(distance)),
                )
            )
        return rows

    def fetch_all_chunks(self) -> list[RetrievedChunk]:
        """Fetch all stored chunks from collection.

        Returns:
            List of chunk records with score set to 0.0.
        """
        logger.debug("Fetching all chunks from collection %r", self.collection_name)
        result = self.collection.get(
            include=["documents", "metadatas"],
        )
        ids = result.get("ids", [])
        docs = result.get("documents", [])
        metas = result.get("metadatas", [])
        rows: list[RetrievedChunk] = []
        for chunk_id, content, metadata in zip(ids, docs, metas):
            rows.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    content=content,
                    metadata=metadata or {},
                    score=0.0,
                )
            )
        logger.debug("Fetched %d chunks from collection %r", len(rows), self.collection_name)
        return rows
