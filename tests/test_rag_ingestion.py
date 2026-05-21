"""Tests for ingestion components."""

import json

from k8s_rag.ingestion.pipeline import IngestionPipeline
from k8s_rag.ingestion.schemas import ChunkRecord
from k8s_rag.ingestion.vector_store import ChromaVectorStore


class FakeEmbedder:
    """Simple deterministic embedder for tests."""

    def embed_texts(self, texts):
        return [[float(len(text))] for text in texts]


class FakeCollection:
    """In-memory collection test double."""

    def __init__(self):
        self.upserts = []
        self.query_response = {
            "ids": [["a"]],
            "documents": [["chunk A"]],
            "metadatas": [[{"source_url": "https://kubernetes.io/docs/a"}]],
            "distances": [[0.25]],
        }

    def upsert(self, ids, documents, metadatas, embeddings):
        self.upserts.append((ids, documents, metadatas, embeddings))

    def query(self, query_embeddings, n_results, include):
        del query_embeddings, n_results, include
        return self.query_response


def test_ingestion_pipeline_loads_and_upserts(tmp_path):
    """Pipeline should load JSONL and upsert all chunks."""
    jsonl = tmp_path / "chunks.jsonl"
    rows = [
        {"chunk_id": "a", "content": "alpha", "source_url": "https://x/a"},
        {"chunk_id": "b", "content": "beta", "source_url": "https://x/b"},
    ]
    jsonl.write_text("".join(json.dumps(row) + "\n" for row in rows))

    collection = FakeCollection()
    store = ChromaVectorStore(
        collection_name="test",
        persist_directory=str(tmp_path / "vec"),
        collection=collection,
    )
    pipeline = IngestionPipeline(embedder=FakeEmbedder(), vector_store=store, batch_size=1)
    stats = pipeline.run(jsonl)

    assert stats["ingested_chunks"] == 2
    assert len(collection.upserts) == 2


def test_vector_store_query_normalizes_response(tmp_path):
    """Vector store query should return typed RetrievedChunk records."""
    collection = FakeCollection()
    store = ChromaVectorStore(
        collection_name="test",
        persist_directory=str(tmp_path / "vec"),
        collection=collection,
    )

    results = store.query([0.1, 0.2], top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "a"
    assert results[0].score > 0


def test_vector_store_upsert_forwards_payload(tmp_path):
    """Upsert should pass ids, docs, metadata, and embeddings to collection."""
    collection = FakeCollection()
    store = ChromaVectorStore(
        collection_name="test",
        persist_directory=str(tmp_path / "vec"),
        collection=collection,
    )
    chunks = [
        ChunkRecord(
            chunk_id="x::1",
            content="hello",
            metadata={"source_url": "https://kubernetes.io/docs/x"},
        )
    ]
    embeddings = [[0.42]]
    store.upsert_chunks(chunks, embeddings)
    ids, documents, metadatas, payload_embeddings = collection.upserts[-1]
    assert ids == ["x::1"]
    assert documents == ["hello"]
    assert metadatas[0]["source_url"].endswith("/x")
    assert payload_embeddings == embeddings


def test_vector_store_normalizes_empty_and_nested_metadata(tmp_path):
    """Metadata normalization should make values Chroma-compatible."""
    collection = FakeCollection()
    store = ChromaVectorStore(
        collection_name="test",
        persist_directory=str(tmp_path / "vec"),
        collection=collection,
    )
    chunks = [
        ChunkRecord(
            chunk_id="x::2",
            content="hello",
            metadata={
                "code_types": [],
                "api_metadata": [{"apiVersion": "v1", "kind": "Pod"}],
                "extra": None,
            },
        )
    ]
    store.upsert_chunks(chunks, [[0.1]])
    _, _, metadatas, _ = collection.upserts[-1]
    assert metadatas[0]["code_types"] == ""
    assert isinstance(metadatas[0]["api_metadata"], str)
    assert metadatas[0]["extra"] == ""
