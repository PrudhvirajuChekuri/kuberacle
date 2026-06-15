"""Tests for the RAG QA system factory."""

from pathlib import Path

import pytest

from kuberacle.config import load_rag_config
from kuberacle.ingestion.embedder import VertexAIEmbedder
from kuberacle.ingestion.vector_store import ChromaVectorStore
from kuberacle.factory import build_qa_system
from kuberacle.gate import VertexAIRelevanceGate
from kuberacle.generator import VertexAIAnswerGenerator
from kuberacle.qa import RAGQASystem
from kuberacle.retrieval.retriever import HybridRetriever


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "rag.yaml"


@pytest.fixture
def config(monkeypatch):
    """Load the real config with stubbed GCP env vars."""
    monkeypatch.setenv("GCP_PROJECT", "test-project")
    monkeypatch.setenv("GCP_LOCATION", "us-central1")
    return load_rag_config(CONFIG_PATH)


@pytest.fixture(autouse=True)
def stub_chroma(monkeypatch):
    """Avoid touching ChromaDB/disk when the factory builds the BM25 index."""
    monkeypatch.setattr(ChromaVectorStore, "fetch_all_chunks", lambda self: [])


def test_build_qa_system_returns_wired_system(config):
    """Factory should return a RAGQASystem with hybrid retriever and generator."""
    qa = build_qa_system(config, PROJECT_ROOT)

    assert isinstance(qa, RAGQASystem)
    assert isinstance(qa.retriever, HybridRetriever)
    assert isinstance(qa.generator, VertexAIAnswerGenerator)
    assert isinstance(qa.retriever.semantic_retriever.embedder, VertexAIEmbedder)


def test_build_qa_system_propagates_config(config):
    """Citation and evidence settings should flow from config into the system."""
    qa = build_qa_system(config, PROJECT_ROOT)

    assert qa.min_evidence_score == config.min_evidence_score
    assert qa.min_supporting_chunks == config.min_supporting_chunks
    assert qa.strict_used_only == config.citation_strict_used_only
    assert qa.deduplicate_citations == config.citation_deduplicate
    assert qa.generator.model_id == config.generation_model_id


def test_build_qa_system_wires_relevance_gate(config):
    """With gate enabled in config, the QA system should carry a wired gate."""
    qa = build_qa_system(config, PROJECT_ROOT)

    assert config.gate_enabled is True
    assert isinstance(qa.relevance_gate, VertexAIRelevanceGate)
    assert qa.relevance_gate.model_id == config.gate_model_id
    assert "{question}" in qa.relevance_gate.prompt_bundle["user"]
    assert qa.relevance_gate.prompt_bundle["system"]


def test_build_qa_system_resolves_persist_directory(config):
    """Vector store persist path should be resolved under the project root."""
    qa = build_qa_system(config, PROJECT_ROOT)
    vector_store = qa.retriever.semantic_retriever.vector_store

    assert vector_store.persist_directory == str(PROJECT_ROOT / config.persist_directory)
