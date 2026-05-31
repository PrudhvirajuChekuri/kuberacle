"""Tests for retrieval and QA orchestration."""

from k8s_rag.ingestion.schemas import RetrievedChunk
from k8s_rag.retrieval.generator import VertexAIAnswerGenerator, extract_citation_indices
from k8s_rag.retrieval.hybrid import merge_hybrid_candidates
from k8s_rag.retrieval.qa import RAGQASystem
from k8s_rag.retrieval.retriever import SemanticRetriever


class FakeEmbedder:
    """Deterministic embedder test double."""

    def embed_text(self, text):
        return [float(len(text))]


class FakeVectorStore:
    """Deterministic retriever backend test double."""

    def query(self, query_embedding, top_k):
        del query_embedding, top_k
        return [
            RetrievedChunk(
                chunk_id="pods::what-is",
                content="A Pod is the smallest deployable unit in Kubernetes.",
                metadata={"source_url": "https://kubernetes.io/docs/concepts/workloads/pods/"},
                score=0.9,
            )
        ]


class FakeGenAIClient:
    """Fake Gen AI client for generator tests."""

    class _Models:
        def generate_content(self, **kwargs):
            del kwargs

            class _Response:
                text = "Pods run one or more containers [1]."

            return _Response()

    def __init__(self):
        self.models = self._Models()


class EmptyVectorStore:
    """Returns no retrievals to test insufficient evidence branch."""

    def query(self, query_embedding, top_k):
        del query_embedding, top_k
        return []


class FixedAnswerGenerator:
    """Deterministic generator for citation enforcement tests."""

    def __init__(self, answer):
        self.answer = answer

    def generate(self, question, chunks):
        del question, chunks
        return self.answer


def test_semantic_retriever_returns_ranked_chunks():
    """Retriever should return rows from vector store."""
    retriever = SemanticRetriever(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        top_k=5,
    )
    rows = retriever.retrieve("What is a Pod?")
    assert len(rows) == 1
    assert rows[0].chunk_id == "pods::what-is"


def test_qa_system_returns_answer_with_citations():
    """QA system should return model answer and citation records."""
    retriever = SemanticRetriever(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        top_k=5,
    )
    generator = VertexAIAnswerGenerator(
        model_id="gemini-2.5-flash-lite",
        gcp_project="test-project",
        gcp_location="us-central1",
        genai_client=FakeGenAIClient(),
    )
    qa = RAGQASystem(retriever=retriever, generator=generator)
    result = qa.ask("What is a Pod?")
    assert "Pods run one or more containers" in result.answer
    assert len(result.citations) == 1
    assert result.citations[0].source_url.startswith("https://kubernetes.io/docs/")


def test_qa_system_handles_insufficient_evidence():
    """QA system should return refusal when no chunks are retrieved."""
    retriever = SemanticRetriever(
        embedder=FakeEmbedder(),
        vector_store=EmptyVectorStore(),
        top_k=5,
    )
    generator = VertexAIAnswerGenerator(
        model_id="gemini-2.5-flash-lite",
        gcp_project="test-project",
        gcp_location="us-central1",
        genai_client=FakeGenAIClient(),
    )
    qa = RAGQASystem(retriever=retriever, generator=generator)
    result = qa.ask("Unknown question?")
    assert result.answer.startswith("INSUFFICIENT_EVIDENCE")
    assert result.citations == []


def test_extract_citation_indices_ordered_unique():
    """Citation parser should return ordered unique indices."""
    assert extract_citation_indices("Fact [2] and [1], again [2].") == [2, 1]


def test_qa_system_outputs_only_used_citations():
    """Citation list should include only indices used by answer."""
    retriever = SemanticRetriever(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        top_k=5,
    )
    generator = FixedAnswerGenerator("Answer text [1].")
    qa = RAGQASystem(
        retriever=retriever,
        generator=generator,
        strict_used_only=True,
        deduplicate_citations=True,
    )
    result = qa.ask("Question?")
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == "pods::what-is"


def test_merge_hybrid_candidates_deduplicates_chunk_ids():
    """Hybrid merge should dedupe matching semantic and lexical ids."""
    semantic = [
        RetrievedChunk("a", "doc a", {"source_url": "u/a"}, 0.9),
        RetrievedChunk("b", "doc b", {"source_url": "u/b"}, 0.8),
    ]
    lexical = [
        RetrievedChunk("a", "doc a", {"source_url": "u/a"}, 12.0),
        RetrievedChunk("c", "doc c", {"source_url": "u/c"}, 11.0),
    ]
    merged = merge_hybrid_candidates(
        semantic_chunks=semantic,
        lexical_chunks=lexical,
        semantic_weight=0.6,
        lexical_weight=0.4,
        top_k=10,
    )
    ids = [chunk.chunk_id for chunk in merged]
    assert len(ids) == len(set(ids))
    assert "a" in ids and "b" in ids and "c" in ids
