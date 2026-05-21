"""Tests for retrieval and QA orchestration."""

from k8s_rag.ingestion.schemas import RetrievedChunk
from k8s_rag.retrieval.generator import BedrockAnswerGenerator
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


class FakeBedrockClient:
    """Fake Bedrock runtime client for generator tests."""

    def converse(self, **kwargs):
        del kwargs
        return {
            "output": {
                "message": {
                    "content": [{"text": "Pods run one or more containers [1]."}]
                }
            }
        }


class EmptyVectorStore:
    """Returns no retrievals to test insufficient evidence branch."""

    def query(self, query_embedding, top_k):
        del query_embedding, top_k
        return []


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
    generator = BedrockAnswerGenerator(
        model_id="anthropic.claude-3-5-haiku-20241022-v1:0",
        region_name="us-east-1",
        bedrock_client=FakeBedrockClient(),
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
    generator = BedrockAnswerGenerator(
        model_id="anthropic.claude-3-5-haiku-20241022-v1:0",
        region_name="us-east-1",
        bedrock_client=FakeBedrockClient(),
    )
    qa = RAGQASystem(retriever=retriever, generator=generator)
    result = qa.ask("Unknown question?")
    assert result.answer.startswith("INSUFFICIENT_EVIDENCE")
    assert result.citations == []
