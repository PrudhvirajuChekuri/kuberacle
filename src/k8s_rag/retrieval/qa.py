"""Question-answer orchestration with citations."""

from dataclasses import dataclass

from k8s_rag.ingestion.schemas import RetrievedChunk


@dataclass(frozen=True)
class Citation:
    """Citation record for generated answer provenance."""

    chunk_id: str
    source_url: str
    score: float


@dataclass(frozen=True)
class QAResult:
    """Output object returned by RAGQASystem."""

    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]


class RAGQASystem:
    """Orchestrate retrieval + generation for grounded answers."""

    def __init__(self, retriever, generator) -> None:
        self.retriever = retriever
        self.generator = generator

    def ask(self, question: str, top_k: int | None = None) -> QAResult:
        """Answer a user question with source citations.

        Args:
            question: User question.
            top_k: Optional retrieval depth override.

        Returns:
            QA result with answer and citations.
        """
        chunks = self.retriever.retrieve(question, top_k=top_k)
        if not chunks:
            return QAResult(
                answer=(
                    "INSUFFICIENT_EVIDENCE. I could not retrieve supporting "
                    "documentation for this question."
                ),
                citations=[],
                retrieved_chunks=[],
            )

        answer = self.generator.generate(question, chunks)
        citations = [
            Citation(
                chunk_id=chunk.chunk_id,
                source_url=str(chunk.metadata.get("source_url", "")),
                score=chunk.score,
            )
            for chunk in chunks
        ]

        return QAResult(
            answer=answer,
            citations=citations,
            retrieved_chunks=chunks,
        )
