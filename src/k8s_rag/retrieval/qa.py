"""Question-answer orchestration with citations."""

from dataclasses import dataclass
from typing import Any

from k8s_rag.ingestion.schemas import RetrievedChunk
from k8s_rag.retrieval.generator import extract_citation_indices


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

    def __init__(
        self,
        retriever: Any,
        generator: Any,
        min_evidence_score: float = 0.0,
        min_supporting_chunks: int = 1,
        strict_used_only: bool = True,
        deduplicate_citations: bool = True,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.min_evidence_score = min_evidence_score
        self.min_supporting_chunks = min_supporting_chunks
        self.strict_used_only = strict_used_only
        self.deduplicate_citations = deduplicate_citations

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
        used_indices = extract_citation_indices(answer)

        if self.strict_used_only:
            if not used_indices:
                return self._insufficient_with_chunks(chunks)
            if any(idx < 1 or idx > len(chunks) for idx in used_indices):
                return self._insufficient_with_chunks(chunks)
            selected_chunks = [chunks[idx - 1] for idx in used_indices]
        else:
            selected_chunks = chunks

        if self.deduplicate_citations:
            deduped: list[RetrievedChunk] = []
            seen: set[str] = set()
            for chunk in selected_chunks:
                if chunk.chunk_id in seen:
                    continue
                seen.add(chunk.chunk_id)
                deduped.append(chunk)
            selected_chunks = deduped

        supporting = [
            chunk for chunk in selected_chunks
            if chunk.score >= self.min_evidence_score
        ]
        if len(supporting) < self.min_supporting_chunks:
            return self._insufficient_with_chunks(chunks)

        citations = [
            Citation(
                chunk_id=chunk.chunk_id,
                source_url=str(chunk.metadata.get("source_url", "unknown")),
                score=chunk.score,
            )
            for chunk in supporting
        ]

        return QAResult(
            answer=answer,
            citations=citations,
            retrieved_chunks=chunks,
        )

    def _insufficient_with_chunks(self, chunks: list[RetrievedChunk]) -> QAResult:
        """Build insufficient evidence result."""
        return QAResult(
            answer=(
                "INSUFFICIENT_EVIDENCE. I could not verify enough supported "
                "citations for this question."
            ),
            citations=[],
            retrieved_chunks=chunks,
        )
