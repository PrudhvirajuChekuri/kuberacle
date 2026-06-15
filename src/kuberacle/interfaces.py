"""Structural interfaces (Protocols) for the pipeline's injected collaborators.

These make the seams between the QA orchestrator and its dependencies explicit
and statically checkable instead of implicit duck-typing. Concrete classes
(``SemanticRetriever``, ``VertexAIAnswerGenerator``, ``VertexAIRelevanceGate``,
``VertexAIEmbedder``, ``ChromaVectorStore``) satisfy these by structure; nothing
needs to inherit from them.
"""

from collections.abc import Iterator
from typing import Protocol

from kuberacle.domain import RetrievedChunk


class Embedder(Protocol):
    """Turns query text into an embedding vector."""

    def embed_text(self, text: str) -> list[float]:
        """Embed a single query string.

        Args:
            text: Query text to embed.

        Returns:
            The embedding vector.
        """
        ...


class VectorStore(Protocol):
    """Vector similarity search over stored chunks."""

    def query(
        self, query_embedding: list[float], top_k: int
    ) -> list[RetrievedChunk]:
        """Return the top-k chunks nearest to a query embedding.

        Args:
            query_embedding: Query vector.
            top_k: Number of chunks to return.

        Returns:
            Ranked chunk list.
        """
        ...


class Retriever(Protocol):
    """Returns ranked chunks for a query."""

    def retrieve(
        self, query: str, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks for a query.

        Args:
            query: User search question.
            top_k: Optional retrieval depth override.

        Returns:
            Ranked chunk list.
        """
        ...


class Generator(Protocol):
    """Generates a grounded answer from retrieved chunks."""

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """Generate a complete answer.

        Args:
            question: User question.
            chunks: Retrieved context chunks.

        Returns:
            Generated answer text.
        """
        ...

    def generate_stream(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> Iterator[str]:
        """Stream a grounded answer as incremental text fragments.

        Args:
            question: User question.
            chunks: Retrieved context chunks.

        Yields:
            Non-empty text fragments in generation order.
        """
        ...


class RelevanceGate(Protocol):
    """Pre-retrieval scope check for a question."""

    def is_relevant(self, question: str) -> bool:
        """Decide whether a question is in scope for the docs corpus.

        Args:
            question: User question.

        Returns:
            True when the question should proceed to retrieval.
        """
        ...
