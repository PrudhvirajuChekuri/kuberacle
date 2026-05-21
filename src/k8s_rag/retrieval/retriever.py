"""Semantic retrieval over embedded chunk collections."""

from k8s_rag.ingestion.schemas import RetrievedChunk


class SemanticRetriever:
    """Retrieve top-k chunks by vector similarity.

    Args:
        embedder: Embedder exposing ``embed_text(str)``.
        vector_store: Vector store exposing ``query(query_embedding, top_k)``.
        top_k: Default retrieval depth.
    """

    def __init__(self, embedder, vector_store, top_k: int = 5) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Retrieve relevant chunks for a query.

        Args:
            query: User search question.
            top_k: Optional override for retrieval depth.

        Returns:
            Ranked chunk list.
        """
        k = top_k if top_k is not None else self.top_k
        query_embedding = self.embedder.embed_text(query)
        return self.vector_store.query(query_embedding, k)
