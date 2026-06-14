"""Request and response schemas for the RAG API."""

from pydantic import BaseModel, Field


#: Upper bound on question length. Generous for a real question (the embedding
#: model truncates near 2048 tokens anyway) while rejecting oversized payloads
#: that would inflate model/reranker cost or pressure memory before any work.
MAX_QUESTION_LENGTH = 2000


class QueryRequest(BaseModel):
    """Incoming question payload for the query endpoint.

    Attributes:
        question: Natural-language question to answer.
    """

    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)


class CitationModel(BaseModel):
    """Serialized citation record for the streamed final event.

    Attributes:
        index: 1-based marker number used in the answer (``[n]``).
        source_url: Source document URL backing the answer.
        chunk_id: Identifier of the supporting chunk.
        score: Relevance score of the supporting chunk.
        title: Document title of the supporting chunk, for source previews.
        snippet: Short text preview of the supporting chunk content.
    """

    index: int
    source_url: str
    chunk_id: str
    score: float
    title: str
    snippet: str
