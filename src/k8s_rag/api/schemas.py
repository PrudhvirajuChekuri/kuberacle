"""Request and response schemas for the RAG API."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Incoming question payload for the query endpoint.

    Attributes:
        question: Natural-language question to answer.
    """

    question: str = Field(min_length=1)


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
