"""Local BM25 retrieval over chunk corpus."""

import logging
import re

from k8s_rag.ingestion.schemas import RetrievedChunk

logger = logging.getLogger(__name__)


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 scoring."""
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Retriever:
    """In-memory BM25 retriever built from chunk records.

    Args:
        chunks: Full chunk corpus fetched from the vector store.
        top_k: Default number of results to return.
    """

    def __init__(self, chunks: list[RetrievedChunk], top_k: int = 8) -> None:
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.top_k = top_k
        if not chunks:
            logger.warning("BM25Retriever initialized with empty corpus")
            self._bm25 = None
            return
        self._tokenized_corpus = [
            _tokenize(
                " ".join(
                    [
                        str(chunk.metadata.get("title", "")),
                        str(chunk.metadata.get("heading_hierarchy", "")),
                        chunk.content,
                    ]
                )
            )
            for chunk in chunks
        ]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.debug("BM25Retriever initialized with %d chunks", len(chunks))

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Retrieve lexical matches for query."""
        if self._bm25 is None:
            return []
        k = top_k if top_k is not None else self.top_k
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            (idx for idx in range(len(scores)) if scores[idx] > 0),
            key=lambda idx: float(scores[idx]),
            reverse=True,
        )[:k]
        rows: list[RetrievedChunk] = []
        for idx in ranked_indices:
            base = self.chunks[idx]
            rows.append(
                RetrievedChunk(
                    chunk_id=base.chunk_id,
                    content=base.content,
                    metadata=base.metadata,
                    score=float(scores[idx]),
                )
            )
        logger.debug("BM25 retrieved %d chunks", len(rows))
        return rows
