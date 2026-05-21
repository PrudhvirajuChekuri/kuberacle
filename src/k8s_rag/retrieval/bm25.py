"""Local BM25 retrieval over chunk corpus."""

import re

from k8s_rag.ingestion.schemas import RetrievedChunk


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 scoring."""
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Retriever:
    """In-memory BM25 retriever built from chunk records."""

    def __init__(self, chunks: list[RetrievedChunk], top_k: int = 8) -> None:
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.top_k = top_k
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

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Retrieve lexical matches for query."""
        k = top_k if top_k is not None else self.top_k
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)),
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
        return rows
