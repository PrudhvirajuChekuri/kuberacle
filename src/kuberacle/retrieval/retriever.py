"""Semantic + hybrid retrieval orchestration."""

import logging
from typing import Any

from kuberacle.domain import RetrievedChunk
from kuberacle.retrieval.bm25 import BM25Retriever
from kuberacle.retrieval.hybrid import merge_hybrid_candidates
from kuberacle.retrieval.reranker import DiscoveryEngineReranker

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """Retrieve top-k chunks by vector similarity.

    Args:
        embedder: Embedder exposing ``embed_text(str)``.
        vector_store: Vector store exposing ``query(query_embedding, top_k)``.
        top_k: Default retrieval depth.
    """

    def __init__(self, embedder: Any, vector_store: Any, top_k: int = 5) -> None:
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
        logger.debug("SemanticRetriever: top_k=%d, query=%r", k, query[:80])
        query_embedding = self.embedder.embed_text(query)
        results = self.vector_store.query(query_embedding, k)
        logger.debug("SemanticRetriever: retrieved %d chunks", len(results))
        return results


class HybridRetriever:
    """Hybrid retrieval pipeline (semantic + BM25 + rerank).

    Args:
        semantic_retriever: Vector similarity retriever.
        bm25_retriever: BM25 lexical retriever.
        reranker: Reranker that re-scores merged candidates.
        semantic_top_k: Candidate depth for semantic search.
        lexical_top_k: Candidate depth for BM25 search.
        merged_top_k: Pool size after hybrid fusion, before rerank.
        final_top_k: Default number of results after rerank.
        semantic_weight: Score weight for semantic candidates.
        lexical_weight: Score weight for lexical candidates.
    """

    def __init__(
        self,
        semantic_retriever: SemanticRetriever,
        bm25_retriever: BM25Retriever,
        reranker: DiscoveryEngineReranker,
        semantic_top_k: int,
        lexical_top_k: int,
        merged_top_k: int,
        final_top_k: int,
        semantic_weight: float,
        lexical_weight: float,
    ) -> None:
        self.semantic_retriever = semantic_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        self.semantic_top_k = semantic_top_k
        self.lexical_top_k = lexical_top_k
        self.merged_top_k = merged_top_k
        self.final_top_k = final_top_k
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Retrieve hybrid candidates and rerank final output.

        Args:
            query: User search question.
            top_k: Optional override for final rerank depth. Capped at the
                number of merged candidates when larger than ``merged_top_k``.

        Returns:
            Reranked chunk list.
        """
        final_k = top_k if top_k is not None else self.final_top_k
        semantic = self.semantic_retriever.retrieve(query, top_k=self.semantic_top_k)
        lexical = self.bm25_retriever.retrieve(query, top_k=self.lexical_top_k)
        merged = merge_hybrid_candidates(
            semantic_chunks=semantic,
            lexical_chunks=lexical,
            semantic_weight=self.semantic_weight,
            lexical_weight=self.lexical_weight,
            top_k=self.merged_top_k,
        )
        logger.debug(
            "HybridRetriever: semantic=%d, lexical=%d, merged=%d, reranking to top_%d",
            len(semantic), len(lexical), len(merged), min(final_k, len(merged)),
        )
        results = self.reranker.rerank(query, merged, top_k=min(final_k, len(merged)))
        logger.debug("HybridRetriever: final %d chunks after rerank", len(results))
        return results
