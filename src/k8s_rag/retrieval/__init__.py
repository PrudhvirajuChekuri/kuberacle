"""Retrieval package for RAG querying."""

from k8s_rag.retrieval.bm25 import BM25Retriever
from k8s_rag.retrieval.factory import build_qa_system
from k8s_rag.retrieval.generator import VertexAIAnswerGenerator
from k8s_rag.retrieval.hybrid import merge_hybrid_candidates
from k8s_rag.retrieval.qa import AnswerDelta, Citation, QAResult, RAGQASystem
from k8s_rag.retrieval.reranker import DiscoveryEngineReranker
from k8s_rag.retrieval.retriever import HybridRetriever, SemanticRetriever

__all__ = [
    "AnswerDelta",
    "BM25Retriever",
    "build_qa_system",
    "VertexAIAnswerGenerator",
    "DiscoveryEngineReranker",
    "Citation",
    "HybridRetriever",
    "QAResult",
    "RAGQASystem",
    "SemanticRetriever",
    "merge_hybrid_candidates",
]
