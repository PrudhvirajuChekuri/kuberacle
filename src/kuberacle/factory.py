"""Assembly of the hybrid RAG question-answer system from runtime config."""

from pathlib import Path

from kuberacle.config import RAGConfig
from kuberacle.ingestion.embedder import VertexAIEmbedder
from kuberacle.ingestion.vector_store import ChromaVectorStore
from kuberacle.retrieval.bm25 import BM25Retriever
from kuberacle.gate import VertexAIRelevanceGate
from kuberacle.generator import VertexAIAnswerGenerator
from kuberacle.observability.prompts import (
    load_answer_prompt,
    load_gate_prompt_managed,
)
from kuberacle.observability.tracing import get_langfuse
from kuberacle.qa import RAGQASystem
from kuberacle.retrieval.reranker import DiscoveryEngineReranker
from kuberacle.retrieval.retriever import HybridRetriever, SemanticRetriever


def build_qa_system(
    config: RAGConfig, project_root: Path, index_dir: Path | None = None
) -> RAGQASystem:
    """Wire the full hybrid retrieval + generation pipeline from config.

    Builds the embedder, vector store, semantic and BM25 retrievers, reranker,
    hybrid retriever, prompt bundle, answer generator, and the optional
    pre-retrieval relevance gate, then composes them
    into a ready-to-query ``RAGQASystem``. The BM25 index is built eagerly from
    all stored chunks, so callers should construct this once and reuse it.

    Args:
        config: Runtime RAG configuration.
        project_root: Project root used to resolve the prompt directory and the
            default on-disk vector store location.
        index_dir: Absolute Chroma persist directory to serve from. When given
            (for example a version pulled from GCS at startup), it overrides the
            config-relative ``vector_store.persist_directory``.

    Returns:
        A configured ``RAGQASystem``.
    """
    embedder = VertexAIEmbedder(
        model_id=config.embedding.model_id,
        gcp_project=config.gcp_project,
        gcp_location=config.gcp_location,
        output_dimensionality=config.embedding.output_dimensionality,
    )
    persist_directory = (
        str(index_dir)
        if index_dir is not None
        else str(project_root / config.vector_store.persist_directory)
    )
    vector_store = ChromaVectorStore(
        collection_name=config.vector_store.collection_name,
        persist_directory=persist_directory,
    )
    semantic = SemanticRetriever(
        embedder=embedder,
        vector_store=vector_store,
        top_k=config.retrieval.semantic_top_k,
    )
    all_chunks = vector_store.fetch_all_chunks()
    lexical = BM25Retriever(chunks=all_chunks, top_k=config.retrieval.lexical_top_k)
    reranker = DiscoveryEngineReranker(
        gcp_project=config.gcp_project,
        ranking_config=config.reranker.ranking_config,
        model=config.reranker.model,
        enabled=config.reranker.enabled,
    )
    retriever = HybridRetriever(
        semantic_retriever=semantic,
        bm25_retriever=lexical,
        reranker=reranker,
        semantic_top_k=config.retrieval.semantic_top_k,
        lexical_top_k=config.retrieval.lexical_top_k,
        merged_top_k=config.retrieval.merged_top_k,
        final_top_k=config.retrieval.final_top_k,
        semantic_weight=config.retrieval.hybrid_weight_semantic,
        lexical_weight=config.retrieval.hybrid_weight_lexical,
    )
    prompt_dir = str(project_root / config.prompts.directory)
    langfuse = get_langfuse()
    prompt_bundle, prompt_ref = load_answer_prompt(
        prompt_dir, config.prompts.version, langfuse
    )
    generator = VertexAIAnswerGenerator(
        model_id=config.generation.model_id,
        gcp_project=config.gcp_project,
        gcp_location=config.gcp_location,
        temperature=config.generation.temperature,
        max_tokens=config.generation.max_tokens,
        prompt_bundle=prompt_bundle,
        prompt_ref=prompt_ref,
    )
    relevance_gate = None
    if config.gate.enabled:
        gate_prompt, gate_ref = load_gate_prompt_managed(
            prompt_dir, config.prompts.version, langfuse
        )
        relevance_gate = VertexAIRelevanceGate(
            model_id=config.gate.model_id,
            gcp_project=config.gcp_project,
            gcp_location=config.gcp_location,
            prompt_bundle=gate_prompt,
            prompt_ref=gate_ref,
        )
    return RAGQASystem(
        retriever=retriever,
        generator=generator,
        min_evidence_score=config.retrieval.min_evidence_score,
        min_supporting_chunks=config.retrieval.min_supporting_chunks,
        strict_used_only=config.citation.strict_used_only,
        deduplicate_citations=config.citation.deduplicate,
        relevance_gate=relevance_gate,
    )
