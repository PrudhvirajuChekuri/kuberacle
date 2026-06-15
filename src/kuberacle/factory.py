"""Assembly of the hybrid RAG question-answer system from runtime config."""

from pathlib import Path

from kuberacle.config import RAGConfig
from kuberacle.ingestion.embedder import VertexAIEmbedder
from kuberacle.ingestion.vector_store import ChromaVectorStore
from kuberacle.retrieval.bm25 import BM25Retriever
from kuberacle.gate import VertexAIRelevanceGate
from kuberacle.generator import VertexAIAnswerGenerator
from kuberacle.prompts import load_gate_prompt, load_prompt_bundle
from kuberacle.qa import RAGQASystem
from kuberacle.retrieval.reranker import DiscoveryEngineReranker
from kuberacle.retrieval.retriever import HybridRetriever, SemanticRetriever


def build_qa_system(config: RAGConfig, project_root: Path) -> RAGQASystem:
    """Wire the full hybrid retrieval + generation pipeline from config.

    Builds the embedder, vector store, semantic and BM25 retrievers, reranker,
    hybrid retriever, prompt bundle, answer generator, and the optional
    pre-retrieval relevance gate, then composes them
    into a ready-to-query ``RAGQASystem``. The BM25 index is built eagerly from
    all stored chunks, so callers should construct this once and reuse it.

    Args:
        config: Runtime RAG configuration.
        project_root: Project root used to resolve the on-disk vector store
            persist directory and the prompt directory.

    Returns:
        A configured ``RAGQASystem``.
    """
    embedder = VertexAIEmbedder(
        model_id=config.embedding_model_id,
        gcp_project=config.gcp_project,
        gcp_location=config.gcp_location,
        output_dimensionality=config.embedding_output_dimensionality,
    )
    vector_store = ChromaVectorStore(
        collection_name=config.collection_name,
        persist_directory=str(project_root / config.persist_directory),
    )
    semantic = SemanticRetriever(
        embedder=embedder,
        vector_store=vector_store,
        top_k=config.semantic_top_k,
    )
    all_chunks = vector_store.fetch_all_chunks()
    lexical = BM25Retriever(chunks=all_chunks, top_k=config.lexical_top_k)
    reranker = DiscoveryEngineReranker(
        gcp_project=config.gcp_project,
        ranking_config=config.reranker_ranking_config,
        model=config.reranker_model,
        enabled=config.reranker_enabled,
    )
    retriever = HybridRetriever(
        semantic_retriever=semantic,
        bm25_retriever=lexical,
        reranker=reranker,
        semantic_top_k=config.semantic_top_k,
        lexical_top_k=config.lexical_top_k,
        merged_top_k=config.merged_top_k,
        final_top_k=config.final_top_k,
        semantic_weight=config.hybrid_weight_semantic,
        lexical_weight=config.hybrid_weight_lexical,
    )
    prompt_bundle = load_prompt_bundle(
        base_dir=str(project_root / config.prompt_directory),
        version=config.prompt_version,
    )
    generator = VertexAIAnswerGenerator(
        model_id=config.generation_model_id,
        gcp_project=config.gcp_project,
        gcp_location=config.gcp_location,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        prompt_bundle=prompt_bundle,
    )
    relevance_gate = None
    if config.gate_enabled:
        gate_prompt = load_gate_prompt(
            base_dir=str(project_root / config.prompt_directory),
            version=config.prompt_version,
        )
        relevance_gate = VertexAIRelevanceGate(
            model_id=config.gate_model_id,
            gcp_project=config.gcp_project,
            gcp_location=config.gcp_location,
            prompt_bundle=gate_prompt,
        )
    return RAGQASystem(
        retriever=retriever,
        generator=generator,
        min_evidence_score=config.min_evidence_score,
        min_supporting_chunks=config.min_supporting_chunks,
        strict_used_only=config.citation_strict_used_only,
        deduplicate_citations=config.citation_deduplicate,
        relevance_gate=relevance_gate,
    )
