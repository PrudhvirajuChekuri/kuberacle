"""Tests for RAG config loader."""

from k8s_rag.ingestion.config import load_rag_config


def test_load_rag_config_parses_expected_fields(tmp_path):
    """Config loader should map YAML values into typed object."""
    config_path = tmp_path / "rag.yaml"
    config_path.write_text(
        "aws:\n"
        "  region: us-east-1\n"
        "models:\n"
        "  embedding: amazon.titan-embed-text-v2:0\n"
        "  generation: anthropic.claude-3-5-haiku-20241022-v1:0\n"
        "  reranker: cohere.rerank-v3-5:0\n"
        "vector_store:\n"
        "  collection_name: k8s_docs\n"
        "  persist_directory: data/vector/chroma\n"
        "retrieval:\n"
        "  semantic_top_k: 7\n"
        "  lexical_top_k: 6\n"
        "  merged_top_k: 9\n"
        "  final_top_k: 4\n"
        "  hybrid_weight_semantic: 0.7\n"
        "  hybrid_weight_lexical: 0.3\n"
        "  min_evidence_score: 0.2\n"
        "  min_supporting_chunks: 2\n"
        "reranker:\n"
        "  enabled: true\n"
        "  top_k: 4\n"
        "citation:\n"
        "  strict_used_only: true\n"
        "  deduplicate: true\n"
        "prompts:\n"
        "  version: v1\n"
        "  directory: configs/prompts\n"
        "generation:\n"
        "  temperature: 0.1\n"
        "  max_tokens: 500\n"
    )

    config = load_rag_config(config_path)
    assert config.aws_region == "us-east-1"
    assert config.collection_name == "k8s_docs"
    assert config.semantic_top_k == 7
    assert config.lexical_top_k == 6
    assert config.final_top_k == 4
    assert config.reranker_enabled is True
    assert config.prompt_version == "v1"
    assert config.max_tokens == 500
