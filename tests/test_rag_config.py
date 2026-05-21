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
        "vector_store:\n"
        "  collection_name: k8s_docs\n"
        "  persist_directory: data/vector/chroma\n"
        "retrieval:\n"
        "  top_k: 7\n"
        "generation:\n"
        "  temperature: 0.1\n"
        "  max_tokens: 500\n"
    )

    config = load_rag_config(config_path)
    assert config.aws_region == "us-east-1"
    assert config.collection_name == "k8s_docs"
    assert config.top_k == 7
    assert config.max_tokens == 500
