"""Tests for RAG config loader."""

import pytest

from k8s_rag.ingestion.config import load_rag_config


def test_load_rag_config_parses_expected_fields(tmp_path, monkeypatch):
    """Config loader should map YAML values into typed object."""
    monkeypatch.setenv("GCP_PROJECT", "test-project")
    monkeypatch.setenv("GCP_LOCATION", "us-central1")

    config_path = tmp_path / "rag.yaml"
    config_path.write_text(
        "models:\n"
        "  embedding: gemini-embedding-001\n"
        "  generation: gemini-2.5-flash-lite\n"
        "embedding:\n"
        "  output_dimensionality: 768\n"
        "vector_store:\n"
        "  collection_name: k8s_docs_chunks_gemini\n"
        "  persist_directory: data/vector/chroma_gemini\n"
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
        "  ranking_config: default_ranking_config\n"
        "  model: semantic-ranker-default@latest\n"
        "citation:\n"
        "  strict_used_only: true\n"
        "  deduplicate: true\n"
        "gate:\n"
        "  enabled: true\n"
        "  model: gemini-2.5-flash\n"
        "prompts:\n"
        "  version: v1\n"
        "  directory: configs/prompts\n"
        "generation:\n"
        "  temperature: 0.1\n"
        "  max_tokens: 500\n"
        "evaluation:\n"
        "  dataset_path: evals/golden/v2.jsonl\n"
        "  retrieval_recall_at_k_threshold: 0.90\n"
        "  mrr_threshold: 0.75\n"
        "  abstention_accuracy_threshold: 0.91\n"
        "  non_empty_answer_rate_threshold: 0.95\n"
        "  faithfulness_threshold: 0.88\n"
        "  faithfulness_judge_model: gemini-2.5-flash\n"
        "  faithfulness_min_parsed: 11\n"
        "  context_precision_threshold: 0.83\n"
        "  context_precision_judge_model: gemini-2.5-flash\n"
        "  context_precision_min_parsed: 9\n"
        "  answer_relevancy_threshold: 0.78\n"
        "  answer_relevancy_judge_model: gemini-2.5-flash\n"
        "  answer_relevancy_embedding_model: gemini-embedding-001\n"
        "  answer_relevancy_min_parsed: 8\n"
    )

    config = load_rag_config(config_path)

    assert config.gcp_project == "test-project"
    assert config.gcp_location == "us-central1"
    assert config.embedding_model_id == "gemini-embedding-001"
    assert config.generation_model_id == "gemini-2.5-flash-lite"
    assert config.embedding_output_dimensionality == 768
    assert config.collection_name == "k8s_docs_chunks_gemini"
    assert config.semantic_top_k == 7
    assert config.lexical_top_k == 6
    assert config.final_top_k == 4
    assert config.reranker_enabled is True
    assert config.reranker_ranking_config == "default_ranking_config"
    assert config.reranker_model == "semantic-ranker-default@latest"
    assert config.prompt_version == "v1"
    assert config.gate_enabled is True
    assert config.gate_model_id == "gemini-2.5-flash"
    assert config.max_tokens == 500
    assert config.evaluation_dataset_path == "evals/golden/v2.jsonl"
    assert config.eval_retrieval_recall_at_k_threshold == 0.90
    assert config.eval_mrr_threshold == 0.75
    assert config.eval_abstention_accuracy_threshold == 0.91
    assert config.eval_non_empty_answer_rate_threshold == 0.95
    assert config.eval_faithfulness_threshold == 0.88
    assert config.eval_faithfulness_judge_model == "gemini-2.5-flash"
    assert config.eval_faithfulness_min_parsed == 11
    assert config.eval_context_precision_threshold == 0.83
    assert config.eval_context_precision_judge_model == "gemini-2.5-flash"
    assert config.eval_context_precision_min_parsed == 9
    assert config.eval_answer_relevancy_threshold == 0.78
    assert config.eval_answer_relevancy_judge_model == "gemini-2.5-flash"
    assert config.eval_answer_relevancy_embedding_model == "gemini-embedding-001"
    assert config.eval_answer_relevancy_min_parsed == 8


def test_load_rag_config_gate_defaults(tmp_path, monkeypatch):
    """Without a gate section, the gate is disabled and reuses the generation model."""
    monkeypatch.setenv("GCP_PROJECT", "test-project")
    monkeypatch.setenv("GCP_LOCATION", "us-central1")

    config_path = tmp_path / "rag.yaml"
    config_path.write_text(
        "models:\n"
        "  embedding: gemini-embedding-001\n"
        "  generation: gemini-2.5-flash-lite\n"
        "vector_store:\n"
        "  collection_name: k8s_docs_chunks_gemini\n"
        "  persist_directory: data/vector/chroma_gemini\n"
        "generation:\n"
        "  temperature: 0.2\n"
        "  max_tokens: 600\n"
    )

    config = load_rag_config(config_path)

    assert config.gate_enabled is False
    assert config.gate_model_id == "gemini-2.5-flash-lite"


def test_load_rag_config_raises_without_gcp_project(tmp_path, monkeypatch):
    """Config loader should raise if GCP_PROJECT is not set."""
    monkeypatch.delenv("GCP_PROJECT", raising=False)
    monkeypatch.setenv("GCP_LOCATION", "us-central1")

    config_path = tmp_path / "rag.yaml"
    config_path.write_text(
        "models:\n"
        "  embedding: gemini-embedding-001\n"
        "  generation: gemini-2.5-flash-lite\n"
        "embedding:\n"
        "  output_dimensionality: 768\n"
        "vector_store:\n"
        "  collection_name: k8s_docs_chunks_gemini\n"
        "  persist_directory: data/vector/chroma_gemini\n"
        "retrieval:\n"
        "  semantic_top_k: 5\n"
        "  lexical_top_k: 5\n"
        "  merged_top_k: 10\n"
        "  final_top_k: 5\n"
        "reranker:\n"
        "  enabled: false\n"
        "  top_k: 5\n"
        "  ranking_config: default_ranking_config\n"
        "  model: semantic-ranker-default@latest\n"
        "generation:\n"
        "  temperature: 0.2\n"
        "  max_tokens: 600\n"
    )

    with pytest.raises(RuntimeError, match="GCP_PROJECT"):
        load_rag_config(config_path)


def test_load_rag_config_raises_without_gcp_location(tmp_path, monkeypatch):
    """Config loader should raise if GCP_LOCATION is not set."""
    monkeypatch.setenv("GCP_PROJECT", "test-project")
    monkeypatch.delenv("GCP_LOCATION", raising=False)

    config_path = tmp_path / "rag.yaml"
    config_path.write_text(
        "models:\n"
        "  embedding: gemini-embedding-001\n"
        "  generation: gemini-2.5-flash-lite\n"
        "embedding:\n"
        "  output_dimensionality: 768\n"
        "vector_store:\n"
        "  collection_name: k8s_docs_chunks_gemini\n"
        "  persist_directory: data/vector/chroma_gemini\n"
        "retrieval:\n"
        "  semantic_top_k: 5\n"
        "  lexical_top_k: 5\n"
        "  merged_top_k: 10\n"
        "  final_top_k: 5\n"
        "reranker:\n"
        "  enabled: false\n"
        "  top_k: 5\n"
        "  ranking_config: default_ranking_config\n"
        "  model: semantic-ranker-default@latest\n"
        "generation:\n"
        "  temperature: 0.2\n"
        "  max_tokens: 600\n"
    )

    with pytest.raises(RuntimeError, match="GCP_LOCATION"):
        load_rag_config(config_path)


def test_load_rag_config_raises_on_missing_yaml_key(tmp_path, monkeypatch):
    """Config loader should raise RuntimeError when a required YAML key is missing."""
    monkeypatch.setenv("GCP_PROJECT", "test-project")
    monkeypatch.setenv("GCP_LOCATION", "us-central1")

    config_path = tmp_path / "rag.yaml"
    config_path.write_text(
        "models:\n"
        "  generation: gemini-2.5-flash-lite\n"
        "vector_store:\n"
        "  collection_name: k8s_docs_chunks_gemini\n"
        "  persist_directory: data/vector/chroma_gemini\n"
        "generation:\n"
        "  temperature: 0.2\n"
        "  max_tokens: 600\n"
    )

    with pytest.raises(RuntimeError, match="models"):
        load_rag_config(config_path)


def test_load_rag_config_raises_on_invalid_hybrid_weights(tmp_path, monkeypatch):
    """Config loader should raise RuntimeError when hybrid weights do not sum to 1.0."""
    monkeypatch.setenv("GCP_PROJECT", "test-project")
    monkeypatch.setenv("GCP_LOCATION", "us-central1")

    config_path = tmp_path / "rag.yaml"
    config_path.write_text(
        "models:\n"
        "  embedding: gemini-embedding-001\n"
        "  generation: gemini-2.5-flash-lite\n"
        "vector_store:\n"
        "  collection_name: k8s_docs_chunks_gemini\n"
        "  persist_directory: data/vector/chroma_gemini\n"
        "retrieval:\n"
        "  hybrid_weight_semantic: 0.6\n"
        "  hybrid_weight_lexical: 0.6\n"
        "generation:\n"
        "  temperature: 0.2\n"
        "  max_tokens: 600\n"
    )

    with pytest.raises(RuntimeError, match="sum to 1.0"):
        load_rag_config(config_path)
