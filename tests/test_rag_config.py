"""Tests for RAG config loader."""

import pytest
import yaml

from kuberacle.config import load_rag_config


def _full_config() -> dict:
    """Return a complete config mapping covering every required key."""
    return {
        "models": {
            "embedding": "gemini-embedding-001",
            "generation": "gemini-2.5-flash-lite",
        },
        "embedding": {"output_dimensionality": 768},
        "vector_store": {
            "collection_name": "k8s_docs_chunks_gemini",
            "persist_directory": "data/vector/chroma_gemini",
        },
        "retrieval": {
            "semantic_top_k": 7,
            "lexical_top_k": 6,
            "merged_top_k": 9,
            "final_top_k": 4,
            "hybrid_weight_semantic": 0.7,
            "hybrid_weight_lexical": 0.3,
            "min_evidence_score": 0.2,
            "min_supporting_chunks": 2,
        },
        "reranker": {
            "enabled": True,
            "ranking_config": "default_ranking_config",
            "model": "semantic-ranker-default@latest",
        },
        "citation": {"strict_used_only": True, "deduplicate": True},
        "gate": {"enabled": True, "model": "gemini-2.5-flash"},
        "prompts": {"version": "v1", "directory": "configs/prompts"},
        "generation": {"temperature": 0.1, "max_tokens": 500},
        "evaluation": {
            "dataset_path": "evals/golden/v2.jsonl",
            "retrieval_recall_at_k_threshold": 0.90,
            "mrr_threshold": 0.75,
            "abstention_accuracy_threshold": 0.91,
            "non_empty_answer_rate_threshold": 0.95,
            "faithfulness_threshold": 0.88,
            "faithfulness_judge_model": "gemini-2.5-flash",
            "faithfulness_min_parsed": 11,
            "context_precision_threshold": 0.83,
            "context_precision_judge_model": "gemini-2.5-flash",
            "context_precision_min_parsed": 9,
            "answer_relevancy_threshold": 0.78,
            "answer_relevancy_judge_model": "gemini-2.5-flash",
            "answer_relevancy_embedding_model": "gemini-embedding-001",
            "answer_relevancy_min_parsed": 8,
        },
        "pricing": {
            "generation_input_per_1m_usd": 0.11,
            "generation_output_per_1m_usd": 0.41,
            "embedding_input_per_1m_usd": 0.16,
            "reranker_per_1k_queries_usd": 1.5,
        },
        "observability": {
            "service_name": "kuberacle-api-test",
            "logging": {"level": "DEBUG", "format": "text"},
            "tracing": {"sample_ratio": 0.25},
        },
    }


def _write_config(tmp_path, data: dict):
    """Write a config mapping to a YAML file and return its path."""
    config_path = tmp_path / "rag.yaml"
    config_path.write_text(yaml.safe_dump(data))
    return config_path


@pytest.fixture
def gcp_env(monkeypatch):
    """Set the GCP env vars the loader requires."""
    monkeypatch.setenv("GCP_PROJECT", "test-project")
    monkeypatch.setenv("GCP_LOCATION", "us-central1")


def test_load_rag_config_parses_expected_fields(tmp_path, gcp_env):
    """Config loader should map YAML values into typed object."""
    config = load_rag_config(_write_config(tmp_path, _full_config()))

    assert config.gcp_project == "test-project"
    assert config.gcp_location == "us-central1"
    assert config.embedding.model_id == "gemini-embedding-001"
    assert config.generation.model_id == "gemini-2.5-flash-lite"
    assert config.embedding.output_dimensionality == 768
    assert config.vector_store.collection_name == "k8s_docs_chunks_gemini"
    assert config.retrieval.semantic_top_k == 7
    assert config.retrieval.lexical_top_k == 6
    assert config.retrieval.merged_top_k == 9
    assert config.retrieval.final_top_k == 4
    assert config.retrieval.hybrid_weight_semantic == 0.7
    assert config.retrieval.hybrid_weight_lexical == 0.3
    assert config.retrieval.min_evidence_score == 0.2
    assert config.retrieval.min_supporting_chunks == 2
    assert config.reranker.enabled is True
    assert config.reranker.ranking_config == "default_ranking_config"
    assert config.reranker.model == "semantic-ranker-default@latest"
    assert config.citation.strict_used_only is True
    assert config.citation.deduplicate is True
    assert config.prompts.version == "v1"
    assert config.prompts.directory == "configs/prompts"
    assert config.gate.enabled is True
    assert config.gate.model_id == "gemini-2.5-flash"
    assert config.generation.temperature == 0.1
    assert config.generation.max_tokens == 500
    assert config.evaluation.dataset_path == "evals/golden/v2.jsonl"
    assert config.evaluation.retrieval_recall_at_k_threshold == 0.90
    assert config.evaluation.mrr_threshold == 0.75
    assert config.evaluation.abstention_accuracy_threshold == 0.91
    assert config.evaluation.non_empty_answer_rate_threshold == 0.95
    assert config.evaluation.faithfulness_threshold == 0.88
    assert config.evaluation.faithfulness_judge_model == "gemini-2.5-flash"
    assert config.evaluation.faithfulness_min_parsed == 11
    assert config.evaluation.context_precision_threshold == 0.83
    assert config.evaluation.context_precision_judge_model == "gemini-2.5-flash"
    assert config.evaluation.context_precision_min_parsed == 9
    assert config.evaluation.answer_relevancy_threshold == 0.78
    assert config.evaluation.answer_relevancy_judge_model == "gemini-2.5-flash"
    assert config.evaluation.answer_relevancy_embedding_model == "gemini-embedding-001"
    assert config.evaluation.answer_relevancy_min_parsed == 8
    assert config.pricing.generation_input_per_1m_usd == 0.11
    assert config.pricing.generation_output_per_1m_usd == 0.41
    assert config.pricing.embedding_input_per_1m_usd == 0.16
    assert config.pricing.reranker_per_1k_queries_usd == 1.5
    assert config.observability.service_name == "kuberacle-api-test"
    assert config.observability.log_level == "DEBUG"
    assert config.observability.log_format == "text"
    assert config.observability.trace_sample_ratio == 0.25


def test_load_rag_config_raises_without_gcp_project(tmp_path, monkeypatch):
    """Config loader should raise if GCP_PROJECT is not set."""
    monkeypatch.delenv("GCP_PROJECT", raising=False)
    monkeypatch.setenv("GCP_LOCATION", "us-central1")

    with pytest.raises(RuntimeError, match="GCP_PROJECT"):
        load_rag_config(_write_config(tmp_path, _full_config()))


def test_load_rag_config_raises_without_gcp_location(tmp_path, monkeypatch):
    """Config loader should raise if GCP_LOCATION is not set."""
    monkeypatch.setenv("GCP_PROJECT", "test-project")
    monkeypatch.delenv("GCP_LOCATION", raising=False)

    with pytest.raises(RuntimeError, match="GCP_LOCATION"):
        load_rag_config(_write_config(tmp_path, _full_config()))


def test_load_rag_config_raises_on_missing_key(tmp_path, gcp_env):
    """A missing leaf key fails loudly instead of falling back to a default."""
    data = _full_config()
    del data["retrieval"]["min_evidence_score"]

    with pytest.raises(RuntimeError, match="retrieval.min_evidence_score"):
        load_rag_config(_write_config(tmp_path, data))


def test_load_rag_config_raises_on_missing_section(tmp_path, gcp_env):
    """A missing section fails loudly, naming the first key looked up in it."""
    data = _full_config()
    del data["pricing"]

    with pytest.raises(RuntimeError, match="pricing"):
        load_rag_config(_write_config(tmp_path, data))


def test_load_rag_config_raises_on_missing_nested_key(tmp_path, gcp_env):
    """A missing nested observability key fails loudly with its dotted path."""
    data = _full_config()
    del data["observability"]["logging"]["level"]

    with pytest.raises(RuntimeError, match="observability.logging.level"):
        load_rag_config(_write_config(tmp_path, data))


def test_load_rag_config_raises_on_missing_model(tmp_path, gcp_env):
    """A missing model id fails loudly."""
    data = _full_config()
    del data["models"]["embedding"]

    with pytest.raises(RuntimeError, match="models.embedding"):
        load_rag_config(_write_config(tmp_path, data))


def test_load_rag_config_raises_on_invalid_hybrid_weights(tmp_path, gcp_env):
    """Config loader should raise RuntimeError when hybrid weights do not sum to 1.0."""
    data = _full_config()
    data["retrieval"]["hybrid_weight_semantic"] = 0.6
    data["retrieval"]["hybrid_weight_lexical"] = 0.6

    with pytest.raises(RuntimeError, match="sum to 1.0"):
        load_rag_config(_write_config(tmp_path, data))
