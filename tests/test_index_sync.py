"""Tests for index source resolution and manifest validation."""

import pytest

import kuberacle.index_sync as index_sync
from kuberacle.index_sync import (
    IndexValidationError,
    ResolvedIndex,
    load_index_settings,
    resolve_index,
    validate_manifest,
)
from kuberacle.provenance import INDEX_CONTRACT_VERSION


class FakeConfig:
    """Minimal config shape for validation/resolution tests."""

    class embedding:
        model_id = "gemini-embedding-001"
        output_dimensionality = 768

    class vector_store:
        collection_name = "k8s_docs_chunks_gemini"
        persist_directory = "data/vector/chroma_gemini"


def _good_manifest(**overrides) -> dict:
    manifest = {
        "embedding_model_id": "gemini-embedding-001",
        "embedding_output_dimensionality": 768,
        "collection_name": "k8s_docs_chunks_gemini",
        "index_contract_version": INDEX_CONTRACT_VERSION,
        "k8s_version": "v1.36",
    }
    manifest.update(overrides)
    return manifest


# --- load_index_settings ---

def test_load_index_settings_defaults_to_local(monkeypatch):
    for var in ("INDEX_SOURCE", "INDEX_BUCKET", "INDEX_VERSION", "INDEX_CACHE_DIR"):
        monkeypatch.delenv(var, raising=False)
    settings = load_index_settings()
    assert settings.source == "local"
    assert settings.version == ""  # no default; gcs must choose explicitly
    assert settings.cache_dir == index_sync.DEFAULT_CACHE_DIR


def test_load_index_settings_reads_env(monkeypatch):
    monkeypatch.setenv("INDEX_SOURCE", "gcs")
    monkeypatch.setenv("INDEX_BUCKET", "my-bucket")
    monkeypatch.setenv("INDEX_VERSION", "20260625T000000Z-abcd1234")
    monkeypatch.setenv("INDEX_CACHE_DIR", "/tmp/custom")
    settings = load_index_settings()
    assert settings.source == "gcs"
    assert settings.bucket == "my-bucket"
    assert settings.version == "20260625T000000Z-abcd1234"
    assert settings.cache_dir == "/tmp/custom"


# --- validate_manifest ---

def test_validate_manifest_accepts_matching(monkeypatch):
    validate_manifest(_good_manifest(), FakeConfig)


def test_validate_manifest_rejects_model_mismatch():
    with pytest.raises(IndexValidationError, match="embedding_model_id"):
        validate_manifest(_good_manifest(embedding_model_id="other-model"), FakeConfig)


def test_validate_manifest_rejects_dimension_mismatch():
    with pytest.raises(IndexValidationError, match="embedding_output_dimensionality"):
        validate_manifest(
            _good_manifest(embedding_output_dimensionality=1536), FakeConfig
        )


def test_validate_manifest_rejects_collection_mismatch():
    with pytest.raises(IndexValidationError, match="collection_name"):
        validate_manifest(_good_manifest(collection_name="other"), FakeConfig)


def test_validate_manifest_rejects_newer_contract():
    with pytest.raises(IndexValidationError, match="index_contract_version"):
        validate_manifest(
            _good_manifest(index_contract_version=INDEX_CONTRACT_VERSION + 1), FakeConfig
        )


def test_validate_manifest_allows_missing_contract_when_not_required(caplog):
    import logging

    manifest = _good_manifest()
    del manifest["index_contract_version"]
    with caplog.at_level(logging.WARNING):
        validate_manifest(manifest, FakeConfig)
    assert any("index_contract_version" in r.message for r in caplog.records)


def test_validate_manifest_requires_contract_when_strict():
    manifest = _good_manifest()
    del manifest["index_contract_version"]
    with pytest.raises(IndexValidationError, match="index_contract_version"):
        validate_manifest(manifest, FakeConfig, require_contract=True)


# --- resolve_index ---

def test_resolve_index_local_reads_version_file(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "k8s_version.txt").write_text("v1.36\n")
    settings = index_sync.IndexSettings("local", "", "latest", "/tmp/x")

    resolved = resolve_index(FakeConfig, settings, tmp_path)

    assert isinstance(resolved, ResolvedIndex)
    assert resolved.persist_directory == tmp_path / "data/vector/chroma_gemini"
    assert resolved.k8s_version == "v1.36"
    assert resolved.manifest is None


def test_resolve_index_local_handles_missing_version_file(tmp_path):
    settings = index_sync.IndexSettings("local", "", "latest", "/tmp/x")
    resolved = resolve_index(FakeConfig, settings, tmp_path)
    assert resolved.k8s_version == "unknown"


def test_resolve_index_gcs_requires_bucket(tmp_path):
    settings = index_sync.IndexSettings("gcs", "", "latest", "/tmp/x")
    with pytest.raises(IndexValidationError, match="INDEX_BUCKET"):
        resolve_index(FakeConfig, settings, tmp_path)


def test_resolve_index_gcs_requires_explicit_version(tmp_path):
    settings = index_sync.IndexSettings("gcs", "b", "", "/tmp/x")
    with pytest.raises(IndexValidationError, match="INDEX_VERSION"):
        resolve_index(FakeConfig, settings, tmp_path)


def test_resolve_index_gcs_pinned_version_uses_versioned_paths(monkeypatch, tmp_path):
    captured = {}

    def fake_download(bucket, tar_object, manifest_object, dest, config, require_contract=False):
        captured.update(
            bucket=bucket, tar=tar_object, manifest=manifest_object, dest=dest,
            require_contract=require_contract,
        )
        return {"k8s_version": "v1.36"}

    monkeypatch.setattr(index_sync, "download_and_extract", fake_download)
    settings = index_sync.IndexSettings("gcs", "b", "20260625T000000Z-abcd1234", "/tmp/cache")

    resolved = resolve_index(FakeConfig, settings, tmp_path)

    assert captured["tar"] == "index/versions/20260625T000000Z-abcd1234.tar.gz"
    assert captured["manifest"] == "index/versions/20260625T000000Z-abcd1234.manifest.json"
    assert captured["dest"] == index_sync.Path("/tmp/cache") / "chroma_gemini"
    # Serving must require a contract version.
    assert captured["require_contract"] is True
    assert resolved.persist_directory == index_sync.Path("/tmp/cache") / "chroma_gemini"
    assert resolved.k8s_version == "v1.36"


def test_resolve_index_gcs_latest_uses_pointer_paths(monkeypatch, tmp_path):
    captured = {}

    def fake_download(bucket, tar_object, manifest_object, dest, config, require_contract=False):
        captured.update(tar=tar_object, manifest=manifest_object)
        return {"k8s_version": "v1.36"}

    monkeypatch.setattr(index_sync, "download_and_extract", fake_download)
    settings = index_sync.IndexSettings("gcs", "b", "latest", "/tmp/cache")

    resolve_index(FakeConfig, settings, tmp_path)

    assert captured["tar"] == index_sync.LATEST_TAR
    assert captured["manifest"] == index_sync.LATEST_MANIFEST
