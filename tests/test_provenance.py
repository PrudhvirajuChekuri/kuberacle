"""Tests for index provenance fingerprints and the contract version."""

from kuberacle.provenance import (
    INDEX_CONTRACT_VERSION,
    build_fingerprint,
    source_fingerprint,
)


# --- source_fingerprint ---

def test_source_fingerprint_is_deterministic():
    files = {
        "content/en/docs/concepts/a.md": {"sha": "aaa", "kind": "page"},
        "content/en/examples/b.yaml": {"sha": "bbb", "kind": "example"},
    }
    assert source_fingerprint(files) == source_fingerprint(dict(files))


def test_source_fingerprint_ignores_insertion_order():
    a = {
        "p1": {"sha": "1", "kind": "page"},
        "p2": {"sha": "2", "kind": "page"},
    }
    b = {
        "p2": {"sha": "2", "kind": "page"},
        "p1": {"sha": "1", "kind": "page"},
    }
    assert source_fingerprint(a) == source_fingerprint(b)


def test_source_fingerprint_changes_when_a_sha_changes():
    base = {"p1": {"sha": "1", "kind": "page"}}
    changed = {"p1": {"sha": "2", "kind": "page"}}
    assert source_fingerprint(base) != source_fingerprint(changed)


def test_source_fingerprint_changes_on_added_or_removed_file():
    base = {"p1": {"sha": "1", "kind": "page"}}
    added = {
        "p1": {"sha": "1", "kind": "page"},
        "p2": {"sha": "2", "kind": "page"},
    }
    assert source_fingerprint(base) != source_fingerprint(added)


# --- build_fingerprint ---

def _make_fake_project(root):
    """Create a minimal project tree with the index-build inputs."""
    pre = root / "src" / "kuberacle" / "preprocessing"
    ing = root / "src" / "kuberacle" / "ingestion"
    cli = root / "src" / "kuberacle" / "cli"
    pre.mkdir(parents=True)
    ing.mkdir(parents=True)
    cli.mkdir(parents=True)
    (pre / "chunker.py").write_text("def chunk():\n    return 1\n")
    (pre / "shortcodes").mkdir()
    (pre / "shortcodes" / "__init__.py").write_text("X = 1\n")
    (ing / "pipeline.py").write_text("def run():\n    return 2\n")
    (cli / "download_data.py").write_text("def scan():\n    return 3\n")
    (cli / "preprocess.py").write_text("def pre():\n    return 4\n")
    (cli / "ingest.py").write_text("def ing():\n    return 5\n")
    (root / "src" / "kuberacle" / "domain.py").write_text("FIELD = 1\n")
    (root / "configs" / "datasets").mkdir(parents=True)
    (root / "configs" / "datasets" / "full.yaml").write_text("chunking:\n  target_tokens: 800\n")


_CFG = {
    "embedding_model_id": "gemini-embedding-001",
    "embedding_output_dimensionality": 768,
    "collection_name": "k8s_docs_chunks_gemini",
}


def test_build_fingerprint_is_deterministic(tmp_path):
    _make_fake_project(tmp_path)
    assert build_fingerprint(tmp_path, _CFG) == build_fingerprint(tmp_path, dict(_CFG))


def test_build_fingerprint_changes_when_code_changes(tmp_path):
    _make_fake_project(tmp_path)
    before = build_fingerprint(tmp_path, _CFG)
    (tmp_path / "src" / "kuberacle" / "preprocessing" / "chunker.py").write_text(
        "def chunk():\n    return 999\n"
    )
    assert build_fingerprint(tmp_path, _CFG) != before


def test_build_fingerprint_changes_when_cli_scanner_changes(tmp_path):
    """download_data holds index-affecting dependency scanners, so it counts."""
    _make_fake_project(tmp_path)
    before = build_fingerprint(tmp_path, _CFG)
    (tmp_path / "src" / "kuberacle" / "cli" / "download_data.py").write_text(
        "def scan():\n    return 999\n"
    )
    assert build_fingerprint(tmp_path, _CFG) != before


def test_build_fingerprint_ignores_publish_cli(tmp_path):
    """push_index only publishes, so a change there must not force a rebuild."""
    _make_fake_project(tmp_path)
    before = build_fingerprint(tmp_path, _CFG)
    (tmp_path / "src" / "kuberacle" / "cli" / "push_index.py").write_text("X = 1\n")
    assert build_fingerprint(tmp_path, _CFG) == before


def test_build_fingerprint_changes_when_domain_changes(tmp_path):
    _make_fake_project(tmp_path)
    before = build_fingerprint(tmp_path, _CFG)
    (tmp_path / "src" / "kuberacle" / "domain.py").write_text("FIELD = 2\n")
    assert build_fingerprint(tmp_path, _CFG) != before


def test_build_fingerprint_changes_when_dataset_config_changes(tmp_path):
    _make_fake_project(tmp_path)
    before = build_fingerprint(tmp_path, _CFG)
    (tmp_path / "configs" / "datasets" / "full.yaml").write_text(
        "chunking:\n  target_tokens: 1000\n"
    )
    assert build_fingerprint(tmp_path, _CFG) != before


def test_build_fingerprint_changes_when_index_config_changes(tmp_path):
    _make_fake_project(tmp_path)
    before = build_fingerprint(tmp_path, _CFG)
    changed = {**_CFG, "embedding_output_dimensionality": 1536}
    assert build_fingerprint(tmp_path, changed) != before


def test_build_fingerprint_ignores_non_build_files(tmp_path):
    _make_fake_project(tmp_path)
    before = build_fingerprint(tmp_path, _CFG)
    # A change to serving/eval code outside the build inputs must not move it.
    (tmp_path / "src" / "kuberacle" / "generator.py").write_text("PROMPT = 1\n")
    assert build_fingerprint(tmp_path, _CFG) == before


# --- contract version ---

def test_index_contract_version_is_a_positive_int():
    assert isinstance(INDEX_CONTRACT_VERSION, int)
    assert INDEX_CONTRACT_VERSION >= 1
