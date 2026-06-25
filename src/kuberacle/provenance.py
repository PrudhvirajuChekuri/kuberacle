"""Index provenance: fingerprints and the served-index contract version.

These helpers let the pipeline decide when a published index is stale and let
the API verify that an index artifact is compatible with the running code.

Three independent signals describe an index:

- ``source_fingerprint`` - the upstream Kubernetes docs content the index was
  built from (derived from recorded git blob SHAs).
- ``build_fingerprint`` - the local code and config that turn that content into
  the index (preprocessing + ingestion + domain types + dataset/runtime config).
- ``INDEX_CONTRACT_VERSION`` - the schema/metadata contract the served index
  exposes, bumped by hand only when the API must understand a new shape.
"""

import hashlib
import json
from pathlib import Path

from kuberacle.config import RAGConfig

#: Bump manually ONLY when the served index's schema/metadata contract changes
#: in a way the API must understand (for example new chunk metadata fields or a
#: different citation data shape). This is independent of upstream content
#: changes (``source_fingerprint``) and index-build code changes
#: (``build_fingerprint``); a rebuild from new docs does not bump it.
INDEX_CONTRACT_VERSION = 1

#: Code directories whose changes alter the bytes or metadata of the built
#: index. Kept deliberately narrow: only inputs to preprocessing + ingestion.
_BUILD_INPUT_DIRS = (
    "src/kuberacle/preprocessing",
    "src/kuberacle/ingestion",
)

#: Individual files (outside the directories above) that feed the build. The CLI
#: commands are included because they carry index-affecting logic: download_data
#: holds the dependency scanners (code_sample/include/glossary), and preprocess
#: and ingest orchestrate the build. push_index/pull_index/serve/query are
#: excluded - they publish or serve, they do not change the index bytes.
_BUILD_INPUT_FILES = (
    "src/kuberacle/domain.py",
    "src/kuberacle/cli/download_data.py",
    "src/kuberacle/cli/preprocess.py",
    "src/kuberacle/cli/ingest.py",
    "configs/datasets/full.yaml",
)


def source_fingerprint(source_files: dict) -> str:
    """Fingerprint upstream source content from recorded blob SHAs.

    Args:
        source_files: The ``files`` map from ``source_files.json``, mapping each
            repo-relative path to a ``{"sha", "kind"}`` record.

    Returns:
        Hex sha256 over the sorted ``(path, blob_sha)`` pairs. Identical content
        yields an identical fingerprint; any added, removed, or changed file
        changes it.
    """
    items = sorted((path, meta.get("sha", "")) for path, meta in source_files.items())
    digest = hashlib.sha256()
    for path, sha in items:
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_index_config(config: RAGConfig) -> dict:
    """Index-affecting runtime config values folded into build_fingerprint.

    Centralized so the publisher (push-index) and the freshness checker
    (docs-check) hash identical inputs; any drift between them would make every
    check report a spurious change.

    Args:
        config: Runtime RAG configuration.

    Returns:
        Dict of the index-affecting fields: embedding model id and dimension,
        collection name, and the persist directory's final name (the tarball's
        top-level directory, which runtime extraction must match).
    """
    return {
        "embedding_model_id": config.embedding.model_id,
        "embedding_output_dimensionality": config.embedding.output_dimensionality,
        "collection_name": config.vector_store.collection_name,
        "persist_directory_name": Path(config.vector_store.persist_directory).name,
    }


def _hash_file(path: Path) -> str:
    """Return the hex sha256 of a file's raw bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_fingerprint(project_root: Path, index_config: dict) -> str:
    """Fingerprint the local code and config that determine the built index.

    Covers preprocessing + ingestion code, the index-affecting CLI commands
    (download/preprocess/ingest), the domain record types, the dataset config,
    and the index-affecting fields of the runtime config. Excludes serving/eval-
    only settings (prompts, retrieval, reranker, generation, gate, observability,
    pricing, citation), which change behavior but not the index.

    Args:
        project_root: Repository root.
        index_config: Index-affecting runtime values (for example embedding
            model id, output dimensionality, and collection name).

    Returns:
        Hex sha256 over the sorted ``(relative_path, content_hash)`` pairs plus
        the index-affecting config values.
    """
    entries: list[tuple[str, str]] = []
    for rel_dir in _BUILD_INPUT_DIRS:
        base = project_root / rel_dir
        for file in base.rglob("*.py"):
            entries.append((file.relative_to(project_root).as_posix(), _hash_file(file)))
    for rel_file in _BUILD_INPUT_FILES:
        file = project_root / rel_file
        if file.exists():
            entries.append((rel_file, _hash_file(file)))
    entries.sort()

    digest = hashlib.sha256()
    for rel_path, content_hash in entries:
        digest.update(rel_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_hash.encode("utf-8"))
        digest.update(b"\n")
    digest.update(b"config\0")
    digest.update(json.dumps(index_config, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()
