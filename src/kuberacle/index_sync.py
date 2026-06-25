"""Resolve the served vector index: a local directory or a pinned GCS artifact.

Production decouples the index from the container image. Instead of baking the
Chroma index into the image, the API pulls a specific index version from GCS at
startup, validates it against the running config and the supported contract
version, extracts it to a writable cache directory, and serves from there.
Local dev, docker-compose, CI, and the CLI keep using an on-disk directory.

Pinning is by ``INDEX_VERSION``: a concrete version (for example
``20260625T120000Z-1c6ea0b9``) maps to the immutable
``index/versions/<version>.{tar.gz,manifest.json}`` artifacts. The convenience
value ``latest`` follows the moving pointer and is meant for local dev only;
production should pin one revision to one exact version.
"""

import hashlib
import json
import logging
import os
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

from kuberacle.config import RAGConfig
from kuberacle.provenance import INDEX_CONTRACT_VERSION

logger = logging.getLogger(__name__)

SOURCE_LOCAL = "local"
SOURCE_GCS = "gcs"
DEFAULT_CACHE_DIR = "/tmp/kuberacle-index"
LATEST_TAR = "index/latest.tar.gz"
LATEST_MANIFEST = "index/manifest.json"
VERSIONS_PREFIX = "index/versions"


class IndexValidationError(RuntimeError):
    """Raised when a published index is incompatible with the running build."""


@dataclass(frozen=True)
class IndexSettings:
    """Where the served index comes from.

    Attributes:
        source: ``local`` (on-disk directory) or ``gcs`` (pull at startup).
        bucket: GCS bucket name (required when source is ``gcs``).
        version: Pinned index version, or ``latest`` for the moving pointer.
        cache_dir: Writable directory the GCS artifact extracts into.
    """

    source: str
    bucket: str
    version: str
    cache_dir: str


@dataclass(frozen=True)
class ResolvedIndex:
    """The index location the pipeline should use.

    Attributes:
        persist_directory: Absolute Chroma persist directory to serve from.
        k8s_version: Kubernetes docs version of the served index.
        manifest: The pulled manifest (``None`` in local mode).
    """

    persist_directory: Path
    k8s_version: str
    manifest: dict | None


def load_index_settings() -> IndexSettings:
    """Read index-source settings from the environment.

    Returns:
        Parsed ``IndexSettings``. Defaults to local source so dev, tests, CI,
        and the CLI stay offline unless GCS is explicitly requested.
    """
    return IndexSettings(
        source=os.environ.get("INDEX_SOURCE", SOURCE_LOCAL).strip().lower() or SOURCE_LOCAL,
        bucket=os.environ.get("INDEX_BUCKET", "").strip(),
        # No default: a GCS deployment must choose a version explicitly (a pinned
        # version, or "latest" for dev), so production never silently follows the
        # moving pointer by forgetting to set it.
        version=os.environ.get("INDEX_VERSION", "").strip(),
        cache_dir=os.environ.get("INDEX_CACHE_DIR", DEFAULT_CACHE_DIR).strip()
        or DEFAULT_CACHE_DIR,
    )


def validate_manifest(
    manifest: dict, config: RAGConfig, require_contract: bool = False
) -> None:
    """Validate a GCS manifest against the running config and contract version.

    Hard-fails on an embedding model/dimension/collection mismatch (the index
    bytes would be unreadable or wrong) or when the index was built with a
    newer contract than this API understands. A missing contract version fails
    when ``require_contract`` is set (production serving) and only warns
    otherwise (transitional CLI pull of a pre-provenance index).

    Args:
        manifest: Parsed index manifest.
        config: Running RAG config.
        require_contract: When true, a missing ``index_contract_version`` is an
            error rather than a warning.

    Raises:
        IndexValidationError: On any incompatibility.
    """
    errors = []

    if manifest.get("embedding_model_id") != config.embedding.model_id:
        errors.append(
            f"embedding_model_id: index={manifest.get('embedding_model_id')!r}, "
            f"config={config.embedding.model_id!r}"
        )
    if manifest.get("collection_name") != config.vector_store.collection_name:
        errors.append(
            f"collection_name: index={manifest.get('collection_name')!r}, "
            f"config={config.vector_store.collection_name!r}"
        )
    if manifest.get("embedding_output_dimensionality") != config.embedding.output_dimensionality:
        errors.append(
            f"embedding_output_dimensionality: "
            f"index={manifest.get('embedding_output_dimensionality')}, "
            f"config={config.embedding.output_dimensionality}"
        )

    contract = manifest.get("index_contract_version")
    if contract is None:
        if require_contract:
            errors.append(
                "index_contract_version: missing; rebuild and republish the index "
                "so it carries a contract version"
            )
        else:
            logger.warning(
                "Index manifest has no index_contract_version - assuming a pre-provenance index"
            )
    elif contract > INDEX_CONTRACT_VERSION:
        errors.append(
            f"index_contract_version: index={contract} is newer than this API "
            f"supports ({INDEX_CONTRACT_VERSION}); deploy a newer image"
        )

    if manifest.get("k8s_version", "unknown") == "unknown":
        logger.warning("k8s_version unknown in manifest - index may be stale")

    if errors:
        raise IndexValidationError(
            "Index is incompatible with current config:\n"
            + "\n".join(f"  {e}" for e in errors)
        )


def download_and_extract(
    bucket_name: str,
    tar_object: str,
    manifest_object: str,
    dest_index_path: Path,
    config: RAGConfig,
    require_contract: bool = False,
) -> dict:
    """Download, validate, and extract a GCS index artifact.

    Extracts into a staging directory and swaps it into place only after the
    expected Chroma directory is confirmed present, so a failed or malformed
    download never destroys the existing index.

    Args:
        bucket_name: GCS bucket name.
        tar_object: Object path of the index tarball.
        manifest_object: Object path of the index manifest JSON.
        dest_index_path: Final Chroma persist directory. Its name must match the
            tarball's top-level directory.
        config: Running RAG config, validated against the manifest.
        require_contract: Forwarded to validate_manifest (strict in serving).

    Returns:
        The parsed manifest.

    Raises:
        IndexValidationError: If the manifest is missing/incompatible or the
            archive lacks the expected directory.
    """
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    manifest_blob = bucket.blob(manifest_object)
    if not manifest_blob.exists():
        raise IndexValidationError(
            f"No manifest at gs://{bucket_name}/{manifest_object} - "
            "index was published without validation."
        )
    manifest = json.loads(manifest_blob.download_as_text())
    logger.info(
        "Manifest: version=%s k8s=%s model=%s collection=%s created=%s",
        manifest.get("index_version", "unknown"),
        manifest.get("k8s_version"),
        manifest.get("embedding_model_id"),
        manifest.get("collection_name"),
        manifest.get("created_at", "unknown"),
    )
    validate_manifest(manifest, config, require_contract=require_contract)

    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".tar.gz")
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    dest_index_path.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=".staging-", dir=dest_index_path.parent))
    try:
        logger.info("Downloading gs://%s/%s", bucket_name, tar_object)
        bucket.blob(tar_object).download_to_filename(str(tmp_path))

        # Verify the tarball matches the manifest's recorded digest before
        # extracting, so a corrupted or swapped artifact under a compatible
        # manifest cannot be served. Older manifests without it only warn.
        expected_sha = manifest.get("artifact_sha256")
        if expected_sha:
            actual_sha = hashlib.sha256(tmp_path.read_bytes()).hexdigest()
            if actual_sha != expected_sha:
                raise IndexValidationError(
                    f"Index tarball digest mismatch: manifest={expected_sha}, "
                    f"download={actual_sha}"
                )
        else:
            logger.warning(
                "Manifest has no artifact_sha256 - cannot verify tarball integrity"
            )

        logger.info("Extracting to staging under %s", dest_index_path.parent)
        with tarfile.open(tmp_path, "r:gz") as tar:
            # filter="data" blocks path-traversal and unsafe members in case the
            # published artifact is ever tampered with.
            tar.extractall(staging_dir, filter="data")

        extracted = staging_dir / dest_index_path.name
        if not extracted.is_dir():
            raise IndexValidationError(
                f"Index archive did not contain the expected directory "
                f"{dest_index_path.name!r}"
            )

        # Swap into place only after a clean extraction.
        if dest_index_path.exists():
            shutil.rmtree(dest_index_path)
        shutil.move(str(extracted), str(dest_index_path))
        logger.info("Index ready at %s", dest_index_path)
    finally:
        tmp_path.unlink(missing_ok=True)
        shutil.rmtree(staging_dir, ignore_errors=True)

    return manifest


def resolve_index(
    config: RAGConfig, settings: IndexSettings, project_root: Path
) -> ResolvedIndex:
    """Resolve the Chroma persist directory to serve, pulling from GCS if needed.

    Args:
        config: Running RAG config.
        settings: Index-source settings.
        project_root: Project root for the local on-disk index.

    Returns:
        The resolved index location and its docs version.

    Raises:
        IndexValidationError: If GCS is requested without a bucket, or the
            pulled index is incompatible.
    """
    persist_name = Path(config.vector_store.persist_directory).name

    if settings.source == SOURCE_GCS:
        if not settings.bucket:
            raise IndexValidationError("INDEX_SOURCE=gcs requires INDEX_BUCKET to be set.")
        if not settings.version:
            raise IndexValidationError(
                "INDEX_SOURCE=gcs requires INDEX_VERSION: a pinned version "
                "(e.g. 20260625T120000Z-1c6ea0b9), or 'latest' for dev only."
            )
        if settings.version == "latest":
            tar_object, manifest_object = LATEST_TAR, LATEST_MANIFEST
        else:
            tar_object = f"{VERSIONS_PREFIX}/{settings.version}.tar.gz"
            manifest_object = f"{VERSIONS_PREFIX}/{settings.version}.manifest.json"
        dest = Path(settings.cache_dir) / persist_name
        logger.info(
            "Pulling index '%s' from gs://%s", settings.version, settings.bucket
        )
        # The serving path is strict: a production index must carry a contract
        # version so the API knows it understands the served schema.
        manifest = download_and_extract(
            settings.bucket, tar_object, manifest_object, dest, config,
            require_contract=True,
        )
        return ResolvedIndex(dest, manifest.get("k8s_version", "unknown"), manifest)

    dest = project_root / config.vector_store.persist_directory
    version_file = project_root / "data" / "k8s_version.txt"
    k8s_version = (
        version_file.read_text(encoding="utf-8").strip()
        if version_file.exists()
        else "unknown"
    )
    return ResolvedIndex(dest, k8s_version, None)
