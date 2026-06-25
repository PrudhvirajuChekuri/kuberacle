"""Archive and upload the persisted Chroma index to GCS."""

import argparse
import hashlib
import json
import logging
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from kuberacle.cli._root import project_root

from dotenv import load_dotenv

load_dotenv(project_root() / ".env")

from kuberacle.config import load_rag_config
from kuberacle.provenance import (
    INDEX_CONTRACT_VERSION,
    build_fingerprint,
    build_index_config,
    source_fingerprint,
)


PROJECT_ROOT = project_root()
CONFIG_PATH = PROJECT_ROOT / "configs" / "rag.yaml"
K8S_VERSION_FILE = PROJECT_ROOT / "data" / "k8s_version.txt"
SOURCE_FILES_PATH = PROJECT_ROOT / "data" / "source_files.json"
MANIFEST_OBJECT = "index/manifest.json"
VERSIONS_PREFIX = "index/versions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload Chroma index to GCS")
    parser.add_argument("--bucket", required=True, help="GCS bucket name")
    parser.add_argument(
        "--object",
        default="index/latest.tar.gz",
        help="GCS object path for the 'latest' pointer (default: index/latest.tar.gz)",
    )
    parser.add_argument(
        "--version-out",
        default=None,
        help="Optional path to write the published index_version to (for CI).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    args = parse_args()
    config = load_rag_config(CONFIG_PATH)
    index_path = PROJECT_ROOT / config.vector_store.persist_directory

    if not index_path.exists():
        raise SystemExit(f"Index directory not found: {index_path}")

    if not SOURCE_FILES_PATH.exists():
        raise SystemExit(
            f"Source inventory not found: {SOURCE_FILES_PATH}. "
            "Run `kuberacle download-data` before publishing so the index carries "
            "its provenance."
        )

    k8s_version = (
        K8S_VERSION_FILE.read_text(encoding="utf-8").strip()
        if K8S_VERSION_FILE.exists()
        else "unknown"
    )

    # The tarball's top-level directory is the persist dir's final name; runtime
    # extraction requires it to match, so a rename is index-artifact-affecting.
    persist_directory_name = Path(config.vector_store.persist_directory).name

    source_files = json.loads(SOURCE_FILES_PATH.read_text(encoding="utf-8"))
    source_fp = source_fingerprint(source_files.get("files", {}))
    build_fp = build_fingerprint(PROJECT_ROOT, build_index_config(config))

    created_at = datetime.now(timezone.utc)
    index_version = f"{created_at:%Y%m%dT%H%M%SZ}-{source_fp[:8]}"
    versioned_tar = f"{VERSIONS_PREFIX}/{index_version}.tar.gz"
    versioned_manifest = f"{VERSIONS_PREFIX}/{index_version}.manifest.json"

    from google.cloud import storage

    tmp_path = Path(tempfile.mktemp(suffix=".tar.gz"))
    try:
        logger.info("Archiving %s", index_path)
        with tarfile.open(tmp_path, "w:gz") as tar:
            tar.add(index_path, arcname=index_path.name)
        # Bind the manifest to the exact bytes so a pull can verify the tarball
        # was not corrupted or swapped under a compatible manifest.
        artifact_sha256 = hashlib.sha256(tmp_path.read_bytes()).hexdigest()

        manifest = {
            "index_version": index_version,
            "index_contract_version": INDEX_CONTRACT_VERSION,
            "k8s_version": k8s_version,
            "embedding_model_id": config.embedding.model_id,
            "embedding_output_dimensionality": config.embedding.output_dimensionality,
            "collection_name": config.vector_store.collection_name,
            "persist_directory_name": persist_directory_name,
            "artifact_sha256": artifact_sha256,
            "source_head_commit": source_files.get("source_head_commit"),
            "source_fingerprint": source_fp,
            "build_fingerprint": build_fp,
            "object": versioned_tar,
            "created_at": created_at.isoformat(),
            "source_files": source_files,
        }
        manifest_json = json.dumps(manifest, indent=2)

        bucket = storage.Client().bucket(args.bucket)

        logger.info("Publishing index version %s", index_version)
        # Versioned artifacts are immutable: create-only (if_generation_match=0)
        # so a re-publish can never overwrite the bytes a pinned revision relies
        # on. A collision raises rather than silently clobbering.
        bucket.blob(versioned_tar).upload_from_filename(
            str(tmp_path), if_generation_match=0
        )
        bucket.blob(versioned_manifest).upload_from_string(
            manifest_json, content_type="application/json", if_generation_match=0
        )
        # The 'latest' pair is a moving pointer (overwrite allowed), updated last
        # so it only ever points at a fully-published version.
        bucket.blob(args.object).upload_from_filename(str(tmp_path))
        bucket.blob(MANIFEST_OBJECT).upload_from_string(
            manifest_json, content_type="application/json"
        )

        logger.info("Done (k8s=%s commit=%s)", k8s_version,
                    (source_files.get("source_head_commit") or "unknown")[:8])
        if args.version_out:
            Path(args.version_out).write_text(index_version, encoding="utf-8")
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
