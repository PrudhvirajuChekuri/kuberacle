"""Download and unpack the persisted Chroma index from GCS."""

import argparse
import json
import logging
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from k8s_rag.ingestion.config import load_rag_config


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "rag.yaml"
MANIFEST_OBJECT = "index/manifest.json"

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Chroma index from GCS")
    parser.add_argument("--bucket", required=True, help="GCS bucket name")
    parser.add_argument(
        "--object",
        default="index/latest.tar.gz",
        help="GCS object path (default: index/latest.tar.gz)",
    )
    return parser.parse_args()


def _validate_manifest(manifest: dict, config) -> None:
    """Validate GCS manifest against current config. Hard-fail on incompatible fields."""
    errors = []
    warnings = []

    if manifest.get("embedding_model_id") != config.embedding_model_id:
        errors.append(
            f"embedding_model_id: index={manifest.get('embedding_model_id')!r}, "
            f"config={config.embedding_model_id!r}"
        )
    if manifest.get("collection_name") != config.collection_name:
        errors.append(
            f"collection_name: index={manifest.get('collection_name')!r}, "
            f"config={config.collection_name!r}"
        )
    if manifest.get("embedding_output_dimensionality") != config.embedding_output_dimensionality:
        errors.append(
            f"embedding_output_dimensionality: index={manifest.get('embedding_output_dimensionality')}, "
            f"config={config.embedding_output_dimensionality}"
        )

    k8s_version = manifest.get("k8s_version", "unknown")
    if k8s_version == "unknown":
        warnings.append("k8s_version unknown in manifest - index may be stale")

    for w in warnings:
        logger.warning("%s", w)

    if errors:
        raise SystemExit(
            "Index is incompatible with current config:\n"
            + "\n".join(f"  {e}" for e in errors)
        )


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    args = parse_args()
    config = load_rag_config(CONFIG_PATH)
    index_path = PROJECT_ROOT / config.persist_directory

    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(args.bucket)

    logger.info("Validating manifest at gs://%s/%s", args.bucket, MANIFEST_OBJECT)
    manifest_blob = bucket.blob(MANIFEST_OBJECT)
    if manifest_blob.exists():
        manifest = json.loads(manifest_blob.download_as_text())
        logger.info(
            "k8s=%s model=%s collection=%s created=%s",
            manifest.get("k8s_version"),
            manifest.get("embedding_model_id"),
            manifest.get("collection_name"),
            manifest.get("created_at", "unknown"),
        )
        _validate_manifest(manifest, config)
    else:
        raise SystemExit(
            "No manifest found at index/manifest.json - index was published without validation. "
            "Run workflow_dispatch to rebuild and re-publish the index."
        )

    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".tar.gz")
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        logger.info("Downloading gs://%s/%s", args.bucket, args.object)
        blob = bucket.blob(args.object)
        blob.download_to_filename(str(tmp_path))

        if index_path.exists():
            shutil.rmtree(index_path)

        index_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Extracting to %s", index_path.parent)
        with tarfile.open(tmp_path, "r:gz") as tar:
            # filter="data" blocks path-traversal and unsafe members in case the
            # published artifact is ever tampered with.
            tar.extractall(index_path.parent, filter="data")
        logger.info("Index ready at %s", index_path)
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
