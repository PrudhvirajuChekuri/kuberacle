"""Archive and upload the persisted Chroma index to GCS."""

import argparse
import json
import logging
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from kuberacle.config import load_rag_config


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "rag.yaml"
K8S_VERSION_FILE = PROJECT_ROOT / "data" / "k8s_version.txt"
MANIFEST_OBJECT = "index/manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload Chroma index to GCS")
    parser.add_argument("--bucket", required=True, help="GCS bucket name")
    parser.add_argument(
        "--object",
        default="index/latest.tar.gz",
        help="GCS object path (default: index/latest.tar.gz)",
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

    k8s_version = (
        K8S_VERSION_FILE.read_text(encoding="utf-8").strip()
        if K8S_VERSION_FILE.exists()
        else "unknown"
    )

    manifest = {
        "k8s_version": k8s_version,
        "embedding_model_id": config.embedding.model_id,
        "embedding_output_dimensionality": config.embedding.output_dimensionality,
        "collection_name": config.vector_store.collection_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    from google.cloud import storage

    tmp_path = Path(tempfile.mktemp(suffix=".tar.gz"))
    try:
        logger.info("Archiving %s", index_path)
        with tarfile.open(tmp_path, "w:gz") as tar:
            tar.add(index_path, arcname=index_path.name)

        client = storage.Client()
        bucket = client.bucket(args.bucket)

        logger.info("Uploading to gs://%s/%s", args.bucket, args.object)
        blob = bucket.blob(args.object)
        blob.upload_from_filename(str(tmp_path))

        logger.info("Uploading manifest to gs://%s/%s", args.bucket, MANIFEST_OBJECT)
        manifest_blob = bucket.blob(MANIFEST_OBJECT)
        manifest_blob.upload_from_string(
            json.dumps(manifest, indent=2), content_type="application/json"
        )

        logger.info("Done")
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
