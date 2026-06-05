"""Archive and upload the persisted Chroma index to GCS."""

import argparse
import json
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from k8s_rag.ingestion.config import load_rag_config


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
    args = parse_args()
    config = load_rag_config(CONFIG_PATH)
    index_path = PROJECT_ROOT / config.persist_directory

    if not index_path.exists():
        raise SystemExit(f"Index directory not found: {index_path}")

    k8s_version = (
        K8S_VERSION_FILE.read_text(encoding="utf-8").strip()
        if K8S_VERSION_FILE.exists()
        else "unknown"
    )

    manifest = {
        "k8s_version": k8s_version,
        "embedding_model_id": config.embedding_model_id,
        "embedding_output_dimensionality": config.embedding_output_dimensionality,
        "collection_name": config.collection_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    from google.cloud import storage

    tmp_path = Path(tempfile.mktemp(suffix=".tar.gz"))
    try:
        print(f"Archiving {index_path} ...")
        with tarfile.open(tmp_path, "w:gz") as tar:
            tar.add(index_path, arcname=index_path.name)

        client = storage.Client()
        bucket = client.bucket(args.bucket)

        print(f"Uploading to gs://{args.bucket}/{args.object} ...")
        blob = bucket.blob(args.object)
        blob.upload_from_filename(str(tmp_path))

        print(f"Uploading manifest to gs://{args.bucket}/{MANIFEST_OBJECT} ...")
        manifest_blob = bucket.blob(MANIFEST_OBJECT)
        manifest_blob.upload_from_string(
            json.dumps(manifest, indent=2), content_type="application/json"
        )

        print("Done.")
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
