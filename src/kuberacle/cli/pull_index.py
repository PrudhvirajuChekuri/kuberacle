"""Download and unpack the persisted Chroma index from GCS.

Pulls the published ``latest`` index into the local on-disk persist directory
for CI eval and local bootstrapping. The API uses ``kuberacle.index_sync``
directly to pull a pinned version at startup.
"""

import argparse
import logging
from kuberacle.cli._root import project_root

from dotenv import load_dotenv

load_dotenv(project_root() / ".env")

from kuberacle.config import load_rag_config
from kuberacle.index_sync import (
    LATEST_MANIFEST,
    LATEST_TAR,
    IndexValidationError,
    download_and_extract,
)


PROJECT_ROOT = project_root()
CONFIG_PATH = PROJECT_ROOT / "configs" / "rag.yaml"

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Chroma index from GCS")
    parser.add_argument("--bucket", required=True, help="GCS bucket name")
    parser.add_argument(
        "--object",
        default=LATEST_TAR,
        help=f"GCS object path (default: {LATEST_TAR})",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    args = parse_args()
    config = load_rag_config(CONFIG_PATH)
    index_path = PROJECT_ROOT / config.vector_store.persist_directory

    try:
        download_and_extract(
            args.bucket, args.object, LATEST_MANIFEST, index_path, config
        )
    except IndexValidationError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
