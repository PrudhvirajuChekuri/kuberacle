"""Push the committed prompts to Langfuse for managed serving.

Git is the source of truth for prompts (``configs/prompts/<version>``); this
command uploads them to Langfuse under the version label so the running service
serves the managed copy (with the files as fallback). Run it after editing the
file prompts or as a deploy step. Requires the Langfuse env vars
(``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` / ``LANGFUSE_HOST``).
"""

import argparse
import logging

from dotenv import load_dotenv

from kuberacle.cli._root import project_root

load_dotenv(project_root() / ".env")

from kuberacle.config import load_rag_config
from kuberacle.observability.prompts import sync_prompts_to_langfuse
from kuberacle.observability.settings import load_observability_settings

PROJECT_ROOT = project_root()
CONFIG_PATH = PROJECT_ROOT / "configs" / "rag.yaml"


def parse_args() -> argparse.Namespace:
    """Parse prompt-sync CLI arguments."""
    parser = argparse.ArgumentParser(description="Sync file prompts to Langfuse")
    parser.add_argument(
        "--version",
        default=None,
        help="Prompt version to sync (defaults to configs/rag.yaml prompts.version)",
    )
    return parser.parse_args()


def main() -> None:
    """Sync the committed prompts for the configured version to Langfuse."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    args = parse_args()
    config = load_rag_config(CONFIG_PATH)
    version = args.version or config.prompts.version

    settings = load_observability_settings()
    if not settings.langfuse_enabled:
        raise SystemExit(
            "Langfuse is not configured. Set OBSERVABILITY_ENABLED=true and the "
            "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY env vars."
        )

    from langfuse import Langfuse

    langfuse = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
    prompt_dir = str(PROJECT_ROOT / config.prompts.directory)
    names = sync_prompts_to_langfuse(prompt_dir, version, langfuse)
    langfuse.flush()
    logger.info("Synced %d prompts for version %s: %s", len(names), version, names)


if __name__ == "__main__":
    main()
