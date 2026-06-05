"""Run the preprocessing pipeline on selected K8s documentation pages."""

import argparse
import logging
from pathlib import Path

import yaml

from k8s_rag.preprocessing.pipeline import run_pipeline, write_jsonl
from k8s_rag.preprocessing.page_selection import resolve_pages


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "datasets" / "full.yaml"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "processed" / "chunks.jsonl"
K8S_VERSION_FILE = DATA_DIR / "k8s_version.txt"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess selected Kubernetes docs into retrieval chunks."
    )
    parser.add_argument(
        "--config",
        default=str(CONFIG_PATH),
        help="Path to dataset selection YAML.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "list", "discover"],
        default="auto",
        help="Page selection mode. `auto` uses config `selection.mode`.",
    )
    parser.add_argument(
        "--sections",
        default=None,
        help="Comma-separated sections (e.g. concepts,tasks,tutorials).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional per-section page cap (useful for partial runs).",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help="Output chunks JSONL path.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Do not fail when some pages fail to preprocess.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.WARNING,
    )
    logger = logging.getLogger(__name__)

    config_path = Path(args.config)
    logger.info("Loading config from %s", config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if "k8s_version" not in config:
        if K8S_VERSION_FILE.exists():
            config["k8s_version"] = K8S_VERSION_FILE.read_text().strip()
        else:
            raise SystemExit(
                f"k8s_version not in config and {K8S_VERSION_FILE} not found. "
                "Run download_data.py first."
            )

    sections = args.sections.split(",") if args.sections else None
    page_map = resolve_pages(
        config=config,
        mode=args.mode,
        sections_override=sections,
        limit_override=args.limit,
    )
    config = {
        **config,
        "pages": page_map,
    }

    logger.info("K8s version: %s", config["k8s_version"])
    page_count = sum(len(pages) for pages in config["pages"].values())
    logger.info("Pages to process: %d", page_count)

    chunks, stats = run_pipeline(config, DATA_DIR)

    output_path = Path(args.output)
    write_jsonl(chunks, output_path)

    unhandled = stats["unhandled_shortcodes"]
    total_unhandled = sum(unhandled.values())

    logger.info("=" * 50)
    logger.info("Pages processed: %d", stats["total_pages"])
    logger.info("Failed pages:    %d", stats["failed_pages"])
    logger.info("Total chunks:    %d", stats["total_chunks"])
    logger.info("Token range:     %d - %d", stats["min_tokens"], stats["max_tokens"])
    logger.info("Avg tokens:      %d", stats["avg_tokens"])
    logger.info(
        "Unhandled:       %d unique shortcodes (%d total appearances)",
        len(unhandled), total_unhandled,
    )
    for name, count in unhandled.items():
        logger.info("                 - %s (%d)", name, count)
    logger.info("Output:          %s", output_path)
    if stats["failed_pages"] > 0 and not args.allow_partial:
        raise SystemExit(1)


if __name__ == "__main__":
    main()