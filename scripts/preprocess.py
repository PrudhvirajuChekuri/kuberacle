"""Run the preprocessing pipeline on selected K8s documentation pages."""

import argparse
from pathlib import Path

import yaml

from k8s_rag.preprocessing.pipeline import run_pipeline, write_jsonl
from k8s_rag.preprocessing.page_selection import resolve_pages


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "selected_pages.yaml"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "processed" / "chunks.jsonl"


def main():
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
        help="Optional per-section page cap (useful for smoke runs).",
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

    config_path = Path(args.config)
    print(f"Loading config from {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
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

    print(f"K8s version: {config['k8s_version']}")
    page_count = sum(len(pages) for pages in config["pages"].values())
    print(f"Pages to process: {page_count}\n")

    chunks, stats = run_pipeline(config, DATA_DIR)

    output_path = Path(args.output)
    write_jsonl(chunks, output_path)

    print(f"\n{'=' * 50}")
    print(f"Pages processed: {stats['total_pages']}")
    print(f"Failed pages:    {stats['failed_pages']}")
    print(f"Total chunks:    {stats['total_chunks']}")
    print(f"Token range:     {stats['min_tokens']} - {stats['max_tokens']}")
    print(f"Avg tokens:      {stats['avg_tokens']}")
    print(f"Output:          {output_path}")
    if stats["failed_pages"] > 0 and not args.allow_partial:
        raise SystemExit(1)


if __name__ == "__main__":
    main()