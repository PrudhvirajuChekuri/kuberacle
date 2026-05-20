"""Run the preprocessing pipeline on configured K8s documentation pages.

Usage:
    python scripts/preprocess.py

Reads configs/selected_pages.yaml, processes all listed pages through
the full pipeline, and writes chunks to data/processed/chunks.jsonl.
"""

import yaml
from pathlib import Path

from k8s_rag.preprocessing.pipeline import run_pipeline, write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "selected_pages.yaml"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "processed" / "chunks.jsonl"


def main():
    print(f"Loading config from {CONFIG_PATH}")
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    print(f"K8s version: {config['k8s_version']}")
    page_count = sum(len(pages) for pages in config["pages"].values())
    print(f"Pages to process: {page_count}\n")

    chunks, stats = run_pipeline(config, DATA_DIR)

    write_jsonl(chunks, OUTPUT_PATH)

    print(f"\n{'=' * 50}")
    print(f"Pages processed: {stats['total_pages']}")
    print(f"Failed pages:    {stats['failed_pages']}")
    print(f"Total chunks:    {stats['total_chunks']}")
    print(f"Token range:     {stats['min_tokens']} - {stats['max_tokens']}")
    print(f"Avg tokens:      {stats['avg_tokens']}")
    print(f"Output:          {OUTPUT_PATH}")


if __name__ == "__main__":
    main()