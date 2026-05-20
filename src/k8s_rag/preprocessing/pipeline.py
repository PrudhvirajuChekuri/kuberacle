"""Orchestrate the full preprocessing pipeline for K8s documentation.

Reads the page selection config, runs each page through all five
preprocessing stages (frontmatter, shortcodes, links, structure,
chunking), and writes the output as JSONL.
"""

import json
from pathlib import Path

from k8s_rag.preprocessing.frontmatter import extract_metadata
from k8s_rag.preprocessing.shortcodes import resolve_shortcodes
from k8s_rag.preprocessing.links import process_links
from k8s_rag.preprocessing.structure import analyze_structure
from k8s_rag.preprocessing.chunker import chunk_document


def process_page(file_path, raw_dir, examples_dir, includes_dir, k8s_version):
    """Run a single page through the full preprocessing pipeline.

    Args:
        file_path: Page path relative to raw_dir
            (e.g., "concepts/workloads/pods/_index.md").
        raw_dir: Path to data/raw/ directory.
        examples_dir: Path to data/examples/ directory.
        includes_dir: Path to data/includes/ directory.
        k8s_version: Kubernetes docs version string.

    Returns:
        List of chunk dicts for this page.
    """
    content = (Path(raw_dir) / file_path).read_text()

    # Stage 1: Frontmatter
    metadata, body = extract_metadata(content, file_path, k8s_version)

    # Stage 2: Shortcodes
    resolved = resolve_shortcodes(body, examples_dir, includes_dir)

    # Stage 3: Links
    resolved, cross_references = process_links(resolved)

    # Stage 4: Structure
    structure = analyze_structure(resolved)

    # Stage 5: Chunking
    chunks = chunk_document(resolved, structure, metadata, cross_references)

    return chunks


def run_pipeline(config, data_dir):
    """Run the full preprocessing pipeline on all configured pages.

    Args:
        config: Parsed config dict from selected_pages.yaml.
        data_dir: Path to the data/ directory.

    Returns:
        Tuple of (all_chunks, stats) where stats is a summary dict.
    """
    raw_dir = Path(data_dir) / "raw"
    examples_dir = Path(data_dir) / "examples"
    includes_dir = Path(data_dir) / "includes"
    k8s_version = config["k8s_version"]

    all_chunks = []
    page_stats = []

    for section, pages in config["pages"].items():
        for page in pages:
            file_path = f"{section}/{page}"
            print(f"Processing {file_path}")

            try:
                chunks = process_page(
                    file_path, raw_dir, examples_dir,
                    includes_dir, k8s_version,
                )
                all_chunks.extend(chunks)

                tokens = [c["token_count"] for c in chunks]
                page_stats.append({
                    "file": file_path,
                    "chunks": len(chunks),
                    "min_tokens": min(tokens),
                    "max_tokens": max(tokens),
                })
            except Exception as e:
                print(f"  ERROR: {e}")
                page_stats.append({
                    "file": file_path,
                    "chunks": 0,
                    "error": str(e),
                })

    all_tokens = [c["token_count"] for c in all_chunks]
    stats = {
        "total_pages": len(page_stats),
        "total_chunks": len(all_chunks),
        "min_tokens": min(all_tokens) if all_tokens else 0,
        "max_tokens": max(all_tokens) if all_tokens else 0,
        "avg_tokens": int(sum(all_tokens) / len(all_tokens)) if all_tokens else 0,
        "failed_pages": sum(1 for p in page_stats if "error" in p),
        "pages": page_stats,
    }

    return all_chunks, stats


def write_jsonl(chunks, output_path):
    """Write chunks to a JSONL file.

    Args:
        chunks: List of chunk dicts.
        output_path: Path to the output .jsonl file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")