"""Orchestrate the full preprocessing pipeline for K8s documentation.

Reads the page selection config, runs each page through all
preprocessing stages (frontmatter, shortcodes, links, structure,
chunking), and writes the output as JSONL.
"""

import json
import logging
import re
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

from kuberacle.preprocessing.frontmatter import extract_metadata
from kuberacle.preprocessing.shortcodes import resolve_shortcodes
from kuberacle.preprocessing.links import process_links, strip_links_to_text, extract_cross_references
from kuberacle.preprocessing.structure import analyze_structure, estimate_tokens
from kuberacle.preprocessing.chunker import chunk_document


_WHATSNEXT_RE = re.compile(
    r'^(#{1,6})\s+[Ww]hat[\u2019\']?s\s+next\b.*$',
    re.MULTILINE,
)


def strip_whatsnext(content: str) -> str:
    """Remove the "What's next" section and everything after it.

    Args:
        content: Markdown string.

    Returns:
        Markdown string with the "What's next" tail removed.
    """
    match = _WHATSNEXT_RE.search(content)
    if match:
        return content[:match.start()].rstrip()
    return content


def strip_inline_formatting(content: str) -> str:
    """Strip inline markdown formatting markers from non-code content.

    Removes backtick, bold (**), and italic (_) markers while
    preserving text inside code fences.

    Args:
        content: Markdown string.

    Returns:
        Markdown string with inline formatting markers removed.
    """
    lines = content.split("\n")
    result: list[str] = []
    in_code_block = False

    for line in lines:
        if line.startswith("```"):
            in_code_block = not in_code_block
            result.append(line)
            continue

        if in_code_block:
            result.append(line)
            continue

        stripped = line
        stripped = re.sub(r'`([^`]+)`', r'\1', stripped)
        stripped = re.sub(r'\*\*([^*]+)\*\*', r'\1', stripped)
        stripped = re.sub(r'(?<!\w)_([^_]+)_(?!\w)', r'\1', stripped)
        result.append(stripped)

    return "\n".join(result)


def process_page(
    file_path: str,
    raw_dir: str | Path,
    examples_dir: str | Path,
    includes_dir: str | Path,
    k8s_version: str,
    glossary_dir: str | Path | None = None,
    target_tokens: int | None = None,
    hard_cap_tokens: int | None = None,
) -> tuple[list[dict], Counter]:
    """Run a single page through the full preprocessing pipeline.

    Args:
        file_path: Page path relative to raw_dir.
        raw_dir: Path to data/raw/ directory.
        examples_dir: Path to data/examples/ directory.
        includes_dir: Path to data/includes/ directory.
        k8s_version: Kubernetes docs version string.
        glossary_dir: Path to glossary files for definition resolution.
        target_tokens: Target chunk size in tokens (optional).
        hard_cap_tokens: Hard cap chunk size in tokens (optional).

    Returns:
        Tuple of (chunks, unhandled_shortcodes) where chunks is
        a list of chunk dicts and unhandled_shortcodes is a Counter
        mapping shortcode names to their occurrence counts.
    """
    content = (Path(raw_dir) / file_path).read_text()

    # Stage 1: Frontmatter
    metadata, body = extract_metadata(content, file_path, k8s_version)

    # Stage 2: Shortcodes
    resolved, unhandled = resolve_shortcodes(
        body, examples_dir, includes_dir, k8s_version,
        glossary_dir=glossary_dir,
    )

    # Stage 3: Strip "What's next"
    resolved = strip_whatsnext(resolved)

    # Stage 4: Links - resolve relative to absolute, extract cross-references,
    # then strip markdown link syntax to plain text
    resolved = process_links(resolved)
    page_cross_references = extract_cross_references(resolved)
    resolved = strip_links_to_text(resolved)

    # Stage 5: Strip inline formatting
    resolved = strip_inline_formatting(resolved)

    # Stage 6: Structure analysis
    structure = analyze_structure(resolved)

    # Stage 7: Chunking
    chunk_kwargs: dict = {}
    if target_tokens is not None:
        chunk_kwargs["target_tokens"] = target_tokens
    if hard_cap_tokens is not None:
        chunk_kwargs["hard_cap_tokens"] = hard_cap_tokens
    chunks = chunk_document(resolved, structure, metadata, **chunk_kwargs)

    # Stage 8: Post-processing - enrich chunks with page-level metadata
    description = metadata.get("description", "").strip()
    total_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        chunk["chunk_index"] = i
        chunk["total_chunks"] = total_chunks
        chunk["cross_references"] = page_cross_references

    # Insert page description into the first chunk, after the breadcrumb
    # line and before the body, so the breadcrumb remains the leading line.
    if chunks and description and description not in chunks[0]["content"]:
        content = chunks[0]["content"]
        if "\n\n" in content:
            breadcrumb, body = content.split("\n\n", 1)
            chunks[0]["content"] = f"{breadcrumb}\n\n{description}\n\n{body}"
        else:
            chunks[0]["content"] = f"{content}\n\n{description}"
        chunks[0]["token_count"] = estimate_tokens(chunks[0]["content"])

    return chunks, unhandled


def run_pipeline(
    config: dict,
    data_dir: str | Path,
) -> tuple[list[dict], dict]:
    """Run the full preprocessing pipeline on all configured pages.

    Args:
        config: Parsed config dict from selected_pages.yaml.
        data_dir: Path to the data/ directory.

    Returns:
        Tuple of (all_chunks, stats) where stats is a summary dict.
    """
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    examples_dir = data_dir / "examples"
    includes_dir = data_dir / "includes"
    glossary_dir = data_dir / "glossary"
    k8s_version = config["k8s_version"]

    chunking_cfg = config.get("chunking", {})
    target_tokens = chunking_cfg.get("target_tokens")
    hard_cap_tokens = chunking_cfg.get("hard_cap_tokens")

    all_chunks: list[dict] = []
    page_stats: list[dict] = []
    all_unhandled: Counter = Counter()

    for section, pages in config["pages"].items():
        for page in pages:
            file_path = f"{section}/{page}"
            logger.info("Processing %s", file_path)

            try:
                chunks, unhandled = process_page(
                    file_path, raw_dir, examples_dir,
                    includes_dir, k8s_version,
                    glossary_dir=glossary_dir,
                    target_tokens=target_tokens,
                    hard_cap_tokens=hard_cap_tokens,
                )
                all_chunks.extend(chunks)
                all_unhandled.update(unhandled)

                tokens = [c["token_count"] for c in chunks]
                page_stats.append({
                    "file": file_path,
                    "chunks": len(chunks),
                    "min_tokens": min(tokens) if tokens else 0,
                    "max_tokens": max(tokens) if tokens else 0,
                })
            except Exception as e:
                logger.error("Failed to process %s: %s", file_path, e, exc_info=True)
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
        "unhandled_shortcodes": dict(sorted(all_unhandled.items(), key=lambda x: -x[1])),
        "pages": page_stats,
    }

    return all_chunks, stats


def write_jsonl(chunks: list[dict], output_path: str | Path) -> None:
    """Write chunks to a JSONL file.

    Args:
        chunks: List of chunk dicts.
        output_path: Path to the output .jsonl file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
