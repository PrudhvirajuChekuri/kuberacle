"""Split resolved markdown documents into retrieval-ready chunks.

Uses the structural map from structure.py to make intelligent splitting
decisions: respects heading hierarchy, keeps code blocks and tables
atomic, and attaches rich metadata to each chunk.
"""

import re
from k8s_rag.preprocessing.structure import analyze_structure, classify_code_block, estimate_tokens


TARGET_TOKENS = 800
HARD_CAP_TOKENS = 1600

# Fields from the raw frontmatter that shouldn't propagate to chunks.
_METADATA_DROP = ("reviewers", "weight", "no_list")

# Sentence boundary: end-of-sentence punctuation followed by whitespace.
_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+')

# Trailing {#anchor-id} marker on a markdown heading line.
_HEADING_ANCHOR = re.compile(r'\s*\{#[^}]+\}\s*$')

# Sentence-overlap tuning for forced paragraph splits.
_OVERLAP_SENTENCES = 2
_OVERLAP_MAX_TOKENS = 150


def _strip_anchor_ids(text: str) -> str:
    """Remove trailing {#anchor-id} markers from markdown heading lines.

    The anchors are already captured into the structural map; keeping
    them in the chunk content adds noise to the embedded text.

    Args:
        text: Markdown content possibly containing heading anchor ids.

    Returns:
        The same text with anchor ids stripped from heading lines.
    """
    def clean(line):
        if line.lstrip().startswith("#"):
            return _HEADING_ANCHOR.sub("", line)
        return line

    return "\n".join(clean(line) for line in text.split("\n"))


def _format_breadcrumb(heading_hierarchy: list[str]) -> str:
    """Render the heading hierarchy as a bracketed breadcrumb line.

    Args:
        heading_hierarchy: List of strings from document title down to
            the chunk's leaf heading.

    Returns:
        A string like "[Pods > Working with Pods > Pod OS]".
    """
    return "[" + " > ".join(heading_hierarchy) + "]"


def _carry_over_sentences(prev_chunk_text: str) -> str:
    """Return the trailing sentences of prev_chunk_text for overlap.

    Used to prepend a short bridge to the start of a continuation
    chunk after a forced paragraph split. Returns an empty string when
    no clean sentence boundary is available or when the carry-over
    would be too large.

    Args:
        prev_chunk_text: The chunk text that immediately preceded the
            continuation chunk.

    Returns:
        A short string (1-2 sentences) or "" if no useful overlap can
        be produced.
    """
    text = prev_chunk_text.rstrip()
    if not text or text[-1] not in ".!?":
        return ""
    sentences = [s for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]
    if not sentences:
        return ""
    candidate = " ".join(sentences[-_OVERLAP_SENTENCES:])
    if estimate_tokens(candidate) > _OVERLAP_MAX_TOKENS:
        return ""
    return candidate


def _heading_slug(text: str) -> str:
    """Generate a URL-friendly anchor slug from a heading text.

    Matches Hugo's default anchor generation: lowercase, strip non-word
    characters (except hyphens), collapse whitespace to hyphens.

    Args:
        text: Raw heading text (without {#anchor} markers).

    Returns:
        Slug string suitable for use as a URL fragment.
    """
    slug = text.lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    return slug.strip("-")


def make_chunk_id(
    file_path: str,
    heading_text: str,
    index: int | None = None,
    heading_hierarchy: list[str] | None = None,
    line_number: int | None = None,
) -> str:
    """Generate a unique chunk ID from file path and heading.

    Args:
        file_path: File path relative to data/raw/
            (e.g., "concepts/workloads/pods/_index.md").
        heading_text: Text of the section heading.
        index: Optional numeric suffix for collision avoidance
            when a section is split into multiple chunks.
        heading_hierarchy: Optional full heading hierarchy path.
            When provided, it is used to disambiguate repeated leaf
            headings under different parent sections.
        line_number: Optional source line number for additional
            disambiguation when the same heading path appears multiple
            times in one document.

    Returns:
        A string ID like "concepts/workloads/pods/_index::pod-lifetime"
        or "concepts/workloads/pods/_index::pod-lifetime--1" if split.
    """
    base = file_path.replace(".md", "")
    slug_source = " > ".join(heading_hierarchy) if heading_hierarchy else heading_text
    slug = re.sub(r'[^a-z0-9]+', '-', slug_source.lower()).strip('-')
    chunk_id = f"{base}::{slug}" if slug else base
    if line_number is not None:
        chunk_id += f"--l{line_number}"
    if index is not None:
        chunk_id += f"--{index}"
    return chunk_id


def build_heading_tree(headings: list[dict], total_lines: int) -> list[dict]:
    """Convert a flat list of headings into a nested tree.

    Each node represents a heading and contains its children (deeper
    headings that appear before the next sibling).

    Args:
        headings: List of heading dicts from analyze_structure.
        total_lines: Total number of lines in the document.

    Returns:
        List of root-level nodes. Each node is a dict with keys:
        heading, children, start_line, end_line.
    """
    if not headings:
        return []

    nodes = []
    stack = []

    for i, heading in enumerate(headings):
        # Find where this heading's content ends
        end_line = total_lines - 1
        for next_h in headings[i + 1:]:
            if next_h["level"] <= heading["level"]:
                end_line = next_h["line"] - 1
                break

        node = {
            "heading": heading,
            "children": [],
            "start_line": heading["line"],
            "end_line": end_line,
        }

        # Pop stack until we find the parent (lower heading level)
        while stack and stack[-1]["heading"]["level"] >= heading["level"]:
            stack.pop()

        if stack:
            stack[-1]["children"].append(node)
        else:
            nodes.append(node)

        stack.append(node)

    return nodes


def _get_own_content(lines: list[str], node: dict) -> str:
    """Get a node's own content, excluding child sections.

    This is the text between the heading line and the start of
    the first child heading.

    Args:
        lines: All document lines.
        node: A heading tree node.

    Returns:
        String of the node's own content (may be empty).
    """
    if node["children"]:
        own_end = node["children"][0]["start_line"] - 1
    else:
        own_end = node["end_line"]

    section_lines = lines[node["start_line"]:own_end + 1]
    return "\n".join(section_lines)


def _get_full_content(lines: list[str], node: dict) -> str:
    """Get all content under a node, including children.

    Args:
        lines: All document lines.
        node: A heading tree node.

    Returns:
        String of the full section content.
    """
    section_lines = lines[node["start_line"]:node["end_line"] + 1]
    return "\n".join(section_lines)


def _is_inside_atomic(line_num: int, code_blocks: list[dict], tables: list[dict]) -> bool:
    """Check if a line is inside a code block or table.

    Args:
        line_num: Line number to check.
        code_blocks: List of code block dicts.
        tables: List of table dicts.

    Returns:
        True if the line is inside an atomic region.
    """
    for cb in code_blocks:
        if cb["start_line"] <= line_num <= cb["end_line"]:
            return True
    for t in tables:
        if t["start_line"] <= line_num <= t["end_line"]:
            return True
    return False


def _split_at_paragraphs(
    text: str,
    start_line: int,
    code_blocks: list[dict],
    tables: list[dict],
    target_tokens: int = TARGET_TOKENS,
) -> list[str]:
    """Split text at paragraph boundaries, respecting atomic units.

    Finds blank-line boundaries that are not inside code blocks or
    tables, and splits there to produce chunks under target_tokens.

    Args:
        text: The text to split.
        start_line: Line number offset for atomic unit checking.
        code_blocks: List of code block dicts.
        tables: List of table dicts.
        target_tokens: Target chunk size in tokens.

    Returns:
        List of text strings, each a chunk.
    """
    lines = text.split("\n")
    if estimate_tokens(text) <= target_tokens:
        return [text]

    # Find safe split points (blank lines not inside atomic units)
    split_points = []
    for i, line in enumerate(lines):
        absolute_line = start_line + i
        if line.strip() == "" and not _is_inside_atomic(absolute_line, code_blocks, tables):
            split_points.append(i)

    if not split_points:
        # No safe split points — return as-is even if oversized
        return [text]

    # Greedily accumulate paragraphs until adding more would exceed target.
    # Track the last split point where the chunk was still under target.
    chunks = []
    current_start = 0
    last_safe_split = None

    for sp in split_points:
        candidate = "\n".join(lines[current_start:sp])

        if estimate_tokens(candidate) <= target_tokens:
            last_safe_split = sp
        else:
            # Adding more would exceed target — split at last safe point
            if last_safe_split is not None:
                chunk_text = "\n".join(lines[current_start:last_safe_split])
                if chunk_text.strip():
                    chunks.append(chunk_text)
                current_start = last_safe_split + 1
                last_safe_split = None
                # Re-check current split point for the new chunk
                new_candidate = "\n".join(lines[current_start:sp])
                if estimate_tokens(new_candidate) <= target_tokens:
                    last_safe_split = sp
            else:
                # Very first paragraph already exceeds target — emit it
                if candidate.strip():
                    chunks.append(candidate)
                current_start = sp + 1

    # Collect whatever is left
    last_chunk = "\n".join(lines[current_start:])
    if last_chunk.strip():
        chunks.append(last_chunk)

    if not chunks:
        return [text]

    # Merge heading-only parts into the following part so a section heading
    # split off from its content never becomes a standalone chunk.
    merged: list[str] = []
    pending_heading: str | None = None
    for chunk in chunks:
        body_lines = [l for l in chunk.split("\n") if l.strip() and not l.startswith("#")]
        if not body_lines:
            pending_heading = chunk if pending_heading is None else f"{pending_heading}\n\n{chunk}"
        else:
            if pending_heading is not None:
                chunk = f"{pending_heading}\n\n{chunk}"
                pending_heading = None
            merged.append(chunk)
    if pending_heading is not None:
        # Trailing heading with no following content — attach to last chunk
        if merged:
            merged[-1] = f"{merged[-1]}\n\n{pending_heading}"
        else:
            merged.append(pending_heading)
    chunks = merged

    if not chunks:
        return [text]

    # Sentence-level overlap: prepend the tail of each chunk to its
    # successor so continuation chunks don't start mid-thought.
    with_overlap = [chunks[0]]
    for i in range(1, len(chunks)):
        carry = _carry_over_sentences(chunks[i - 1])
        if carry:
            with_overlap.append(f"{carry}\n\n{chunks[i]}")
        else:
            with_overlap.append(chunks[i])
    return with_overlap


def _force_split(text: str, hard_cap_tokens: int = HARD_CAP_TOKENS) -> list[str]:
    """Last resort: split oversized text at line boundaries.

    Used only when a chunk exceeds hard_cap_tokens and no
    paragraph or heading boundary is available.

    Args:
        text: The oversized text to split.
        hard_cap_tokens: Maximum tokens per chunk.

    Returns:
        List of text strings. Each part is kept under hard_cap_tokens
        where line boundaries allow it. A single line that already
        exceeds hard_cap_tokens is emitted as-is rather than truncated.
    """
    lines = text.split("\n")
    chunks = []
    current_lines = []
    current_text = ""

    for line in lines:
        test_text = current_text + "\n" + line if current_text else line
        if estimate_tokens(test_text) > hard_cap_tokens and current_text:
            chunks.append(current_text)
            current_lines = [line]
            current_text = line
        else:
            current_lines.append(line)
            current_text = test_text

    if current_text.strip():
        chunks.append(current_text)

    return chunks


def _collect_content_flags(raw_content: str) -> tuple[bool, list[str], bool]:
    """Collect has_code, code_types, has_table for a chunk's own text.

    Uses a direct line scan rather than analyze_structure to avoid false
    "unclosed code fence" warnings when a fence spans a chunk boundary.

    Args:
        raw_content: Raw chunk content before breadcrumb prefixing.

    Returns:
        Tuple of (has_code, code_types, has_table).
    """
    code_types: set[str] = set()
    has_table = False
    in_fence = False
    lang = ""
    fence_content: list[str] = []

    for line in raw_content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            if not in_fence:
                in_fence = True
                lang = stripped[3:].strip()
                fence_content = []
            else:
                code_types.add(classify_code_block(lang, "\n".join(fence_content)))
                in_fence = False
                fence_content = []
        elif in_fence:
            fence_content.append(line)
        elif "|" in stripped and not stripped.startswith(">"):
            has_table = True

    # Fence open at chunk boundary — still classify what we have so far
    if in_fence and fence_content:
        code_types.add(classify_code_block(lang, "\n".join(fence_content)))

    return bool(code_types), sorted(code_types), has_table


def _make_chunk(
    chunk_id: str,
    heading_hierarchy: list[str],
    raw_content: str,
    doc_metadata: dict,
) -> dict:
    """Build a chunk dict with all post-processing applied.

    Strips heading anchor markers, prepends the breadcrumb context
    line, derives chunk-local cross-references from the cleaned text,
    and computes content flags from the source line range.

    Args:
        chunk_id: Unique chunk id.
        heading_hierarchy: Breadcrumb list from doc title to leaf
            heading.
        raw_content: Untrimmed chunk text taken from the resolved
            document.
        doc_metadata: Document-level metadata dict.
    Returns:
        A chunk dict ready for serialization.
    """
    cleaned = _strip_anchor_ids(raw_content).strip()
    breadcrumb = _format_breadcrumb(heading_hierarchy)
    final_content = f"{breadcrumb}\n\n{cleaned}" if cleaned else breadcrumb
    has_code, code_types, has_table = _collect_content_flags(cleaned)
    return {
        "chunk_id": chunk_id,
        "heading_hierarchy": heading_hierarchy,
        "content": final_content,
        "token_count": estimate_tokens(final_content),
        "has_code": has_code,
        "code_types": code_types,
        "has_table": has_table,
        **{k: v for k, v in doc_metadata.items() if k not in _METADATA_DROP},
        "cross_references": [],  # populated by process_page after chunking
    }


def _chunk_node(
    node: dict,
    lines: list[str],
    doc_metadata: dict,
    structure: dict,
    parent_hierarchy: list[str],
    target_tokens: int = TARGET_TOKENS,
    hard_cap_tokens: int = HARD_CAP_TOKENS,
) -> list[dict]:
    """Recursively chunk a heading tree node.

    If the node's full content fits in target_tokens, it becomes one
    chunk. Otherwise, the node's own content becomes a chunk and
    children are recursed into. Oversized leaf content is split at
    paragraph boundaries.

    Args:
        node: A heading tree node.
        lines: All document lines.
        doc_metadata: Document-level metadata dict.
        structure: Full structural map from analyze_structure.
        parent_hierarchy: Heading hierarchy from parent nodes.
        target_tokens: Target chunk size in tokens.
        hard_cap_tokens: Maximum tokens per chunk.

    Returns:
        List of chunk dicts.
    """
    full_text = _get_full_content(lines, node)
    full_tokens = estimate_tokens(full_text)
    hierarchy = parent_hierarchy + [node["heading"]["text"]]
    file_path = doc_metadata.get("file_path", "")

    # Build node-specific metadata: append the heading anchor to source_url
    # so each chunk deep-links to its exact section rather than the page root.
    anchor = node["heading"].get("anchor") or _heading_slug(node["heading"]["text"])
    source_url = doc_metadata.get("source_url", "")
    if anchor and source_url and "#" not in source_url:
        base = source_url if source_url.endswith("/") else f"{source_url}/"
        node_metadata = {**doc_metadata, "source_url": f"{base}#{anchor}"}
    else:
        node_metadata = doc_metadata

    # If the whole section fits, make one chunk
    if full_tokens <= target_tokens:
        return [_make_chunk(
            chunk_id=make_chunk_id(
                file_path,
                node["heading"]["text"],
                heading_hierarchy=hierarchy,
                line_number=node["start_line"],
            ),
            heading_hierarchy=hierarchy,
            raw_content=full_text,
            doc_metadata=node_metadata,
        )]

    chunks = []

    if node["children"]:
        # Node's own content (between heading and first child)
        own_text = _get_own_content(lines, node)
        # Require actual body content below the heading line — the heading
        # itself doesn't count, so a section with no prose before its first
        # child produces no chunk here (children are still recursed into).
        own_body = any(
            l.strip() and not l.startswith("#")
            for l in own_text.split("\n")[1:]
        )
        if own_body:
            if estimate_tokens(own_text) <= target_tokens:
                chunks.append(_make_chunk(
                    chunk_id=make_chunk_id(
                        file_path,
                        node["heading"]["text"],
                        heading_hierarchy=hierarchy,
                        line_number=node["start_line"],
                    ),
                    heading_hierarchy=hierarchy,
                    raw_content=own_text,
                    doc_metadata=node_metadata,
                ))
            else:
                # Own content is too large — split at paragraphs
                parts = _split_at_paragraphs(
                    own_text,
                    node["start_line"],
                    structure["code_blocks"],
                    structure["tables"],
                    target_tokens=target_tokens,
                )
                for i, part in enumerate(parts):
                    idx = i if len(parts) > 1 else None
                    chunks.append(_make_chunk(
                        chunk_id=make_chunk_id(
                            file_path,
                            node["heading"]["text"],
                            idx,
                            heading_hierarchy=hierarchy,
                            line_number=node["start_line"],
                        ),
                        heading_hierarchy=hierarchy,
                        raw_content=part,
                        doc_metadata=node_metadata,
                    ))

        # Recurse into children
        for child in node["children"]:
            chunks.extend(_chunk_node(
                child, lines, doc_metadata, structure, hierarchy,
                target_tokens=target_tokens,
                hard_cap_tokens=hard_cap_tokens,
            ))
    else:
        # Leaf node that's too big — split at paragraphs
        parts = _split_at_paragraphs(
            full_text,
            node["start_line"],
            structure["code_blocks"],
            structure["tables"],
            target_tokens=target_tokens,
        )

        flat_idx = 0
        for part in parts:
            # Apply hard cap
            if estimate_tokens(part) > hard_cap_tokens:
                sub_parts = _force_split(part, hard_cap_tokens=hard_cap_tokens)
            else:
                sub_parts = [part]

            for sub in sub_parts:
                idx = flat_idx if len(parts) > 1 or len(sub_parts) > 1 else None
                flat_idx += 1
                chunks.append(_make_chunk(
                    chunk_id=make_chunk_id(
                        file_path,
                        node["heading"]["text"],
                        idx,
                        heading_hierarchy=hierarchy,
                        line_number=node["start_line"],
                    ),
                    heading_hierarchy=hierarchy,
                    raw_content=sub,
                    doc_metadata=node_metadata,
                ))

    return chunks


def chunk_document(
    content: str,
    structure: dict,
    metadata: dict,
    *,
    target_tokens: int = TARGET_TOKENS,
    hard_cap_tokens: int = HARD_CAP_TOKENS,
) -> list[dict]:
    """Split a resolved document into retrieval-ready chunks.

    This is the main entry point for the chunker. It builds a heading
    tree, recursively chunks each section, and handles intro content
    before the first heading. Cross-references are derived from each
    chunk's own content.

    Args:
        content: Resolved markdown string.
        structure: Structural map from analyze_structure().
        metadata: Document metadata from extract_metadata().
        target_tokens: Target chunk size in tokens (default 800).
        hard_cap_tokens: Hard cap chunk size in tokens (default 1600).

    Returns:
        List of chunk dicts, each containing content, metadata,
        and retrieval flags.
    """
    lines = content.split("\n")
    total_lines = len(lines)
    headings = structure["headings"]
    file_path = metadata.get("file_path", "")
    doc_title = metadata.get("title", "Untitled")

    tree = build_heading_tree(headings, total_lines)
    chunks = []

    # Handle intro content before the first heading
    first_heading_line = headings[0]["line"] if headings else total_lines
    if first_heading_line > 0:
        intro_text = "\n".join(lines[:first_heading_line]).strip()
        if intro_text:
            if estimate_tokens(intro_text) <= target_tokens:
                chunks.append(_make_chunk(
                    chunk_id=make_chunk_id(
                        file_path,
                        "intro",
                        heading_hierarchy=[doc_title],
                        line_number=0,
                    ),
                    heading_hierarchy=[doc_title],
                    raw_content=intro_text,
                    doc_metadata=metadata,
                ))
            else:
                parts = _split_at_paragraphs(
                    intro_text, 0,
                    structure["code_blocks"],
                    structure["tables"],
                    target_tokens=target_tokens,
                )
                flat_idx = 0
                for part in parts:
                    if estimate_tokens(part) > hard_cap_tokens:
                        sub_parts = _force_split(
                            part, hard_cap_tokens=hard_cap_tokens,
                        )
                    else:
                        sub_parts = [part]
                    for sub in sub_parts:
                        idx = flat_idx if len(parts) > 1 or len(sub_parts) > 1 else None
                        flat_idx += 1
                        chunks.append(_make_chunk(
                            chunk_id=make_chunk_id(
                                file_path,
                                "intro",
                                idx,
                                heading_hierarchy=[doc_title],
                                line_number=0,
                            ),
                            heading_hierarchy=[doc_title],
                            raw_content=sub,
                            doc_metadata=metadata,
                        ))

    # Recursively chunk each root-level section
    for node in tree:
        chunks.extend(_chunk_node(
            node, lines, metadata, structure, [doc_title],
            target_tokens=target_tokens,
            hard_cap_tokens=hard_cap_tokens,
        ))

    return chunks