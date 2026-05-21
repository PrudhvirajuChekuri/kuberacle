"""Split resolved markdown documents into retrieval-ready chunks.

Uses the structural map from structure.py to make intelligent splitting
decisions: respects heading hierarchy, keeps code blocks and tables
atomic, and attaches rich metadata to each chunk.
"""

import re
from k8s_rag.preprocessing.structure import analyze_structure, estimate_tokens
from k8s_rag.preprocessing.links import extract_cross_references


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


def _strip_anchor_ids(text):
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


def _format_breadcrumb(heading_hierarchy):
    """Render the heading hierarchy as a bracketed breadcrumb line.

    Args:
        heading_hierarchy: List of strings from document title down to
            the chunk's leaf heading.

    Returns:
        A string like "[Pods > Working with Pods > Pod OS]".
    """
    return "[" + " > ".join(heading_hierarchy) + "]"


def _carry_over_sentences(prev_chunk_text):
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


def make_chunk_id(
    file_path,
    heading_text,
    index=None,
    heading_hierarchy=None,
    line_number=None,
):
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


def build_heading_tree(headings, total_lines):
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


def _get_own_content(lines, node):
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


def _get_full_content(lines, node):
    """Get all content under a node, including children.

    Args:
        lines: All document lines.
        node: A heading tree node.

    Returns:
        String of the full section content.
    """
    section_lines = lines[node["start_line"]:node["end_line"] + 1]
    return "\n".join(section_lines)


def _is_inside_atomic(line_num, code_blocks, tables):
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


def _split_at_paragraphs(text, start_line, code_blocks, tables):
    """Split text at paragraph boundaries, respecting atomic units.

    Finds blank-line boundaries that are not inside code blocks or
    tables, and splits there to produce chunks under TARGET_TOKENS.

    Args:
        text: The text to split.
        start_line: Line number offset for atomic unit checking.
        code_blocks: List of code block dicts.
        tables: List of table dicts.

    Returns:
        List of text strings, each a chunk.
    """
    lines = text.split("\n")
    if estimate_tokens(text) <= TARGET_TOKENS:
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

        if estimate_tokens(candidate) <= TARGET_TOKENS:
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
                if estimate_tokens(new_candidate) <= TARGET_TOKENS:
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


def _force_split(text):
    """Last resort: split oversized text at line boundaries.

    Used only when a chunk exceeds HARD_CAP_TOKENS and no
    paragraph or heading boundary is available.

    Args:
        text: The oversized text to split.

    Returns:
        List of text strings, each under HARD_CAP_TOKENS.
    """
    lines = text.split("\n")
    chunks = []
    current_lines = []
    current_text = ""

    for line in lines:
        test_text = current_text + "\n" + line if current_text else line
        if estimate_tokens(test_text) > HARD_CAP_TOKENS and current_text:
            chunks.append(current_text)
            current_lines = [line]
            current_text = line
        else:
            current_lines.append(line)
            current_text = test_text

    if current_text.strip():
        chunks.append(current_text)

    return chunks


def _get_heading_hierarchy(node, doc_title):
    """Build the heading breadcrumb trail for a node.

    Walks up from the node to the root, collecting heading texts,
    then prepends the document title.

    Args:
        node: A heading tree node.
        doc_title: The document's title from frontmatter.

    Returns:
        List of strings like ["Pods", "Pod lifetime", "Fault recovery"].
    """
    return [doc_title, node["heading"]["text"]]


def _collect_content_flags(raw_content):
    """Collect has_code, code_types, has_table for a chunk's own text.

    Args:
        raw_content: Raw chunk content before breadcrumb prefixing.

    Returns:
        Tuple of (has_code, code_types, has_table).
    """
    chunk_structure = analyze_structure(raw_content)
    section_code = chunk_structure["code_blocks"]
    code_types = sorted(set(cb["code_type"] for cb in section_code))
    section_tables = chunk_structure["tables"]
    return bool(section_code), code_types, bool(section_tables)


def _make_chunk(chunk_id, heading_hierarchy, raw_content, doc_metadata):
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
    cross_refs = extract_cross_references(cleaned)
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
        "cross_references": cross_refs,
    }


def _chunk_node(node, lines, doc_metadata, structure, parent_hierarchy):
    """Recursively chunk a heading tree node.

    If the node's full content fits in TARGET_TOKENS, it becomes one
    chunk. Otherwise, the node's own content becomes a chunk and
    children are recursed into. Oversized leaf content is split at
    paragraph boundaries.

    Args:
        node: A heading tree node.
        lines: All document lines.
        doc_metadata: Document-level metadata dict.
        structure: Full structural map from analyze_structure.
        parent_hierarchy: Heading hierarchy from parent nodes.

    Returns:
        List of chunk dicts.
    """
    full_text = _get_full_content(lines, node)
    full_tokens = estimate_tokens(full_text)
    hierarchy = parent_hierarchy + [node["heading"]["text"]]
    file_path = doc_metadata.get("file_path", "")
    # If the whole section fits, make one chunk
    if full_tokens <= TARGET_TOKENS:
        return [_make_chunk(
            chunk_id=make_chunk_id(
                file_path,
                node["heading"]["text"],
                heading_hierarchy=hierarchy,
                line_number=node["start_line"],
            ),
            heading_hierarchy=hierarchy,
            raw_content=full_text,
            doc_metadata=doc_metadata,
        )]

    chunks = []

    if node["children"]:
        # Node's own content (between heading and first child)
        own_text = _get_own_content(lines, node)
        if own_text.strip():
            if estimate_tokens(own_text) <= TARGET_TOKENS:
                chunks.append(_make_chunk(
                    chunk_id=make_chunk_id(
                        file_path,
                        node["heading"]["text"],
                        heading_hierarchy=hierarchy,
                        line_number=node["start_line"],
                    ),
                    heading_hierarchy=hierarchy,
                    raw_content=own_text,
                    doc_metadata=doc_metadata,
                ))
            else:
                # Own content is too large — split at paragraphs
                parts = _split_at_paragraphs(
                    own_text,
                    node["start_line"],
                    structure["code_blocks"],
                    structure["tables"],
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
                        doc_metadata=doc_metadata,
                    ))

        # Recurse into children
        for child in node["children"]:
            chunks.extend(_chunk_node(
                child, lines, doc_metadata, structure, hierarchy,
            ))
    else:
        # Leaf node that's too big — split at paragraphs
        parts = _split_at_paragraphs(
            full_text,
            node["start_line"],
            structure["code_blocks"],
            structure["tables"],
        )

        for i, part in enumerate(parts):
            # Apply hard cap
            if estimate_tokens(part) > HARD_CAP_TOKENS:
                sub_parts = _force_split(part)
            else:
                sub_parts = [part]

            for j, sub in enumerate(sub_parts):
                idx = i if len(parts) > 1 else None
                if j > 0:
                    idx = (i * 10) + j
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
                    doc_metadata=doc_metadata,
                ))

    return chunks


def chunk_document(content, structure, metadata, cross_references=None):
    """Split a resolved document into retrieval-ready chunks.

    This is the main entry point for the chunker. It builds a heading
    tree, recursively chunks each section, and handles intro content
    before the first heading. Cross-references are derived from each
    chunk's own content; the optional cross_references argument is
    retained for backward compatibility but ignored.

    Args:
        content: Resolved markdown string.
        structure: Structural map from analyze_structure().
        metadata: Document metadata from extract_metadata().
        cross_references: Deprecated. Kept for backward compatibility.

    Returns:
        List of chunk dicts, each containing content, metadata,
        and retrieval flags.
    """
    del cross_references  # superseded by chunk-local extraction
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

    # Recursively chunk each root-level section
    for node in tree:
        chunks.extend(_chunk_node(
            node, lines, metadata, structure, [doc_title],
        ))

    return chunks