"""Split resolved markdown documents into retrieval-ready chunks.

Uses the structural map from structure.py to make intelligent splitting
decisions: respects heading hierarchy, keeps code blocks and tables
atomic, and attaches rich metadata to each chunk.
"""

import re
from k8s_rag.preprocessing.structure import estimate_tokens


TARGET_TOKENS = 800
HARD_CAP_TOKENS = 1600


def make_chunk_id(file_path, heading_text, index=None):
    """Generate a unique chunk ID from file path and heading.

    Args:
        file_path: File path relative to data/raw/
            (e.g., "concepts/workloads/pods/_index.md").
        heading_text: Text of the section heading.
        index: Optional numeric suffix for collision avoidance
            when a section is split into multiple chunks.

    Returns:
        A string ID like "concepts/workloads/pods/_index::pod-lifetime"
        or "concepts/workloads/pods/_index::pod-lifetime--1" if split.
    """
    base = file_path.replace(".md", "")
    slug = re.sub(r'[^a-z0-9]+', '-', heading_text.lower()).strip('-')
    chunk_id = f"{base}::{slug}" if slug else base
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

    return chunks if chunks else [text]


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


def _collect_content_flags(start_line, end_line, code_blocks, tables):
    """Collect has_code, code_types, has_table for a line range.

    Args:
        start_line: First line of the range.
        end_line: Last line of the range.
        code_blocks: All code blocks in the document.
        tables: All tables in the document.

    Returns:
        Tuple of (has_code, code_types, has_table).
    """
    section_code = [
        cb for cb in code_blocks
        if cb["start_line"] >= start_line and cb["end_line"] <= end_line
    ]
    code_types = sorted(set(cb["code_type"] for cb in section_code))
    section_tables = [
        t for t in tables
        if t["start_line"] >= start_line and t["end_line"] <= end_line
    ]
    return bool(section_code), code_types, bool(section_tables)


def _chunk_node(node, lines, doc_metadata, cross_references,
                structure, parent_hierarchy):
    """Recursively chunk a heading tree node.

    If the node's full content fits in TARGET_TOKENS, it becomes one
    chunk. Otherwise, the node's own content becomes a chunk and
    children are recursed into. Oversized leaf content is split at
    paragraph boundaries.

    Args:
        node: A heading tree node.
        lines: All document lines.
        doc_metadata: Document-level metadata dict.
        cross_references: List of cross-reference URLs.
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
        has_code, code_types, has_table = _collect_content_flags(
            node["start_line"], node["end_line"],
            structure["code_blocks"], structure["tables"],
        )
        return [{
            "chunk_id": make_chunk_id(file_path, node["heading"]["text"]),
            "heading_hierarchy": hierarchy,
            "content": full_text.strip(),
            "token_count": full_tokens,
            "has_code": has_code,
            "code_types": code_types,
            "has_table": has_table,
            **{k: v for k, v in doc_metadata.items()
               if k not in ("reviewers", "weight", "no_list")},
            "cross_references": cross_references,
        }]

    chunks = []

    # Node's own content (between heading and first child)
    if node["children"]:
        own_text = _get_own_content(lines, node)
        if own_text.strip():
            own_end = node["children"][0]["start_line"] - 1
            has_code, code_types, has_table = _collect_content_flags(
                node["start_line"], own_end,
                structure["code_blocks"], structure["tables"],
            )

            if estimate_tokens(own_text) <= TARGET_TOKENS:
                chunks.append({
                    "chunk_id": make_chunk_id(file_path, node["heading"]["text"]),
                    "heading_hierarchy": hierarchy,
                    "content": own_text.strip(),
                    "token_count": estimate_tokens(own_text),
                    "has_code": has_code,
                    "code_types": code_types,
                    "has_table": has_table,
                    **{k: v for k, v in doc_metadata.items()
                       if k not in ("reviewers", "weight", "no_list")},
                    "cross_references": cross_references,
                })
            else:
                # Own content is too large — split at paragraphs
                parts = _split_at_paragraphs(
                    own_text, node["start_line"],
                    structure["code_blocks"], structure["tables"],
                )
                for i, part in enumerate(parts):
                    idx = i if len(parts) > 1 else None
                    chunks.append({
                        "chunk_id": make_chunk_id(
                            file_path, node["heading"]["text"], idx,
                        ),
                        "heading_hierarchy": hierarchy,
                        "content": part.strip(),
                        "token_count": estimate_tokens(part),
                        "has_code": has_code,
                        "code_types": code_types,
                        "has_table": has_table,
                        **{k: v for k, v in doc_metadata.items()
                           if k not in ("reviewers", "weight", "no_list")},
                        "cross_references": cross_references,
                    })

        # Recurse into children
        for child in node["children"]:
            chunks.extend(_chunk_node(
                child, lines, doc_metadata, cross_references,
                structure, hierarchy,
            ))
    else:
        # Leaf node that's too big — split at paragraphs
        parts = _split_at_paragraphs(
            full_text, node["start_line"],
            structure["code_blocks"], structure["tables"],
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
                has_code, code_types, has_table = _collect_content_flags(
                    node["start_line"], node["end_line"],
                    structure["code_blocks"], structure["tables"],
                )
                chunks.append({
                    "chunk_id": make_chunk_id(
                        file_path, node["heading"]["text"], idx,
                    ),
                    "heading_hierarchy": hierarchy,
                    "content": sub.strip(),
                    "token_count": estimate_tokens(sub),
                    "has_code": has_code,
                    "code_types": code_types,
                    "has_table": has_table,
                    **{k: v for k, v in doc_metadata.items()
                       if k not in ("reviewers", "weight", "no_list")},
                    "cross_references": cross_references,
                })

    return chunks


def chunk_document(content, structure, metadata, cross_references):
    """Split a resolved document into retrieval-ready chunks.

    This is the main entry point for the chunker. It builds a heading
    tree, recursively chunks each section, and handles intro content
    before the first heading.

    Args:
        content: Resolved markdown string.
        structure: Structural map from analyze_structure().
        metadata: Document metadata from extract_metadata().
        cross_references: List of cross-reference URLs.

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
            has_code, code_types, has_table = _collect_content_flags(
                0, first_heading_line - 1,
                structure["code_blocks"], structure["tables"],
            )
            chunks.append({
                "chunk_id": make_chunk_id(file_path, "intro"),
                "heading_hierarchy": [doc_title],
                "content": intro_text,
                "token_count": estimate_tokens(intro_text),
                "has_code": has_code,
                "code_types": code_types,
                "has_table": has_table,
                **{k: v for k, v in metadata.items()
                   if k not in ("reviewers", "weight", "no_list")},
                "cross_references": cross_references,
            })

    # Recursively chunk each root-level section
    for node in tree:
        chunks.extend(_chunk_node(
            node, lines, metadata, cross_references,
            structure, [doc_title],
        ))

    return chunks