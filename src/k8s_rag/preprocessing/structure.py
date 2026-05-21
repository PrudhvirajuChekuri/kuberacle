"""Analyze the structure of resolved markdown documents.

Builds a structural map of each document: heading tree, code blocks,
tables, and section boundaries with token estimates. The chunker uses
this map to determine safe split points and atomic regions.
"""

import re


# A GFM table separator row: optional leading/trailing pipes, two or
# more cells of three-plus dashes (with optional alignment colons),
# pipe-separated. Used to distinguish real tables from prose that
# happens to contain pipe characters.
_TABLE_SEPARATOR = re.compile(
    r'^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$'
)


def _is_table_separator(line):
    """Return True if a line is a GFM table separator row.

    Args:
        line: A single line of markdown.

    Returns:
        True if the line matches the separator pattern (e.g.,
        "|---|---|" or "| :--- | ---: |").
    """
    return bool(_TABLE_SEPARATOR.match(line))


def estimate_tokens(text):
    """Estimate token count for a text string.

    Uses a rough heuristic of words * 1.3, which is close enough
    for chunking decisions without needing a tokenizer dependency.

    Args:
        text: String to estimate tokens for.

    Returns:
        Estimated token count as an integer.
    """
    words = len(text.split())
    return int(words * 1.3)


def classify_code_block(language, content):
    """Classify a fenced code block by its content type.

    Args:
        language: Language tag from the code fence (e.g., "yaml", "shell").
        content: The code block content.

    Returns:
        One of: "yaml-manifest", "kubectl-command", "shell", or the
        original language tag.
    """
    if language in ("yaml", "yml"):
        if "apiVersion:" in content or "kind:" in content:
            return "yaml-manifest"
        return "yaml"
    if language in ("shell", "bash", "sh"):
        if "kubectl" in content:
            return "kubectl-command"
        return "shell"
    return language or "text"


def analyze_structure(content):
    """Build a structural map of a resolved markdown document.

    Scans the document line by line, tracking whether we're inside
    a fenced code block to avoid false detections. Produces a map
    of headings, code blocks, tables, and sections with token counts.

    Args:
        content: Resolved markdown string (after shortcode and link
            processing).

    Returns:
        Dict with keys:
            headings: List of dicts with level, text, line, anchor.
            code_blocks: List of dicts with start_line, end_line,
                language, code_type.
            tables: List of dicts with start_line, end_line.
            sections: List of dicts with heading_text, heading_level,
                start_line, end_line, token_count, has_code,
                code_types, has_table.
    """
    lines = content.split("\n")
    headings = []
    code_blocks = []
    tables = []

    in_code_block = False
    code_start = None
    code_language = ""
    code_content_lines = []

    in_table = False
    table_start = None

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Track fenced code blocks
        if stripped.startswith("```"):
            if not in_code_block:
                in_code_block = True
                code_start = i
                code_language = stripped[3:].strip()
                code_content_lines = []
            else:
                code_content = "\n".join(code_content_lines)
                code_type = classify_code_block(code_language, code_content)
                code_blocks.append({
                    "start_line": code_start,
                    "end_line": i,
                    "language": code_language,
                    "code_type": code_type,
                })
                in_code_block = False
                code_start = None
                code_content_lines = []
            continue

        if in_code_block:
            code_content_lines.append(line)
            continue

        # Detect headings (only outside code blocks)
        heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if heading_match:
            level = len(heading_match.group(1))
            raw_text = heading_match.group(2).strip()

            # Extract anchor if present: ## Text {#anchor-id}
            anchor = None
            anchor_match = re.search(r'\{#([^}]+)\}\s*$', raw_text)
            if anchor_match:
                anchor = anchor_match.group(1)
                raw_text = raw_text[:anchor_match.start()].strip()

            headings.append({
                "level": level,
                "text": raw_text,
                "line": i,
                "anchor": anchor,
            })

            # End any open table at a heading boundary
            if in_table:
                tables.append({
                    "start_line": table_start,
                    "end_line": i - 1,
                })
                in_table = False
            continue

        # Detect tables (pipe-delimited rows). A region only counts as
        # a table when a header row is followed by a GFM separator row,
        # so that prose lines containing "|" (inline code, shell pipes,
        # cross-references) don't get flagged.
        is_pipe_line = "|" in stripped and not stripped.startswith(">")
        if in_table:
            if not is_pipe_line:
                tables.append({
                    "start_line": table_start,
                    "end_line": i - 1,
                })
                in_table = False
        else:
            if (
                is_pipe_line
                and i + 1 < len(lines)
                and _is_table_separator(lines[i + 1])
            ):
                in_table = True
                table_start = i

    # Close any open table at end of document
    if in_table:
        tables.append({
            "start_line": table_start,
            "end_line": len(lines) - 1,
        })

    # Build sections from headings
    sections = _build_sections(lines, headings, code_blocks, tables)

    return {
        "headings": headings,
        "code_blocks": code_blocks,
        "tables": tables,
        "sections": sections,
    }


def _build_sections(lines, headings, code_blocks, tables):
    """Build section boundaries from headings with content analysis.

    Each section spans from one heading to the next heading of the
    same or higher level. The first section covers any content before
    the first heading.

    Args:
        lines: List of document lines.
        headings: List of heading dicts from analyze_structure.
        code_blocks: List of code block dicts from analyze_structure.
        tables: List of table dicts from analyze_structure.

    Returns:
        List of section dicts with heading info, line range,
        token count, and content flags.
    """
    sections = []
    total_lines = len(lines)

    # Content before the first heading (if any)
    first_heading_line = headings[0]["line"] if headings else total_lines
    if first_heading_line > 0:
        section_text = "\n".join(lines[:first_heading_line])
        if section_text.strip():
            sections.append(_make_section(
                heading_text="(intro)",
                heading_level=0,
                start_line=0,
                end_line=first_heading_line - 1,
                section_text=section_text,
                code_blocks=code_blocks,
                tables=tables,
            ))

    # Each heading starts a section that ends at the next same-or-higher level
    for idx, heading in enumerate(headings):
        start_line = heading["line"]

        # Find where this section ends
        end_line = total_lines - 1
        for next_heading in headings[idx + 1:]:
            if next_heading["level"] <= heading["level"]:
                end_line = next_heading["line"] - 1
                break
        else:
            # Last heading of this level — extends to the end
            if idx + 1 < len(headings):
                # But only if the next heading is deeper
                end_line = total_lines - 1

        section_text = "\n".join(lines[start_line:end_line + 1])

        sections.append(_make_section(
            heading_text=heading["text"],
            heading_level=heading["level"],
            start_line=start_line,
            end_line=end_line,
            section_text=section_text,
            code_blocks=code_blocks,
            tables=tables,
        ))

    return sections


def _make_section(heading_text, heading_level, start_line, end_line,
                  section_text, code_blocks, tables):
    """Create a section dict with content analysis.

    Args:
        heading_text: Text of the section heading.
        heading_level: Heading level (0 for intro, 1-6 for H1-H6).
        start_line: First line of the section.
        end_line: Last line of the section.
        section_text: Full text content of the section.
        code_blocks: All code blocks in the document.
        tables: All tables in the document.

    Returns:
        Section dict with metadata and content flags.
    """
    # Find code blocks within this section
    section_code = [
        cb for cb in code_blocks
        if cb["start_line"] >= start_line and cb["end_line"] <= end_line
    ]
    code_types = sorted(set(cb["code_type"] for cb in section_code))

    # Find tables within this section
    section_tables = [
        t for t in tables
        if t["start_line"] >= start_line and t["end_line"] <= end_line
    ]

    return {
        "heading_text": heading_text,
        "heading_level": heading_level,
        "start_line": start_line,
        "end_line": end_line,
        "token_count": estimate_tokens(section_text),
        "has_code": len(section_code) > 0,
        "code_types": code_types,
        "has_table": len(section_tables) > 0,
    }