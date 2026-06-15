"""Tests for the document chunking module."""

import re

from kuberacle.preprocessing.chunker import (
    make_chunk_id,
    build_heading_tree,
    chunk_document,
    TARGET_TOKENS,
    HARD_CAP_TOKENS,
)
from kuberacle.preprocessing.structure import analyze_structure, estimate_tokens


# --- make_chunk_id ---

def test_chunk_id_basic():
    result = make_chunk_id("concepts/workloads/pods/_index.md", "What is a Pod?")
    assert result == "concepts/workloads/pods/_index::what-is-a-pod"


def test_chunk_id_with_index():
    result = make_chunk_id("concepts/pods/_index.md", "Overview", index=1)
    assert "--1" in result


def test_chunk_id_with_hierarchy_disambiguates_repeated_headings():
    """Hierarchy context should avoid collisions for repeated leaf headings."""
    a = make_chunk_id(
        "concepts/pods/_index.md",
        "Overview",
        heading_hierarchy=["Pods", "Parent A", "Overview"],
    )
    b = make_chunk_id(
        "concepts/pods/_index.md",
        "Overview",
        heading_hierarchy=["Pods", "Parent B", "Overview"],
    )
    assert a != b


def test_chunk_id_strips_special_chars():
    result = make_chunk_id("tasks/debug.md", "`Waiting` {#state}")
    assert "`" not in result
    assert "{" not in result


# --- build_heading_tree ---

def test_heading_tree_nesting():
    headings = [
        {"level": 2, "text": "A", "line": 0, "anchor": None},
        {"level": 3, "text": "A1", "line": 5, "anchor": None},
        {"level": 3, "text": "A2", "line": 10, "anchor": None},
        {"level": 2, "text": "B", "line": 15, "anchor": None},
    ]
    tree = build_heading_tree(headings, 20)
    assert len(tree) == 2
    assert tree[0]["heading"]["text"] == "A"
    assert len(tree[0]["children"]) == 2
    assert tree[1]["heading"]["text"] == "B"
    assert len(tree[1]["children"]) == 0


def test_heading_tree_empty():
    assert build_heading_tree([], 10) == []


# --- chunk_document ---

def _make_metadata(file_path="test/doc.md"):
    return {
        "file_path": file_path,
        "title": "Test Doc",
        "content_type": "concept",
        "section_path": ["test"],
        "source_url": "https://kubernetes.io/docs/test/doc/",
        "k8s_version": "v1.36",
    }


def test_small_document_single_chunk():
    """A short document should become one or two chunks (intro + section)."""
    content = "Intro text.\n\n## Section\n\nSmall content."
    structure = analyze_structure(content)
    chunks = chunk_document(content, structure, _make_metadata())
    total_tokens = sum(c["token_count"] for c in chunks)
    assert total_tokens < TARGET_TOKENS
    assert all(c["token_count"] <= TARGET_TOKENS for c in chunks)


def test_no_chunks_exceed_hard_cap():
    """Generate a large document and verify no chunk exceeds HARD_CAP."""
    # Build a document with a large section (no sub-headings)
    paragraphs = ["Paragraph text. " * 40 + "\n" for _ in range(10)]
    content = "## Big Section\n\n" + "\n".join(paragraphs)
    structure = analyze_structure(content)
    chunks = chunk_document(content, structure, _make_metadata())
    for c in chunks:
        assert c["token_count"] <= HARD_CAP_TOKENS, (
            f"Chunk '{c['chunk_id']}' has {c['token_count']} tokens"
        )


def test_code_blocks_stay_atomic():
    """A code block should not be split across chunks."""
    yaml_block = "```yaml\n" + "key: value\n" * 20 + "```"
    content = f"## Section\n\nSome text.\n\n{yaml_block}\n\nMore text."
    structure = analyze_structure(content)
    chunks = chunk_document(content, structure, _make_metadata())
    # The yaml content should appear entirely in one chunk
    for c in chunks:
        if "key: value" in c["content"]:
            assert c["content"].count("key: value") == 20
            break
    else:
        raise AssertionError("YAML block not found in any chunk")


def test_metadata_propagated_to_chunks():
    """Document metadata should appear in every chunk."""
    content = "## Section A\n\nContent A.\n\n## Section B\n\nContent B."
    structure = analyze_structure(content)
    metadata = _make_metadata()
    chunks = chunk_document(content, structure, metadata)
    for c in chunks:
        assert c["title"] == "Test Doc"
        assert c["content_type"] == "concept"
        assert c["k8s_version"] == "v1.36"
        # source_url has the heading anchor appended
        assert c["source_url"].startswith("https://kubernetes.io/docs/test/doc/")
        assert "cross_references" in c


def test_chunk_ids_unique():
    """All chunks from a document should have unique IDs."""
    content = (
        "## Section A\n\nContent.\n\n"
        "## Section B\n\nContent.\n\n"
        "## Section C\n\nContent."
    )
    structure = analyze_structure(content)
    chunks = chunk_document(content, structure, _make_metadata())
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), f"Duplicate IDs found: {ids}"


def test_chunk_ids_unique_for_repeated_child_headings_when_parents_split():
    """Repeated child headings remain unique even after parent splitting."""
    filler = " ".join(["word"] * 900)
    content = (
        "## Parent A\n\n"
        f"{filler}\n\n"
        "### Overview\n\n"
        "Text A.\n\n"
        "## Parent B\n\n"
        f"{filler}\n\n"
        "### Overview\n\n"
        "Text B.\n"
    )
    structure = analyze_structure(content)
    chunks = chunk_document(content, structure, _make_metadata())
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), f"Duplicate IDs found: {ids}"


def test_chunk_ids_unique_for_repeated_identical_heading_hierarchy():
    """Repeated identical heading paths must still produce unique chunk IDs."""
    content = (
        "## A\n\n"
        "First section text.\n\n"
        "## A\n\n"
        "Second section text.\n"
    )
    structure = analyze_structure(content)
    chunks = chunk_document(content, structure, _make_metadata())
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), f"Duplicate IDs found: {ids}"


def test_heading_hierarchy_includes_doc_title():
    """Every chunk's heading_hierarchy should start with the doc title."""
    content = "## Section\n\n### Subsection\n\nContent."
    structure = analyze_structure(content)
    chunks = chunk_document(content, structure, _make_metadata())
    for c in chunks:
        assert c["heading_hierarchy"][0] == "Test Doc"


def test_content_flags_accurate():
    """has_code and has_table should reflect actual chunk content."""
    content = (
        "## Code Section\n\n```yaml\napiVersion: v1\n```\n\n"
        "## Table Section\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\n"
        "## Text Section\n\nJust words."
    )
    structure = analyze_structure(content)
    chunks = chunk_document(content, structure, _make_metadata())

    code_chunk = [c for c in chunks if "Code" in c["heading_hierarchy"][-1]][0]
    assert code_chunk["has_code"] is True

    table_chunk = [c for c in chunks if "Table" in c["heading_hierarchy"][-1]][0]
    assert table_chunk["has_table"] is True

    text_chunk = [c for c in chunks if "Text" in c["heading_hierarchy"][-1]][0]
    assert text_chunk["has_code"] is False
    assert text_chunk["has_table"] is False


# --- chunk-quality improvements ---

def test_heading_anchor_id_stripped_from_content():
    """Heading anchors {#anchor} must not appear in chunk content."""
    content = (
        "## Pod readiness {#pod-ready}\n\n"
        "Some content about readiness."
    )
    structure = analyze_structure(content)
    chunks = chunk_document(content, structure, _make_metadata())
    for c in chunks:
        assert "{#" not in c["content"]
        assert "}" not in c["content"] or "{" in c["content"]


def test_cross_references_field_present():
    """Every chunk must have a cross_references field (populated by process_page)."""
    content = "## Pods\n\nSome content.\n\n## Services\n\nMore content."
    structure = analyze_structure(content)
    chunks = chunk_document(content, structure, _make_metadata())
    for chunk in chunks:
        assert "cross_references" in chunk
        assert isinstance(chunk["cross_references"], list)


def test_breadcrumb_prepended_to_content():
    """Every chunk's content must start with a heading breadcrumb line."""
    content = "## Section A\n\n### Subsection\n\nLeaf content."
    structure = analyze_structure(content)
    chunks = chunk_document(content, structure, _make_metadata())
    for c in chunks:
        expected = "[" + " > ".join(c["heading_hierarchy"]) + "]"
        assert c["content"].startswith(expected), (
            f"Chunk '{c['chunk_id']}' content did not start with {expected!r}"
        )


def test_intro_chunk_breadcrumb_is_doc_title_only():
    """Intro chunks get a single-item breadcrumb with just the doc title."""
    content = "Some intro paragraph.\n\n## Section\n\nMore text."
    structure = analyze_structure(content)
    chunks = chunk_document(content, structure, _make_metadata())
    intro = [c for c in chunks if c["heading_hierarchy"] == ["Test Doc"]][0]
    assert intro["content"].startswith("[Test Doc]\n\n")


def test_paragraph_split_overlap_carries_sentences():
    """Forced paragraph splits prepend the prior chunk's tail sentences."""
    # Build a section whose own content exceeds TARGET_TOKENS to force
    # paragraph-level splitting. Each paragraph ends with a unique
    # sentinel sentence so we can detect carry-over precisely.
    filler = "Filler content to reach a paragraph worth of tokens here. " * 8
    paragraphs = [
        f"Paragraph {i} body. " + filler + f"Sentinel sentence {i}."
        for i in range(20)
    ]
    content = "## Big Section\n\n" + "\n\n".join(paragraphs)
    structure = analyze_structure(content)
    chunks = chunk_document(content, structure, _make_metadata())

    section_chunks = [
        c for c in chunks
        if c["heading_hierarchy"][-1] == "Big Section"
    ]
    assert len(section_chunks) >= 2, "expected the section to be split"

    # Some continuation chunk must contain a Sentinel sentence whose
    # paragraph number is strictly smaller than its own first paragraph.
    found_overlap = False
    for prev, curr in zip(section_chunks, section_chunks[1:]):
        prev_sentinels = [
            int(m) for m in re.findall(r"Sentinel sentence (\d+)\.", prev["content"])
        ]
        curr_sentinels = [
            int(m) for m in re.findall(r"Sentinel sentence (\d+)\.", curr["content"])
        ]
        if not prev_sentinels or not curr_sentinels:
            continue
        # A successful overlap puts at least one sentinel from prev at
        # the start of curr, before curr's own first paragraph.
        carryover = set(curr_sentinels) & set(prev_sentinels)
        if carryover and min(curr_sentinels) in prev_sentinels:
            found_overlap = True
            break
    assert found_overlap, "no sentence overlap detected between split chunks"


def test_split_chunk_flags_are_chunk_local():
    """Split chunks should not inherit has_code from sibling chunk parts."""
    prose = "\n\n".join(" ".join(["text"] * 250) for _ in range(6))
    content = (
        "## Big\n\n"
        f"{prose}\n\n"
        "```yaml\n"
        "apiVersion: v1\n"
        "kind: Pod\n"
        "```\n"
    )
    structure = analyze_structure(content)
    chunks = chunk_document(content, structure, _make_metadata())

    code_chunks = [c for c in chunks if "apiVersion: v1" in c["content"]]
    prose_chunks = [c for c in chunks if "apiVersion: v1" not in c["content"]]

    assert code_chunks, "expected at least one chunk containing code"
    assert all(c["has_code"] is True for c in code_chunks)
    assert all(c["has_code"] is False for c in prose_chunks)


def test_oversized_intro_is_split():
    """Intro content exceeding target_tokens should be split."""
    paragraphs = ["Intro paragraph. " * 40 for _ in range(20)]
    intro = "\n\n".join(paragraphs)
    content = f"{intro}\n\n## Section\n\nBody text."
    structure = analyze_structure(content)
    chunks = chunk_document(content, structure, _make_metadata())
    intro_chunks = [c for c in chunks if c["heading_hierarchy"] == ["Test Doc"]]
    assert len(intro_chunks) >= 2, "expected oversized intro to be split"


def test_custom_token_limits():
    """Custom target_tokens and hard_cap_tokens should be respected."""
    content = "## Section\n\n" + "Some text. " * 100
    structure = analyze_structure(content)
    chunks_default = chunk_document(content, structure, _make_metadata())
    chunks_small = chunk_document(
        content, structure, _make_metadata(),
        target_tokens=50, hard_cap_tokens=200,
    )
    assert len(chunks_small) >= len(chunks_default)