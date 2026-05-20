"""Tests for the document chunking module."""

from k8s_rag.preprocessing.chunker import (
    make_chunk_id,
    build_heading_tree,
    chunk_document,
    TARGET_TOKENS,
    HARD_CAP_TOKENS,
)
from k8s_rag.preprocessing.structure import analyze_structure, estimate_tokens


# --- make_chunk_id ---

def test_chunk_id_basic():
    result = make_chunk_id("concepts/workloads/pods/_index.md", "What is a Pod?")
    assert result == "concepts/workloads/pods/_index::what-is-a-pod"


def test_chunk_id_with_index():
    result = make_chunk_id("concepts/pods/_index.md", "Overview", index=1)
    assert "--1" in result


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
    chunks = chunk_document(content, structure, _make_metadata(), [])
    total_tokens = sum(c["token_count"] for c in chunks)
    assert total_tokens < TARGET_TOKENS
    assert all(c["token_count"] <= TARGET_TOKENS for c in chunks)


def test_no_chunks_exceed_hard_cap():
    """Generate a large document and verify no chunk exceeds HARD_CAP."""
    # Build a document with a large section (no sub-headings)
    paragraphs = ["Paragraph text. " * 40 + "\n" for _ in range(10)]
    content = "## Big Section\n\n" + "\n".join(paragraphs)
    structure = analyze_structure(content)
    chunks = chunk_document(content, structure, _make_metadata(), [])
    for c in chunks:
        assert c["token_count"] <= HARD_CAP_TOKENS, (
            f"Chunk '{c['chunk_id']}' has {c['token_count']} tokens"
        )


def test_code_blocks_stay_atomic():
    """A code block should not be split across chunks."""
    yaml_block = "```yaml\n" + "key: value\n" * 20 + "```"
    content = f"## Section\n\nSome text.\n\n{yaml_block}\n\nMore text."
    structure = analyze_structure(content)
    chunks = chunk_document(content, structure, _make_metadata(), [])
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
    chunks = chunk_document(content, structure, metadata, ["https://k8s.io/ref"])
    for c in chunks:
        assert c["title"] == "Test Doc"
        assert c["content_type"] == "concept"
        assert c["k8s_version"] == "v1.36"
        assert c["source_url"] == "https://kubernetes.io/docs/test/doc/"
        assert "https://k8s.io/ref" in c["cross_references"]


def test_chunk_ids_unique():
    """All chunks from a document should have unique IDs."""
    content = (
        "## Section A\n\nContent.\n\n"
        "## Section B\n\nContent.\n\n"
        "## Section C\n\nContent."
    )
    structure = analyze_structure(content)
    chunks = chunk_document(content, structure, _make_metadata(), [])
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), f"Duplicate IDs found: {ids}"


def test_heading_hierarchy_includes_doc_title():
    """Every chunk's heading_hierarchy should start with the doc title."""
    content = "## Section\n\n### Subsection\n\nContent."
    structure = analyze_structure(content)
    chunks = chunk_document(content, structure, _make_metadata(), [])
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
    chunks = chunk_document(content, structure, _make_metadata(), [])

    code_chunk = [c for c in chunks if "Code" in c["heading_hierarchy"][-1]][0]
    assert code_chunk["has_code"] is True

    table_chunk = [c for c in chunks if "Table" in c["heading_hierarchy"][-1]][0]
    assert table_chunk["has_table"] is True

    text_chunk = [c for c in chunks if "Text" in c["heading_hierarchy"][-1]][0]
    assert text_chunk["has_code"] is False
    assert text_chunk["has_table"] is False