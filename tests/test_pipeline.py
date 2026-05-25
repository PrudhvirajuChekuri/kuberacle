"""Tests for the preprocessing pipeline orchestrator."""

import json

import pytest

from k8s_rag.preprocessing.pipeline import process_page, run_pipeline, write_jsonl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_page(raw_dir, file_path, content):
    """Write a raw doc file under raw_dir, creating directories as needed."""
    dest = raw_dir / file_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content)


def _minimal_page(title="Test Doc", section="concepts"):
    """Return a minimal but valid K8s-style markdown page."""
    return (
        "---\n"
        f"title: {title}\n"
        "weight: 10\n"
        "---\n\n"
        "<!-- overview -->\n\n"
        "An overview paragraph.\n\n"
        "## First Section\n\n"
        "First section content.\n\n"
        "## Second Section\n\n"
        "Second section content.\n"
    )


# ---------------------------------------------------------------------------
# process_page
# ---------------------------------------------------------------------------

def test_process_page_returns_chunks(tmp_path):
    """process_page should return at least one chunk for a valid doc."""
    raw_dir = tmp_path / "raw"
    examples_dir = tmp_path / "examples"
    includes_dir = tmp_path / "includes"
    examples_dir.mkdir()
    includes_dir.mkdir()

    file_path = "concepts/overview/test.md"
    _write_page(raw_dir, file_path, _minimal_page())

    chunks = process_page(
        file_path, raw_dir, examples_dir, includes_dir, "v1.36",
    )

    assert len(chunks) >= 1


def test_process_page_chunk_schema(tmp_path):
    """Every chunk must carry the required metadata fields."""
    raw_dir = tmp_path / "raw"
    examples_dir = tmp_path / "examples"
    includes_dir = tmp_path / "includes"
    examples_dir.mkdir()
    includes_dir.mkdir()

    file_path = "concepts/overview/test.md"
    _write_page(raw_dir, file_path, _minimal_page("Schema Doc"))

    chunks = process_page(
        file_path, raw_dir, examples_dir, includes_dir, "v1.36",
    )

    required = {
        "chunk_id", "heading_hierarchy", "content", "token_count",
        "has_code", "code_types", "has_table",
        "title", "content_type", "file_path", "source_url", "k8s_version",
        "cross_references",
    }
    for chunk in chunks:
        missing = required - chunk.keys()
        assert not missing, f"Chunk {chunk['chunk_id']!r} missing fields: {missing}"


def test_process_page_content_type_derived_from_path(tmp_path):
    """content_type must reflect the top-level section in the file path."""
    raw_dir = tmp_path / "raw"
    examples_dir = tmp_path / "examples"
    includes_dir = tmp_path / "includes"
    examples_dir.mkdir()
    includes_dir.mkdir()

    for section, expected_type in [
        ("concepts", "concept"),
        ("tasks", "task"),
        ("tutorials", "tutorial"),
    ]:
        file_path = f"{section}/example/page.md"
        _write_page(raw_dir, file_path, _minimal_page())
        chunks = process_page(
            file_path, raw_dir, examples_dir, includes_dir, "v1.36",
        )
        for chunk in chunks:
            assert chunk["content_type"] == expected_type, (
                f"Expected content_type={expected_type!r}, got {chunk['content_type']!r}"
            )


def test_process_page_k8s_version_threaded(tmp_path):
    """k8s_version from the config must appear in chunk metadata."""
    raw_dir = tmp_path / "raw"
    examples_dir = tmp_path / "examples"
    includes_dir = tmp_path / "includes"
    examples_dir.mkdir()
    includes_dir.mkdir()

    file_path = "concepts/overview/test.md"
    _write_page(raw_dir, file_path, _minimal_page())

    chunks = process_page(
        file_path, raw_dir, examples_dir, includes_dir, "v1.99",
    )

    for chunk in chunks:
        assert chunk["k8s_version"] == "v1.99"


def test_process_page_resolves_code_sample(tmp_path):
    """A code_sample shortcode must be inlined with a Source comment."""
    raw_dir = tmp_path / "raw"
    examples_dir = tmp_path / "examples"
    includes_dir = tmp_path / "includes"
    includes_dir.mkdir()
    (examples_dir / "pods").mkdir(parents=True)
    (examples_dir / "pods" / "simple-pod.yaml").write_text(
        "apiVersion: v1\nkind: Pod\n"
    )

    page = (
        "---\ntitle: Pod Demo\n---\n\n"
        "## Using a Pod\n\n"
        '{{% code_sample file="pods/simple-pod.yaml" %}}\n'
    )
    file_path = "concepts/demo/page.md"
    _write_page(raw_dir, file_path, page)

    chunks = process_page(
        file_path, raw_dir, examples_dir, includes_dir, "v1.36",
    )

    all_content = " ".join(c["content"] for c in chunks)
    assert "# Source: pods/simple-pod.yaml" in all_content
    assert "apiVersion: v1" in all_content


def test_process_page_resolves_include(tmp_path):
    """An include shortcode must be inlined with the referenced file content."""
    raw_dir = tmp_path / "raw"
    examples_dir = tmp_path / "examples"
    includes_dir = tmp_path / "includes"
    examples_dir.mkdir()
    includes_dir.mkdir()
    (includes_dir / "prereqs.md").write_text("You need a cluster.")

    page = (
        "---\ntitle: Task Page\n---\n\n"
        "## Before you begin\n\n"
        '{{< include "prereqs.md" >}}\n'
    )
    file_path = "tasks/demo/page.md"
    _write_page(raw_dir, file_path, page)

    chunks = process_page(
        file_path, raw_dir, examples_dir, includes_dir, "v1.36",
    )

    all_content = " ".join(c["content"] for c in chunks)
    assert "You need a cluster." in all_content


def test_process_page_missing_file_raises(tmp_path):
    """process_page must raise when the raw file does not exist."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    examples_dir = tmp_path / "examples"
    includes_dir = tmp_path / "includes"
    examples_dir.mkdir()
    includes_dir.mkdir()

    with pytest.raises(Exception):
        process_page(
            "concepts/nonexistent/page.md",
            raw_dir, examples_dir, includes_dir, "v1.36",
        )


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------

def _make_config(tmp_path, pages_by_section):
    """Write raw files and return a config dict for run_pipeline."""
    raw_dir = tmp_path / "raw"
    examples_dir = tmp_path / "examples"
    includes_dir = tmp_path / "includes"
    examples_dir.mkdir(exist_ok=True)
    includes_dir.mkdir(exist_ok=True)

    pages_config = {}
    for section, page_names in pages_by_section.items():
        pages_config[section] = []
        for name in page_names:
            file_path = f"{section}/{name}"
            _write_page(raw_dir, file_path, _minimal_page(title=name.replace(".md", "")))
            pages_config[section].append(name)

    return {
        "k8s_version": "v1.36",
        "pages": pages_config,
    }


def test_run_pipeline_aggregates_all_pages(tmp_path):
    """run_pipeline must return chunks from every configured page."""
    config = _make_config(tmp_path, {
        "concepts": ["pods.md", "workloads.md"],
        "tasks": ["configure.md"],
    })

    chunks, stats = run_pipeline(config, tmp_path)

    assert stats["total_pages"] == 3
    assert stats["failed_pages"] == 0
    assert stats["total_chunks"] == len(chunks)
    assert len(chunks) > 0


def test_run_pipeline_stats_token_range(tmp_path):
    """Stats must report a valid token range (min <= avg <= max)."""
    config = _make_config(tmp_path, {"concepts": ["page.md"]})

    _, stats = run_pipeline(config, tmp_path)

    assert stats["min_tokens"] <= stats["avg_tokens"] <= stats["max_tokens"]


def test_run_pipeline_records_failed_page(tmp_path):
    """A missing file must be counted as a failed page, not raise."""
    config = {
        "k8s_version": "v1.36",
        "pages": {"concepts": ["nonexistent.md"]},
    }
    # raw dir exists but the file does not
    (tmp_path / "raw").mkdir()
    (tmp_path / "examples").mkdir()
    (tmp_path / "includes").mkdir()

    chunks, stats = run_pipeline(config, tmp_path)

    assert stats["failed_pages"] == 1
    assert len(chunks) == 0


def test_run_pipeline_empty_config(tmp_path):
    """An empty pages config must return zero chunks and zero pages."""
    config = {"k8s_version": "v1.36", "pages": {}}
    (tmp_path / "raw").mkdir()
    (tmp_path / "examples").mkdir()
    (tmp_path / "includes").mkdir()

    chunks, stats = run_pipeline(config, tmp_path)

    assert stats["total_pages"] == 0
    assert stats["total_chunks"] == 0
    assert len(chunks) == 0


def test_run_pipeline_allows_empty_chunk_page(tmp_path, monkeypatch):
    """Pages that produce zero chunks should not be marked as failed."""
    config = _make_config(tmp_path, {"concepts": ["empty.md"]})

    def fake_process_page(*args, **kwargs):
        del args, kwargs
        return []

    monkeypatch.setattr(
        "k8s_rag.preprocessing.pipeline.process_page",
        fake_process_page,
    )

    chunks, stats = run_pipeline(config, tmp_path)

    assert len(chunks) == 0
    assert stats["total_pages"] == 1
    assert stats["failed_pages"] == 0
    assert stats["pages"][0]["chunks"] == 0


# ---------------------------------------------------------------------------
# write_jsonl
# ---------------------------------------------------------------------------

def test_write_jsonl_creates_file(tmp_path):
    """write_jsonl must create the output file and its parent directories."""
    output_path = tmp_path / "nested" / "dir" / "out.jsonl"
    chunks = [{"chunk_id": "a::b", "content": "hello", "token_count": 5}]

    write_jsonl(chunks, output_path)

    assert output_path.exists()


def test_write_jsonl_one_line_per_chunk(tmp_path):
    """Each chunk must occupy exactly one line in the JSONL file."""
    output_path = tmp_path / "out.jsonl"
    chunks = [
        {"chunk_id": "a::1", "content": "first"},
        {"chunk_id": "a::2", "content": "second"},
        {"chunk_id": "a::3", "content": "third"},
    ]

    write_jsonl(chunks, output_path)

    lines = output_path.read_text().splitlines()
    assert len(lines) == 3


def test_write_jsonl_valid_json_per_line(tmp_path):
    """Every line in the JSONL output must be valid JSON."""
    output_path = tmp_path / "out.jsonl"
    chunks = [
        {"chunk_id": "a::1", "content": "hello", "cross_references": []},
        {"chunk_id": "a::2", "content": "world", "cross_references": ["https://k8s.io"]},
    ]

    write_jsonl(chunks, output_path)

    for line in output_path.read_text().splitlines():
        parsed = json.loads(line)
        assert "chunk_id" in parsed


def test_write_jsonl_preserves_unicode(tmp_path):
    """Unicode characters must be written as-is, not escaped."""
    output_path = tmp_path / "out.jsonl"
    chunks = [{"chunk_id": "x::y", "content": "Kubernetes — \u201cpods\u201d"}]

    write_jsonl(chunks, output_path)

    raw = output_path.read_text()
    assert "\u2014" in raw
    assert "\u201c" in raw


def test_write_jsonl_roundtrip(tmp_path):
    """Data written by write_jsonl must be recoverable verbatim."""
    output_path = tmp_path / "out.jsonl"
    chunks = [
        {
            "chunk_id": "concepts/pods/_index::intro",
            "heading_hierarchy": ["Pods"],
            "content": "Some content.",
            "token_count": 12,
            "has_code": False,
            "cross_references": ["https://kubernetes.io/docs/concepts/"],
        }
    ]

    write_jsonl(chunks, output_path)

    recovered = [
        json.loads(line) for line in output_path.read_text().splitlines()
    ]
    assert recovered == chunks
