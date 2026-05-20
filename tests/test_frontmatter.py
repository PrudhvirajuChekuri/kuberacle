"""Tests for the frontmatter parsing module."""

import pytest
import yaml
from k8s_rag.preprocessing.frontmatter import (
    parse_frontmatter,
    derive_metadata,
    extract_metadata,
)


# --- parse_frontmatter ---

def test_parse_frontmatter_standard():
    """Standard K8s doc frontmatter with multiple field types."""
    content = (
        "---\n"
        "title: Pods\n"
        "weight: 10\n"
        "reviewers:\n"
        "- erictune\n"
        "---\n"
        "Body content here."
    )
    metadata, body = parse_frontmatter(content)
    assert metadata["title"] == "Pods"
    assert metadata["weight"] == 10
    assert metadata["reviewers"] == ["erictune"]
    assert body == "Body content here."


def test_parse_frontmatter_no_frontmatter():
    """File with no frontmatter should return empty dict and full content."""
    content = "Just a markdown file with no frontmatter."
    metadata, body = parse_frontmatter(content)
    assert metadata == {}
    assert body == content


def test_parse_frontmatter_empty_frontmatter():
    """Empty frontmatter block (just two --- delimiters) returns empty dict."""
    content = "---\n---\nBody after empty frontmatter."
    metadata, body = parse_frontmatter(content)
    assert metadata == {}
    assert body == "Body after empty frontmatter."


def test_parse_frontmatter_invalid_yaml():
    """Malformed YAML in frontmatter should raise an error."""
    content = "---\ntitle: [unclosed bracket\n---\nBody."
    with pytest.raises(yaml.YAMLError):
        parse_frontmatter(content)


def test_parse_frontmatter_triple_dashes_in_body():
    """Triple dashes in the body should not be confused with frontmatter."""
    content = (
        "---\n"
        "title: Test\n"
        "---\n"
        "Some text.\n"
        "---\n"
        "This is a horizontal rule, not frontmatter."
    )
    metadata, body = parse_frontmatter(content)
    assert metadata["title"] == "Test"
    assert "---" in body
    assert "horizontal rule" in body


# --- derive_metadata ---

def test_derive_metadata_index_file():
    """_index.md files should produce a URL ending at the parent directory."""
    result = derive_metadata("concepts/workloads/pods/_index.md")
    assert result["source_url"] == "https://kubernetes.io/docs/concepts/workloads/pods/"
    assert result["content_type"] == "concept"
    assert result["section_path"] == ["concepts", "workloads", "pods"]
    assert result["file_path"] == "concepts/workloads/pods/_index.md"


def test_derive_metadata_regular_file():
    """Regular .md files should include the filename (without .md) in the URL."""
    result = derive_metadata("concepts/workloads/pods/pod-lifecycle.md")
    assert result["source_url"] == "https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/"


def test_derive_metadata_task():
    result = derive_metadata("tasks/debug/debug-application/debug-pods.md")
    assert result["content_type"] == "task"
    assert result["section_path"] == ["tasks", "debug", "debug-application"]


def test_derive_metadata_tutorial():
    result = derive_metadata("tutorials/stateless-application/expose-external-ip-address.md")
    assert result["content_type"] == "tutorial"


def test_derive_metadata_custom_version():
    result = derive_metadata("concepts/workloads/pods/_index.md", k8s_version="v1.35")
    assert result["k8s_version"] == "v1.35"


# --- extract_metadata ---

def test_extract_metadata_merges_frontmatter_and_derived():
    """Frontmatter fields and derived fields should coexist in the result."""
    content = "---\ntitle: Pods\nweight: 10\n---\nBody here."
    metadata, body = extract_metadata(content, "concepts/workloads/pods/_index.md")

    # From frontmatter
    assert metadata["title"] == "Pods"
    assert metadata["weight"] == 10
    # From derived
    assert metadata["source_url"] == "https://kubernetes.io/docs/concepts/workloads/pods/"
    assert metadata["content_type"] == "concept"
    # Body is returned cleanly
    assert body == "Body here."


def test_extract_metadata_content_type_overlap():
    """When frontmatter has content_type, derived value should match."""
    content = "---\ntitle: Pods\ncontent_type: concept\n---\nBody."
    metadata, _ = extract_metadata(content, "concepts/workloads/pods/_index.md")
    assert metadata["content_type"] == "concept"