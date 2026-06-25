"""Tests for the source-inventory recording in download_data."""

import kuberacle.cli.download_data as dd


CONFIG = {
    "source_repo": "https://github.com/kubernetes/website",
    "source_branch": "main",
    "docs_path": "content/en/docs",
    "examples_path": "content/en/examples",
    "includes_path": "content/en/includes",
    "glossary_path": "content/en/docs/reference/glossary",
}

PAGE = (
    "Intro text.\n"
    '{{< code_sample file="pods/simple-pod.yaml" >}}\n'
    '{{< glossary_definition term_id="pod" length="all" >}}\n'
    '{{< include "task-tutorial-prereqs.md" >}}\n'
)

BLOB_SHAS = {
    "content/en/docs/concepts/a.md": "sha-page",
    "content/en/examples/pods/simple-pod.yaml": "sha-example",
    "content/en/includes/task-tutorial-prereqs.md": "sha-include",
    "content/en/docs/reference/glossary/pod.md": "sha-glossary",
}


def _fake_fetch_file(url, quiet=False):
    del quiet
    if url.endswith("content/en/docs/concepts/a.md"):
        return PAGE
    if "examples/pods/simple-pod.yaml" in url:
        return "example-bytes"
    if "includes/task-tutorial-prereqs.md" in url:
        return "include-bytes"
    if "glossary/pod.md" in url:
        return "glossary-bytes"
    return None


def test_download_pages_records_inventory_and_edges(monkeypatch):
    monkeypatch.setattr(dd, "fetch_file", _fake_fetch_file)
    monkeypatch.setattr(dd, "save_file", lambda content, path: None)

    counts, inventory = dd.download_pages(
        CONFIG, {"concepts": ["a.md"]}, BLOB_SHAS
    )

    assert counts == {
        "pages": 1, "examples": 1, "includes": 1, "glossary": 1,
        "failed": 0, "missing_sha": 0,
    }

    # Every fetched file is stamped with its blob sha and kind.
    assert inventory["files"] == {
        "content/en/docs/concepts/a.md": {"sha": "sha-page", "kind": "page"},
        "content/en/examples/pods/simple-pod.yaml": {"sha": "sha-example", "kind": "example"},
        "content/en/includes/task-tutorial-prereqs.md": {"sha": "sha-include", "kind": "include"},
        "content/en/docs/reference/glossary/pod.md": {"sha": "sha-glossary", "kind": "glossary"},
    }

    # The page's dependency edges point at its example, glossary, and include.
    assert inventory["dependencies"] == {
        "content/en/docs/concepts/a.md": [
            "content/en/docs/reference/glossary/pod.md",
            "content/en/examples/pods/simple-pod.yaml",
            "content/en/includes/task-tutorial-prereqs.md",
        ],
    }


def test_download_pages_warns_on_missing_blob_sha(monkeypatch, caplog):
    import logging

    monkeypatch.setattr(dd, "fetch_file", _fake_fetch_file)
    monkeypatch.setattr(dd, "save_file", lambda content, path: None)

    # Drop the page's sha so its provenance is incomplete.
    shas = dict(BLOB_SHAS)
    del shas["content/en/docs/concepts/a.md"]

    with caplog.at_level(logging.WARNING):
        counts, inventory = dd.download_pages(CONFIG, {"concepts": ["a.md"]}, shas)

    assert inventory["files"]["content/en/docs/concepts/a.md"] == {"sha": "", "kind": "page"}
    assert counts["missing_sha"] == 1
    assert any("No blob SHA" in r.message for r in caplog.records)
