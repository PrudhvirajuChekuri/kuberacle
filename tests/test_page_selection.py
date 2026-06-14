"""Tests for page selection resolution logic."""

import pytest

from kuberacle.preprocessing.page_selection import (
    _owner_repo_from_url,
    _fetch_repo_tree,
    resolve_pages,
)


# --- _owner_repo_from_url ---

def test_owner_repo_from_valid_url():
    assert _owner_repo_from_url("https://github.com/kubernetes/website") == "kubernetes/website"


def test_owner_repo_from_url_rejects_non_github():
    with pytest.raises(ValueError, match="Expected a GitHub HTTPS URL"):
        _owner_repo_from_url("https://gitlab.com/org/repo")


def test_owner_repo_from_url_rejects_ssh():
    with pytest.raises(ValueError, match="Expected a GitHub HTTPS URL"):
        _owner_repo_from_url("git@github.com:org/repo.git")


# --- _fetch_repo_tree ---

def test_fetch_repo_tree_raises_on_truncated(monkeypatch):
    """A truncated tree response must raise rather than silently drop pages."""
    import kuberacle.preprocessing.page_selection as mod

    def fake_urlopen(*args, **kwargs):
        import io
        import json as _json
        body = _json.dumps({"tree": [], "truncated": True}).encode()
        resp = io.BytesIO(body)
        resp.__enter__ = lambda s: s
        resp.__exit__ = lambda s, *a: None
        resp.read = lambda: body
        return resp

    monkeypatch.setattr(mod, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="truncated"):
        _fetch_repo_tree("https://github.com/kubernetes/website", "main")


def test_resolve_pages_list_mode_returns_config_pages():
    """List mode should return pages exactly as configured."""
    config = {
        "pages": {
            "concepts": ["a.md"],
            "tasks": ["b.md"],
        }
    }
    resolved = resolve_pages(config=config, mode="list")
    assert resolved == {"concepts": ["a.md"], "tasks": ["b.md"]}


def test_resolve_pages_discover_mode_uses_repo_tree(monkeypatch):
    """Discover mode should map markdown files by selected section."""
    config = {
        "source_repo": "https://github.com/kubernetes/website",
        "source_branch": "main",
        "docs_path": "content/en/docs",
        "selection": {
            "mode": "discover",
            "sections": ["concepts", "tasks"],
        },
    }

    def fake_fetch_tree(repo_url, branch):
        del repo_url, branch
        return [
            {"type": "blob", "path": "content/en/docs/concepts/a.md"},
            {"type": "blob", "path": "content/en/docs/tasks/b.md"},
            {"type": "blob", "path": "content/en/docs/tasks/c.txt"},
            {"type": "tree", "path": "content/en/docs/concepts/nested"},
        ]

    monkeypatch.setattr(
        "kuberacle.preprocessing.page_selection._fetch_repo_tree",
        fake_fetch_tree,
    )

    resolved = resolve_pages(config=config, mode="discover")
    assert resolved == {"concepts": ["a.md"], "tasks": ["b.md"]}


# --- resolve_pages warnings ---

def test_resolve_pages_warns_on_missing_section(caplog):
    """A typo in --sections should produce a visible warning."""
    import logging
    config = {
        "pages": {
            "concepts": ["a.md"],
            "tasks": ["b.md"],
        }
    }
    with caplog.at_level(logging.WARNING):
        resolved = resolve_pages(
            config=config, mode="list", sections_override=["concepts", "typo"],
        )
    assert resolved["typo"] == []
    assert any("typo" in r.message for r in caplog.records)
