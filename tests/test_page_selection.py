"""Tests for page selection resolution logic."""

from k8s_rag.preprocessing.page_selection import resolve_pages


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
        "k8s_rag.preprocessing.page_selection._fetch_repo_tree",
        fake_fetch_tree,
    )

    resolved = resolve_pages(config=config, mode="discover")
    assert resolved == {"concepts": ["a.md"], "tasks": ["b.md"]}
