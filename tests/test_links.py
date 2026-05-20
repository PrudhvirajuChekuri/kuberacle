"""Tests for the link resolution module."""

from k8s_rag.preprocessing.links import (
    resolve_relative_links,
    extract_cross_references,
    process_links,
)


# --- resolve_relative_links ---

def test_relative_docs_link_becomes_absolute():
    content = "[Pods](/docs/concepts/workloads/pods/)"
    result = resolve_relative_links(content)
    assert result == "[Pods](https://kubernetes.io/docs/concepts/workloads/pods/)"


def test_relative_blog_link_becomes_absolute():
    content = "[post](/blog/2015/06/some-post/)"
    result = resolve_relative_links(content)
    assert result == "[post](https://kubernetes.io/blog/2015/06/some-post/)"


def test_anchor_link_unchanged():
    content = "[section](#my-section)"
    result = resolve_relative_links(content)
    assert result == "[section](#my-section)"


def test_external_link_unchanged():
    content = "[GitHub](https://github.com/kubernetes/kubernetes)"
    result = resolve_relative_links(content)
    assert result == "[GitHub](https://github.com/kubernetes/kubernetes)"


def test_relative_link_with_anchor():
    content = "[names](/docs/concepts/overview/names#dns-label-names)"
    result = resolve_relative_links(content)
    assert result == "[names](https://kubernetes.io/docs/concepts/overview/names#dns-label-names)"


def test_multiple_links_in_one_line():
    content = "See [Pods](/docs/pods/) and [Jobs](/docs/jobs/) for details."
    result = resolve_relative_links(content)
    assert "https://kubernetes.io/docs/pods/" in result
    assert "https://kubernetes.io/docs/jobs/" in result


# --- extract_cross_references ---

def test_extracts_absolute_urls():
    content = "[Pods](https://kubernetes.io/docs/pods/) and [Jobs](https://kubernetes.io/docs/jobs/)"
    refs = extract_cross_references(content)
    assert "https://kubernetes.io/docs/pods/" in refs
    assert "https://kubernetes.io/docs/jobs/" in refs


def test_excludes_anchor_only_links():
    content = "[section](#my-section)"
    refs = extract_cross_references(content)
    assert refs == []


def test_deduplicates_urls_with_different_anchors():
    content = (
        "[overview](https://kubernetes.io/docs/page/#section-a) "
        "and [details](https://kubernetes.io/docs/page/#section-b)"
    )
    refs = extract_cross_references(content)
    assert refs == ["https://kubernetes.io/docs/page/"]


def test_returns_sorted_list():
    content = "[B](https://b.com) then [A](https://a.com)"
    refs = extract_cross_references(content)
    assert refs == ["https://a.com", "https://b.com"]


# --- process_links ---

def test_process_links_resolves_and_extracts():
    content = "See [Pods](/docs/pods/) and [section](#anchor)."
    resolved, refs = process_links(content)

    assert "https://kubernetes.io/docs/pods/" in resolved
    assert "(#anchor)" in resolved
    assert "https://kubernetes.io/docs/pods/" in refs
    assert len(refs) == 1