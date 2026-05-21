"""Resolve relative links and extract cross-references from K8s docs.

Converts relative links (e.g., /docs/concepts/workloads/pods/) to
absolute kubernetes.io URLs and collects all outgoing links as
cross-references for chunk metadata.
"""

import re

BASE_URL = "https://kubernetes.io"

# Inline markdown links, excluding image links.
_INLINE_LINK = re.compile(r'(?<!!)\[([^\]]*)\]\(([^)]*)\)')
# Markdown reference-style definitions: [id]: /docs/...
_REFERENCE_DEF = re.compile(r'^\s*\[([^\]]+)\]:\s*(\S+)\s*$', flags=re.MULTILINE)
# Markdown reference-style usage: [text][id]
_REFERENCE_USE = re.compile(r'(?<!!)\[([^\]]+)\]\[([^\]]*)\]')
# Autolinks like <https://kubernetes.io/docs/...>
_AUTOLINK = re.compile(r'<(https?://[^>\s]+)>')


def _to_absolute_url(url):
    """Normalize supported relative URLs to absolute kubernetes.io URLs."""
    if url.startswith("/"):
        return f"{BASE_URL}{url}"
    return url


def resolve_relative_links(content):
    """Convert relative links to absolute kubernetes.io URLs.

    Handles markdown links like [text](/docs/...) and [text](/blog/...).
    Leaves anchor-only links (#section) and external links (http...) unchanged.

    Args:
        content: Markdown string with relative links.

    Returns:
        Markdown string with relative links converted to absolute URLs.
    """
    def replace_link(match):
        text = match.group(1)
        url = match.group(2)
        return f"[{text}]({_to_absolute_url(url)})"

    resolved = _INLINE_LINK.sub(replace_link, content)

    # Also normalize reference-style link definitions.
    def replace_ref_def(match):
        label = match.group(1)
        url = match.group(2)
        return f"[{label}]: {_to_absolute_url(url)}"

    return _REFERENCE_DEF.sub(replace_ref_def, resolved)


def extract_cross_references(content):
    """Extract all outgoing links from markdown content.

    Collects unique URLs that point to other pages, which can be used
    as cross-reference metadata during retrieval. Excludes same-page
    anchor links since they don't point to other documents.

    Args:
        content: Markdown string (after link resolution).

    Returns:
        Sorted list of unique URLs found in the content.
    """
    urls = set()

    # Inline links
    for _, url in _INLINE_LINK.findall(content):
        # Skip same-page anchors
        if url.startswith("#"):
            continue
        # Strip any trailing anchor from the URL for deduplication
        base_url = url.split("#")[0]
        if base_url:
            urls.add(base_url)

    # Reference-style links: only include definitions that are actually used.
    definitions = {
        label.lower(): url for label, url in _REFERENCE_DEF.findall(content)
    }
    for text, label in _REFERENCE_USE.findall(content):
        key = (label or text).lower()
        url = definitions.get(key)
        if not url:
            continue
        # Skip same-page anchors
        if url.startswith("#"):
            continue
        # Strip any trailing anchor from the URL for deduplication
        base_url = url.split("#")[0]
        if base_url:
            urls.add(base_url)

    # Autolinks
    for url in _AUTOLINK.findall(content):
        base_url = url.split("#")[0]
        if base_url:
            urls.add(base_url)

    return sorted(urls)


def process_links(content):
    """Resolve links and extract cross-references in one pass.

    This is the main entry point for this module. Resolves relative
    links to absolute URLs, then extracts the full list of
    cross-references.

    Args:
        content: Markdown string with potentially relative links.

    Returns:
        Tuple of (resolved_content, cross_references) where
        resolved_content has absolute URLs and cross_references
        is a sorted list of unique outgoing URLs.
    """
    resolved = resolve_relative_links(content)
    cross_refs = extract_cross_references(resolved)
    return resolved, cross_refs