"""Strip-only shortcode resolvers for blocks with no retrieval value."""

import re


def resolve_thirdparty_content(content: str) -> str:
    """Strip thirdparty-content shortcodes entirely.

    Handles both delimiter styles ({{% %}} and {{< >}}) and optional
    attributes (single="true", vendor="true").

    Args:
        content: Markdown string containing thirdparty-content shortcodes.

    Returns:
        Markdown string with thirdparty-content shortcodes removed.
    """
    return re.sub(r'{{[<%]\s*thirdparty-content[^%>]*[%>]}}', '', content)


def resolve_legacy_repos_deprecation(content: str) -> str:
    """Strip legacy-repos-deprecation shortcodes entirely.

    These render a deprecation notice about old package repositories
    with no retrieval-relevant content.

    Args:
        content: Markdown string containing legacy-repos-deprecation shortcodes.

    Returns:
        Markdown string with legacy-repos-deprecation shortcodes removed.
    """
    return re.sub(r'{{[<%]\s*legacy-repos-deprecation\s*[%>]}}', '', content)
