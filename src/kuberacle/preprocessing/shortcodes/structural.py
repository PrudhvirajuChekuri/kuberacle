"""Structural/block-wrapper shortcode resolvers.

Flatten or strip layout wrappers (tabs, tables, mermaid, comments, tutorial
carousels) that carry no retrieval-relevant text of their own.
"""

import re


def resolve_tabs(content: str) -> str:
    """Flatten tab blocks into sequential labeled sections.

    Handles both percent ({{% %}}) and angle-bracket ({{< >}}) delimiter
    styles for both outer tabs wrappers and inner tab elements.

    Args:
        content: Markdown string containing tab shortcodes.

    Returns:
        Markdown string with tab structures flattened.
    """
    # Strip outer tabs wrappers (both delimiter styles, with or without attrs)
    content = re.sub(r'{{[<%]\s*tabs\b[^%>]*[%>]}}\n?', '', content)
    content = re.sub(r'{{[<%]\s*/tabs\s*[%>]}}\n?', '', content)
    # Percent-style tabs: {{% tab name="..." %}}
    content = re.sub(
        r'{{% tab\s+name="([^"]*)"\s*%}}\n?',
        r'Tab: \1\n',
        content,
    )
    content = re.sub(r'{{% /tab\s*%}}\n?', '', content)
    # Angle-bracket-style tabs: {{< tab name="..." codelang="..." >}}
    content = re.sub(
        r'{{<\s*tab\s+name="([^"]*)"[^>]*>}}\n?',
        r'Tab: \1\n',
        content,
    )
    content = re.sub(r'{{<\s*/tab\s*>}}\n?', '', content)
    return content


def resolve_table(content: str) -> str:
    """Strip {{< table >}} wrapper tags, keeping the markdown table content.

    Args:
        content: Markdown string containing table shortcodes.

    Returns:
        Markdown string with table wrapper tags removed, content preserved.
    """
    content = re.sub(r'{{<\s*table\b[^>]*>}}\n?', '', content)
    content = re.sub(r'{{<\s*/table\s*>}}\n?', '', content)
    return content


def resolve_mermaid(content: str) -> str:
    """Strip mermaid diagram blocks entirely.

    Mermaid blocks contain diagram code that is not useful as retrieval text.
    Handles both spaced ({{< mermaid >}}) and compact ({{<mermaid>}}) forms,
    and closing tags with space after </ ({{</ mermaid >}}).

    Args:
        content: Markdown string containing mermaid shortcodes.

    Returns:
        Markdown string with mermaid blocks removed.
    """
    return re.sub(
        r'{{<\s*mermaid\s*>}}.*?{{<\s*/\s*mermaid\s*>}}',
        '',
        content,
        flags=re.DOTALL,
    )


def resolve_comments(content: str) -> str:
    """Strip Hugo comment blocks entirely.

    Comment blocks contain author notes not intended for readers.

    Args:
        content: Markdown string containing comment shortcodes.

    Returns:
        Markdown string with comment blocks removed.
    """
    return re.sub(
        r'{{<\s*comment\s*>}}.*?{{<\s*/comment\s*>}}',
        '',
        content,
        flags=re.DOTALL,
    )


def resolve_tutorial_shortcodes(content: str) -> str:
    """Strip tutorial carousel and module wrapper blocks.

    These shortcodes render as interactive UI components (carousels,
    navigation modules) containing image references with no retrieval value.

    Args:
        content: Markdown string containing tutorial shortcodes.

    Returns:
        Markdown string with tutorial shortcode blocks removed.
    """
    for block in ('tutorials/modules', 'tutorials/carousel'):
        escaped = re.escape(block)
        opening = r'{{<\s*' + escaped + r'[^>]*>}}'
        closing = r'{{<\s*/' + escaped + r'\s*>}}'
        content = re.sub(opening + r'.*?' + closing, '', content, flags=re.DOTALL)
    return content
