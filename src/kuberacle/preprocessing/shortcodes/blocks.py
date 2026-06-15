"""Content-preserving block shortcode resolvers.

Convert admonition and disclosure blocks (note/caution/warning, alert, details,
pageinfo, example) into plain labeled text while keeping their inner content.
"""

import re


def resolve_notes(content: str) -> str:
    """Convert note, caution, and warning blocks to labeled text.

    Handles both standard ({{< /note >}}) and compact ({{</ note >}})
    closing tag formats.

    Example:
        {{< note >}}
        Some important info.
        {{< /note >}}
        becomes:
        NOTE: Some important info.

    Args:
        content: Markdown string containing note/caution/warning blocks.

    Returns:
        Markdown string with admonition blocks converted to labeled text.
    """
    for admonition in ("note", "caution", "warning"):
        pattern = (
            r'{{[<%]\s*'
            + admonition
            + r'\s*[%>]}}\n?'
            + r'(.*?)'
            + r'{{[<%]\s*/\s*'   # allow optional space between / and name
            + admonition
            + r'\s*[%>]}}'
        )
        label = admonition.upper()

        def make_labeled(match, label=label):
            inner = match.group(1).strip()
            return f"{label}: {inner}"

        content = re.sub(pattern, make_labeled, content, flags=re.DOTALL)

    return content


def resolve_alerts(content: str) -> str:
    """Convert alert blocks to labeled text, preserving the title if present.

    Example:
        {{% alert title="Removed feature" color="warning" %}}
        PodSecurityPolicy was removed in v1.25.
        {{% /alert %}}
        becomes:
        ALERT (Removed feature): PodSecurityPolicy was removed in v1.25.

    Args:
        content: Markdown string containing alert shortcodes.

    Returns:
        Markdown string with alert blocks converted to labeled text.
    """
    pattern = r'{{[<%]\s*alert([^%>]*)[%>]}}\n?(.*?){{[<%]\s*/alert\s*[%>]}}'

    def replace_alert(match):
        attrs = match.group(1)
        inner = match.group(2).strip()
        title_m = re.search(r'title="([^"]*)"', attrs)
        if title_m:
            return f"ALERT ({title_m.group(1)}): {inner}"
        return f"ALERT: {inner}"

    return re.sub(pattern, replace_alert, content, flags=re.DOTALL)


def resolve_details(content: str) -> str:
    """Strip details shortcode wrapper tags, keeping inner content.

    Args:
        content: Markdown string containing details shortcodes.

    Returns:
        Markdown string with details wrapper tags removed.
    """
    return re.sub(
        r'{{<\s*details\b[^>]*>}}\n?(.*?){{<\s*/details\s*>}}',
        lambda m: m.group(1).strip(),
        content,
        flags=re.DOTALL,
    )


def resolve_pageinfo(content: str) -> str:
    """Strip pageinfo shortcode wrapper tags, keeping inner content.

    Args:
        content: Markdown string containing pageinfo shortcodes.

    Returns:
        Markdown string with pageinfo wrapper tags removed, content preserved.
    """
    return re.sub(
        r'{{[<%]\s*pageinfo[^%>]*[%>]}}\n?(.*?){{[<%]\s*/pageinfo\s*[%>]}}',
        lambda m: m.group(1).strip(),
        content,
        flags=re.DOTALL,
    )


def resolve_examples(content: str) -> str:
    """Replace example shortcodes with their display text.

    Example:
        {{< example file="policy/privileged-psp.yaml" >}}privileged PSP{{< /example >}}
        becomes:
        privileged PSP

    Args:
        content: Markdown string containing example shortcodes.

    Returns:
        Markdown string with example shortcodes replaced by their inner text.
    """
    return re.sub(
        r'{{<\s*example\s+[^>]*>}}(.*?){{<\s*/example\s*>}}',
        r'\1',
        content,
        flags=re.DOTALL,
    )
