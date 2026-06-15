"""Post-resolution cleanup: HTML comments, leftover detection, whitespace."""

import re
from collections import Counter


def strip_html_comments(content: str) -> str:
    """Remove HTML comments like <!-- overview --> and <!-- body -->.

    Args:
        content: Markdown string containing HTML comments.

    Returns:
        Markdown string with HTML comments removed.
    """
    return re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)


def find_unhandled_shortcodes(content: str) -> Counter:
    """Find any remaining Hugo shortcode patterns after resolution.

    Uses a character-class pattern that stops at `>`, `%`, `}`, and
    whitespace, preventing greedy capture of closing `>}}` in compact
    shortcode forms like {{<mermaid>}}.

    Args:
        content: Markdown string after all resolvers have run.

    Returns:
        Counter mapping unhandled shortcode names to their occurrence counts.
    """
    return Counter(
        match.group(1)
        for match in re.finditer(r'{{[<%]\s*/?([a-zA-Z][a-zA-Z0-9_/-]*)', content)
    )


def clean_extra_whitespace(content: str) -> str:
    """Collapse excessive blank lines left after shortcode removal.

    Args:
        content: Markdown string that may have extra blank lines.

    Returns:
        Markdown string with at most two consecutive newlines.
    """
    return re.sub(r'\n{3,}', '\n\n', content).strip()
