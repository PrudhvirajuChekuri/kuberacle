"""Inline shortcode resolvers that replace a shortcode with display text."""

import re

HEADING_MAP = {
    "prerequisites": "Before you begin",
    "whatsnext": "What's next",
    "objectives": "Objectives",
    "cleanup": "Cleaning up",
}

API_REFERENCE_BASE = "https://kubernetes.io/docs/reference/kubernetes-api"


def resolve_glossary_tooltips(content: str) -> str:
    """Replace glossary_tooltip shortcodes with their display text.

    Args:
        content: Markdown string containing glossary_tooltip shortcodes.

    Returns:
        Markdown string with tooltips replaced by their text attribute.
    """
    pattern = r'{{<\s*glossary_tooltip\s+[^>]*?>}}'

    def replace_tooltip(match):
        attrs = match.group(0)
        text_match = re.search(r'text="([^"]*)"', attrs)
        if text_match:
            return text_match.group(1)
        term_match = re.search(r'term_id="([^"]*)"', attrs)
        if term_match:
            return term_match.group(1).replace("-", " ")
        return ""

    return re.sub(pattern, replace_tooltip, content)


def resolve_headings(content: str) -> str:
    """Convert heading shortcodes to standard markdown headings.

    Args:
        content: Markdown string containing heading shortcodes.

    Returns:
        Markdown string with heading shortcodes replaced.
    """
    pattern = r'{{% heading\s+"([^"]*)"\s*%}}'

    def replace_heading(match):
        key = match.group(1)
        return HEADING_MAP.get(key, key.replace("-", " ").title())

    return re.sub(pattern, replace_heading, content)


def resolve_params(content: str, k8s_version: str) -> str:
    """Replace param shortcodes with known values.

    Args:
        content: Markdown string containing param shortcodes.
        k8s_version: Kubernetes version string (e.g. "v1.36").

    Returns:
        Markdown string with param shortcodes replaced.
    """
    param_map = {
        "version": k8s_version,
        "latest": k8s_version,
    }
    pattern = r'{{<\s*param\s+"([^"]*)"\s*>}}'

    def replace_param(match):
        key = match.group(1)
        return param_map.get(key, key)

    return re.sub(pattern, replace_param, content)


def resolve_figures(content: str) -> str:
    """Convert figure shortcodes to plain labeled text.

    Args:
        content: Markdown string containing figure shortcodes.

    Returns:
        Markdown string with figure shortcodes replaced by
        plain "Figure: ..." labels.
    """
    pattern = r'{{<\s*figure\s+([^>]*)>}}'

    def replace_figure(match):
        attrs = match.group(1)

        caption_match = re.search(r'caption="([^"]*)"', attrs)
        title_match = re.search(r'title="([^"]*)"', attrs)

        parts = []
        if title_match:
            parts.append(title_match.group(1))
        if caption_match:
            parts.append(caption_match.group(1))

        if parts:
            return "Figure: " + " - ".join(parts)

        alt_match = re.search(r'alt="([^"]*)"', attrs)
        if alt_match and alt_match.group(1).strip():
            return f"Figure: {alt_match.group(1)}"
        return ""

    return re.sub(pattern, replace_figure, content)


def resolve_skew(content: str, k8s_version: str) -> str:
    """Resolve skew shortcodes to the current Kubernetes version.

    Handles both simple variants (currentVersion) and arithmetic variants
    (currentVersionAddMinor N) with an optional separator argument.

    Args:
        content: Markdown string containing skew shortcodes.
        k8s_version: Kubernetes docs version string (e.g., "v1.36").

    Returns:
        Markdown string with skew shortcodes replaced.
    """
    version_number = k8s_version.lstrip("v")
    try:
        major, minor_str = version_number.split(".", 1)
        minor = int(minor_str)
    except (ValueError, AttributeError):
        major, minor = "1", 0

    simple_map = {
        "currentVersion": version_number,
        "latestVersion": version_number,
        "currentPatchVersion": version_number,
    }

    # Non-greedy capture of all args; handles both {{< >}} and {{% %}} delimiters
    pattern = r'{{[<%]\s*skew\s+(.*?)\s*[%>]}}'

    def replace_skew(match):
        args = match.group(1).strip().split()
        if not args:
            return version_number
        name = args[0]
        if name in simple_map:
            return simple_map[name]
        if name == "currentVersionAddMinor" and len(args) >= 2:
            try:
                offset = int(args[1])
                sep = args[2].strip('"') if len(args) >= 3 else "."
                return f"{major}{sep}{minor + offset}"
            except (ValueError, IndexError):
                return version_number
        return version_number

    return re.sub(pattern, replace_skew, content)


def resolve_feature_states(content: str) -> str:
    """Convert feature-state shortcodes to labeled text.

    Args:
        content: Markdown string containing feature-state shortcodes.

    Returns:
        Markdown string with feature-state shortcodes replaced.
    """
    pattern = r'{{<\s*feature-state\s+([^>]*?)>}}'

    def replace_feature_state(match):
        attrs = match.group(1)
        version = re.search(r'for_k8s_version="([^"]*)"', attrs)
        state = re.search(r'state="([^"]*)"', attrs)
        gate = re.search(r'feature_gate_name="([^"]*)"', attrs)

        if version and state:
            return f"[FEATURE STATE: {version.group(1)} {state.group(1)}]"
        if gate:
            return f"[FEATURE STATE: {gate.group(1)}]"
        return ""

    return re.sub(pattern, replace_feature_state, content)


def _api_reference_link_text(page: str) -> str:
    """Build a human-readable link label from an api-reference page path.

    Args:
        page: The page attribute from an api-reference shortcode.

    Returns:
        Display text for the resulting markdown link.
    """
    last_segment = page.rsplit("/", 1)[-1]
    resource_slug = re.sub(r'-v\d+\w*$', '', last_segment)
    words = [w for w in resource_slug.split("-") if w]
    resource_label = " ".join(w.capitalize() for w in words) if words else "API"
    return f"{resource_label} API reference"


def resolve_api_reference(content: str) -> str:
    """Resolve api-reference shortcodes to markdown links.

    Args:
        content: Markdown string containing api-reference shortcodes.

    Returns:
        Markdown string with api-reference shortcodes replaced by
        labeled links to the API reference docs.
    """
    pattern = r'{{<\s*api-reference\s+([^>]*?)>}}'

    def replace_api_reference(match):
        attrs = match.group(1)
        page_match = re.search(r'page="([^"]*)"', attrs)
        if not page_match:
            return ""
        page = page_match.group(1)
        label = _api_reference_link_text(page)
        url = f"{API_REFERENCE_BASE}/{page}/"
        return f"[{label}]({url})"

    return re.sub(pattern, replace_api_reference, content)


def resolve_link_shortcodes(content: str) -> str:
    """Replace link shortcodes with their display text.

    Args:
        content: Markdown string containing link shortcodes.

    Returns:
        Markdown string with link shortcodes replaced by text.
    """
    pattern = r'{{<\s*link\s+([^>]*?)>}}'

    def replace_link(match):
        attrs = match.group(1)
        text_match = re.search(r'text="([^"]*)"', attrs)
        if text_match:
            return text_match.group(1)
        return ""

    return re.sub(pattern, replace_link, content)


def resolve_highlight(content: str) -> str:
    """Strip highlight shortcode tags, keeping inner content.

    Args:
        content: Markdown string containing highlight shortcodes.

    Returns:
        Markdown string with highlight tags removed.
    """
    content = re.sub(r'{{<\s*/?\s*highlight[^>]*>}}\n?', '', content)
    return content


def resolve_version_check(content: str) -> str:
    """Strip version-check shortcodes.

    These render as a prerequisite note about the minimum required
    Kubernetes version and carry no retrieval-relevant content.
    Handles both angle-bracket ({{< >}}) and percent ({{% %}}) delimiters.

    Args:
        content: Markdown string containing version-check shortcodes.

    Returns:
        Markdown string with version-check shortcodes removed.
    """
    return re.sub(r'{{[<%]\s*version-check\s*[%>]}}', '', content)


def resolve_relref(content: str) -> str:
    """Replace relref shortcodes with their path text.

    Args:
        content: Markdown string containing relref shortcodes.

    Returns:
        Markdown string with relref shortcodes replaced by their path.
    """
    return re.sub(r'{{<\s*relref\s+"([^"]*)"\s*>}}', r'\1', content)
