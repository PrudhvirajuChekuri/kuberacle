"""Resolve Hugo shortcodes in Kubernetes documentation markdown.

Transforms Hugo-specific shortcodes into clean markdown so the content
can be chunked and embedded for retrieval. Each shortcode type has a
dedicated resolver function, and the main resolve_shortcodes function
orchestrates them in the correct order.
"""

import re
from pathlib import Path


# Standard heading shortcode mappings used by the K8s docs
HEADING_MAP = {
    "prerequisites": "Before you begin",
    "whatsnext": "What's next",
    "objectives": "Objectives",
    "cleanup": "Cleaning up",
}

# Known Hugo params from the K8s site config
PARAM_MAP = {
    "version": "v1.36",
    "latest": "v1.36",
}


def resolve_glossary_tooltips(content):
    """Replace glossary_tooltip shortcodes with their display text.

    Example:
        {{< glossary_tooltip text="containers" term_id="container" >}}
        becomes: containers

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
        # Fall back to term_id if no text attribute
        term_match = re.search(r'term_id="([^"]*)"', attrs)
        if term_match:
            return term_match.group(1).replace("-", " ")
        return ""

    return re.sub(pattern, replace_tooltip, content)


def resolve_notes(content):
    """Convert note, caution, and warning blocks to blockquotes.

    Example:
        {{< note >}}
        Some important info.
        {{< /note >}}
        becomes:
        > **NOTE:** Some important info.

    Args:
        content: Markdown string containing note/caution/warning blocks.

    Returns:
        Markdown string with admonition blocks converted to blockquotes.
    """
    for admonition in ("note", "caution", "warning"):
        pattern = (
            r'{{[<%]\s*'
            + admonition
            + r'\s*[%>]}}\n?'
            + r'(.*?)'
            + r'{{[<%]\s*/'
            + admonition
            + r'\s*[%>]}}'
        )
        label = admonition.upper()

        def make_blockquote(match, label=label):
            inner = match.group(1).strip()
            # Prefix each line with > for blockquote formatting
            quoted = "\n".join(
                f"> {line}" if line.strip() else ">" for line in inner.split("\n")
            )
            return f"> **{label}:** {quoted.lstrip('> ')}"

        content = re.sub(pattern, make_blockquote, content, flags=re.DOTALL)

    return content


def resolve_code_samples(content, examples_dir):
    """Inline code_sample shortcodes with the referenced file content.

    Reads the referenced YAML/JSON file from the examples directory
    and inserts it as a fenced code block.

    Example:
        {{% code_sample file="pods/simple-pod.yaml" %}}
        becomes:
        ```yaml
        apiVersion: v1
        kind: Pod
        ...
        ```

    Args:
        content: Markdown string containing code_sample shortcodes.
        examples_dir: Path to the directory containing example files.

    Returns:
        Markdown string with code_sample shortcodes replaced by
        fenced code blocks.
    """
    pattern = r'{{% code_sample\s+file="([^"]*)"\s*%}}'

    def replace_code_sample(match):
        file_path = match.group(1)
        full_path = Path(examples_dir) / file_path

        if not full_path.exists():
            return f"<!-- code_sample not found: {file_path} -->"

        code = full_path.read_text().rstrip("\n")
        # Detect language from file extension
        suffix = full_path.suffix.lstrip(".")
        lang = {"yaml": "yaml", "yml": "yaml", "json": "json", "go": "go",
                "sh": "shell", "bash": "shell"}.get(suffix, suffix)

        return f"```{lang}\n{code}\n```"

    return re.sub(pattern, replace_code_sample, content)


def resolve_includes(content, includes_dir):
    """Inline include shortcodes with the referenced file content.

    Example:
        {{< include "task-tutorial-prereqs.md" >}}
        becomes the content of that file.

    Args:
        content: Markdown string containing include shortcodes.
        includes_dir: Path to the directory containing include files.

    Returns:
        Markdown string with include shortcodes replaced by file content.
    """
    pattern = r'{{<\s*include\s+"([^"]*)"\s*>}}'

    def replace_include(match):
        filename = match.group(1)
        full_path = Path(includes_dir) / filename

        if not full_path.exists():
            return f"<!-- include not found: {filename} -->"

        return full_path.read_text().rstrip("\n")

    return re.sub(pattern, replace_include, content)


def resolve_headings(content):
    """Convert heading shortcodes to standard markdown headings.

    Example:
        ## {{% heading "prerequisites" %}}
        becomes:
        ## Before you begin

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


def resolve_feature_states(content):
    """Convert feature-state shortcodes to labeled text.

    Handles two formats:
        {{< feature-state for_k8s_version="v1.14" state="stable" >}}
        becomes: [FEATURE STATE: v1.14 stable]

        {{< feature-state feature_gate_name="ContainerRestartRules" >}}
        becomes: [FEATURE STATE: ContainerRestartRules]

    Args:
        content: Markdown string containing feature-state shortcodes.

    Returns:
        Markdown string with feature-state shortcodes replaced.
    """
    # Match the whole shortcode, then extract attributes flexibly
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


def resolve_tabs(content):
    """Flatten tab blocks into sequential labeled sections.

    Example:
        {{< tabs name="example" >}}
        {{% tab name="Option A" %}}
        Content A
        {{% /tab %}}
        {{< /tabs >}}
        becomes:
        **Tab: Option A**
        Content A

    Args:
        content: Markdown string containing tab shortcodes.

    Returns:
        Markdown string with tab structures flattened.
    """
    # Remove tabs wrapper (opening and closing)
    content = re.sub(r'{{<\s*tabs\s+[^>]*>}}\n?', '', content)
    content = re.sub(r'{{<\s*/tabs\s*>}}\n?', '', content)

    # Convert individual tab markers to bold labels
    content = re.sub(
        r'{{% tab\s+name="([^"]*)"\s*%}}\n?',
        r'**Tab: \1**\n',
        content,
    )
    content = re.sub(r'{{% /tab\s*%}}\n?', '', content)

    return content


def resolve_params(content):
    """Replace param shortcodes with known values.

    Example:
        {{< param "version" >}}
        becomes: v1.36

    Args:
        content: Markdown string containing param shortcodes.

    Returns:
        Markdown string with param shortcodes replaced.
    """
    pattern = r'{{<\s*param\s+"([^"]*)"\s*>}}'

    def replace_param(match):
        key = match.group(1)
        return PARAM_MAP.get(key, key)

    return re.sub(pattern, replace_param, content)


def resolve_figures(content):
    """Convert figure shortcodes to descriptive text.

    Extracts caption and title if available, since the actual image
    files are not useful for text-based retrieval.

    Args:
        content: Markdown string containing figure shortcodes.

    Returns:
        Markdown string with figure shortcodes replaced by captions.
    """
    pattern = r'{{<\s*figure\s+([^>]*)>}}'

    def replace_figure(match):
        attrs = match.group(1)

        caption_match = re.search(r'caption="([^"]*)"', attrs)
        title_match = re.search(r'title="([^"]*)"', attrs)

        parts = []
        if title_match:
            parts.append(f"**{title_match.group(1)}**")
        if caption_match:
            parts.append(caption_match.group(1))

        if parts:
            return " ".join(parts)
        return ""

    return re.sub(pattern, replace_figure, content)


def strip_skew(content):
    """Remove skew shortcodes entirely.

    These reference version skew policies and are not useful
    for RAG retrieval.

    Args:
        content: Markdown string containing skew shortcodes.

    Returns:
        Markdown string with skew shortcodes removed.
    """
    return re.sub(r'{{<\s*skew\s+[^>]*>}}', '', content)


def strip_api_reference(content):
    """Remove api-reference shortcodes.

    These render as links to the API reference docs. The page
    attribute is not useful for RAG content.

    Args:
        content: Markdown string containing api-reference shortcodes.

    Returns:
        Markdown string with api-reference shortcodes removed.
    """
    return re.sub(r'{{<\s*api-reference\s+[^>]*>}}', '', content)


def strip_html_comments(content):
    """Remove HTML comments like <!-- overview --> and <!-- body -->.

    These are Hugo section markers that have no content value.

    Args:
        content: Markdown string containing HTML comments.

    Returns:
        Markdown string with HTML comments removed.
    """
    return re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)


def clean_extra_whitespace(content):
    """Collapse excessive blank lines left after shortcode removal.

    Args:
        content: Markdown string that may have extra blank lines.

    Returns:
        Markdown string with at most two consecutive newlines.
    """
    return re.sub(r'\n{3,}', '\n\n', content).strip()


def resolve_shortcodes(content, examples_dir, includes_dir):
    """Resolve all Hugo shortcodes in a markdown document.

    Runs each resolver in the correct order:
    1. Structural (tabs) — unwrap before processing inner content
    2. File inlining (code_sample, include) — bring in external content
    3. Inline replacements (glossary, heading, feature-state, param, figure)
    4. Block replacements (note, caution, warning)
    5. Cleanup (skew, HTML comments, whitespace)

    Args:
        content: Raw markdown body (after frontmatter extraction).
        examples_dir: Path to data/examples/ for code_sample resolution.
        includes_dir: Path to data/includes/ for include resolution.

    Returns:
        Clean markdown string with all shortcodes resolved.
    """
    # 1. Structural
    content = resolve_tabs(content)

    # 2. File inlining
    content = resolve_code_samples(content, examples_dir)
    content = resolve_includes(content, includes_dir)

    # 3. Inline replacements
    content = resolve_glossary_tooltips(content)
    content = resolve_headings(content)
    content = resolve_feature_states(content)
    content = resolve_params(content)
    content = resolve_figures(content)

    # 4. Block replacements
    content = resolve_notes(content)

    # 5. Cleanup
    content = strip_skew(content)
    content = strip_api_reference(content)
    content = strip_html_comments(content)
    content = clean_extra_whitespace(content)

    return content