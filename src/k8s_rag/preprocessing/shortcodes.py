"""Resolve Hugo shortcodes in Kubernetes documentation markdown.

Transforms Hugo-specific shortcodes into clean markdown so the content
can be chunked and embedded for retrieval. Each shortcode type has a
dedicated resolver function, and the main resolve_shortcodes function
orchestrates them in the correct order.
"""

import logging
import re
from collections import Counter
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


HEADING_MAP = {
    "prerequisites": "Before you begin",
    "whatsnext": "What's next",
    "objectives": "Objectives",
    "cleanup": "Cleaning up",
}

SLASH_COMMENT_LANGS = {
    "go", "javascript", "js", "typescript", "ts",
    "java", "c", "cpp", "csharp", "rust", "swift", "kotlin",
}

API_REFERENCE_BASE = "https://kubernetes.io/docs/reference/kubernetes-api"

_INCLUDE_PATTERN = re.compile(r'{{[<%]\s*include\s+"([^"]*)"\s*[>%]}}')


def _safe_relative_path(path: str) -> Path | None:
    """Return a safe relative path or None when path is unsafe."""
    cleaned = path.strip().lstrip("/")
    if not cleaned:
        return None
    candidate = Path(cleaned)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    return candidate


def _source_comment(lang: str, file_path: str) -> str:
    """Format a 'Source:' comment line for an inlined code sample.

    Args:
        lang: Code fence language identifier.
        file_path: Original file path of the inlined sample.

    Returns:
        A single-line comment in syntax appropriate for the language.
    """
    if lang in SLASH_COMMENT_LANGS:
        return f"// Source: {file_path}"
    return f"# Source: {file_path}"


# ---- File inlining resolvers (run first) ----

def resolve_code_samples(content: str, examples_dir: str | Path) -> str:
    """Inline code_sample shortcodes with the referenced file content.

    Handles both delimiter styles ({{% %}} and {{< >}}) and any attribute
    order (language= may appear before file=).

    Args:
        content: Markdown string containing code_sample shortcodes.
        examples_dir: Path to the directory containing example files.

    Returns:
        Markdown string with code_sample shortcodes replaced by
        fenced code blocks.
    """
    # Matches both "code_sample" and the older "code" shortcode name
    pattern = r'{{[<%]\s*code(?:_sample)?\s+(.*?)\s*[%>]}}'

    def replace_code_sample(match):
        attrs = match.group(1)
        file_match = re.search(r'file="([^"]*)"', attrs)
        if not file_match:
            return ""
        file_path = file_match.group(1)
        rel_path = _safe_relative_path(file_path)
        if rel_path is None:
            logger.warning("code_sample not found: %s", file_path)
            return ""
        full_path = Path(examples_dir) / rel_path

        if not full_path.exists():
            logger.warning("code_sample not found: %s", file_path)
            return ""

        code = full_path.read_text().rstrip("\n")
        suffix = full_path.suffix.lstrip(".")
        lang = {"yaml": "yaml", "yml": "yaml", "json": "json", "go": "go",
                "sh": "shell", "bash": "shell"}.get(suffix, suffix)

        source_line = _source_comment(lang, file_path)
        return f"```{lang}\n{source_line}\n{code}\n```"

    return re.sub(pattern, replace_code_sample, content)


def resolve_includes(content: str, includes_dir: str | Path) -> str:
    """Inline include shortcodes with the referenced file content.

    Performs a single pass (depth 1). Any include shortcodes remaining
    after inlining (e.g. nested includes) are stripped with a warning.

    Args:
        content: Markdown string containing include shortcodes.
        includes_dir: Path to the directory containing include files.

    Returns:
        Markdown string with include shortcodes replaced by file content.
    """
    def replace_include(match):
        filename = match.group(1)
        rel_path = _safe_relative_path(filename)
        if rel_path is None:
            logger.warning("include not found: %s", filename)
            return ""
        full_path = Path(includes_dir) / rel_path

        if not full_path.exists():
            logger.warning("include not found: %s", filename)
            return ""

        return full_path.read_text().rstrip("\n")

    content = _INCLUDE_PATTERN.sub(replace_include, content)

    remaining = _INCLUDE_PATTERN.findall(content)
    if remaining:
        for nested in remaining:
            logger.warning("stripping nested include: %s", nested)
        content = _INCLUDE_PATTERN.sub("", content)

    return content


def resolve_glossary_definitions(content: str, glossary_dir: str | Path) -> str:
    """Inline glossary_definition shortcodes with definition text.

    Reads the glossary markdown file for each term, extracts the
    short_description from frontmatter (length=short) or full body
    (length=all), and inlines the text.

    Args:
        content: Markdown string containing glossary_definition shortcodes.
        glossary_dir: Path to the glossary files directory.

    Returns:
        Markdown string with glossary_definition shortcodes replaced.
    """
    pattern = r'{{<\s*glossary_definition\s+([^>]*?)>}}'

    def replace_definition(match):
        attrs = match.group(1)
        term_match = re.search(r'term_id="([^"]*)"', attrs)
        if not term_match:
            return ""

        term_id = term_match.group(1)
        length_match = re.search(r'length="([^"]*)"', attrs)
        length = length_match.group(1) if length_match else "short"

        glossary_file = Path(glossary_dir) / f"{term_id}.md"
        if not glossary_file.exists():
            logger.warning("glossary file not found: %s.md", term_id)
            return ""

        file_content = glossary_file.read_text()
        if not file_content.startswith("---"):
            return file_content.strip()

        lines = file_content.splitlines(keepends=True)
        closing_line = None
        for idx, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                closing_line = idx
                break

        if closing_line is None:
            return file_content.strip()

        yaml_block = "".join(lines[1:closing_line])
        body = "".join(lines[closing_line + 1:]).strip()
        frontmatter = yaml.safe_load(yaml_block) or {}

        if length == "short":
            return frontmatter.get("short_description", body or "")
        return body if body else frontmatter.get("short_description", "")

    return re.sub(pattern, replace_definition, content)


# ---- Structural resolvers ----

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


# ---- Inline shortcode resolvers ----

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
            return "Figure: " + " — ".join(parts)

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


# ---- Block shortcode resolvers ----

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


# ---- Strip resolvers ----

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


# ---- Cleanup ----

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


# ---- Orchestrator ----

def resolve_shortcodes(
    content: str,
    examples_dir: str | Path,
    includes_dir: str | Path,
    k8s_version: str,
    glossary_dir: str | Path | None = None,
) -> tuple[str, Counter]:
    """Resolve all Hugo shortcodes in a markdown document.

    Runs each resolver in the correct order:
    1. File inlining (code_sample, include, glossary_definition)
    2. Structural/block wrappers (tabs, table, mermaid, comment,
       tutorial shortcodes)
    3. Inline replacements (glossary_tooltip, heading, feature_state,
       param, figure, skew, api-reference, link, highlight,
       version_check, relref)
    4. Block replacements (note, caution, warning, alert, details,
       pageinfo, example)
    5. Strip (thirdparty-content, legacy-repos-deprecation)
    6. Cleanup (HTML comments, catch-all, whitespace)

    Args:
        content: Raw markdown body (after frontmatter extraction).
        examples_dir: Path to data/examples/ for code_sample resolution.
        includes_dir: Path to data/includes/ for include resolution.
        k8s_version: Kubernetes docs version (e.g., "v1.36").
        glossary_dir: Path to glossary files for glossary_definition
            resolution. Optional for backward compatibility.

    Returns:
        Tuple of (resolved_content, unhandled_shortcodes) where
        unhandled_shortcodes is a Counter mapping shortcode names to
        their occurrence counts for names not resolved by any handler.
    """
    # 1. File inlining (depth 1, remaining includes stripped)
    content = resolve_code_samples(content, examples_dir)
    content = resolve_includes(content, includes_dir)
    if glossary_dir:
        content = resolve_glossary_definitions(content, glossary_dir)

    # 2. Structural wrappers (strip wrappers; some strip entire block)
    content = resolve_tabs(content)
    content = resolve_table(content)
    content = resolve_mermaid(content)
    content = resolve_comments(content)
    content = resolve_tutorial_shortcodes(content)

    # 3. Inline replacements
    content = resolve_glossary_tooltips(content)
    content = resolve_headings(content)
    content = resolve_feature_states(content)
    content = resolve_params(content, k8s_version)
    content = resolve_figures(content)
    content = resolve_skew(content, k8s_version)
    content = resolve_api_reference(content)
    content = resolve_link_shortcodes(content)
    content = resolve_highlight(content)
    content = resolve_version_check(content)
    content = resolve_relref(content)

    # 4. Block replacements (content-preserving)
    content = resolve_notes(content)
    content = resolve_alerts(content)
    content = resolve_details(content)
    content = resolve_pageinfo(content)
    content = resolve_examples(content)

    # 5. Strip
    content = resolve_thirdparty_content(content)
    content = resolve_legacy_repos_deprecation(content)

    # 6. Cleanup
    content = strip_html_comments(content)
    unhandled = find_unhandled_shortcodes(content)
    content = clean_extra_whitespace(content)

    return content, unhandled
