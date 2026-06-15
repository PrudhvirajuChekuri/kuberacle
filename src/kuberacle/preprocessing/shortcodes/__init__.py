"""Resolve Hugo shortcodes in Kubernetes documentation markdown.

Transforms Hugo-specific shortcodes into clean markdown so the content can be
chunked and embedded for retrieval. Each shortcode type has a dedicated resolver
function (grouped by stage into submodules), and ``resolve_shortcodes``
orchestrates them in the correct order.
"""

from collections import Counter
from pathlib import Path

from kuberacle.preprocessing.shortcodes.blocks import (
    resolve_alerts,
    resolve_details,
    resolve_examples,
    resolve_notes,
    resolve_pageinfo,
)
from kuberacle.preprocessing.shortcodes.cleanup import (
    clean_extra_whitespace,
    find_unhandled_shortcodes,
    strip_html_comments,
)
from kuberacle.preprocessing.shortcodes.file_inlining import (
    resolve_code_samples,
    resolve_glossary_definitions,
    resolve_includes,
)
from kuberacle.preprocessing.shortcodes.inline import (
    resolve_api_reference,
    resolve_feature_states,
    resolve_figures,
    resolve_glossary_tooltips,
    resolve_headings,
    resolve_highlight,
    resolve_link_shortcodes,
    resolve_params,
    resolve_relref,
    resolve_skew,
    resolve_version_check,
)
from kuberacle.preprocessing.shortcodes.strip import (
    resolve_legacy_repos_deprecation,
    resolve_thirdparty_content,
)
from kuberacle.preprocessing.shortcodes.structural import (
    resolve_comments,
    resolve_mermaid,
    resolve_table,
    resolve_tabs,
    resolve_tutorial_shortcodes,
)

__all__ = [
    "resolve_code_samples",
    "resolve_includes",
    "resolve_glossary_definitions",
    "resolve_tabs",
    "resolve_table",
    "resolve_mermaid",
    "resolve_comments",
    "resolve_tutorial_shortcodes",
    "resolve_glossary_tooltips",
    "resolve_headings",
    "resolve_params",
    "resolve_figures",
    "resolve_skew",
    "resolve_feature_states",
    "resolve_api_reference",
    "resolve_link_shortcodes",
    "resolve_highlight",
    "resolve_version_check",
    "resolve_relref",
    "resolve_notes",
    "resolve_alerts",
    "resolve_details",
    "resolve_pageinfo",
    "resolve_examples",
    "resolve_thirdparty_content",
    "resolve_legacy_repos_deprecation",
    "strip_html_comments",
    "find_unhandled_shortcodes",
    "clean_extra_whitespace",
    "resolve_shortcodes",
]


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
