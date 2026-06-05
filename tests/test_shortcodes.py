"""Tests for the shortcode resolution module."""

import pytest
from collections import Counter

from k8s_rag.preprocessing.shortcodes import (
    resolve_glossary_tooltips,
    resolve_glossary_definitions,
    resolve_notes,
    resolve_code_samples,
    resolve_includes,
    resolve_headings,
    resolve_feature_states,
    resolve_tabs,
    resolve_table,
    resolve_mermaid,
    resolve_comments,
    resolve_tutorial_shortcodes,
    resolve_params,
    resolve_figures,
    resolve_skew,
    resolve_api_reference,
    resolve_link_shortcodes,
    resolve_highlight,
    resolve_version_check,
    resolve_relref,
    resolve_alerts,
    resolve_details,
    resolve_pageinfo,
    resolve_examples,
    resolve_thirdparty_content,
    resolve_legacy_repos_deprecation,
    strip_html_comments,
    find_unhandled_shortcodes,
    clean_extra_whitespace,
    resolve_shortcodes,
)


# --- resolve_glossary_tooltips ---

def test_glossary_tooltip_text_before_term_id():
    content = '{{< glossary_tooltip text="containers" term_id="container" >}}'
    assert resolve_glossary_tooltips(content) == "containers"


def test_glossary_tooltip_term_id_before_text():
    content = '{{< glossary_tooltip term_id="kubelet" text="kubelet" >}}'
    assert resolve_glossary_tooltips(content) == "kubelet"


def test_glossary_tooltip_no_text_attribute():
    """Falls back to term_id, replacing hyphens with spaces."""
    content = '{{< glossary_tooltip term_id="init-container" >}}'
    assert resolve_glossary_tooltips(content) == "init container"


def test_glossary_tooltip_no_space_before_closing():
    content = '{{< glossary_tooltip term_id="operator-pattern" text="operators">}}'
    assert resolve_glossary_tooltips(content) == "operators"


def test_glossary_tooltip_in_surrounding_text():
    content = 'A group of {{< glossary_tooltip text="containers" term_id="container" >}} running together.'
    assert resolve_glossary_tooltips(content) == "A group of containers running together."


# --- resolve_glossary_definitions ---

def test_glossary_definition_short(tmp_path):
    gfile = tmp_path / "ingress.md"
    gfile.write_text(
        "---\ntitle: Ingress\nshort_description: An API object.\n---\nFull body here."
    )
    content = '{{< glossary_definition term_id="ingress" length="short" >}}'
    result = resolve_glossary_definitions(content, tmp_path)
    assert result == "An API object."


def test_glossary_definition_all(tmp_path):
    gfile = tmp_path / "ingress.md"
    gfile.write_text(
        "---\ntitle: Ingress\nshort_description: An API object.\n---\nFull body here."
    )
    content = '{{< glossary_definition term_id="ingress" length="all" >}}'
    result = resolve_glossary_definitions(content, tmp_path)
    assert result == "Full body here."


def test_glossary_definition_missing_file(tmp_path, caplog):
    import logging
    content = '{{< glossary_definition term_id="missing" length="short" >}}'
    with caplog.at_level(logging.WARNING):
        result = resolve_glossary_definitions(content, tmp_path)
    assert result == ""
    assert any("missing" in r.message for r in caplog.records)


# --- resolve_notes ---

def test_note_block():
    content = "{{< note >}}\nThis is important.\n{{< /note >}}"
    result = resolve_notes(content)
    assert "NOTE:" in result
    assert "**" not in result
    assert "This is important." in result
    assert ">" not in result


def test_caution_block():
    content = "{{< caution >}}\nBe careful here.\n{{< /caution >}}"
    result = resolve_notes(content)
    assert "CAUTION:" in result
    assert "**" not in result


def test_warning_block():
    content = "{{< warning >}}\nDanger ahead.\n{{< /warning >}}"
    result = resolve_notes(content)
    assert "WARNING:" in result
    assert "**" not in result


def test_note_block_multiline():
    content = "{{< note >}}\nLine one.\nLine two.\nLine three.\n{{< /note >}}"
    result = resolve_notes(content)
    assert "Line one." in result
    assert "Line three." in result
    assert not any(line.startswith(">") for line in result.strip().split("\n"))


# --- resolve_code_samples ---

def test_code_sample_inlines_yaml(tmp_path):
    example_file = tmp_path / "pods" / "simple-pod.yaml"
    example_file.parent.mkdir(parents=True)
    example_file.write_text("apiVersion: v1\nkind: Pod\n")

    content = '{{% code_sample file="pods/simple-pod.yaml" %}}'
    result = resolve_code_samples(content, tmp_path)
    assert "```yaml" in result
    assert "apiVersion: v1" in result
    assert "kind: Pod" in result
    assert "```" in result


def test_code_sample_includes_source_path(tmp_path):
    """The inlined code block should preserve the source file path."""
    example_file = tmp_path / "pods" / "simple-pod.yaml"
    example_file.parent.mkdir(parents=True)
    example_file.write_text("apiVersion: v1\nkind: Pod\n")

    content = '{{% code_sample file="pods/simple-pod.yaml" %}}'
    result = resolve_code_samples(content, tmp_path)
    assert "# Source: pods/simple-pod.yaml" in result
    fence_start = result.index("```yaml") + len("```yaml") + 1
    body = result[fence_start:]
    assert body.startswith("# Source: pods/simple-pod.yaml")


def test_code_sample_source_uses_slash_comment_for_go(tmp_path):
    """Go samples should use // for the source comment, not #."""
    example_file = tmp_path / "main.go"
    example_file.write_text("package main\n")

    content = '{{% code_sample file="main.go" %}}'
    result = resolve_code_samples(content, tmp_path)
    assert "// Source: main.go" in result
    assert "# Source: main.go" not in result


def test_code_sample_missing_file(tmp_path, caplog):
    import logging
    content = '{{% code_sample file="missing/file.yaml" %}}'
    with caplog.at_level(logging.WARNING):
        result = resolve_code_samples(content, tmp_path)
    assert result == ""
    assert any("missing/file.yaml" in r.message for r in caplog.records)


def test_code_sample_leading_slash_is_treated_as_relative(tmp_path):
    """Leading slash references should resolve within examples_dir."""
    example_file = tmp_path / "controllers" / "example.yaml"
    example_file.parent.mkdir(parents=True)
    example_file.write_text("apiVersion: v1\nkind: ConfigMap\n")

    content = '{{% code_sample file="/controllers/example.yaml" %}}'
    result = resolve_code_samples(content, tmp_path)
    assert "```yaml" in result
    assert "kind: ConfigMap" in result


# --- resolve_includes ---

def test_include_inlines_content(tmp_path):
    include_file = tmp_path / "prereqs.md"
    include_file.write_text("You need a Kubernetes cluster.")

    content = '{{< include "prereqs.md" >}}'
    result = resolve_includes(content, tmp_path)
    assert result == "You need a Kubernetes cluster."


def test_include_percent_syntax_inlines_content(tmp_path):
    """Percent-delimited include shortcode should also resolve."""
    include_file = tmp_path / "prereqs.md"
    include_file.write_text("You need a Kubernetes cluster.")

    content = '{{% include "prereqs.md" %}}'
    result = resolve_includes(content, tmp_path)
    assert result == "You need a Kubernetes cluster."


def test_include_missing_file(tmp_path, caplog):
    import logging
    content = '{{< include "missing.md" >}}'
    with caplog.at_level(logging.WARNING):
        result = resolve_includes(content, tmp_path)
    assert result == ""
    assert any("missing.md" in r.message for r in caplog.records)


def test_include_leading_slash_is_treated_as_relative(tmp_path):
    """Leading slash include references should resolve under includes_dir."""
    include_file = tmp_path / "shared" / "snippet.md"
    include_file.parent.mkdir(parents=True)
    include_file.write_text("Shared include text.")

    content = '{{< include "/shared/snippet.md" >}}'
    result = resolve_includes(content, tmp_path)
    assert result == "Shared include text."


def test_include_strips_nested_includes(tmp_path, caplog):
    """Nested include shortcodes in inlined content should be stripped."""
    import logging
    outer = tmp_path / "outer.md"
    outer.write_text('Outer content.\n{{< include "inner.md" >}}')

    content = '{{< include "outer.md" >}}'
    with caplog.at_level(logging.WARNING):
        result = resolve_includes(content, tmp_path)
    assert "Outer content." in result
    assert "{{" not in result
    assert any("nested include" in r.message.lower() for r in caplog.records)


# --- resolve_headings ---

def test_heading_prerequisites():
    content = '## {{% heading "prerequisites" %}}'
    assert resolve_headings(content) == "## Before you begin"


def test_heading_whatsnext():
    content = '## {{% heading "whatsnext" %}}'
    assert resolve_headings(content) == "## What's next"


def test_heading_cleanup():
    content = '## {{% heading "cleanup" %}}'
    assert resolve_headings(content) == "## Cleaning up"


def test_heading_unknown_falls_back_to_title_case():
    content = '{{% heading "some-custom-heading" %}}'
    assert resolve_headings(content) == "Some Custom Heading"


# --- resolve_feature_states ---

def test_feature_state_versioned():
    content = '{{< feature-state for_k8s_version="v1.14" state="stable" >}}'
    assert resolve_feature_states(content) == "[FEATURE STATE: v1.14 stable]"


def test_feature_state_reversed_order():
    content = '{{< feature-state state="stable" for_k8s_version="v1.25" >}}'
    assert resolve_feature_states(content) == "[FEATURE STATE: v1.25 stable]"


def test_feature_state_gate_name():
    content = '{{< feature-state feature_gate_name="ContainerRestartRules" >}}'
    assert resolve_feature_states(content) == "[FEATURE STATE: ContainerRestartRules]"


# --- resolve_tabs ---

def test_tabs_flattened():
    content = (
        '{{< tabs name="example" >}}\n'
        '{{% tab name="Option A" %}}\n'
        'Content A\n'
        '{{% /tab %}}\n'
        '{{% tab name="Option B" %}}\n'
        'Content B\n'
        '{{% /tab %}}\n'
        '{{< /tabs >}}'
    )
    result = resolve_tabs(content)
    assert "Tab: Option A" in result
    assert "Content A" in result
    assert "Tab: Option B" in result
    assert "Content B" in result
    assert "{{" not in result


# --- resolve_params ---

def test_param_version():
    content = 'Kubernetes {{< param "version" >}} docs'
    assert resolve_params(content, "v1.36") == "Kubernetes v1.36 docs"


def test_param_uses_dynamic_version():
    content = 'Kubernetes {{< param "version" >}} docs'
    assert resolve_params(content, "v1.99") == "Kubernetes v1.99 docs"


def test_param_unknown_returns_key():
    content = '{{< param "unknown_key" >}}'
    assert resolve_params(content, "v1.36") == "unknown_key"


# --- resolve_figures ---

def test_figure_with_title_and_caption():
    content = '{{< figure src="/img.svg" title="Figure 1." caption="A pod diagram." >}}'
    result = resolve_figures(content)
    assert "Figure: Figure 1." in result
    assert "A pod diagram." in result


def test_figure_falls_back_to_alt():
    """A figure with only an alt attribute should emit a bracketed label."""
    content = '{{< figure src="/img.svg" alt="Pod creation diagram" class="medium" >}}'
    assert resolve_figures(content) == "Figure: Pod creation diagram"


def test_figure_no_text_attributes_returns_empty():
    """A figure with no title, caption, or alt still returns empty string."""
    content = '{{< figure src="/img.svg" class="medium" >}}'
    assert resolve_figures(content) == ""


# --- resolve_skew ---

def test_resolve_skew_current_version():
    content = "Version v{{< skew currentVersion >}} is required."
    assert resolve_skew(content, "v1.36") == "Version v1.36 is required."


def test_resolve_skew_latest_version():
    content = "Use {{< skew latestVersion >}} or later."
    assert resolve_skew(content, "v1.36") == "Use 1.36 or later."


def test_resolve_skew_current_patch_version():
    content = "Patch {{< skew currentPatchVersion >}} available."
    assert resolve_skew(content, "v1.36") == "Patch 1.36 available."


def test_resolve_skew_strips_v_prefix_from_config():
    """The 'v' prefix on the configured version must not be duplicated."""
    content = "In Kubernetes {{< skew currentVersion >}}, the behavior changes."
    assert resolve_skew(content, "v1.36") == "In Kubernetes 1.36, the behavior changes."


# --- resolve_api_reference ---

def test_resolve_api_reference_emits_link():
    content = '{{< api-reference page="workload-resources/pod-v1" >}}'
    result = resolve_api_reference(content)
    assert result == (
        "[Pod API reference]"
        "(https://kubernetes.io/docs/reference/kubernetes-api/"
        "workload-resources/pod-v1/)"
    )


def test_resolve_api_reference_multiword_resource():
    """Multi-word resources keep word boundaries in the link label."""
    content = '{{< api-reference page="workload-resources/replica-set-v1" >}}'
    result = resolve_api_reference(content)
    assert "[Replica Set API reference]" in result
    assert "workload-resources/replica-set-v1/" in result


def test_resolve_api_reference_inline_in_sentence():
    """The resolver must produce a sentence that reads naturally."""
    content = (
        "Read the {{< api-reference page=\"workload-resources/deployment-v1\" >}} "
        "to understand the Deployment API."
    )
    result = resolve_api_reference(content)
    assert "Read the [Deployment API reference]" in result
    assert "to understand the Deployment API." in result


# --- resolve_link_shortcodes ---

def test_link_shortcode_extracts_text():
    content = '{{< link text="services" url="/docs/concepts/services-networking/service/" >}}'
    assert resolve_link_shortcodes(content) == "services"


def test_link_shortcode_no_text_returns_empty():
    content = '{{< link url="/docs/concepts/services-networking/service/" >}}'
    assert resolve_link_shortcodes(content) == ""


# --- resolve_highlight ---

def test_highlight_strips_tags():
    content = '{{< highlight yaml >}}\napiVersion: v1\n{{< /highlight >}}'
    result = resolve_highlight(content)
    assert "{{" not in result
    assert "apiVersion: v1" in result


# --- resolve_thirdparty_content ---

def test_thirdparty_content_angle_bracket_stripped():
    content = "Some text.\n{{< thirdparty-content >}}\nMore text."
    result = resolve_thirdparty_content(content)
    assert "thirdparty" not in result
    assert "Some text." in result
    assert "More text." in result


def test_thirdparty_content_percent_stripped():
    content = "Some text.\n{{% thirdparty-content %}}\nMore text."
    result = resolve_thirdparty_content(content)
    assert "thirdparty" not in result
    assert "Some text." in result
    assert "More text." in result


def test_thirdparty_content_with_attribute_stripped():
    content = '{{% thirdparty-content single="true" %}}'
    assert resolve_thirdparty_content(content) == ""


# --- resolve_skew (multi-arg) ---

def test_resolve_skew_add_minor_negative():
    content = "v{{< skew currentVersionAddMinor -1 >}} control planes."
    assert resolve_skew(content, "v1.36") == "v1.35 control planes."


def test_resolve_skew_add_minor_positive():
    content = "v{{< skew currentVersionAddMinor 1 >}} control planes."
    assert resolve_skew(content, "v1.36") == "v1.37 control planes."


def test_resolve_skew_add_minor_zero():
    content = "{{< skew currentVersionAddMinor 0 >}}"
    assert resolve_skew(content, "v1.36") == "1.36"


def test_resolve_skew_add_minor_with_separator():
    content = "https://v{{< skew currentVersionAddMinor -1 \"-\" >}}.docs.kubernetes.io/"
    assert resolve_skew(content, "v1.36") == "https://v1-35.docs.kubernetes.io/"


# --- resolve_code_samples (language= before file=) ---

def test_code_sample_language_before_file(tmp_path):
    example_file = tmp_path / "pods" / "simple-pod.yaml"
    example_file.parent.mkdir(parents=True)
    example_file.write_text("apiVersion: v1\nkind: Pod\n")

    content = '{{% code_sample language="yaml" file="pods/simple-pod.yaml" %}}'
    result = resolve_code_samples(content, tmp_path)
    assert "```yaml" in result
    assert "apiVersion: v1" in result


def test_code_sample_angle_bracket_syntax(tmp_path):
    example_file = tmp_path / "controllers" / "example.yaml"
    example_file.parent.mkdir(parents=True)
    example_file.write_text("kind: Job\n")

    content = '{{< code_sample file="/controllers/example.yaml" >}}'
    result = resolve_code_samples(content, tmp_path)
    assert "kind: Job" in result


# --- resolve_tabs (angle-bracket form) ---

def test_tabs_angle_bracket_form():
    content = (
        '{{< tabs name="install" >}}\n'
        '{{< tab name="x86-64" codelang="bash" >}}\n'
        'Content for x86\n'
        '{{< /tab >}}\n'
        '{{< tab name="ARM64" codelang="bash" >}}\n'
        'Content for ARM\n'
        '{{< /tab >}}\n'
        '{{< /tabs >}}'
    )
    result = resolve_tabs(content)
    assert "Tab: x86-64" in result
    assert "Content for x86" in result
    assert "Tab: ARM64" in result
    assert "{{" not in result


# --- resolve_notes (non-standard closing tag) ---

def test_note_compact_closing_tag():
    """{{</ note >}} (space after </) should be handled like {{< /note >}}."""
    content = "{{< note >}}\nImportant info.\n{{</ note >}}"
    result = resolve_notes(content)
    assert "NOTE: Important info." in result
    assert "{{" not in result


def test_caution_compact_closing_tag():
    content = "{{< caution >}}\nBe careful.\n{{</ caution >}}"
    result = resolve_notes(content)
    assert "CAUTION: Be careful." in result
    assert "{{" not in result


# --- resolve_alerts ---

def test_alert_with_title():
    content = '{{% alert title="Removed feature" color="warning" %}}\nThis was removed.\n{{% /alert %}}'
    result = resolve_alerts(content)
    assert "ALERT (Removed feature): This was removed." in result
    assert "{{" not in result


def test_alert_without_title():
    content = '{{% alert %}}\nSome note here.\n{{% /alert %}}'
    result = resolve_alerts(content)
    assert "ALERT: Some note here." in result
    assert "{{" not in result


def test_alert_title_only():
    content = '{{% alert title="Note" %}}\nNew clusters only.\n{{% /alert %}}'
    result = resolve_alerts(content)
    assert "ALERT (Note): New clusters only." in result


# --- resolve_mermaid ---

def test_mermaid_block_stripped():
    content = "Before.\n{{< mermaid >}}\ngraph TB\n  A --> B\n{{< /mermaid >}}\nAfter."
    result = resolve_mermaid(content)
    assert "graph TB" not in result
    assert "Before." in result
    assert "After." in result
    assert "{{" not in result


def test_mermaid_compact_form_stripped():
    content = "Before.\n{{<mermaid>}}\ngraph BT\n  X --> Y\n{{</mermaid>}}\nAfter."
    result = resolve_mermaid(content)
    assert "graph BT" not in result
    assert "Before." in result
    assert "After." in result


# --- resolve_table ---

def test_table_wrapper_stripped_content_preserved():
    content = (
        '{{< table caption="Pod conditions" >}}\n'
        '| Field | Description |\n'
        '| ----- | ----------- |\n'
        '| type  | Condition type |\n'
        '{{< /table >}}'
    )
    result = resolve_table(content)
    assert "| Field | Description |" in result
    assert "| type  | Condition type |" in result
    assert "{{" not in result
    assert "caption" not in result


# --- resolve_comments ---

def test_comment_block_stripped():
    content = "Real content.\n{{< comment >}}\nTODO: fix this.\n{{< /comment >}}\nMore content."
    result = resolve_comments(content)
    assert "TODO" not in result
    assert "Real content." in result
    assert "More content." in result
    assert "{{" not in result


# --- resolve_version_check ---

def test_version_check_stripped():
    content = "Prerequisites.\n{{< version-check >}}\nNext section."
    result = resolve_version_check(content)
    assert "version-check" not in result
    assert "{{" not in result
    assert "Prerequisites." in result


# --- resolve_relref ---

def test_relref_replaced_with_path():
    content = 'See the {{< relref "/docs/reference/config-api/kubelet-config.v1beta1" >}} for details.'
    result = resolve_relref(content)
    assert "/docs/reference/config-api/kubelet-config.v1beta1" in result
    assert "{{" not in result


# --- resolve_details ---

def test_details_wrapper_stripped_content_preserved():
    content = '{{< details summary="About this architecture" >}}\nThe diagram shows...\n{{< /details >}}'
    result = resolve_details(content)
    assert "The diagram shows..." in result
    assert "{{" not in result
    assert "summary" not in result


# --- resolve_pageinfo ---

def test_pageinfo_wrapper_stripped_content_preserved():
    content = '{{% pageinfo color="primary" %}}\nDashboard is deprecated.\n{{% /pageinfo %}}'
    result = resolve_pageinfo(content)
    assert "Dashboard is deprecated." in result
    assert "{{" not in result


# --- resolve_examples ---

def test_example_replaced_with_inner_text():
    content = '{{< example file="policy/privileged-psp.yaml" >}}privileged PSP{{< /example >}}'
    result = resolve_examples(content)
    assert result == "privileged PSP"


# --- resolve_legacy_repos_deprecation ---

def test_legacy_repos_deprecation_stripped():
    content = "Before.\n{{% legacy-repos-deprecation %}}\nAfter."
    result = resolve_legacy_repos_deprecation(content)
    assert "legacy-repos-deprecation" not in result
    assert "Before." in result
    assert "After." in result


# --- resolve_tutorial_shortcodes ---

def test_tutorial_carousel_block_stripped():
    content = (
        "Before.\n"
        '{{< tutorials/carousel id="myCarousel" interval="3000" >}}\n'
        "  Animation content\n"
        "{{< /tutorials/carousel >}}\n"
        "After."
    )
    result = resolve_tutorial_shortcodes(content)
    assert "Animation content" not in result
    assert "Before." in result
    assert "After." in result
    assert "{{" not in result


def test_tutorial_modules_block_stripped():
    content = (
        "Before.\n"
        "{{< tutorials/modules >}}\n"
        "  Nav content\n"
        "{{< /tutorials/modules >}}\n"
        "After."
    )
    result = resolve_tutorial_shortcodes(content)
    assert "Nav content" not in result
    assert "Before." in result
    assert "After." in result


# --- find_unhandled_shortcodes (fixed regex) ---

def test_find_unhandled_compact_form_no_false_suffix():
    """{{<mermaid>}} should report 'mermaid', not 'mermaid>}}'."""
    content = "{{<mermaid>}}\ngraph TB\n  A-->B\n{{</mermaid>}}"
    result = find_unhandled_shortcodes(content)
    assert "mermaid>}}" not in result
    assert "mermaid" in result


def test_find_unhandled_compact_closing_slash_space():
    """{{</ note >}} should report 'note', not '/'."""
    content = "{{< note >}}\nText.\n{{</ note >}}"
    result = find_unhandled_shortcodes(content)
    assert "/" not in result
    assert "note" in result


# --- strip functions ---

def test_strip_html_comments():
    content = "<!-- overview -->\nActual content.\n<!-- body -->"
    result = strip_html_comments(content)
    assert "overview" not in result
    assert "Actual content." in result


# --- find_unhandled_shortcodes ---

def test_find_unhandled_shortcodes():
    content = "Some text {{< unknown_shortcode >}} more {{< another >}} text."
    result = find_unhandled_shortcodes(content)
    assert set(result.keys()) == {"unknown_shortcode", "another"}


def test_find_unhandled_shortcodes_clean():
    content = "No shortcodes here."
    assert find_unhandled_shortcodes(content) == Counter()


# --- clean_extra_whitespace ---

def test_collapses_multiple_blank_lines():
    content = "Line one.\n\n\n\n\nLine two."
    assert clean_extra_whitespace(content) == "Line one.\n\nLine two."


# --- resolve_shortcodes (integration) ---

def test_resolve_shortcodes_full_pipeline(tmp_path):
    """Multiple shortcode types in one document resolve correctly."""
    examples_dir = tmp_path / "examples"
    examples_dir.mkdir()
    (examples_dir / "test.yaml").write_text("apiVersion: v1\nkind: Pod\n")

    includes_dir = tmp_path / "includes"
    includes_dir.mkdir()
    (includes_dir / "prereqs.md").write_text("You need a cluster.")

    content = (
        "<!-- overview -->\n"
        "A {{< glossary_tooltip text=\"pod\" term_id=\"pod\" >}} runs "
        "{{< glossary_tooltip term_id=\"container\" >}}.\n\n"
        "{{< note >}}\nImportant info.\n{{< /note >}}\n\n"
        '## {{% heading "prerequisites" %}}\n\n'
        '{{< include "prereqs.md" >}}\n\n'
        '{{% code_sample file="test.yaml" %}}\n\n'
        '{{< feature-state for_k8s_version="v1.28" state="stable" >}}\n\n'
        "In Kubernetes {{< skew currentVersion >}} the API is stable.\n\n"
        'See the {{< api-reference page="workload-resources/pod-v1" >}}.\n'
    )

    result, unhandled = resolve_shortcodes(content, examples_dir, includes_dir, "v1.36")

    assert "pod" in result
    assert "container" in result
    assert "NOTE:" in result
    assert "## Before you begin" in result
    assert "You need a cluster." in result
    assert "apiVersion: v1" in result
    assert "# Source: test.yaml" in result
    assert "[FEATURE STATE: v1.28 stable]" in result
    assert "In Kubernetes 1.36 the API is stable." in result
    assert "[Pod API reference]" in result
    assert "{{" not in result
    assert "<!--" not in result
    assert unhandled == Counter()
