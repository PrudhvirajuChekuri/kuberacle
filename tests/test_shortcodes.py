"""Tests for the shortcode resolution module."""

from k8s_rag.preprocessing.shortcodes import (
    resolve_glossary_tooltips,
    resolve_notes,
    resolve_code_samples,
    resolve_includes,
    resolve_headings,
    resolve_feature_states,
    resolve_tabs,
    resolve_params,
    resolve_figures,
    strip_skew,
    strip_api_reference,
    strip_html_comments,
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


# --- resolve_notes ---

def test_note_block():
    content = "{{< note >}}\nThis is important.\n{{< /note >}}"
    result = resolve_notes(content)
    assert "> **NOTE:**" in result
    assert "This is important." in result


def test_caution_block():
    content = "{{< caution >}}\nBe careful here.\n{{< /caution >}}"
    result = resolve_notes(content)
    assert "> **CAUTION:**" in result


def test_warning_block():
    content = "{{< warning >}}\nDanger ahead.\n{{< /warning >}}"
    result = resolve_notes(content)
    assert "> **WARNING:**" in result


def test_note_block_multiline():
    content = "{{< note >}}\nLine one.\nLine two.\nLine three.\n{{< /note >}}"
    result = resolve_notes(content)
    assert "Line one." in result
    assert "Line three." in result
    # Every line should be in a blockquote
    for line in result.strip().split("\n"):
        assert line.startswith(">")


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


def test_code_sample_missing_file(tmp_path):
    content = '{{% code_sample file="missing/file.yaml" %}}'
    result = resolve_code_samples(content, tmp_path)
    assert "not found" in result


# --- resolve_includes ---

def test_include_inlines_content(tmp_path):
    include_file = tmp_path / "prereqs.md"
    include_file.write_text("You need a Kubernetes cluster.")

    content = '{{< include "prereqs.md" >}}'
    result = resolve_includes(content, tmp_path)
    assert result == "You need a Kubernetes cluster."


def test_include_missing_file(tmp_path):
    content = '{{< include "missing.md" >}}'
    result = resolve_includes(content, tmp_path)
    assert "not found" in result


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
    assert "**Tab: Option A**" in result
    assert "Content A" in result
    assert "**Tab: Option B**" in result
    assert "Content B" in result
    assert "{{" not in result


# --- resolve_params ---

def test_param_version():
    content = 'Kubernetes {{< param "version" >}} docs'
    assert resolve_params(content) == "Kubernetes v1.36 docs"


def test_param_unknown_returns_key():
    content = '{{< param "unknown_key" >}}'
    assert resolve_params(content) == "unknown_key"


# --- resolve_figures ---

def test_figure_with_title_and_caption():
    content = '{{< figure src="/img.svg" title="Figure 1." caption="A pod diagram." >}}'
    result = resolve_figures(content)
    assert "**Figure 1.**" in result
    assert "A pod diagram." in result


def test_figure_no_caption_no_title():
    content = '{{< figure src="/img.svg" alt="diagram" class="medium" >}}'
    assert resolve_figures(content) == ""


# --- strip functions ---

def test_strip_skew():
    content = "Version {{< skew currentVersion >}} is required."
    assert strip_skew(content) == "Version  is required."


def test_strip_api_reference():
    content = '{{< api-reference page="workload-resources/pod-v1" >}}'
    assert strip_api_reference(content) == ""


def test_strip_html_comments():
    content = "<!-- overview -->\nActual content.\n<!-- body -->"
    result = strip_html_comments(content)
    assert "overview" not in result
    assert "Actual content." in result


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
        '{{< feature-state for_k8s_version="v1.28" state="stable" >}}\n'
    )

    result = resolve_shortcodes(content, examples_dir, includes_dir)

    assert "pod" in result
    assert "container" in result
    assert "> **NOTE:**" in result
    assert "## Before you begin" in result
    assert "You need a cluster." in result
    assert "apiVersion: v1" in result
    assert "[FEATURE STATE: v1.28 stable]" in result
    assert "{{" not in result
    assert "<!--" not in result