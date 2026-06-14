"""Tests for the structure analysis module."""

from kuberacle.preprocessing.structure import (
    estimate_tokens,
    classify_code_block,
    analyze_structure,
)


# --- estimate_tokens ---

def test_estimate_tokens_rough_accuracy():
    text = "This is a ten word sentence with some extra words."
    tokens = estimate_tokens(text)
    # 10 words * 1.3 = 13
    assert 10 <= tokens <= 16


def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0


# --- classify_code_block ---

def test_classify_yaml_manifest():
    assert classify_code_block("yaml", "apiVersion: v1\nkind: Pod") == "yaml-manifest"


def test_classify_yaml_without_api_version():
    assert classify_code_block("yaml", "maxContainerRestartPeriod: 100s") == "yaml"


def test_classify_kubectl_command():
    assert classify_code_block("shell", "kubectl get pods -n default") == "kubectl-command"


def test_classify_plain_shell():
    assert classify_code_block("shell", "echo hello") == "shell"


def test_classify_unknown_language():
    assert classify_code_block("go", "func main() {}") == "go"


def test_classify_no_language():
    assert classify_code_block("", "some text") == "text"


# --- analyze_structure ---

def test_detects_headings():
    content = "# Title\n\nSome text.\n\n## Section A\n\nMore text."
    result = analyze_structure(content)
    assert len(result["headings"]) == 2
    assert result["headings"][0]["level"] == 1
    assert result["headings"][0]["text"] == "Title"
    assert result["headings"][1]["level"] == 2
    assert result["headings"][1]["text"] == "Section A"


def test_extracts_heading_anchors():
    content = "## Pod lifetime {#pod-lifetime}\n\nSome text."
    result = analyze_structure(content)
    assert result["headings"][0]["anchor"] == "pod-lifetime"
    assert result["headings"][0]["text"] == "Pod lifetime"


def test_ignores_headings_inside_code_blocks():
    content = "## Real heading\n\n```yaml\n# This is a YAML comment\nkind: Pod\n```\n\nMore text."
    result = analyze_structure(content)
    assert len(result["headings"]) == 1
    assert result["headings"][0]["text"] == "Real heading"


def test_detects_code_blocks():
    content = "Text.\n\n```yaml\napiVersion: v1\nkind: Pod\n```\n\nMore text."
    result = analyze_structure(content)
    assert len(result["code_blocks"]) == 1
    assert result["code_blocks"][0]["code_type"] == "yaml-manifest"
    assert result["code_blocks"][0]["language"] == "yaml"


def test_detects_tables():
    content = "Text.\n\n| Col A | Col B |\n|-------|-------|\n| 1 | 2 |\n\nMore text."
    result = analyze_structure(content)
    assert len(result["tables"]) == 1


def test_table_not_detected_inside_code_block():
    content = "```\n| A | B |\n|---|---|\n| 1 | 2 |\n```"
    result = analyze_structure(content)
    assert len(result["tables"]) == 0


def test_pipe_in_prose_not_detected_as_table():
    """A sentence with stray pipes must not be treated as a table."""
    content = (
        "## Section\n\n"
        "Run `kubectl get pods | grep web` to filter the output.\n\n"
        "More prose follows."
    )
    result = analyze_structure(content)
    assert len(result["tables"]) == 0


def test_pipe_in_inline_code_not_detected():
    """Backticked pipe characters in markdown must not start a table."""
    content = (
        "## Section\n\n"
        "Use the `|` operator to combine filters.\n\n"
        "Another prose line."
    )
    result = analyze_structure(content)
    assert len(result["tables"]) == 0


def test_table_with_alignment_separator_detected():
    """Separator rows with alignment colons still register as tables."""
    content = (
        "## Section\n\n"
        "| Left | Right |\n"
        "| :--- | ---: |\n"
        "| a    | b     |\n"
    )
    result = analyze_structure(content)
    assert len(result["tables"]) == 1


def test_sections_have_token_counts():
    content = "## Section A\n\nSome content here.\n\n## Section B\n\nMore content."
    result = analyze_structure(content)
    for section in result["sections"]:
        assert "token_count" in section
        assert section["token_count"] >= 0


def test_section_flags_code_and_tables():
    content = (
        "## With code\n\n```yaml\napiVersion: v1\nkind: Pod\n```\n\n"
        "## With table\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\n"
        "## Plain\n\nJust text."
    )
    result = analyze_structure(content)

    code_section = [s for s in result["sections"] if s["heading_text"] == "With code"][0]
    assert code_section["has_code"] is True
    assert "yaml-manifest" in code_section["code_types"]

    table_section = [s for s in result["sections"] if s["heading_text"] == "With table"][0]
    assert table_section["has_table"] is True

    plain_section = [s for s in result["sections"] if s["heading_text"] == "Plain"][0]
    assert plain_section["has_code"] is False
    assert plain_section["has_table"] is False


def test_intro_section_before_first_heading():
    content = "Some intro text.\n\n## First heading\n\nContent."
    result = analyze_structure(content)
    assert result["sections"][0]["heading_text"] == "(intro)"
    assert result["sections"][0]["heading_level"] == 0


def test_no_intro_section_when_heading_first():
    content = "## First heading\n\nContent."
    result = analyze_structure(content)
    assert result["sections"][0]["heading_text"] == "First heading"


def test_unclosed_code_fence_warning(caplog):
    """An unclosed code fence should warn and record the block to end of doc."""
    import logging
    content = "## Section\n\n```yaml\napiVersion: v1\nkind: Pod"
    with caplog.at_level(logging.WARNING):
        result = analyze_structure(content)
    assert len(result["code_blocks"]) == 1
    assert result["code_blocks"][0]["code_type"] == "yaml-manifest"
    assert result["code_blocks"][0]["end_line"] == len(content.split("\n")) - 1
    assert any("unclosed" in r.message.lower() for r in caplog.records)