"""File-inlining shortcode resolvers (run first).

Inline external file content referenced by ``code_sample``, ``include``, and
``glossary_definition`` shortcodes.
"""

import logging
import re
from pathlib import Path

import yaml

from kuberacle.preprocessing.shortcodes._common import (
    _safe_relative_path,
    _source_comment,
)

logger = logging.getLogger(__name__)

_INCLUDE_PATTERN = re.compile(r'{{[<%]\s*include\s+"([^"]*)"\s*[>%]}}')


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
