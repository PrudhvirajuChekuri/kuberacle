"""Shared helpers for the file-inlining shortcode resolvers."""

from pathlib import Path

SLASH_COMMENT_LANGS = {
    "go", "javascript", "js", "typescript", "ts",
    "java", "c", "cpp", "csharp", "rust", "swift", "kotlin",
}


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
