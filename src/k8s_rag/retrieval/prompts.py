"""Prompt loading utilities for versioned RAG prompts."""

from pathlib import Path

import yaml


def load_prompt_bundle(base_dir: str, version: str) -> dict[str, str]:
    """Load answer + citation prompt artifacts for a version.

    Args:
        base_dir: Prompt root directory (e.g. configs/prompts).
        version: Prompt version (e.g. v1).

    Returns:
        Dict with keys `system`, `user`, `citation_rules`.
    """
    version_dir = Path(base_dir) / version
    answer_path = version_dir / "answer.yaml"
    citation_path = version_dir / "citation_enforcement.yaml"

    with open(answer_path, "r") as file:
        answer_cfg = yaml.safe_load(file)
    with open(citation_path, "r") as file:
        citation_cfg = yaml.safe_load(file)

    rules = citation_cfg.get("rules", [])
    citation_rules = "\n".join(f"- {rule}" for rule in rules)
    return {
        "system": str(answer_cfg.get("system", "")),
        "user": str(answer_cfg.get("user", "")),
        "citation_rules": citation_rules,
    }
