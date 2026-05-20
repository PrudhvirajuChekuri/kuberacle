"""Parse YAML frontmatter and derive metadata from Kubernetes doc files.

Splits a raw markdown file into its YAML frontmatter (parsed into a dict)
and the remaining markdown body. Derives additional metadata from the
file's path within the docs directory structure.
"""

import yaml
from pathlib import Path


# Maps directory names to content type labels
SECTION_TO_CONTENT_TYPE = {
    "concepts": "concept",
    "tasks": "task",
    "tutorials": "tutorial",
}


def parse_frontmatter(content):
    """Split a markdown file into its YAML frontmatter and body.

    Expects the file to start with '---', followed by YAML, followed
    by a closing '---'. Everything after the closing delimiter is the body.

    Args:
        content: Raw markdown file content as a string.

    Returns:
        Tuple of (frontmatter_dict, body_string). Returns an empty dict
        for frontmatter if none is found.

    Raises:
        yaml.YAMLError: If the frontmatter contains invalid YAML.
    """
    if not content.startswith("---"):
        return {}, content

    # Find the closing '---' (skip the opening one)
    closing_index = content.index("---", 3)
    yaml_block = content[3:closing_index]
    body = content[closing_index + 3:].lstrip("\n")

    frontmatter = yaml.safe_load(yaml_block) or {}
    return frontmatter, body


def derive_metadata(file_path, k8s_version="v1.36"):
    """Derive metadata from a doc file's path within the data/raw/ directory.

    The path structure (e.g., concepts/workloads/pods/_index.md) encodes
    the content type, section hierarchy, and maps to a kubernetes.io URL.

    Args:
        file_path: Path to the file relative to data/raw/
            (e.g., "concepts/workloads/pods/_index.md").
        k8s_version: Kubernetes docs version string.

    Returns:
        Dict with derived metadata: content_type, section_path,
        source_url, and k8s_version.
    """
    parts = Path(file_path).parts

    # First directory is the section (concepts, tasks, tutorials)
    section = parts[0]
    content_type = SECTION_TO_CONTENT_TYPE.get(section, section)

    # Section path is everything except the filename
    section_path = list(parts[:-1])

    # Build the kubernetes.io URL
    # _index.md -> parent directory URL, other files -> filename without .md
    filename = parts[-1]
    if filename == "_index.md":
        url_path = "/".join(parts[:-1])
    else:
        url_path = "/".join(parts[:-1]) + "/" + filename.replace(".md", "")
    source_url = f"https://kubernetes.io/docs/{url_path}/"

    return {
        "file_path": file_path,
        "content_type": content_type,
        "section_path": section_path,
        "source_url": source_url,
        "k8s_version": k8s_version,
    }


def extract_metadata(content, file_path, k8s_version="v1.36"):
    """Parse frontmatter and combine with path-derived metadata.

    This is the main entry point for this module. It combines the YAML
    frontmatter from the file content with metadata derived from the
    file's location in the directory structure.

    Args:
        content: Raw markdown file content as a string.
        file_path: Path to the file relative to data/raw/
            (e.g., "concepts/workloads/pods/_index.md").
        k8s_version: Kubernetes docs version string.

    Returns:
        Tuple of (metadata_dict, body_string) where metadata_dict
        contains both parsed frontmatter fields and derived fields.
    """
    frontmatter, body = parse_frontmatter(content)
    derived = derive_metadata(file_path, k8s_version)

    # Derived fields are added alongside frontmatter fields.
    # Derived fields use distinct names so there's no collision.
    metadata = {**frontmatter, **derived}

    return metadata, body