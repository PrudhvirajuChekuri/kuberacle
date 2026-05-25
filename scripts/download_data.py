"""Download Kubernetes documentation files from the official GitHub repository."""

import argparse
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml

from k8s_rag.preprocessing.page_selection import resolve_pages


# Repository root (assumes script is run from project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "selected_pages.yaml"
DATA_DIR = PROJECT_ROOT / "data"


def load_config(config_path):
    """Load and return the page selection config.

    Args:
        config_path: Path to the selected_pages.yaml file.

    Returns:
        Dict containing source repo info and page lists.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_raw_url(repo_url, branch, path):
    """Convert a GitHub repo URL and file path to a raw content URL.

    Args:
        repo_url: GitHub repo URL (e.g., https://github.com/kubernetes/website).
        branch: Git branch name (e.g., main).
        path: File path within the repo.

    Returns:
        Raw githubusercontent URL for direct file download.
    """
    # https://github.com/kubernetes/website -> kubernetes/website
    owner_repo = repo_url.replace("https://github.com/", "")
    return f"https://raw.githubusercontent.com/{owner_repo}/{branch}/{path}"


def fetch_file(url):
    """Fetch a file's content from a URL.

    Args:
        url: The URL to fetch.

    Returns:
        File content as a string, or None if the fetch failed.
    """
    try:
        request = Request(url, headers={"User-Agent": "k8s-docs-rag"})
        with urlopen(request, timeout=15) as response:
            return response.read().decode("utf-8")
    except HTTPError as e:
        print(f"  HTTP {e.code}: {url}")
        return None
    except URLError as e:
        print(f"  Network error: {e.reason} — {url}")
        return None


def save_file(content, path):
    """Save content to a local file, creating directories as needed.

    Args:
        content: String content to write.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def scan_code_samples(content):
    """Extract code_sample file references from markdown content.

    Finds patterns like {{% code_sample file="pods/simple-pod.yaml" %}}
    and returns the file paths.

    Args:
        content: Raw markdown string.

    Returns:
        List of file paths referenced by code_sample shortcodes.
    """
    return re.findall(r'code_sample\s+file="([^"]+)"', content)


def scan_includes(content):
    """Extract include file references from markdown content.

    Finds patterns like {{< include "task-tutorial-prereqs.md" >}}
    and returns the filenames.

    Args:
        content: Raw markdown string.

    Returns:
        List of filenames referenced by include shortcodes.
    """
    pattern = r'{{[<%]\s*include\s+"([^"]+)"\s*[>%]}}'
    return re.findall(pattern, content)


def normalize_repo_relative_path(path: str, path_type: str) -> str:
    """Normalize a repo-relative path from shortcode references.

    Shortcodes may reference files with a leading slash (for example
    `/controllers/example.yaml`). Treat those as repo-relative, and reject
    traversal paths.

    Args:
        path: Raw path extracted from markdown shortcode.
        path_type: Human-readable label for error messages.

    Returns:
        Clean relative POSIX path suitable for URL and local joins.
    """
    cleaned = path.strip().lstrip("/")
    if not cleaned:
        raise ValueError(f"Empty {path_type} path reference.")
    candidate = Path(cleaned)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Unsafe {path_type} path reference: {path}")
    return cleaned


def download_pages(config, page_map):
    """Download all pages listed in the config and their dependencies.

    Fetches raw markdown files, then scans each for code_sample and
    include references and fetches those too.

    Args:
        config: Parsed config dict from selected_pages.yaml.

    Returns:
        Dict with counts of downloaded files by type.
    """
    repo_url = config["source_repo"]
    branch = config["source_branch"]
    docs_path = config["docs_path"]
    examples_path = config["examples_path"]
    includes_path = config["includes_path"]

    referenced_examples = set()
    referenced_includes: dict[str, set[str]] = {}
    counts = {"pages": 0, "examples": 0, "includes": 0, "failed": 0}

    # Download doc pages
    for section, pages in page_map.items():
        for page in pages:
            remote_path = f"{docs_path}/{section}/{page}"
            local_path = DATA_DIR / "raw" / section / page
            url = build_raw_url(repo_url, branch, remote_path)

            print(f"Fetching {section}/{page}")
            content = fetch_file(url)

            if content is None:
                counts["failed"] += 1
                continue

            save_file(content, local_path)
            counts["pages"] += 1

            # Scan for dependencies
            referenced_examples.update(scan_code_samples(content))
            source_dir = str(Path(section) / Path(page).parent)
            for include_ref in scan_includes(content):
                referenced_includes.setdefault(include_ref, set()).add(source_dir)

    # Download referenced example files
    for example_file in sorted(referenced_examples):
        try:
            normalized_example = normalize_repo_relative_path(example_file, "example")
        except ValueError as exc:
            print(f"  Invalid example reference: {exc}")
            counts["failed"] += 1
            continue
        remote_path = f"{examples_path}/{normalized_example}"
        local_path = DATA_DIR / "examples" / normalized_example
        url = build_raw_url(repo_url, branch, remote_path)

        print(f"Fetching example: {normalized_example}")
        content = fetch_file(url)

        if content is None:
            counts["failed"] += 1
            continue

        save_file(content, local_path)
        counts["examples"] += 1

    # Download referenced include files
    for include_file in sorted(referenced_includes):
        try:
            normalized_include = normalize_repo_relative_path(include_file, "include")
        except ValueError as exc:
            print(f"  Invalid include reference: {exc}")
            counts["failed"] += 1
            continue
        source_dirs = sorted(referenced_includes.get(include_file, set()))
        candidate_remote_paths = [f"{includes_path}/{normalized_include}"]
        for source_dir in source_dirs:
            if source_dir in ("", "."):
                continue
            candidate_remote_paths.append(
                f"{docs_path}/{source_dir}/{normalized_include}"
            )
        candidate_remote_paths.append(f"{docs_path}/{normalized_include}")

        local_path = DATA_DIR / "includes" / normalized_include
        fetched_content = None
        print(f"Fetching include: {normalized_include}")
        for remote_path in candidate_remote_paths:
            url = build_raw_url(repo_url, branch, remote_path)
            fetched_content = fetch_file(url)
            if fetched_content is not None:
                break

        if fetched_content is None:
            counts["failed"] += 1
            continue

        save_file(fetched_content, local_path)
        counts["includes"] += 1

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Download selected Kubernetes docs pages and dependencies."
    )
    parser.add_argument(
        "--config",
        default=str(CONFIG_PATH),
        help="Path to dataset selection YAML.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "list", "discover"],
        default="auto",
        help="Page selection mode. `auto` uses config `selection.mode`.",
    )
    parser.add_argument(
        "--sections",
        default=None,
        help="Comma-separated sections (e.g. concepts,tasks,tutorials).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional per-section page cap (useful for smoke runs).",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Do not fail when some pages/examples/includes fail to download.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    print(f"Loading config from {config_path}")
    config = load_config(config_path)
    print(f"Source: {config['source_repo']} ({config['source_branch']})")
    print(f"K8s version: {config['k8s_version']}")
    sections = args.sections.split(",") if args.sections else None
    page_map = resolve_pages(
        config=config,
        mode=args.mode,
        sections_override=sections,
        limit_override=args.limit,
    )
    total_pages = sum(len(values) for values in page_map.values())
    print(f"Resolved pages: {total_pages}\n")

    counts = download_pages(config, page_map)

    print(f"\nDone: {counts['pages']} pages, {counts['examples']} examples, "
          f"{counts['includes']} includes, {counts['failed']} failed")
    if counts["failed"] > 0 and not args.allow_partial:
        raise SystemExit(1)


if __name__ == "__main__":
    main()