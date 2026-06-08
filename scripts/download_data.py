"""Download Kubernetes documentation files from the official GitHub repository."""

import argparse
import logging
import re
import time
import tomllib
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml
from tqdm import tqdm

from k8s_rag.preprocessing.page_selection import resolve_pages, _owner_repo_from_url


# Repository root (assumes script is run from project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)
CONFIG_PATH = PROJECT_ROOT / "configs" / "datasets" / "full.yaml"
DATA_DIR = PROJECT_ROOT / "data"


def load_config(config_path: str | Path) -> dict:
    """Load and return the page selection config.

    Args:
        config_path: Path to the dataset config YAML file.

    Returns:
        Dict containing source repo info and page lists.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_raw_url(repo_url: str, branch: str, path: str) -> str:
    """Convert a GitHub repo URL and file path to a raw content URL.

    Args:
        repo_url: GitHub repo URL (e.g., https://github.com/kubernetes/website).
        branch: Git branch name (e.g., main).
        path: File path within the repo.

    Returns:
        Raw githubusercontent URL for direct file download.
    """
    owner_repo = _owner_repo_from_url(repo_url)
    return f"https://raw.githubusercontent.com/{owner_repo}/{branch}/{path}"


def fetch_file(url: str, quiet: bool = False) -> str | None:
    """Fetch a file's content from a URL with retries.

    Retries up to 3 times on network errors with exponential backoff.
    Does not retry on HTTP errors (e.g., 404) since those are definitive.

    Args:
        url: The URL to fetch.
        quiet: If True, suppress HTTP error output (useful when
            fallback paths will be tried).

    Returns:
        File content as a string, or None if all attempts failed.
    """
    retries = 3
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={"User-Agent": "k8s-docs-rag"})
            with urlopen(request, timeout=15) as response:
                return response.read().decode("utf-8")
        except HTTPError as e:
            log = logger.debug if quiet else logger.warning
            log("HTTP %d: %s", e.code, url)
            return None
        except URLError as e:
            if attempt == retries:
                log = logger.debug if quiet else logger.warning
                log("Network error: %s - %s", e.reason, url)
                return None
            sleep_seconds = 2 ** (attempt - 1)
            logger.warning(
                "Fetch attempt %d/%d failed (%s); retrying in %ds",
                attempt, retries, e.reason, sleep_seconds,
            )
            time.sleep(sleep_seconds)
    return None


def fetch_k8s_version(config: dict) -> str:
    """Fetch the current Kubernetes docs version from hugo.toml in the source repo.

    The Hugo site config is the authoritative source for the version param that
    {{< param "version" >}} and {{< skew currentVersion >}} shortcodes resolve to.
    Fetching it dynamically keeps the pipeline in sync with source_branch: main.

    Args:
        config: Parsed config dict with source_repo and source_branch.

    Returns:
        Version string with leading "v" (e.g., "v1.36").

    Raises:
        RuntimeError: If hugo.toml cannot be fetched or version is not found.
    """
    url = build_raw_url(config["source_repo"], config["source_branch"], "hugo.toml")
    content = fetch_file(url)
    if content is None:
        raise RuntimeError(f"Failed to fetch hugo.toml from {url}")
    hugo_config = tomllib.loads(content)
    version = hugo_config.get("params", {}).get("version", "")
    if not version:
        raise RuntimeError("Could not find params.version in hugo.toml")
    version = str(version)
    if not version.startswith("v"):
        version = f"v{version}"
    return version


def save_file(content: str, path: str | Path) -> None:
    """Save content to a local file, creating directories as needed.

    Args:
        content: String content to write.
        path: Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def scan_code_samples(content: str) -> list[str]:
    """Extract code_sample file references from markdown content.

    Handles both shortcode names (code_sample and code), both delimiter
    styles ({{< >}} and {{% %}}), and any attribute order (language= may
    appear before file=).

    Args:
        content: Raw markdown string.

    Returns:
        List of file paths referenced by code_sample shortcodes.
    """
    return re.findall(r'{{[<%]\s*code(?:_sample)?\s+[^%>]*?file="([^"]+)"', content)


def scan_includes(content: str) -> list[str]:
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


def scan_glossary_definitions(content: str) -> list[str]:
    """Extract term_id references from glossary_definition shortcodes.

    Handles any attribute order (prepend= or length= may appear before term_id=).

    Args:
        content: Raw markdown string.

    Returns:
        List of term IDs referenced by glossary_definition shortcodes.
    """
    return re.findall(r'glossary_definition\b[^>]*?term_id="([^"]+)"', content)


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


def download_pages(config: dict, page_map: dict[str, list[str]]) -> dict[str, int]:
    """Download all pages listed in the config and their dependencies.

    Fetches raw markdown files, then scans each for code_sample, include,
    and glossary_definition references and fetches those too.

    Args:
        config: Parsed config dict from the dataset YAML.
        page_map: Section-to-pages map from resolve_pages().

    Returns:
        Dict with counts of downloaded files by type.
    """
    repo_url = config["source_repo"]
    branch = config["source_branch"]
    docs_path = config["docs_path"]
    examples_path = config["examples_path"]
    includes_path = config["includes_path"]
    glossary_path = config.get("glossary_path", "")

    referenced_examples = set()
    referenced_includes: dict[str, set[str]] = {}
    referenced_glossary_terms: set[str] = set()
    counts = {"pages": 0, "examples": 0, "includes": 0, "glossary": 0, "failed": 0}

    # Download doc pages
    all_pages = [
        (section, page)
        for section, pages in page_map.items()
        for page in pages
    ]
    with tqdm(all_pages, desc="Downloading pages", unit="page") as progress:
        for section, page in progress:
            remote_path = f"{docs_path}/{section}/{page}"
            local_path = DATA_DIR / "raw" / section / page
            url = build_raw_url(repo_url, branch, remote_path)

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
            referenced_glossary_terms.update(scan_glossary_definitions(content))

    # Download referenced example files
    logger.info("Fetching %d referenced examples", len(referenced_examples))
    for example_file in sorted(referenced_examples):
        try:
            normalized_example = normalize_repo_relative_path(example_file, "example")
        except ValueError as exc:
            logger.warning("Invalid example reference: %s", exc)
            counts["failed"] += 1
            continue
        remote_path = f"{examples_path}/{normalized_example}"
        local_path = DATA_DIR / "examples" / normalized_example
        url = build_raw_url(repo_url, branch, remote_path)

        content = fetch_file(url)

        if content is None:
            counts["failed"] += 1
            continue

        save_file(content, local_path)
        counts["examples"] += 1

    # Download referenced include files
    logger.info("Fetching %d referenced includes", len(referenced_includes))
    for include_file in sorted(referenced_includes):
        try:
            normalized_include = normalize_repo_relative_path(include_file, "include")
        except ValueError as exc:
            logger.warning("Invalid include reference: %s", exc)
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
        for remote_path in candidate_remote_paths:
            url = build_raw_url(repo_url, branch, remote_path)
            fetched_content = fetch_file(url, quiet=True)
            if fetched_content is not None:
                break

        if fetched_content is None:
            logger.warning("Not found at any candidate path: %s", normalized_include)
            counts["failed"] += 1
            continue

        save_file(fetched_content, local_path)
        counts["includes"] += 1

    # Download referenced glossary term files
    if glossary_path:
        logger.info("Fetching %d referenced glossary terms", len(referenced_glossary_terms))
        for term_id in sorted(referenced_glossary_terms):
            try:
                normalized_term = normalize_repo_relative_path(
                    f"{term_id}.md", "glossary term"
                )
            except ValueError as exc:
                logger.warning("Invalid glossary term reference: %s", exc)
                counts["failed"] += 1
                continue
            remote_path = f"{glossary_path}/{normalized_term}"
            local_path = DATA_DIR / "glossary" / normalized_term
            url = build_raw_url(repo_url, branch, remote_path)

            content = fetch_file(url)

            if content is None:
                counts["failed"] += 1
                continue

            save_file(content, local_path)
            counts["glossary"] += 1

    return counts


def main() -> None:
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
        help="Optional per-section page cap (useful for partial runs).",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Do not fail when some pages/examples/includes fail to download.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    config_path = Path(args.config)
    logger.info("Loading config from %s", config_path)
    config = load_config(config_path)
    logger.info("Source: %s (%s)", config["source_repo"], config["source_branch"])

    logger.info("Fetching k8s version from source repo...")
    k8s_version = fetch_k8s_version(config)
    logger.info("K8s version: %s", k8s_version)
    save_file(k8s_version, DATA_DIR / "k8s_version.txt")

    sections = args.sections.split(",") if args.sections else None
    page_map = resolve_pages(
        config=config,
        mode=args.mode,
        sections_override=sections,
        limit_override=args.limit,
    )
    total_pages = sum(len(values) for values in page_map.values())
    logger.info("Resolved pages: %d", total_pages)

    counts = download_pages(config, page_map)

    logger.info(
        "Done: %d pages, %d examples, %d includes, %d glossary terms, %d failed",
        counts["pages"], counts["examples"], counts["includes"],
        counts["glossary"], counts["failed"],
    )
    if counts["failed"] > 0 and not args.allow_partial:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
