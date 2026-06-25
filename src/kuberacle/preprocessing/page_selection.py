"""Resolve page selection from static lists or repository discovery."""

import json
import logging
import os
import ssl
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


def _owner_repo_from_url(repo_url: str) -> str:
    """Extract owner/repository from a GitHub repository URL.

    Args:
        repo_url: Full HTTPS GitHub URL
            (e.g., "https://github.com/kubernetes/website").

    Returns:
        Owner/repo string like "kubernetes/website".

    Raises:
        ValueError: If the URL is not an https://github.com/ URL.
    """
    prefix = "https://github.com/"
    if not repo_url.startswith(prefix):
        raise ValueError(
            f"Expected a GitHub HTTPS URL starting with {prefix!r}, "
            f"got {repo_url!r}"
        )
    return repo_url.removeprefix(prefix).strip("/")


def _github_get_json(url: str) -> dict:
    """Fetch and decode a JSON response from the GitHub API with retries.

    Adds the optional ``GITHUB_TOKEN`` bearer auth. Does not retry on explicit
    HTTP responses (for example 404/401), which are definitive.

    Args:
        url: Full GitHub API URL.

    Returns:
        Decoded JSON payload.

    Raises:
        RuntimeError: On an HTTP error or after exhausting network retries.
    """
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "kuberacle",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    retries = 3
    for attempt in range(1, retries + 1):
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(
                f"GitHub API request failed (HTTP {exc.code}) for {url}"
            ) from exc
        except (URLError, ssl.SSLError) as exc:
            if attempt == retries:
                raise RuntimeError(
                    f"GitHub API request failed after {retries} attempts ({exc}) for {url}"
                ) from exc
            sleep_seconds = 2 ** (attempt - 1)
            logger.warning(
                "GitHub API attempt %d/%d failed (%s); retrying in %ds",
                attempt, retries, exc, sleep_seconds,
            )
            time.sleep(sleep_seconds)
    raise RuntimeError(f"GitHub API request failed for {url}")  # pragma: no cover


def _fetch_repo_tree(repo_url: str, branch: str) -> list[dict]:
    """Fetch recursive git tree entries from GitHub API."""
    owner_repo = _owner_repo_from_url(repo_url)
    url = f"https://api.github.com/repos/{owner_repo}/git/trees/{branch}?recursive=1"
    payload = _github_get_json(url)

    if payload.get("truncated"):
        raise RuntimeError(
            f"GitHub tree response was truncated for {url}. "
            "The repository may have too many entries for a single recursive "
            "tree request. Consider using list mode instead."
        )

    tree = payload.get("tree")
    if not isinstance(tree, list):
        raise RuntimeError(f"Unexpected tree response from GitHub API: {url}")
    return tree


def fetch_blob_shas(repo_url: str, branch: str) -> dict[str, str]:
    """Return a map of repo-relative file path to its git blob SHA.

    Fetches the recursive git tree once and retains the blob SHAs, which give a
    cheap content fingerprint for every file in the repository without
    downloading any file contents. Used to record source provenance for the
    index and to detect upstream changes.

    The ``branch`` argument is any tree-ish accepted by GitHub's trees endpoint:
    a branch ref (as discover mode uses) or a commit SHA (as download-data pins
    to). GitHub resolves a commit SHA to its root tree; the returned blob SHAs
    are content-addressed, so they are identical either way (verified live).

    Args:
        repo_url: Full HTTPS GitHub repository URL.
        branch: Branch ref or commit SHA to read the tree at.

    Returns:
        Mapping of blob path to its git blob SHA (tree entries only).
    """
    return {
        str(entry["path"]): str(entry["sha"])
        for entry in _fetch_repo_tree(repo_url, branch)
        if entry.get("type") == "blob" and entry.get("path") and entry.get("sha")
    }


def fetch_head_commit(repo_url: str, branch: str) -> str:
    """Fetch the current HEAD commit SHA of a branch via the GitHub API.

    Recorded for human traceability and rollback labeling. Change detection is
    driven by content fingerprints, not this value.

    Args:
        repo_url: Full HTTPS GitHub repository URL.
        branch: Git branch name.

    Returns:
        The commit SHA at the tip of the branch.

    Raises:
        RuntimeError: If the response does not contain a commit SHA.
    """
    owner_repo = _owner_repo_from_url(repo_url)
    url = f"https://api.github.com/repos/{owner_repo}/commits/{branch}"
    payload = _github_get_json(url)
    sha = payload.get("sha")
    if not isinstance(sha, str) or not sha:
        raise RuntimeError(f"No commit SHA in GitHub response for {url}")
    return sha


def discover_pages(
    repo_url: str,
    branch: str,
    docs_path: str,
    sections: list[str],
    limit: int | None = None,
) -> dict[str, list[str]]:
    """Discover markdown files under selected docs sections."""
    normalized_docs_path = docs_path.strip("/")
    section_prefixes = {
        section: f"{normalized_docs_path}/{section}/"
        for section in sections
    }
    page_map: dict[str, list[str]] = {section: [] for section in sections}

    for entry in _fetch_repo_tree(repo_url, branch):
        if entry.get("type") != "blob":
            continue
        path = str(entry.get("path", ""))
        if not path.endswith(".md"):
            continue
        for section, prefix in section_prefixes.items():
            if not path.startswith(prefix):
                continue
            rel_path = path.removeprefix(prefix)
            page_map[section].append(rel_path)
            break

    for section in sections:
        page_map[section] = sorted(page_map[section])
        if limit is not None:
            page_map[section] = page_map[section][:limit]
    return page_map


def resolve_pages(
    config: dict,
    mode: str = "auto",
    sections_override: list[str] | None = None,
    limit_override: int | None = None,
) -> dict[str, list[str]]:
    """Resolve section-to-pages map from config and runtime flags."""
    selection = config.get("selection", {})
    configured_mode = str(selection.get("mode", "list"))
    effective_mode = configured_mode if mode == "auto" else mode

    if effective_mode == "list":
        pages = config.get("pages")
        if not isinstance(pages, dict) or not pages:
            raise ValueError("List mode requires `pages` in config.")
        if sections_override:
            for section in sections_override:
                if section not in pages:
                    logger.warning("Requested section %r not found in config pages", section)
            return {
                section: list(pages.get(section, []))
                for section in sections_override
            }
        return {section: list(values) for section, values in pages.items()}

    if effective_mode != "discover":
        raise ValueError(f"Unsupported selection mode: {effective_mode}")

    selected_sections = (
        sections_override
        or selection.get("sections")
        or list(config.get("pages", {}).keys())
        or ["concepts", "tasks", "tutorials"]
    )
    if not selected_sections:
        raise ValueError("Discover mode requires at least one section.")

    configured_limit = selection.get("limit")
    effective_limit = (
        limit_override
        if limit_override is not None
        else (int(configured_limit) if configured_limit is not None else None)
    )
    return discover_pages(
        repo_url=config["source_repo"],
        branch=config["source_branch"],
        docs_path=config["docs_path"],
        sections=[str(section) for section in selected_sections],
        limit=effective_limit,
    )
