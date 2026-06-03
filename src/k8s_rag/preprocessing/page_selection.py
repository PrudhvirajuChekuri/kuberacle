"""Resolve page selection from static lists or repository discovery."""

import json
import ssl
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _owner_repo_from_url(repo_url: str) -> str:
    """Extract owner/repository from a GitHub repository URL."""
    return repo_url.replace("https://github.com/", "").strip("/")


def _fetch_repo_tree(repo_url: str, branch: str) -> list[dict]:
    """Fetch recursive git tree entries from GitHub API."""
    owner_repo = _owner_repo_from_url(repo_url)
    url = f"https://api.github.com/repos/{owner_repo}/git/trees/{branch}?recursive=1"
    retries = 3
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        request = Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "k8s-docs-rag",
            },
        )
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except HTTPError as exc:
            # Don't retry on explicit HTTP responses like 404/401.
            raise RuntimeError(
                f"Failed to fetch repo tree (HTTP {exc.code}) from {url}"
            ) from exc
        except (URLError, ssl.SSLError) as exc:
            last_error = exc
            if attempt == retries:
                raise RuntimeError(
                    f"Failed to fetch repo tree after {retries} attempts ({exc}) from {url}"
                ) from exc
            sleep_seconds = 2 ** (attempt - 1)
            print(
                f"GitHub tree fetch attempt {attempt}/{retries} failed ({exc}); "
                f"retrying in {sleep_seconds}s..."
            )
            time.sleep(sleep_seconds)

    tree = payload.get("tree")
    if not isinstance(tree, list):
        raise RuntimeError(f"Unexpected tree response from GitHub API: {url}")
    return tree


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
