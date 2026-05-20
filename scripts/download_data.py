"""Download Kubernetes documentation files from the official GitHub repository.

Reads configs/selected_pages.yaml to determine which pages to fetch,
then downloads raw markdown files along with their referenced examples
and includes.
"""

import re
import yaml
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError


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
    return re.findall(r'include\s+"([^"]+)"', content)


def download_pages(config):
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
    referenced_includes = set()
    counts = {"pages": 0, "examples": 0, "includes": 0, "failed": 0}

    # Download doc pages
    for section, pages in config["pages"].items():
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
            referenced_includes.update(scan_includes(content))

    # Download referenced example files
    for example_file in sorted(referenced_examples):
        remote_path = f"{examples_path}/{example_file}"
        local_path = DATA_DIR / "examples" / example_file
        url = build_raw_url(repo_url, branch, remote_path)

        print(f"Fetching example: {example_file}")
        content = fetch_file(url)

        if content is None:
            counts["failed"] += 1
            continue

        save_file(content, local_path)
        counts["examples"] += 1

    # Download referenced include files
    for include_file in sorted(referenced_includes):
        remote_path = f"{includes_path}/{include_file}"
        local_path = DATA_DIR / "includes" / include_file
        url = build_raw_url(repo_url, branch, remote_path)

        print(f"Fetching include: {include_file}")
        content = fetch_file(url)

        if content is None:
            counts["failed"] += 1
            continue

        save_file(content, local_path)
        counts["includes"] += 1

    return counts


def main():
    print(f"Loading config from {CONFIG_PATH}")
    config = load_config(CONFIG_PATH)
    print(f"Source: {config['source_repo']} ({config['source_branch']})")
    print(f"K8s version: {config['k8s_version']}\n")

    counts = download_pages(config)

    print(f"\nDone: {counts['pages']} pages, {counts['examples']} examples, "
          f"{counts['includes']} includes, {counts['failed']} failed")


if __name__ == "__main__":
    main()