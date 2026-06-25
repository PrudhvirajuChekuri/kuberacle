"""Detect whether the published index is stale vs upstream docs or local code.

Compares the published index manifest's ``source_fingerprint`` and
``build_fingerprint`` against freshly computed values. The check is cheap: it
reads the upstream git tree once (blob SHAs, no file contents) and hashes local
index-build code. It mutates nothing - it writes a ``changed`` decision to
``GITHUB_OUTPUT`` so the scheduled workflow can decide whether to rebuild.

The recomputed source fingerprint must cover the same file set the publisher
recorded: the primary docs pages under the configured sections (rediscovered
from the current tree, so additions/removals are caught) plus the dependency
files (examples/includes/glossary/hugo.toml) recorded in the last manifest. A
newly referenced dependency only appears after the page that references it
changes, and that page's blob SHA changing already triggers a rebuild that
rediscovers it.
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass
from kuberacle.cli._root import project_root

import yaml
from dotenv import load_dotenv

load_dotenv(project_root() / ".env")

from kuberacle.config import load_rag_config
from kuberacle.index_sync import LATEST_MANIFEST
from kuberacle.preprocessing.page_selection import fetch_blob_shas, fetch_head_commit
from kuberacle.provenance import build_fingerprint, build_index_config, source_fingerprint


PROJECT_ROOT = project_root()
RAG_CONFIG_PATH = PROJECT_ROOT / "configs" / "rag.yaml"
DATASET_CONFIG_PATH = PROJECT_ROOT / "configs" / "datasets" / "full.yaml"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StalenessResult:
    """Outcome of comparing the published index against current inputs."""

    changed: bool
    source_changed: bool
    build_changed: bool
    reason: str


def evaluate_staleness(
    manifest: dict | None,
    blob_shas: dict[str, str],
    docs_path: str,
    sections: list[str],
    build_fingerprint_now: str,
) -> StalenessResult:
    """Decide whether the published index is stale.

    Args:
        manifest: Published index manifest, or ``None`` if none exists.
        blob_shas: Current upstream path -> blob SHA map.
        docs_path: Repo path under which docs sections live.
        sections: Configured doc sections (concepts/tasks/tutorials).
        build_fingerprint_now: Freshly computed local build fingerprint.

    Returns:
        The staleness decision and a human-readable reason.
    """
    if not manifest:
        return StalenessResult(True, True, True, "no published index manifest")

    prefixes = tuple(f"{docs_path.strip('/')}/{section}/" for section in sections)
    primary = {
        path for path in blob_shas
        if path.endswith(".md") and path.startswith(prefixes)
    }
    published_files = (manifest.get("source_files") or {}).get("files", {})
    dependencies = {
        path for path, meta in published_files.items() if meta.get("kind") != "page"
    }

    files_now = {path: {"sha": blob_shas.get(path, "")} for path in primary | dependencies}
    source_fp_now = source_fingerprint(files_now)

    source_changed = source_fp_now != manifest.get("source_fingerprint")
    build_changed = build_fingerprint_now != manifest.get("build_fingerprint")

    reasons = []
    if source_changed:
        reasons.append("upstream docs changed")
    if build_changed:
        reasons.append("index-build code/config changed")
    reason = "; ".join(reasons) if reasons else "up to date"

    return StalenessResult(
        changed=source_changed or build_changed,
        source_changed=source_changed,
        build_changed=build_changed,
        reason=reason,
    )


def _read_manifest(bucket_name: str) -> dict | None:
    """Read the published 'latest' manifest from GCS, or None if absent."""
    from google.cloud import storage

    blob = storage.Client().bucket(bucket_name).blob(LATEST_MANIFEST)
    if not blob.exists():
        return None
    return json.loads(blob.download_as_text())


def _write_github_output(result: StalenessResult, commit: str) -> None:
    """Emit the decision to GITHUB_OUTPUT when running under Actions."""
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as handle:
        handle.write(f"changed={'true' if result.changed else 'false'}\n")
        handle.write(f"source_changed={'true' if result.source_changed else 'false'}\n")
        handle.write(f"build_changed={'true' if result.build_changed else 'false'}\n")
        handle.write(f"reason={result.reason}\n")
        handle.write(f"upstream_commit={commit}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect whether the published index is stale."
    )
    parser.add_argument("--bucket", required=True, help="GCS bucket name")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    args = parse_args()
    config = load_rag_config(RAG_CONFIG_PATH)
    dataset = yaml.safe_load(DATASET_CONFIG_PATH.read_text(encoding="utf-8"))
    repo_url = dataset["source_repo"]
    branch = dataset["source_branch"]
    docs_path = dataset["docs_path"]
    sections = dataset["selection"]["sections"]

    manifest = _read_manifest(args.bucket)
    commit = fetch_head_commit(repo_url, branch)
    blob_shas = fetch_blob_shas(repo_url, commit)
    build_fp_now = build_fingerprint(PROJECT_ROOT, build_index_config(config))

    result = evaluate_staleness(manifest, blob_shas, docs_path, sections, build_fp_now)

    logger.info("Upstream commit: %s", commit)
    logger.info(
        "Staleness: changed=%s (source=%s, build=%s) - %s",
        result.changed, result.source_changed, result.build_changed, result.reason,
    )
    _write_github_output(result, commit)


if __name__ == "__main__":
    main()
