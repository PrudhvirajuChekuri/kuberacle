"""Command-line entry points for kuberacle.

Each command module exposes a ``main()``. They are exposed three ways:
- console scripts (``kuberacle-ingest`` ...), declared in pyproject.toml,
- the unified dispatcher ``python -m kuberacle <command>``,
- and the thin ``scripts/*.py`` shims (kept for existing docs and CI).

``COMMANDS`` maps the dispatcher's command names to their module paths; modules
are imported lazily so a single command never pulls in unrelated heavy deps.
"""

COMMANDS = {
    "download-data": "kuberacle.cli.download_data",
    "preprocess": "kuberacle.cli.preprocess",
    "ingest": "kuberacle.cli.ingest",
    "query": "kuberacle.cli.query",
    "evaluate": "kuberacle.cli.evaluate",
    "serve": "kuberacle.cli.serve",
    "push-index": "kuberacle.cli.push_index",
    "pull-index": "kuberacle.cli.pull_index",
}
