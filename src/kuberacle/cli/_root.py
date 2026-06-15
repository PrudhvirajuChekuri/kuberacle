"""Project-root resolution shared by the CLI commands."""

import os
from pathlib import Path


def project_root() -> Path:
    """Resolve the project root for locating ``configs/`` and ``data/``.

    Uses the ``RAG_PROJECT_ROOT`` environment variable when set, otherwise the
    current working directory. Mirrors the API's resolution so every entry point
    agrees on where the project lives.

    Returns:
        Absolute path to the project root.
    """
    return Path(os.environ.get("RAG_PROJECT_ROOT", Path.cwd())).resolve()
