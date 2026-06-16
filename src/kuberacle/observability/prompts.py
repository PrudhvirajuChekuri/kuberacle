"""Langfuse-backed prompt loading with file fallback.

Git stays the source of truth: prompts live in ``configs/prompts/`` and are
pushed to Langfuse with the prompt-sync utility. At runtime these loaders fetch
the managed version from Langfuse (so prompt edits can ship without a redeploy,
within the cache TTL) and fall back to the committed file content when Langfuse
is unavailable or the prompt has not been pushed. The fetched prompt object is
returned alongside the bundle so the orchestration layer can link it to the
generation/gate observation in traces.

When no Langfuse client is configured (local dev, tests, CLI) these delegate to
the plain file loaders and return no prompt object.
"""

import logging
from typing import Any

from kuberacle.prompts import load_gate_prompt, load_prompt_bundle

logger = logging.getLogger(__name__)

# Names under which the file prompts are managed in Langfuse.
ANSWER_SYSTEM = "kuberacle-answer-system"
ANSWER_USER = "kuberacle-answer-user"
ANSWER_CITATION_RULES = "kuberacle-answer-citation-rules"
GATE_SYSTEM = "kuberacle-gate-system"
GATE_USER = "kuberacle-gate-user"

# Cache managed prompts briefly so warm instances pick up edits without a
# redeploy while still avoiding a fetch on every request.
_CACHE_TTL_SECONDS = 60


def _managed_text(
    langfuse: Any, name: str, version: str, fallback: str
) -> tuple[str, Any]:
    """Fetch a managed text prompt, falling back to the committed content.

    Args:
        langfuse: Langfuse client.
        name: Managed prompt name.
        version: Prompt version, used as the Langfuse label.
        fallback: Committed file content used when the fetch fails.

    Returns:
        A ``(text, prompt_object)`` tuple; ``prompt_object`` is None on fallback.
    """
    try:
        prompt = langfuse.get_prompt(
            name,
            label=version,
            type="text",
            fallback=fallback,
            cache_ttl_seconds=_CACHE_TTL_SECONDS,
        )
        return prompt.prompt, prompt
    except Exception:
        logger.warning(
            "Langfuse prompt %r fetch failed; using file fallback", name,
            exc_info=True,
        )
        return fallback, None


def load_answer_prompt(
    base_dir: str, version: str, langfuse: Any = None
) -> tuple[dict[str, str], Any]:
    """Load the answer prompt bundle, managed by Langfuse when available.

    Args:
        base_dir: Prompt root directory.
        version: Prompt version.
        langfuse: Optional Langfuse client; when None, files are used directly.

    Returns:
        A ``(bundle, system_prompt_object)`` tuple. ``system_prompt_object`` is
        the Langfuse prompt to link to the generation, or None.
    """
    file_bundle = load_prompt_bundle(base_dir, version)
    if langfuse is None:
        return file_bundle, None
    system, system_obj = _managed_text(
        langfuse, ANSWER_SYSTEM, version, file_bundle["system"]
    )
    user, _ = _managed_text(langfuse, ANSWER_USER, version, file_bundle["user"])
    citation, _ = _managed_text(
        langfuse, ANSWER_CITATION_RULES, version, file_bundle["citation_rules"]
    )
    return (
        {"system": system, "user": user, "citation_rules": citation},
        system_obj,
    )


def sync_prompts_to_langfuse(base_dir: str, version: str, langfuse: Any) -> list[str]:
    """Push the committed file prompts to Langfuse under the version label.

    Git is the source of truth: this uploads each ``configs/prompts`` part as a
    managed Langfuse text prompt labelled with the version, so the runtime
    loaders serve the managed copy (with the files still as fallback). Run on
    deploy or whenever the file prompts change.

    Args:
        base_dir: Prompt root directory.
        version: Prompt version, applied as the Langfuse label.
        langfuse: Langfuse client.

    Returns:
        The list of prompt names that were created/updated.
    """
    answer = load_prompt_bundle(base_dir, version)
    gate = load_gate_prompt(base_dir, version)
    items = {
        ANSWER_SYSTEM: answer["system"],
        ANSWER_USER: answer["user"],
        ANSWER_CITATION_RULES: answer["citation_rules"],
        GATE_SYSTEM: gate["system"],
        GATE_USER: gate["user"],
    }
    for name, text in items.items():
        langfuse.create_prompt(
            name=name,
            prompt=text,
            type="text",
            labels=[version],
            commit_message=f"sync from configs/prompts/{version}",
        )
        logger.info("Synced prompt %r to Langfuse label %r", name, version)
    return list(items)


def load_gate_prompt_managed(
    base_dir: str, version: str, langfuse: Any = None
) -> tuple[dict[str, str], Any]:
    """Load the gate prompt bundle, managed by Langfuse when available.

    Args:
        base_dir: Prompt root directory.
        version: Prompt version.
        langfuse: Optional Langfuse client; when None, files are used directly.

    Returns:
        A ``(bundle, system_prompt_object)`` tuple. ``system_prompt_object`` is
        the Langfuse prompt to link to the gate observation, or None.
    """
    file_bundle = load_gate_prompt(base_dir, version)
    if langfuse is None:
        return file_bundle, None
    system, system_obj = _managed_text(
        langfuse, GATE_SYSTEM, version, file_bundle["system"]
    )
    user, _ = _managed_text(langfuse, GATE_USER, version, file_bundle["user"])
    return {"system": system, "user": user}, system_obj
