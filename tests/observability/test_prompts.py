"""Tests for Langfuse-backed prompt loading with file fallback."""

import pytest

from kuberacle.observability import prompts


@pytest.fixture
def prompt_dir(tmp_path):
    """Create a minimal versioned prompt directory and return its base."""
    version_dir = tmp_path / "v1"
    version_dir.mkdir()
    (version_dir / "answer.yaml").write_text(
        "system: FILE answer system\nuser: 'Q: {question}\\nC: {context}'\n"
    )
    (version_dir / "citation_enforcement.yaml").write_text(
        "rules:\n  - cite every fact\n"
    )
    (version_dir / "gate.yaml").write_text(
        "system: FILE gate system\nuser: 'Classify: {question}'\n"
    )
    return str(tmp_path)


class _FakePrompt:
    def __init__(self, text):
        self.prompt = text


class _FakeLangfuse:
    """Stub returning managed text, or raising to exercise the fallback."""

    def __init__(self, mapping=None, fail=False):
        self._mapping = mapping or {}
        self._fail = fail
        self.created = []

    def get_prompt(self, name, *, label, type, fallback, cache_ttl_seconds):
        del label, type, cache_ttl_seconds
        if self._fail:
            raise RuntimeError("langfuse down")
        return _FakePrompt(self._mapping.get(name, fallback))

    def create_prompt(self, *, name, prompt, type, labels, commit_message):
        del type, commit_message
        self.created.append((name, prompt, tuple(labels)))


def test_load_answer_prompt_files_without_langfuse(prompt_dir):
    """With no client, the file bundle is returned and no prompt object."""
    bundle, ref = prompts.load_answer_prompt(prompt_dir, "v1", None)
    assert bundle["system"] == "FILE answer system"
    assert "{question}" in bundle["user"]
    assert bundle["citation_rules"] == "- cite every fact"
    assert ref is None


def test_load_answer_prompt_uses_managed_text(prompt_dir):
    """Managed content overrides files, and the system prompt object is returned."""
    client = _FakeLangfuse({prompts.ANSWER_SYSTEM: "MANAGED system"})
    bundle, ref = prompts.load_answer_prompt(prompt_dir, "v1", client)
    assert bundle["system"] == "MANAGED system"
    assert ref is not None


def test_load_answer_prompt_falls_back_on_error(prompt_dir):
    """A Langfuse failure falls back to file content with no prompt object."""
    bundle, ref = prompts.load_answer_prompt(
        prompt_dir, "v1", _FakeLangfuse(fail=True)
    )
    assert bundle["system"] == "FILE answer system"
    assert ref is None


def test_load_gate_prompt_managed_files(prompt_dir):
    """Gate loader returns the file bundle without a client."""
    bundle, ref = prompts.load_gate_prompt_managed(prompt_dir, "v1", None)
    assert bundle["system"] == "FILE gate system"
    assert "{question}" in bundle["user"]
    assert ref is None


def test_sync_pushes_all_named_prompts(prompt_dir):
    """Sync uploads all five managed prompts under the version label."""
    client = _FakeLangfuse()
    names = prompts.sync_prompts_to_langfuse(prompt_dir, "v1", client)
    assert set(names) == {
        prompts.ANSWER_SYSTEM,
        prompts.ANSWER_USER,
        prompts.ANSWER_CITATION_RULES,
        prompts.GATE_SYSTEM,
        prompts.GATE_USER,
    }
    created_names = {c[0] for c in client.created}
    assert created_names == set(names)
    assert all(labels == ("v1",) for _, _, labels in client.created)
