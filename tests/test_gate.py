"""Tests for the pre-retrieval relevance gate."""

from kuberacle.gate import ScopeLabel, VertexAIRelevanceGate
from kuberacle.prompts import load_gate_prompt


PROMPT_BUNDLE = {
    "system": "Classify the message.",
    "user": 'Message to classify:\n"{question}"\n\nLabel:',
}


class FakeGenAIClient:
    """Gen AI client stub returning a fixed classification label."""

    class _Models:
        def __init__(self, text):
            self._text = text
            self.last_kwargs = None

        def generate_content(self, **kwargs):
            self.last_kwargs = kwargs

            class _Response:
                text = self._text

            return _Response()

    def __init__(self, text):
        self.models = self._Models(text)


class FailingGenAIClient:
    """Gen AI client stub whose calls always raise."""

    class _Models:
        def generate_content(self, **kwargs):
            raise RuntimeError("model unavailable")

    def __init__(self):
        self.models = self._Models()


def _make_gate(client) -> VertexAIRelevanceGate:
    """Build a gate with an injected fake client."""
    return VertexAIRelevanceGate(
        model_id="gemini-2.5-flash-lite",
        gcp_project="test-project",
        gcp_location="us-central1",
        prompt_bundle=PROMPT_BUNDLE,
        genai_client=client,
    )


def test_gate_in_scope_label_returns_true():
    """An IN_SCOPE label should pass the question through."""
    gate = _make_gate(FakeGenAIClient("IN_SCOPE"))
    assert gate.is_relevant("What is a Pod?") is True


def test_gate_out_of_scope_label_returns_false():
    """An OUT_OF_SCOPE label should block the question."""
    gate = _make_gate(FakeGenAIClient("OUT_OF_SCOPE"))
    assert gate.is_relevant("hello there") is False


def test_gate_parses_label_with_surrounding_whitespace():
    """Labels with surrounding whitespace or different casing should parse."""
    gate = _make_gate(FakeGenAIClient("  out_of_scope\n"))
    assert gate.is_relevant("what is your name?") is False


def test_gate_unparseable_label_fails_open():
    """An unrecognized label should fail open and allow the question."""
    gate = _make_gate(FakeGenAIClient("MAYBE"))
    assert gate.is_relevant("What is a Pod?") is True


def test_gate_empty_response_fails_open():
    """An empty model response should fail open."""
    gate = _make_gate(FakeGenAIClient(""))
    assert gate.is_relevant("What is a Pod?") is True


def test_gate_model_error_fails_open():
    """A model call failure should fail open and allow the question."""
    gate = _make_gate(FailingGenAIClient())
    assert gate.is_relevant("What is a Pod?") is True


def test_gate_formats_question_into_user_prompt():
    """The question should be substituted into the user prompt template."""
    client = FakeGenAIClient("IN_SCOPE")
    gate = _make_gate(client)
    gate.is_relevant("How do taints work?")

    contents = client.models.last_kwargs["contents"]
    assert "How do taints work?" in contents[0].parts[0].text


def test_gate_constrains_response_to_scope_label_enum():
    """The model call should enforce the ScopeLabel enum via response schema."""
    client = FakeGenAIClient("IN_SCOPE")
    gate = _make_gate(client)
    gate.is_relevant("What is a Pod?")

    config = client.models.last_kwargs["config"]
    assert config.response_mime_type == "text/x.enum"
    assert config.response_schema is ScopeLabel


def test_load_gate_prompt_reads_system_and_user(tmp_path):
    """Gate prompt loader should return system and user strings."""
    version_dir = tmp_path / "v1"
    version_dir.mkdir()
    (version_dir / "gate.yaml").write_text(
        "system: |\n"
        "  Classify the message.\n"
        "user: |\n"
        '  Message: "{question}"\n'
        "  Label:\n",
        encoding="utf-8",
    )

    bundle = load_gate_prompt(base_dir=str(tmp_path), version="v1")

    assert bundle["system"].startswith("Classify")
    assert "{question}" in bundle["user"]
