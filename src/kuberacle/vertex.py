"""Shared Vertex AI Gen AI client construction."""

from typing import Any


def make_vertex_client(gcp_project: str, gcp_location: str) -> Any:
    """Create a google-genai client bound to Vertex AI.

    Centralizes the client construction shared by the embedder, generator, and
    relevance gate so the Vertex wiring lives in one place.

    Args:
        gcp_project: GCP project ID.
        gcp_location: GCP region.

    Returns:
        A configured ``google.genai.Client`` using Vertex AI.
    """
    from google import genai

    return genai.Client(
        vertexai=True,
        project=gcp_project,
        location=gcp_location,
    )


def extract_token_usage(usage_metadata: Any) -> tuple[int, int]:
    """Pull (input, output) token counts from a Gen AI usage metadata object.

    Shared by the generator and the relevance gate. Output includes any
    thinking/reasoning tokens, which Gemini bills as output tokens.

    Args:
        usage_metadata: A response ``usage_metadata`` object, or None.

    Returns:
        A ``(input_tokens, output_tokens)`` tuple; ``(0, 0)`` when unavailable.
    """
    if usage_metadata is None:
        return 0, 0
    prompt = getattr(usage_metadata, "prompt_token_count", 0) or 0
    candidates = getattr(usage_metadata, "candidates_token_count", 0) or 0
    thoughts = getattr(usage_metadata, "thoughts_token_count", 0) or 0
    return prompt, candidates + thoughts
