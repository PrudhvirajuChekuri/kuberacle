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
