"""Embedding adapter for ingestion and retrieval using Vertex AI."""

from typing import Any


class VertexAIEmbedder:
    """Generate text embeddings with the Vertex AI Gemini embedding model.

    Uses RETRIEVAL_DOCUMENT task type for batch ingestion and RETRIEVAL_QUERY
    for single-query retrieval, which improves asymmetric retrieval quality.

    Args:
        model_id: Embedding model id (e.g. ``gemini-embedding-001``).
        gcp_project: GCP project ID.
        gcp_location: GCP region.
        output_dimensionality: Embedding vector dimension.
        genai_client: Optional injected client for testing.
    """

    def __init__(
        self,
        model_id: str,
        gcp_project: str,
        gcp_location: str,
        output_dimensionality: int = 768,
        genai_client: Any = None,
    ) -> None:
        self.model_id = model_id
        self.gcp_project = gcp_project
        self.gcp_location = gcp_location
        self.output_dimensionality = output_dimensionality
        self._client = genai_client

    @property
    def client(self) -> Any:
        """Lazily initialize and return the Gen AI client."""
        if self._client is None:
            from google import genai

            self._client = genai.Client(
                vertexai=True,
                project=self.gcp_project,
                location=self.gcp_location,
            )
        return self._client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts for document ingestion.

        Uses RETRIEVAL_DOCUMENT task type.

        Args:
            texts: Input text list.

        Returns:
            List of embedding vectors aligned to input order.
        """
        from google.genai import types

        response = self.client.models.embed_content(
            model=self.model_id,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=self.output_dimensionality,
            ),
        )
        return [embedding.values for embedding in response.embeddings]

    def embed_text(self, text: str) -> list[float]:
        """Embed a single query string for retrieval.

        Uses RETRIEVAL_QUERY task type.

        Args:
            text: Input query text.

        Returns:
            Embedding vector.
        """
        from google.genai import types

        response = self.client.models.embed_content(
            model=self.model_id,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=self.output_dimensionality,
            ),
        )
        return response.embeddings[0].values
