"""Embedding adapter for ingestion and retrieval using Vertex AI."""

import logging
from typing import Any

from kuberacle.vertex import make_vertex_client

logger = logging.getLogger(__name__)


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
            self._client = make_vertex_client(self.gcp_project, self.gcp_location)
        return self._client

    def _embed(self, contents: str | list[str], task_type: str) -> list[list[float]]:
        from google.genai import types

        response = self.client.models.embed_content(
            model=self.model_id,
            contents=contents,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.output_dimensionality,
            ),
        )
        return [embedding.values for embedding in response.embeddings]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts for document ingestion.

        Uses RETRIEVAL_DOCUMENT task type.

        Args:
            texts: Input text list.

        Returns:
            List of embedding vectors aligned to input order.
        """
        logger.debug("Embedding %d texts (RETRIEVAL_DOCUMENT)", len(texts))
        embeddings = self._embed(texts, "RETRIEVAL_DOCUMENT")
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"API returned {len(embeddings)} embeddings for {len(texts)} inputs."
            )
        return embeddings

    def embed_text(self, text: str) -> list[float]:
        """Embed a single query string for retrieval.

        Uses RETRIEVAL_QUERY task type.

        Args:
            text: Input query text.

        Returns:
            Embedding vector.
        """
        logger.debug("Embedding query text (RETRIEVAL_QUERY)")
        embeddings = self._embed(text, "RETRIEVAL_QUERY")
        if not embeddings:
            raise RuntimeError("API returned no embeddings for query.")
        return embeddings[0]
