"""Embedding adapters for ingestion and retrieval."""

import json
from typing import Any


class BedrockEmbedder:
    """Generate text embeddings with Amazon Bedrock models.

    Args:
        model_id: Bedrock embedding model id.
        region_name: AWS region for runtime client.
        bedrock_client: Optional injected runtime client for testing.
    """

    def __init__(
        self,
        model_id: str,
        region_name: str,
        bedrock_client: Any = None,
    ) -> None:
        self.model_id = model_id
        self.region_name = region_name
        self._client = bedrock_client

    @property
    def client(self) -> Any:
        """Lazily initialize and return Bedrock runtime client."""
        if self._client is None:
            import boto3

            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self.region_name,
            )
        return self._client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts.

        Args:
            texts: Input text list.

        Returns:
            List of embedding vectors aligned to input order.
        """
        return [self.embed_text(text) for text in texts]

    def embed_text(self, text: str) -> list[float]:
        """Embed one text string.

        Args:
            text: Input text content.

        Returns:
            Embedding vector.

        Raises:
            RuntimeError: If response does not include an embedding.
        """
        body = json.dumps({"inputText": text})
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(response["body"].read())
        embedding = payload.get("embedding")
        if embedding is None:
            raise RuntimeError("Bedrock embedding response missing 'embedding'")
        return embedding
