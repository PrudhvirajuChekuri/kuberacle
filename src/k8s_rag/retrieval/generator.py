"""Answer generation on top of retrieved chunks."""

from typing import Any

from k8s_rag.ingestion.schemas import RetrievedChunk


def _build_context(chunks: list[RetrievedChunk]) -> str:
    """Build numbered context block for prompting."""
    parts = []
    for idx, chunk in enumerate(chunks, start=1):
        source_url = chunk.metadata.get("source_url", "unknown")
        parts.append(
            f"[{idx}] source_url: {source_url}\n"
            f"[{idx}] chunk_id: {chunk.chunk_id}\n"
            f"[{idx}] content:\n{chunk.content}"
        )
    return "\n\n".join(parts)


class BedrockAnswerGenerator:
    """Generate grounded answers with Bedrock chat/completion models.

    Args:
        model_id: Bedrock generation model id.
        region_name: AWS region for runtime calls.
        temperature: Generation temperature.
        max_tokens: Maximum generated tokens.
        bedrock_client: Optional injected Bedrock runtime client.
    """

    def __init__(
        self,
        model_id: str,
        region_name: str,
        temperature: float = 0.2,
        max_tokens: int = 600,
        bedrock_client: Any = None,
    ) -> None:
        self.model_id = model_id
        self.region_name = region_name
        self.temperature = temperature
        self.max_tokens = max_tokens
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

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """Generate an answer grounded in retrieved chunks.

        Args:
            question: User question.
            chunks: Retrieved context chunks.

        Returns:
            Generated answer text.
        """
        context = _build_context(chunks)
        system_prompt = (
            "You are a Kubernetes docs assistant. Answer only using the provided "
            "context. If context is insufficient, say exactly: INSUFFICIENT_EVIDENCE."
        )
        user_prompt = (
            "Question:\n"
            f"{question}\n\n"
            "Context:\n"
            f"{context}\n\n"
            "Answer in concise prose and include inline citations like [1], [2] "
            "that refer to the numbered context entries."
        )

        return self._generate_with_converse(system_prompt, user_prompt)

    def _generate_with_converse(self, system_prompt: str, user_prompt: str) -> str:
        """Invoke Bedrock model via Converse API."""
        response = self.client.converse(
            modelId=self.model_id,
            system=[{"text": system_prompt}],
            messages=[
                {
                    "role": "user",
                    "content": [{"text": user_prompt}],
                }
            ],
            inferenceConfig={
                "temperature": self.temperature,
                "maxTokens": self.max_tokens,
            },
        )
        output = response.get("output", {})
        message = output.get("message", {})
        content = message.get("content", [])
        if not content:
            return "INSUFFICIENT_EVIDENCE."
        return content[0].get("text", "").strip()
