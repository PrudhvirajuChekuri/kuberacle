"""Answer generation on top of retrieved chunks."""

import re
from typing import Any

from k8s_rag.ingestion.schemas import RetrievedChunk


_CITATION_INDEX = re.compile(r"\[(\d+)\]")


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
        prompt_bundle: dict[str, str] | None = None,
        bedrock_client: Any = None,
    ) -> None:
        self.model_id = model_id
        self.region_name = region_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_bundle = prompt_bundle or {}
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
        citation_rules = self.prompt_bundle.get("citation_rules", "")
        system_prompt = self.prompt_bundle.get(
            "system",
            "You are a Kubernetes docs assistant. Answer only using context.",
        )
        user_template = self.prompt_bundle.get(
            "user",
            "Question:\n{question}\n\nContext:\n{context}\n",
        )
        user_prompt = user_template.format(question=question, context=context)
        if citation_rules:
            user_prompt += f"\n\nCitation rules:\n{citation_rules}"

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


def extract_citation_indices(answer: str) -> list[int]:
    """Extract ordered citation indices from answer text."""
    indices = [int(match) for match in _CITATION_INDEX.findall(answer)]
    ordered_unique: list[int] = []
    seen: set[int] = set()
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            ordered_unique.append(idx)
    return ordered_unique
