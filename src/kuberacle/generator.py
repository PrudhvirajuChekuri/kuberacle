"""Answer generation on top of retrieved chunks."""

import logging
import re
from collections.abc import Iterator
from typing import Any

from kuberacle.domain import RetrievedChunk

logger = logging.getLogger(__name__)


_CITATION_GROUP = re.compile(r"\[([\d,\s]+)\]")


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


class VertexAIAnswerGenerator:
    """Generate grounded answers with a Vertex AI Gemini model.

    Args:
        model_id: Gemini generation model id.
        gcp_project: GCP project ID.
        gcp_location: GCP region.
        temperature: Generation temperature.
        max_tokens: Maximum generated tokens.
        prompt_bundle: Versioned prompt strings keyed by role.
        genai_client: Optional injected Gen AI client for testing.
    """

    def __init__(
        self,
        model_id: str,
        gcp_project: str,
        gcp_location: str,
        temperature: float = 0.2,
        max_tokens: int = 600,
        prompt_bundle: dict[str, str] | None = None,
        genai_client: Any = None,
    ) -> None:
        self.model_id = model_id
        self.gcp_project = gcp_project
        self.gcp_location = gcp_location
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_bundle = prompt_bundle or {}
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

    def _build_prompts(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> tuple[str, str]:
        """Assemble the system and user prompts for a question.

        Args:
            question: User question.
            chunks: Retrieved context chunks.

        Returns:
            Tuple of ``(system_prompt, user_prompt)``.
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
        return system_prompt, user_prompt

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """Generate an answer grounded in retrieved chunks.

        Args:
            question: User question.
            chunks: Retrieved context chunks.

        Returns:
            Generated answer text.
        """
        logger.debug("Generating answer from %d context chunks", len(chunks))
        system_prompt, user_prompt = self._build_prompts(question, chunks)
        return self._generate_with_gemini(system_prompt, user_prompt)

    def generate_stream(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> Iterator[str]:
        """Stream a grounded answer as incremental text deltas.

        Args:
            question: User question.
            chunks: Retrieved context chunks.

        Yields:
            Non-empty text fragments in generation order.
        """
        logger.debug("Streaming answer from %d context chunks", len(chunks))
        system_prompt, user_prompt = self._build_prompts(question, chunks)
        yield from self._stream_with_gemini(system_prompt, user_prompt)

    def _generate_with_gemini(self, system_prompt: str, user_prompt: str) -> str:
        """Invoke Gemini model via the Gen AI SDK.

        Args:
            system_prompt: System instruction text.
            user_prompt: User turn text.

        Returns:
            Generated answer string.
        """
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part(text=user_prompt)],
                )
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )
        text = response.text
        if not text:
            logger.warning("Generator returned empty response from model %r", self.model_id)
            return "INSUFFICIENT_EVIDENCE."
        return text.strip()

    def _stream_with_gemini(
        self, system_prompt: str, user_prompt: str
    ) -> Iterator[str]:
        """Stream a Gemini response as text deltas via the Gen AI SDK.

        Args:
            system_prompt: System instruction text.
            user_prompt: User turn text.

        Yields:
            Non-empty text fragments in arrival order.
        """
        from google.genai import types

        stream = self.client.models.generate_content_stream(
            model=self.model_id,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part(text=user_prompt)],
                )
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )
        for chunk in stream:
            text = chunk.text
            if text:
                yield text


def extract_citation_indices(answer: str) -> list[int]:
    """Extract ordered citation indices from answer text.

    Handles both single markers (``[1]``) and grouped markers (``[1, 4]``),
    parsing every index inside a bracket.

    Args:
        answer: Generated answer containing bracketed citation markers.

    Returns:
        Ordered list of unique citation indices.
    """
    ordered_unique: list[int] = []
    seen: set[int] = set()
    for group in _CITATION_GROUP.findall(answer):
        for token in group.split(","):
            token = token.strip()
            if not token.isdigit():
                continue
            idx = int(token)
            if idx not in seen:
                seen.add(idx)
                ordered_unique.append(idx)
    return ordered_unique
