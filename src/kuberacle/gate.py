"""Pre-retrieval relevance gate for scoping questions to the docs corpus."""

import logging
from enum import Enum
from typing import Any

from kuberacle.observability import context as obs
from kuberacle.observability.instrumentation import link_prompt
from kuberacle.vertex import extract_token_usage, make_vertex_client

logger = logging.getLogger(__name__)


class ScopeLabel(Enum):
    """Allowed gate classification labels, enforced via constrained decoding."""

    IN_SCOPE = "IN_SCOPE"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class VertexAIRelevanceGate:
    """Classify whether a question is answerable from the Kubernetes docs.

    Runs a cheap single-label classification call before retrieval so that
    off-topic and conversational messages can be refused without spending the
    full retrieval + generation pipeline. The response is constrained to the
    ``ScopeLabel`` enum via schema-enforced decoding. Fails open: any model
    error or unparseable label lets the question through to the normal
    pipeline.

    Args:
        model_id: Gemini classification model id.
        gcp_project: GCP project ID.
        gcp_location: GCP region.
        prompt_bundle: Gate prompt strings with keys ``system`` and ``user``;
            the ``user`` template must contain a ``{question}`` placeholder.
        prompt_ref: Optional managed prompt object to link in traces.
        genai_client: Optional injected Gen AI client for testing.
    """

    def __init__(
        self,
        model_id: str,
        gcp_project: str,
        gcp_location: str,
        prompt_bundle: dict[str, str],
        prompt_ref: Any = None,
        genai_client: Any = None,
    ) -> None:
        self.model_id = model_id
        self.gcp_project = gcp_project
        self.gcp_location = gcp_location
        self.prompt_bundle = prompt_bundle
        self.prompt_ref = prompt_ref
        self._client = genai_client

    @property
    def client(self) -> Any:
        """Lazily initialize and return the Gen AI client."""
        if self._client is None:
            self._client = make_vertex_client(self.gcp_project, self.gcp_location)
        return self._client

    def is_relevant(self, question: str) -> bool:
        """Decide whether a question is in scope for the docs corpus.

        Args:
            question: User question.

        Returns:
            False when the model labels the question OUT_OF_SCOPE; True when
            it labels it IN_SCOPE or when the call fails or the label cannot
            be parsed (fail-open).
        """
        from google.genai import types

        link_prompt(self.prompt_ref)
        user_prompt = self.prompt_bundle["user"].format(question=question)
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text=user_prompt)],
                    )
                ],
                config=types.GenerateContentConfig(
                    system_instruction=self.prompt_bundle["system"],
                    temperature=0.0,
                    response_mime_type="text/x.enum",
                    response_schema=ScopeLabel,
                ),
            )
        except Exception:
            logger.warning(
                "Relevance gate call failed; failing open for question: %r",
                question[:100],
                exc_info=True,
            )
            return True

        obs.record_model_usage(
            "gate", *extract_token_usage(getattr(response, "usage_metadata", None))
        )
        label = (response.text or "").strip().upper()
        if label == ScopeLabel.OUT_OF_SCOPE.value:
            logger.info("Relevance gate: out of scope: %r", question[:100])
            return False
        if label == ScopeLabel.IN_SCOPE.value:
            return True
        logger.warning(
            "Relevance gate returned unparseable label %r; failing open", label
        )
        return True
