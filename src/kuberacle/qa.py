"""Question-answer orchestration with citations."""

import json
import logging
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from kuberacle.domain import RetrievedChunk
from kuberacle.generator import extract_citation_indices

logger = logging.getLogger(__name__)

_NO_RETRIEVAL_ANSWER = (
    "INSUFFICIENT_EVIDENCE. I could not retrieve supporting "
    "documentation for this question."
)
_UNVERIFIED_ANSWER = (
    "INSUFFICIENT_EVIDENCE. I could not verify enough supported "
    "citations for this question."
)
_OUT_OF_SCOPE_ANSWER = (
    "INSUFFICIENT_EVIDENCE. This question is outside the scope of the "
    "Kubernetes documentation, so I cannot answer it from my sources."
)


def _chunk_title(metadata: Mapping[str, Any]) -> str:
    """Pick the most specific heading for a chunk, for source previews.

    Prefers the deepest entry of ``heading_hierarchy`` (the chunk's own section,
    e.g. "What is a Pod?") so chunks from the same page are distinguishable,
    falling back to the page ``title``. Handles ``heading_hierarchy`` stored as a
    list or as a JSON-encoded string.

    Args:
        metadata: Chunk metadata map.

    Returns:
        The section heading, or the page title if no hierarchy is present.
    """
    hierarchy = metadata.get("heading_hierarchy")
    if isinstance(hierarchy, str):
        try:
            hierarchy = json.loads(hierarchy)
        except (ValueError, TypeError):
            hierarchy = None
    if isinstance(hierarchy, (list, tuple)) and hierarchy:
        last = str(hierarchy[-1]).strip()
        if last:
            return last
    return str(metadata.get("title", ""))


_ADMONITION_RE = re.compile(
    r"^(?:NOTE|TIP|INFO|WARNING|CAUTION|ALERT|DANGER|FEATURE STATE):\s*",
    re.IGNORECASE,
)


def _strip_inline_markup(text: str) -> str:
    """Remove inline Markdown emphasis and code markers, keeping the text.

    Drops backticks and asterisk emphasis unconditionally; for underscores,
    only strips emphasis used at word boundaries (``_word_``) so identifiers
    such as ``revision_history_limit`` are left intact.

    Args:
        text: A single line of text.

    Returns:
        The line with inline emphasis and code markers removed.
    """
    text = text.replace("`", "")
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"(?<![A-Za-z0-9_])_{1,3}([^_]+?)_{1,3}(?![A-Za-z0-9_])", r"\1", text)
    return text


def _make_snippet(content: str, limit: int = 200, title: str = "") -> str:
    """Build a short, single-line preview snippet from chunk content.

    Drops fenced code blocks entirely, strips leading Markdown markers
    (headings, blockquotes, list bullets), a leading ``[Heading]`` breadcrumb
    marker, leading admonition labels (``NOTE:``/``ALERT:``/...), and inline
    emphasis/code markers, line by line. Collapses whitespace and truncates to
    ``limit`` characters with an ellipsis. Skips a leading line that merely
    repeats ``title`` (the chunk's section heading shown above the snippet), so
    the preview starts at the body rather than echoing the title.

    Args:
        content: Raw chunk text content.
        limit: Maximum length of the returned snippet.
        title: Chunk section title shown alongside the snippet; a leading content
            line equal to it (case-insensitively) is dropped to avoid repetition.

    Returns:
        A cleaned, truncated preview string.
    """
    title_norm = title.strip().casefold()
    cleaned: list[str] = []
    in_fence = False
    for line in content.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        stripped = re.sub(r"^\s*[#>*+\-]+\s+", "", line).strip()
        if stripped.startswith("["):
            end = stripped.find("]")
            if end != -1:
                stripped = stripped[end + 1 :].strip()
        stripped = _ADMONITION_RE.sub("", stripped).strip()
        if not stripped:
            continue
        if not cleaned and title_norm and stripped.casefold() == title_norm:
            continue
        cleaned.append(stripped)
    # Strip inline emphasis on the joined text so emphasis pairs that span
    # multiple source lines (e.g. a multi-line italic blurb) are matched.
    text = " ".join(" ".join(cleaned).split())
    text = " ".join(_strip_inline_markup(text).split())
    if len(text) > limit:
        text = text[:limit].rstrip() + "…"
    return text


@dataclass(frozen=True)
class Citation:
    """Citation record for generated answer provenance.

    Attributes:
        index: 1-based context-chunk number, matching the ``[n]`` marker the
            answer uses to reference this source.
        chunk_id: Identifier of the supporting chunk.
        source_url: Source document URL backing the answer.
        score: Relevance score of the supporting chunk.
        title: Document title of the supporting chunk, for source previews.
        snippet: Short text preview of the supporting chunk content.
    """

    index: int
    chunk_id: str
    source_url: str
    score: float
    title: str = ""
    snippet: str = ""


@dataclass(frozen=True)
class QAResult:
    """Output object returned by RAGQASystem."""

    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]


@dataclass(frozen=True)
class AnswerDelta:
    """Incremental answer text fragment emitted during streaming."""

    text: str


class RAGQASystem:
    """Orchestrate retrieval + generation for grounded answers."""

    def __init__(
        self,
        retriever: Any,
        generator: Any,
        min_evidence_score: float = 0.0,
        min_supporting_chunks: int = 1,
        strict_used_only: bool = True,
        deduplicate_citations: bool = True,
        relevance_gate: Any = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.min_evidence_score = min_evidence_score
        self.min_supporting_chunks = min_supporting_chunks
        self.strict_used_only = strict_used_only
        self.deduplicate_citations = deduplicate_citations
        self.relevance_gate = relevance_gate

    def _is_out_of_scope(self, question: str) -> bool:
        """Check the question against the optional pre-retrieval gate.

        Args:
            question: User question.

        Returns:
            True when a gate is configured and classifies the question as out
            of scope for the docs corpus; False otherwise.
        """
        if self.relevance_gate is None:
            return False
        return not self.relevance_gate.is_relevant(question)

    def ask(self, question: str, top_k: int | None = None) -> QAResult:
        """Answer a user question with source citations.

        Args:
            question: User question.
            top_k: Optional retrieval depth override.

        Returns:
            QA result with answer and citations.
        """
        logger.info("Processing question: %r", question[:100])
        if self._is_out_of_scope(question):
            return QAResult(
                answer=_OUT_OF_SCOPE_ANSWER,
                citations=[],
                retrieved_chunks=[],
            )
        chunks = self.retriever.retrieve(question, top_k=top_k)
        if not chunks:
            logger.warning("No chunks retrieved for question: %r", question[:100])
            return QAResult(
                answer=_NO_RETRIEVAL_ANSWER,
                citations=[],
                retrieved_chunks=[],
            )

        logger.debug("Retrieved %d chunks; generating answer", len(chunks))
        answer = self.generator.generate(question, chunks)
        citations = self._compute_citations(chunks, answer)
        if citations is None:
            return self._insufficient_with_chunks(chunks)

        return QAResult(
            answer=answer,
            citations=citations,
            retrieved_chunks=chunks,
        )

    def ask_stream(
        self, question: str, top_k: int | None = None
    ) -> Iterator[AnswerDelta | QAResult]:
        """Answer a question, streaming answer text then a final result.

        Yields ``AnswerDelta`` fragments as the answer is generated, followed by
        exactly one terminal ``QAResult`` carrying the validated citations. When
        citations cannot be validated the terminal result has empty citations
        while the already-streamed answer text is preserved (ungrounded), so
        callers can flag the answer rather than discard it.

        Args:
            question: User question.
            top_k: Optional retrieval depth override.

        Yields:
            Zero or more ``AnswerDelta`` items, then one terminal ``QAResult``.
        """
        logger.info("Processing question (stream): %r", question[:100])
        if self._is_out_of_scope(question):
            yield AnswerDelta(text=_OUT_OF_SCOPE_ANSWER)
            yield QAResult(
                answer=_OUT_OF_SCOPE_ANSWER,
                citations=[],
                retrieved_chunks=[],
            )
            return
        chunks = self.retriever.retrieve(question, top_k=top_k)
        if not chunks:
            logger.warning("No chunks retrieved for question: %r", question[:100])
            yield AnswerDelta(text=_NO_RETRIEVAL_ANSWER)
            yield QAResult(
                answer=_NO_RETRIEVAL_ANSWER,
                citations=[],
                retrieved_chunks=[],
            )
            return

        logger.debug("Retrieved %d chunks; streaming answer", len(chunks))
        parts: list[str] = []
        for delta in self.generator.generate_stream(question, chunks):
            parts.append(delta)
            yield AnswerDelta(text=delta)
        answer = "".join(parts)

        citations = self._compute_citations(chunks, answer)
        if citations is None:
            logger.warning("Insufficient evidence: streamed answer is ungrounded")
        yield QAResult(
            answer=answer,
            citations=citations or [],
            retrieved_chunks=chunks,
        )

    def _compute_citations(
        self, chunks: list[RetrievedChunk], answer: str
    ) -> list[Citation] | None:
        """Validate the answer's citations against retrieved chunks.

        Args:
            chunks: Retrieved context chunks, in prompt order.
            answer: Generated answer text containing bracketed citation markers.

        Returns:
            The list of supporting citations, or ``None`` when the answer fails
            citation validation (insufficient evidence).
        """
        used_indices = extract_citation_indices(answer)
        logger.debug("Extracted %d citation indices from answer", len(used_indices))

        if self.strict_used_only:
            if not used_indices:
                return None
            if any(idx < 1 or idx > len(chunks) for idx in used_indices):
                return None
            selected = [(idx, chunks[idx - 1]) for idx in used_indices]
        else:
            selected = [(i + 1, chunk) for i, chunk in enumerate(chunks)]

        if self.deduplicate_citations:
            deduped: list[tuple[int, RetrievedChunk]] = []
            seen: set[str] = set()
            for idx, chunk in selected:
                if chunk.chunk_id in seen:
                    continue
                seen.add(chunk.chunk_id)
                deduped.append((idx, chunk))
            selected = deduped

        # Evidence score gates abstention only: if no cited chunk clears the
        # threshold the answer is treated as unsupported. Otherwise every cited
        # chunk is kept, so citations always match the markers in the answer.
        supporting = [
            chunk for _, chunk in selected
            if chunk.score >= self.min_evidence_score
        ]
        if len(supporting) < self.min_supporting_chunks:
            return None

        citations = []
        for idx, chunk in selected:
            title = _chunk_title(chunk.metadata)
            citations.append(
                Citation(
                    index=idx,
                    chunk_id=chunk.chunk_id,
                    source_url=str(chunk.metadata.get("source_url", "unknown")),
                    score=chunk.score,
                    title=title,
                    snippet=_make_snippet(chunk.content, title=title),
                )
            )
        return citations

    def _insufficient_with_chunks(self, chunks: list[RetrievedChunk]) -> QAResult:
        """Build insufficient evidence result."""
        logger.warning("Insufficient evidence: could not verify supported citations")
        return QAResult(
            answer=_UNVERIFIED_ANSWER,
            citations=[],
            retrieved_chunks=chunks,
        )
