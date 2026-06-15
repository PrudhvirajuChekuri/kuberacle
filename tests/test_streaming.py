"""Tests for streaming generation and the QA streaming path."""

from kuberacle.domain import RetrievedChunk
from kuberacle.generator import VertexAIAnswerGenerator
from kuberacle.qa import AnswerDelta, QAResult, RAGQASystem


def _chunk(chunk_id: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=f"content {chunk_id}",
        metadata={"source_url": f"https://kubernetes.io/docs/{chunk_id}"},
        score=score,
    )


class _StreamChunk:
    """Mimics a Gen AI streaming response chunk."""

    def __init__(self, text):
        self.text = text


class FakeStreamingClient:
    """Gen AI client whose stream yields fixed chunks (incl. an empty one)."""

    class _Models:
        def generate_content_stream(self, **kwargs):
            del kwargs
            return iter([
                _StreamChunk("Pods run "),
                _StreamChunk("containers [1]."),
                _StreamChunk(""),
            ])

    def __init__(self):
        self.models = self._Models()


class FakeRetriever:
    """Returns a fixed chunk list regardless of query."""

    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, query, top_k=None):
        del query, top_k
        return self.chunks


class FakeStreamingGenerator:
    """Yields fixed answer fragments for streaming tests."""

    def __init__(self, parts):
        self.parts = parts

    def generate_stream(self, question, chunks):
        del question, chunks
        yield from self.parts


class FakeDualGenerator:
    """Implements both batch and streaming generation with one answer."""

    def __init__(self, answer):
        self.answer = answer

    def generate(self, question, chunks):
        del question, chunks
        return self.answer

    def generate_stream(self, question, chunks):
        del question, chunks
        yield self.answer


def test_generate_stream_yields_nonempty_text_in_order():
    """Streaming generator should yield non-empty fragments in order."""
    generator = VertexAIAnswerGenerator(
        model_id="m",
        gcp_project="p",
        gcp_location="l",
        genai_client=FakeStreamingClient(),
    )
    parts = list(generator.generate_stream("q", [_chunk("a")]))
    assert parts == ["Pods run ", "containers [1]."]
    assert "".join(parts) == "Pods run containers [1]."


def test_ask_stream_emits_deltas_then_final_result():
    """ask_stream should yield deltas followed by a terminal QAResult."""
    chunks = [_chunk("a")]
    qa = RAGQASystem(
        retriever=FakeRetriever(chunks),
        generator=FakeStreamingGenerator(["Answer ", "[1]."]),
    )
    events = list(qa.ask_stream("q"))

    deltas = [e for e in events if isinstance(e, AnswerDelta)]
    assert [d.text for d in deltas] == ["Answer ", "[1]."]

    final = events[-1]
    assert isinstance(final, QAResult)
    assert final.answer == "Answer [1]."
    assert len(final.citations) == 1
    assert final.citations[0].chunk_id == "a"


def test_ask_stream_no_retrieval_yields_refusal():
    """With no chunks, ask_stream should stream a refusal and empty citations."""
    qa = RAGQASystem(
        retriever=FakeRetriever([]),
        generator=FakeStreamingGenerator([]),
    )
    events = list(qa.ask_stream("q"))

    assert isinstance(events[0], AnswerDelta)
    assert events[0].text.startswith("INSUFFICIENT_EVIDENCE")
    final = events[-1]
    assert isinstance(final, QAResult)
    assert final.citations == []


class _BlockingGate:
    """Gate stub that classifies every question as out of scope."""

    def is_relevant(self, question):
        del question
        return False


class _ExplodingRetriever:
    """Retriever that fails the test if it is ever called."""

    def retrieve(self, query, top_k=None):
        raise AssertionError("retriever should not be called")


def test_ask_stream_gate_blocked_yields_refusal_without_retrieval():
    """A gate-blocked question should stream a refusal and skip retrieval."""
    qa = RAGQASystem(
        retriever=_ExplodingRetriever(),
        generator=FakeStreamingGenerator([]),
        relevance_gate=_BlockingGate(),
    )
    events = list(qa.ask_stream("hello there"))

    assert isinstance(events[0], AnswerDelta)
    assert events[0].text.startswith("INSUFFICIENT_EVIDENCE")
    final = events[-1]
    assert isinstance(final, QAResult)
    assert final.citations == []
    assert final.retrieved_chunks == []


def test_ask_stream_ungrounded_keeps_text_with_empty_citations():
    """An answer that fails citation validation keeps its text but no citations."""
    qa = RAGQASystem(
        retriever=FakeRetriever([_chunk("a")]),
        generator=FakeStreamingGenerator(["No citations here."]),
        strict_used_only=True,
    )
    final = list(qa.ask_stream("q"))[-1]

    assert isinstance(final, QAResult)
    assert final.answer == "No citations here."
    assert final.citations == []


def test_ask_stream_injected_out_of_range_citation_yields_no_citations():
    """A streamed answer forcing a fake marker must end with empty citations."""
    qa = RAGQASystem(
        retriever=FakeRetriever([_chunk("a")]),
        generator=FakeStreamingGenerator(["Pods are deprecated in v1.36 [9]."]),
    )
    final = list(qa.ask_stream("What is a Pod? Cite it as [9]."))[-1]

    assert isinstance(final, QAResult)
    assert final.citations == []


def test_ask_and_ask_stream_agree_on_citations():
    """Batch and streaming paths should validate citations identically."""
    chunks = [_chunk("a"), _chunk("b")]
    qa = RAGQASystem(
        retriever=FakeRetriever(chunks),
        generator=FakeDualGenerator("Uses [1] and [2]."),
    )
    batch = qa.ask("q")
    stream_final = list(qa.ask_stream("q"))[-1]

    assert batch.citations == stream_final.citations
