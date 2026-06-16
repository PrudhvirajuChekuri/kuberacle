"""Tests that the pipeline records usage, cost, and outcomes into metrics."""

from kuberacle.config import PricingConfig
from kuberacle.domain import RetrievedChunk
from kuberacle.gate import VertexAIRelevanceGate
from kuberacle.generator import VertexAIAnswerGenerator
from kuberacle.observability import context as ctx
from kuberacle.qa import RAGQASystem

PRICING = PricingConfig(0.10, 0.40, 0.15, 1.00)


class _Usage:
    def __init__(self, prompt, candidates, thoughts=0):
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates
        self.thoughts_token_count = thoughts


class _StreamGenAIClient:
    """Gen AI client stub streaming text chunks with terminal usage metadata."""

    class _Models:
        def generate_content_stream(self, **kwargs):
            class _Chunk:
                def __init__(self, text, usage=None):
                    self.text = text
                    self.usage_metadata = usage

            yield _Chunk("Pods ")
            yield _Chunk("run [1].", _Usage(3000, 600, thoughts=100))

    def __init__(self):
        self.models = self._Models()


class _GateGenAIClient:
    """Gen AI client stub returning a fixed gate label with usage metadata."""

    class _Models:
        def __init__(self, label):
            self._label = label

        def generate_content(self, **kwargs):
            label = self._label

            class _Response:
                text = label
                usage_metadata = _Usage(120, 1)

            return _Response()

    def __init__(self, label):
        self.models = self._Models(label)


class _FakeRetriever:
    def __init__(self, chunks):
        self._chunks = chunks

    def retrieve(self, query, top_k=None):
        del query, top_k
        return self._chunks


class _FakeGenerator:
    def __init__(self, answer):
        self._answer = answer
        self.model_id = "gemini-2.5-flash-lite"

    def generate(self, question, chunks):
        del question, chunks
        return self._answer

    def generate_stream(self, question, chunks):
        del question, chunks
        yield self._answer


def _chunk(idx="a", score=0.9):
    return RetrievedChunk(
        chunk_id=idx,
        content="A Pod is the smallest deployable unit.",
        metadata={"source_url": "https://kubernetes.io/docs/a"},
        score=score,
    )


def _bind():
    metrics = ctx.RequestMetrics(pricing=PRICING)
    return metrics, ctx.set_metrics(metrics)


def test_gate_records_usage_and_cost():
    """The relevance gate records its token usage and cost."""
    gate = VertexAIRelevanceGate(
        model_id="gemini-2.5-flash-lite",
        gcp_project="p",
        gcp_location="us-central1",
        prompt_bundle={"system": "s", "user": "{question}"},
        genai_client=_GateGenAIClient("IN_SCOPE"),
    )
    metrics, token = _bind()
    try:
        assert gate.is_relevant("What is a Pod?") is True
        assert metrics.tokens["gate_in"] == 120
        assert metrics.tokens["gate_out"] == 1
        assert metrics.cost_usd["gate"] > 0
    finally:
        ctx.reset_metrics(token)


def test_generator_stream_records_output_tokens_including_thoughts():
    """Streaming generation records usage with thinking tokens billed as output."""
    generator = VertexAIAnswerGenerator(
        model_id="gemini-2.5-flash-lite",
        gcp_project="p",
        gcp_location="us-central1",
        genai_client=_StreamGenAIClient(),
    )
    metrics, token = _bind()
    try:
        list(generator.generate_stream("q", [_chunk()]))
        assert metrics.tokens["generation_in"] == 3000
        # candidates (600) + thoughts (100) billed as output.
        assert metrics.tokens["generation_out"] == 700
        assert metrics.cost_usd["generation"] > 0
    finally:
        ctx.reset_metrics(token)


def test_ask_stream_records_answered_outcome():
    """A grounded streamed answer records the answered outcome and counts."""
    system = RAGQASystem(
        retriever=_FakeRetriever([_chunk()]),
        generator=_FakeGenerator("Pods run [1]."),
        min_evidence_score=0.0,
    )
    metrics, token = _bind()
    try:
        list(system.ask_stream("What is a Pod?"))
        assert metrics.outcome == ctx.OUTCOME_ANSWERED
        assert metrics.chunks_retrieved == 1
        assert metrics.citations_count == 1
        assert metrics.gate_decision == ctx.GATE_SKIPPED
    finally:
        ctx.reset_metrics(token)


def test_ask_records_no_retrieval_outcome():
    """No retrieved chunks records the no_retrieval outcome."""
    system = RAGQASystem(
        retriever=_FakeRetriever([]),
        generator=_FakeGenerator("unused"),
    )
    metrics, token = _bind()
    try:
        system.ask("q")
        assert metrics.outcome == ctx.OUTCOME_NO_RETRIEVAL
        assert metrics.chunks_retrieved == 0
    finally:
        ctx.reset_metrics(token)


def test_ask_records_unverified_outcome():
    """An answer without valid citations records the unverified outcome."""
    system = RAGQASystem(
        retriever=_FakeRetriever([_chunk()]),
        generator=_FakeGenerator("No citation markers here."),
        min_evidence_score=0.0,
    )
    metrics, token = _bind()
    try:
        system.ask("q")
        assert metrics.outcome == ctx.OUTCOME_UNVERIFIED
        assert metrics.insufficient_evidence is True
    finally:
        ctx.reset_metrics(token)


class _OutOfScopeGate:
    def is_relevant(self, question):
        del question
        return False


def test_ask_records_gate_abstained_outcome():
    """An out-of-scope question records the gate_abstained outcome."""
    system = RAGQASystem(
        retriever=_FakeRetriever([_chunk()]),
        generator=_FakeGenerator("Pods run [1]."),
        relevance_gate=_OutOfScopeGate(),
    )
    metrics, token = _bind()
    try:
        system.ask("bake a cake")
        assert metrics.outcome == ctx.OUTCOME_GATE_ABSTAINED
        assert metrics.gate_decision == ctx.GATE_OUT_OF_SCOPE
    finally:
        ctx.reset_metrics(token)
