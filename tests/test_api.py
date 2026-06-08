"""Tests for the FastAPI streaming query endpoint.

The TestClient is used without its context manager so the lifespan hook (which
builds the real GCP-backed pipeline) does not run; ``app.state.qa_system`` is
stubbed instead.
"""

from fastapi.testclient import TestClient

from k8s_rag.api.app import create_app
from k8s_rag.retrieval.qa import AnswerDelta, Citation, QAResult


class FakeQA:
    """Stub QA system yielding pre-baked stream events."""

    def __init__(self, events):
        self._events = events

    def ask_stream(self, question, top_k=None):
        del question, top_k
        yield from self._events


def _client_with(events) -> TestClient:
    app = create_app()
    app.state.qa_system = FakeQA(events)
    return TestClient(app)


def test_health_ok():
    """Health endpoint should report ok without touching the model."""
    client = TestClient(create_app())
    assert client.get("/health").json() == {"status": "ok"}


def test_query_streams_tokens_then_final():
    """Query should emit token events then a grounded final event."""
    events = [
        AnswerDelta("Pods "),
        AnswerDelta("run [1]."),
        QAResult(
            answer="Pods run [1].",
            citations=[
                Citation(
                    index=1,
                    chunk_id="a",
                    source_url="https://kubernetes.io/docs/a",
                    score=0.9,
                    title="Pods",
                    snippet="A Pod is the smallest deployable unit.",
                )
            ],
            retrieved_chunks=[],
        ),
    ]
    resp = _client_with(events).post("/query", json={"question": "What is a Pod?"})

    assert resp.status_code == 200
    body = resp.text
    assert "event: token" in body
    assert '"text": "Pods "' in body
    assert "event: final" in body
    assert '"insufficient_evidence": false' in body
    assert '"chunk_id": "a"' in body
    assert '"title": "Pods"' in body
    assert "A Pod is the smallest deployable unit." in body


def test_query_final_flags_insufficient_when_no_citations():
    """A final event without citations should be flagged ungrounded."""
    events = [
        AnswerDelta("Ungrounded text."),
        QAResult(answer="Ungrounded text.", citations=[], retrieved_chunks=[]),
    ]
    resp = _client_with(events).post("/query", json={"question": "q"})

    assert resp.status_code == 200
    assert '"insufficient_evidence": true' in resp.text


def test_query_rejects_empty_question():
    """An empty question should be rejected by request validation."""
    resp = _client_with([]).post("/query", json={"question": ""})
    assert resp.status_code == 422
