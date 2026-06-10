"""Tests for the FastAPI streaming query endpoint.

The TestClient is used without its context manager so the lifespan hook (which
builds the real GCP-backed pipeline) does not run; ``app.state.qa_system`` is
stubbed instead.
"""

from fastapi.testclient import TestClient

from k8s_rag.api.app import create_app
from k8s_rag.api.guardrails import GuardrailError
from k8s_rag.retrieval.qa import AnswerDelta, Citation, QAResult


class FakeQA:
    """Stub QA system yielding pre-baked stream events."""

    def __init__(self, events):
        self._events = events

    def ask_stream(self, question, top_k=None):
        del question, top_k
        yield from self._events


class FakeGuardrails:
    """Stub guardrails that record the request and optionally reject it."""

    def __init__(self, error=None):
        self._error = error
        self.calls = []

    def enforce(self, client_ip, turnstile_token):
        self.calls.append((client_ip, turnstile_token))
        if self._error is not None:
            raise self._error


def _client_with(events, guardrails=None) -> TestClient:
    app = create_app()
    app.state.qa_system = FakeQA(events)
    if guardrails is not None:
        app.state.guardrails = guardrails
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


def test_query_rejects_overlong_question():
    """A question past the length cap is rejected before any model call."""
    from k8s_rag.api.schemas import MAX_QUESTION_LENGTH

    resp = _client_with([]).post(
        "/query", json={"question": "a" * (MAX_QUESTION_LENGTH + 1)}
    )
    assert resp.status_code == 422


def test_docs_disabled_by_default():
    """The interactive docs and OpenAPI schema are off unless explicitly enabled."""
    client = TestClient(create_app())
    assert client.get("/openapi.json").status_code == 404
    assert client.get("/docs").status_code == 404


def test_query_runs_guardrails_with_forwarded_headers():
    """When guardrails are set, the request is enforced before streaming."""
    events = [
        AnswerDelta("Pods run."),
        QAResult(answer="Pods run.", citations=[], retrieved_chunks=[]),
    ]
    guardrails = FakeGuardrails()
    client = _client_with(events, guardrails=guardrails)

    resp = client.post(
        "/query",
        json={"question": "q"},
        headers={"X-Client-IP": "1.2.3.4", "X-Turnstile-Token": "tok"},
    )

    assert resp.status_code == 200
    assert "event: final" in resp.text
    assert guardrails.calls == [("1.2.3.4", "tok")]


def test_query_blocked_returns_status_and_sse_error():
    """A rejected request returns the guardrail status with an SSE error frame."""
    guardrails = FakeGuardrails(
        error=GuardrailError(429, "You have reached your daily query limit.")
    )
    client = _client_with([], guardrails=guardrails)

    resp = client.post("/query", json={"question": "q"})

    assert resp.status_code == 429
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert "event: error" in resp.text
    assert "daily query limit" in resp.text


def test_query_bot_check_failure_returns_403():
    """A failed bot check surfaces as a 403 SSE error."""
    guardrails = FakeGuardrails(
        error=GuardrailError(403, "Bot check failed. Please reload and try again.")
    )
    client = _client_with([], guardrails=guardrails)

    resp = client.post("/query", json={"question": "q"})

    assert resp.status_code == 403
    assert "Bot check failed" in resp.text
