"""Tests for the FastAPI streaming query endpoint.

The TestClient is used without its context manager so the lifespan hook (which
builds the real GCP-backed pipeline) does not run; ``app.state.qa_system`` is
stubbed instead.
"""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from kuberacle.api.app import _cache_outcome, create_app
from kuberacle.api.cache import CachedAnswer
from kuberacle.api.guardrails import GuardrailError
from kuberacle.observability import context as ctx
from kuberacle.qa import AnswerDelta, Citation, QAResult


class FakeQA:
    """Stub QA system yielding pre-baked stream events."""

    def __init__(self, events):
        self._events = events

    def ask_stream(self, question, top_k=None):
        del question, top_k
        yield from self._events


class FakeGuardrails:
    """Stub guardrails recording each gate and optionally rejecting it."""

    def __init__(
        self,
        turnstile_error=None,
        peek_error=None,
        ip_error=None,
        combined_error=None,
    ):
        self._turnstile_error = turnstile_error
        self._peek_error = peek_error
        self._ip_error = ip_error
        self._combined_error = combined_error
        self.turnstile = []
        self.peeks = []
        self.ip_charges = []
        self.combined = []

    def verify_turnstile(self, client_ip, turnstile_token):
        self.turnstile.append((client_ip, turnstile_token))
        if self._turnstile_error is not None:
            raise self._turnstile_error

    def check_ip_rate_limit(self, client_ip):
        self.peeks.append(client_ip)
        if self._peek_error is not None:
            raise self._peek_error

    def charge_ip(self, client_ip):
        self.ip_charges.append(client_ip)
        if self._ip_error is not None:
            raise self._ip_error

    def charge_ip_and_global(self, client_ip):
        self.combined.append(client_ip)
        if self._combined_error is not None:
            raise self._combined_error


class FakeCache:
    """Stub answer cache recording reads/writes and serving a preset hit."""

    def __init__(self, hit=None):
        self._hit = hit
        self.gets = []
        self.puts = []

    def get(self, key):
        self.gets.append(key)
        return self._hit

    def put(self, key, value):
        self.puts.append((key, value))


def _client_with(events, guardrails=None, cache=None) -> TestClient:
    app = create_app()
    app.state.qa_system = FakeQA(events)
    if guardrails is not None:
        app.state.guardrails = guardrails
    if cache is not None:
        app.state.answer_cache = cache
        app.state.index_version = "idx-1"
        app.state.answer_config_version = "cfg-1"
    return TestClient(app)


def test_meta_returns_k8s_version():
    """The /meta endpoint exposes the served index's docs version at runtime."""
    app = create_app()
    app.state.k8s_version = "v1.36"
    client = TestClient(app)
    response = client.get("/meta")
    assert response.status_code == 200
    assert response.json() == {"k8s_version": "v1.36"}


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
    assert '"abstained": false' in body
    assert '"chunk_id": "a"' in body
    assert '"title": "Pods"' in body
    assert "A Pod is the smallest deployable unit." in body


def test_query_final_flags_insufficient_when_no_citations():
    """A final event without citations should be flagged ungrounded, not abstained."""
    events = [
        AnswerDelta("Ungrounded text."),
        QAResult(answer="Ungrounded text.", citations=[], retrieved_chunks=[]),
    ]
    resp = _client_with(events).post("/query", json={"question": "q"})

    assert resp.status_code == 200
    assert '"insufficient_evidence": true' in resp.text
    assert '"abstained": false' in resp.text


def test_query_final_flags_abstained_on_sentinel_answer():
    """An answer that is the abstention sentinel should set abstained true."""
    events = [
        AnswerDelta("INSUFFICIENT_EVIDENCE. Out of scope."),
        QAResult(
            answer="INSUFFICIENT_EVIDENCE. Out of scope.",
            citations=[],
            retrieved_chunks=[],
        ),
    ]
    resp = _client_with(events).post("/query", json={"question": "q"})

    assert resp.status_code == 200
    assert '"abstained": true' in resp.text


def test_query_rejects_empty_question():
    """An empty question should be rejected by request validation."""
    resp = _client_with([]).post("/query", json={"question": ""})
    assert resp.status_code == 422


def test_query_rejects_overlong_question():
    """A question past the length cap is rejected before any model call."""
    from kuberacle.api.schemas import MAX_QUESTION_LENGTH

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
    """Turnstile + per-IP peek run before the pipeline; a miss charges both caps."""
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
    assert guardrails.turnstile == [("1.2.3.4", "tok")]
    assert guardrails.peeks == ["1.2.3.4"]
    # No cache present, so the miss path charges per-IP and global together.
    assert guardrails.combined == ["1.2.3.4"]
    assert guardrails.ip_charges == []


def test_query_blocked_returns_status_and_sse_error():
    """A per-IP peek rejection returns 429 with an SSE error and skips the pipeline."""
    guardrails = FakeGuardrails(
        peek_error=GuardrailError(429, "You have reached your daily query limit.")
    )
    client = _client_with([], guardrails=guardrails)

    resp = client.post("/query", json={"question": "q"})

    assert resp.status_code == 429
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert "event: error" in resp.text
    assert "daily query limit" in resp.text
    # An over-cap peek never charges anything.
    assert guardrails.combined == [] and guardrails.ip_charges == []


def test_query_bot_check_failure_returns_403():
    """A failed bot check surfaces as a 403 SSE error before the per-IP peek."""
    guardrails = FakeGuardrails(
        turnstile_error=GuardrailError(
            403, "Bot check failed. Please reload and try again."
        )
    )
    client = _client_with([], guardrails=guardrails)

    resp = client.post("/query", json={"question": "q"})

    assert resp.status_code == 403
    assert "Bot check failed" in resp.text
    assert guardrails.peeks == []


def test_query_global_cap_rejects_after_cache_miss():
    """The combined miss charge rejects with a 429 SSE error, skipping the pipeline."""
    guardrails = FakeGuardrails(
        combined_error=GuardrailError(
            429, "The daily query limit for this demo has been reached."
        )
    )
    client = _client_with([], guardrails=guardrails)

    resp = client.post("/query", json={"question": "q"})

    assert resp.status_code == 429
    assert "demo" in resp.text
    assert guardrails.peeks == [""]
    assert guardrails.combined == [""]


def test_cache_hit_replays_without_running_pipeline():
    """A cache hit replays token+final, charges per-IP only, and skips the pipeline."""
    cached = CachedAnswer(
        answer="Pods run [1].",
        citations=[{"index": 1, "chunk_id": "a", "source_url": "u"}],
        insufficient_evidence=False,
        abstained=False,
        outcome="answered",
        cost_usd=0.0012,
    )
    guardrails = FakeGuardrails()
    cache = FakeCache(hit=cached)
    # No QA events: a hit must never touch the pipeline.
    client = _client_with([], guardrails=guardrails, cache=cache)

    resp = client.post("/query", json={"question": "What is a Pod?"})

    assert resp.status_code == 200
    assert "event: token" in resp.text
    assert "Pods run [1]." in resp.text
    assert '"chunk_id": "a"' in resp.text
    assert '"abstained": false' in resp.text
    # Per-IP charged (hit path), global cap bypassed for the free hit.
    assert guardrails.peeks == [""]
    assert guardrails.ip_charges == [""]
    assert guardrails.combined == []
    assert cache.gets and not cache.puts


def test_cache_hit_blocked_when_per_ip_charge_loses_race():
    """If the per-IP charge fails on a hit, the request is rejected, not served."""
    cached = CachedAnswer(
        answer="Pods run [1].",
        citations=[{"index": 1, "chunk_id": "a", "source_url": "u"}],
        insufficient_evidence=False,
        abstained=False,
        outcome="answered",
        cost_usd=0.0,
    )
    guardrails = FakeGuardrails(
        ip_error=GuardrailError(429, "You have reached your daily query limit.")
    )
    cache = FakeCache(hit=cached)
    client = _client_with([], guardrails=guardrails, cache=cache)

    resp = client.post("/query", json={"question": "What is a Pod?"})

    assert resp.status_code == 429
    assert "Pods run" not in resp.text
    assert guardrails.combined == []


def test_cache_miss_writes_answer_on_clean_completion():
    """A miss runs the pipeline and writes the completed answer to the cache."""
    events = [
        AnswerDelta("Pods run [1]."),
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
    cache = FakeCache(hit=None)
    client = _client_with(events, cache=cache)

    resp = client.post("/query", json={"question": "What is a Pod?"})

    assert resp.status_code == 200
    assert "event: final" in resp.text
    assert len(cache.puts) == 1
    key, value = cache.puts[0]
    assert key == cache.gets[0]
    assert value.answer == "Pods run [1]."
    assert value.outcome == "answered"
    assert value.citations[0]["chunk_id"] == "a"


def test_cache_skips_write_on_error():
    """A failed generation must not be cached."""

    def boom():
        raise RuntimeError("kaboom")
        yield  # pragma: no cover

    app_cache = FakeCache(hit=None)
    app = create_app()

    class ExplodingQA:
        def ask_stream(self, question, top_k=None):
            return boom()

    app.state.qa_system = ExplodingQA()
    app.state.answer_cache = app_cache
    app.state.index_version = "idx-1"
    app.state.answer_config_version = "cfg-1"

    resp = TestClient(app).post("/query", json={"question": "q"})

    assert "event: error" in resp.text
    assert app_cache.puts == []


def test_model_emitted_sentinel_is_not_cached():
    """A model-emitted INSUFFICIENT_EVIDENCE answer (unverified) is never cached."""
    events = [
        AnswerDelta("INSUFFICIENT_EVIDENCE. Not enough supporting docs."),
        QAResult(
            answer="INSUFFICIENT_EVIDENCE. Not enough supporting docs.",
            citations=[],
            retrieved_chunks=[],
        ),
    ]
    cache = FakeCache(hit=None)
    client = _client_with(events, cache=cache)

    resp = client.post("/query", json={"question": "What is a frobnicator?"})

    assert resp.status_code == 200
    assert '"abstained": true' in resp.text
    # Ungrounded/unverified: streamed to the client but not persisted.
    assert cache.puts == []


def test_empty_normalized_question_skips_cache():
    """A punctuation-only question is never read from or written to the cache."""
    events = [
        AnswerDelta("Pods run [1]."),
        QAResult(
            answer="Pods run [1].",
            citations=[
                Citation(
                    index=1,
                    chunk_id="a",
                    source_url="https://kubernetes.io/docs/a",
                    score=0.9,
                )
            ],
            retrieved_chunks=[],
        ),
    ]
    cache = FakeCache(hit=None)
    client = _client_with(events, cache=cache)

    resp = client.post("/query", json={"question": "???"})

    assert resp.status_code == 200
    # "???" normalizes to "" -> all such questions would collide on one key.
    assert cache.gets == []
    assert cache.puts == []


def test_cache_outcome_trusts_pipeline_outcome():
    """With metrics, cacheability follows the pipeline's authoritative outcome."""
    assert (
        _cache_outcome(SimpleNamespace(outcome=ctx.OUTCOME_ANSWERED), True)
        == ctx.OUTCOME_ANSWERED
    )
    assert (
        _cache_outcome(SimpleNamespace(outcome=ctx.OUTCOME_GATE_ABSTAINED), False)
        == ctx.OUTCOME_GATE_ABSTAINED
    )
    assert (
        _cache_outcome(SimpleNamespace(outcome=ctx.OUTCOME_NO_RETRIEVAL), False)
        == ctx.OUTCOME_NO_RETRIEVAL
    )
    # A model-emitted sentinel is classified UNVERIFIED -> not cacheable.
    assert _cache_outcome(SimpleNamespace(outcome=ctx.OUTCOME_UNVERIFIED), False) is None


def test_cache_outcome_conservative_without_metrics():
    """Without metrics, only cited answers are cached, never an abstention."""
    assert _cache_outcome(None, True) == ctx.OUTCOME_ANSWERED
    assert _cache_outcome(None, False) is None
