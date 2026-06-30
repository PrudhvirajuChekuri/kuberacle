"""Tests for the Firestore-backed answer cache.

The key/normalization/fingerprint helpers are pure and tested directly. The
store is exercised against a fake in-memory Firestore client (no ``firestore``
package or live database).
"""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from kuberacle.api.cache import (
    AnswerCache,
    CachedAnswer,
    answer_cache_key,
    answer_config_version,
    normalize_question,
)


def test_normalize_lowercases_collapses_and_strips_punctuation():
    assert normalize_question("  What  is a   Pod? ") == "what is a pod"
    assert normalize_question("What is a Pod") == "what is a pod"
    assert normalize_question("HOW does etcd work?!") == "how does etcd work"


def test_normalize_variants_map_to_one_key():
    a = answer_cache_key("What is a Pod?", "v1", "c1")
    b = answer_cache_key("  what is a   pod ", "v1", "c1")
    assert a == b


def test_key_changes_with_index_or_config_version():
    base = answer_cache_key("q", "v1", "c1")
    assert answer_cache_key("q", "v2", "c1") != base
    assert answer_cache_key("q", "v1", "c2") != base


class _Config:
    """Minimal stand-in exposing only the attributes the fingerprint reads."""

    class _N:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    def __init__(self):
        self.prompts = self._N(version="v1")
        self.generation = self._N(
            model_id="gemini", temperature=0.2, max_tokens=1024
        )
        self.retrieval = self._N(
            semantic_top_k=5,
            lexical_top_k=5,
            merged_top_k=10,
            final_top_k=5,
            hybrid_weight_semantic=0.6,
            hybrid_weight_lexical=0.4,
            min_evidence_score=0.0,
            min_supporting_chunks=1,
        )
        self.citation = self._N(strict_used_only=True, deduplicate=True)
        self.reranker = self._N(
            enabled=True, model="ranker@latest", ranking_config="default"
        )
        self.gate = self._N(enabled=True, model_id="gemini")


_PROMPTS = {
    "answer": {"system": "Answer grounded.", "user": "{question}"},
    "gate": {"system": "Classify.", "user": "{question}"},
}


def test_config_version_is_stable_and_short():
    cfg = _Config()
    first = answer_config_version(cfg, _PROMPTS)
    assert first == answer_config_version(cfg, _PROMPTS)
    assert len(first) == 16


def test_config_version_changes_when_answer_affecting_field_changes():
    cfg = _Config()
    base = answer_config_version(cfg, _PROMPTS)
    cfg.generation.temperature = 0.9
    assert answer_config_version(cfg, _PROMPTS) != base
    cfg.generation.temperature = 0.2
    cfg.prompts.version = "v2"
    assert answer_config_version(cfg, _PROMPTS) != base


def test_config_version_changes_when_prompt_text_changes():
    """A prompt edit under the same version label invalidates the cache."""
    cfg = _Config()
    base = answer_config_version(cfg, _PROMPTS)
    edited = {
        "answer": {"system": "Answer grounded. Be concise.", "user": "{question}"},
        "gate": _PROMPTS["gate"],
    }
    assert answer_config_version(cfg, edited) != base


# --- Store, exercised against a fake Firestore client ------------------------


class FakeDoc:
    def __init__(self, store, doc_id):
        self._store = store
        self.id = doc_id

    def get(self):
        return FakeSnapshot(self._store.get(self.id))

    def set(self, data):
        self._store[self.id] = dict(data)


class FakeSnapshot:
    def __init__(self, data):
        self._data = data

    @property
    def exists(self):
        return self._data is not None

    def to_dict(self):
        return dict(self._data)


class FakeCollection:
    def __init__(self, store):
        self._store = store

    def document(self, doc_id):
        return FakeDoc(self._store, doc_id)


class FakeClient:
    def __init__(self, store):
        self._store = store

    def collection(self, name):
        return FakeCollection(self._store)


def _cache(store, ttl_days=14):
    return AnswerCache(project="p", client=FakeClient(store), ttl_days=ttl_days)


_VALUE = CachedAnswer(
    answer="Pods run [1].",
    citations=[{"index": 1, "chunk_id": "a"}],
    insufficient_evidence=False,
    abstained=False,
    outcome="answered",
    cost_usd=0.0012,
)


def test_get_returns_none_when_absent():
    assert _cache({}).get("missing") is None


def test_put_then_get_round_trips():
    store = {}
    cache = _cache(store)
    cache.put("k", _VALUE)
    got = cache.get("k")
    assert got == _VALUE
    # An expiry timestamp is written for the Firestore TTL policy.
    assert "expires_at" in store["k"]


def test_get_treats_expired_entry_as_miss():
    store = {}
    cache = _cache(store)
    cache.put("k", _VALUE)
    store["k"]["expires_at"] = datetime.now(timezone.utc) - timedelta(seconds=1)
    assert cache.get("k") is None


def test_get_handles_naive_expiry_timestamp():
    store = {
        "k": {
            "answer": "x",
            "citations": [],
            "insufficient_evidence": False,
            "abstained": False,
            "outcome": "answered",
            "cost_usd": 0.0,
            "expires_at": datetime.now(timezone.utc).replace(tzinfo=None)
            + timedelta(days=1),
        }
    }
    got = _cache(store).get("k")
    assert got is not None
    assert got.answer == "x"


def test_abstention_round_trips():
    store = {}
    cache = _cache(store)
    value = replace(
        _VALUE, answer="abstained", abstained=True, outcome="gate_abstained"
    )
    cache.put("k", value)
    assert cache.get("k") == value
