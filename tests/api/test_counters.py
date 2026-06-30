"""Tests for the Firestore-backed daily counters.

A fake in-memory Firestore client exercises the transaction logic without the
``firestore`` package or a live database. The per-IP counter is peeked
read-only before the cache and charged alone (hit) or with the global counter
in one transaction (miss).
"""

from kuberacle.api.counters import Decision, FirestoreCounters, _decide


class FakeSnapshot:
    """Stand-in for a Firestore DocumentSnapshot."""

    def __init__(self, data):
        self._data = data

    @property
    def exists(self):
        return self._data is not None

    def to_dict(self):
        return dict(self._data) if self._data else {}


class FakeDocRef:
    """Stand-in for a Firestore DocumentReference backed by a dict store."""

    def __init__(self, store, doc_id):
        self._store = store
        self.id = doc_id

    def get(self, transaction=None):
        return FakeSnapshot(self._store.get(self.id))


class FakeTransaction:
    """Stand-in for a Firestore Transaction applying writes to the store."""

    def __init__(self, store):
        self._store = store

    def set(self, ref, data, merge=False):
        existing = self._store.get(ref.id) or {}
        self._store[ref.id] = {**existing, **data} if merge else dict(data)


class FakeCollection:
    def __init__(self, store):
        self._store = store

    def document(self, doc_id):
        return FakeDocRef(self._store, doc_id)


class FakeClient:
    """In-memory Firestore client; all documents share one dict store."""

    def __init__(self, store):
        self._store = store

    def collection(self, name):
        return FakeCollection(self._store)

    def transaction(self):
        return FakeTransaction(self._store)


def _identity_transactional(fn):
    """Stand-in for firestore.transactional: run the function directly."""
    return fn


def _counters(store):
    return FirestoreCounters(
        project="p",
        client=FakeClient(store),
        transactional=_identity_transactional,
    )


DAY = "2026-06-08"


def test_decide_allows_under_caps():
    assert _decide(0, 0, 10, 300) == Decision(True, None)
    assert _decide(299, 9, 10, 300) == Decision(True, None)


def test_decide_denies_global_first():
    assert _decide(300, 0, 10, 300) == Decision(False, "global")
    # Both exhausted -> attributed to global (a site-wide outage).
    assert _decide(300, 10, 10, 300) == Decision(False, "global")


def test_decide_denies_per_ip():
    assert _decide(0, 10, 10, 300) == Decision(False, "per_ip")


def test_ip_under_cap_is_read_only():
    store = {f"ip_{DAY}_h": {"count": 5}}
    counters = _counters(store)
    assert counters.ip_under_cap("h", 10, today=DAY) is True
    assert counters.ip_under_cap("h", 5, today=DAY) is False
    # Peeking never advances the counter.
    assert store[f"ip_{DAY}_h"]["count"] == 5


def test_ip_under_cap_treats_absent_counter_as_zero():
    assert _counters({}).ip_under_cap("h", 10, today=DAY) is True


def test_check_and_increment_ip_charges_only_per_ip():
    store = {}
    allowed = _counters(store).check_and_increment_ip("h", 10, today=DAY)
    assert allowed is True
    assert store[f"ip_{DAY}_h"]["count"] == 1
    # The hit path must not touch the global counter.
    assert f"global_{DAY}" not in store


def test_check_and_increment_ip_cap_hit_does_not_increment():
    store = {f"ip_{DAY}_h": {"count": 10}}
    allowed = _counters(store).check_and_increment_ip("h", 10, today=DAY)
    assert allowed is False
    assert store[f"ip_{DAY}_h"]["count"] == 10


def test_combined_increments_both_when_allowed():
    store = {f"global_{DAY}": {"count": 5}, f"ip_{DAY}_h": {"count": 2}}
    decision = _counters(store).check_and_increment("h", 10, 300, today=DAY)
    assert decision == Decision(True, None)
    assert store[f"global_{DAY}"]["count"] == 6
    assert store[f"ip_{DAY}_h"]["count"] == 3


def test_combined_global_cap_hit_increments_neither():
    store = {f"global_{DAY}": {"count": 300}}
    decision = _counters(store).check_and_increment("h", 10, 300, today=DAY)
    assert decision == Decision(False, "global")
    # The global rejection must not burn per-IP budget.
    assert store[f"global_{DAY}"]["count"] == 300
    assert f"ip_{DAY}_h" not in store


def test_combined_per_ip_cap_hit_increments_neither():
    store = {f"global_{DAY}": {"count": 5}, f"ip_{DAY}_h": {"count": 10}}
    decision = _counters(store).check_and_increment("h", 10, 300, today=DAY)
    assert decision == Decision(False, "per_ip")
    assert store[f"global_{DAY}"]["count"] == 5
    assert store[f"ip_{DAY}_h"]["count"] == 10
