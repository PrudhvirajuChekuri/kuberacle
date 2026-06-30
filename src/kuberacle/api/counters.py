"""Firestore-backed daily request counters for rate limiting.

Two UTC-daily counters bound the demo:

    global_<YYYY-MM-DD>        - total queries that day across all clients
    ip_<YYYY-MM-DD>_<ip_hash>  - queries that day from one (hashed) client IP

The answer cache splits how these are charged. The per-IP counter is consulted
read-only before the cache lookup (so an over-cap client never reaches the
cache), then charged once the request is served: alone on a cache hit (which
bypasses the global cap, being free), or together with the global counter in a
single transaction on a cache miss (so the paid pipeline draws down both caps
atomically with no overshoot, and a global rejection charges neither counter).
"""

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class Decision:
    """Outcome of a combined cap check.

    Attributes:
        allowed: Whether the request may proceed.
        reason: ``"global"`` or ``"per_ip"`` when denied, otherwise None.
    """

    allowed: bool
    reason: str | None


def _decide(
    global_count: int, ip_count: int, per_ip_cap: int, global_cap: int
) -> Decision:
    """Decide whether a request is allowed given current counts.

    The global cap is checked first so that when both caps are exhausted the
    denial is attributed to the global limit (a site-wide outage), not the
    caller's personal quota.

    Args:
        global_count: Current global count for the day.
        ip_count: Current per-IP count for the day.
        per_ip_cap: Maximum allowed per IP per day.
        global_cap: Maximum allowed globally per day.

    Returns:
        A Decision describing whether to allow the request and why not.
    """
    if global_count >= global_cap:
        return Decision(allowed=False, reason="global")
    if ip_count >= per_ip_cap:
        return Decision(allowed=False, reason="per_ip")
    return Decision(allowed=True, reason=None)


def _utc_date() -> str:
    """Return the current UTC date as ``YYYY-MM-DD``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class FirestoreCounters:
    """Daily request counters persisted in Firestore.

    The Firestore client and transactional decorator are injectable so the
    logic can be unit-tested without a live database or the ``firestore``
    package installed.
    """

    def __init__(
        self,
        project: str,
        database: str = "(default)",
        collection: str = "daily_counters",
        client=None,
        transactional=None,
    ):
        """Initialize the counter store.

        Args:
            project: GCP project ID (used only when creating a real client).
            database: Firestore database name.
            collection: Firestore collection holding the counter documents.
            client: Optional pre-built Firestore client (for tests).
            transactional: Optional ``firestore.transactional`` decorator (for
                tests); defaults to the real one when a client is created.
        """
        if client is None:
            from google.cloud import firestore

            client = firestore.Client(project=project, database=database)
            transactional = firestore.transactional
        self._client = client
        self._transactional = transactional
        self._collection = collection

    def _document(self, doc_id: str):
        """Return the counter document reference for ``doc_id``."""
        return self._client.collection(self._collection).document(doc_id)

    @staticmethod
    def _count(snapshot) -> int:
        """Read the ``count`` field from a snapshot, defaulting to 0."""
        return snapshot.to_dict().get("count", 0) if snapshot.exists else 0

    def ip_under_cap(
        self, ip_hash: str, per_ip_cap: int, today: str | None = None
    ) -> bool:
        """Read-only check of whether a client is below its per-IP daily cap.

        Used as the pre-cache gate so an over-cap client is rejected before the
        cache is even consulted. Charges nothing; the per-IP counter is only
        advanced later by :meth:`check_and_increment_ip` (cache hit) or
        :meth:`check_and_increment` (cache miss).

        Args:
            ip_hash: Salted hash of the client IP.
            per_ip_cap: Maximum allowed per IP per day.
            today: UTC date key override (defaults to the current UTC date).

        Returns:
            True when the client is below its cap, False when at or above it.
        """
        day = today or _utc_date()
        snap = self._document(f"ip_{day}_{ip_hash}").get()
        return self._count(snap) < per_ip_cap

    def check_and_increment_ip(
        self, ip_hash: str, per_ip_cap: int, today: str | None = None
    ) -> bool:
        """Atomically charge one unit against a client's per-IP counter.

        Used on the cache-hit path, where only the per-IP cap applies (a hit
        bypasses the global cap). Re-checks the cap inside the transaction so
        concurrent hits cannot overshoot it.

        Args:
            ip_hash: Salted hash of the client IP.
            per_ip_cap: Maximum allowed per IP per day.
            today: UTC date key override (defaults to the current UTC date).

        Returns:
            True when the request is allowed (counter incremented); False when
            the per-IP cap is reached.
        """
        day = today or _utc_date()
        ip_ref = self._document(f"ip_{day}_{ip_hash}")

        @self._transactional
        def _run(transaction) -> bool:
            ip_count = self._count(ip_ref.get(transaction=transaction))
            if ip_count >= per_ip_cap:
                return False
            transaction.set(ip_ref, {"count": ip_count + 1}, merge=True)
            return True

        return _run(self._client.transaction())

    def check_and_increment(
        self,
        ip_hash: str,
        per_ip_cap: int,
        global_cap: int,
        today: str | None = None,
    ) -> Decision:
        """Atomically check both caps and charge both counters when allowed.

        Used on the cache-miss path, where the request can trigger the paid
        pipeline and so counts against the global budget too. Both counters are
        read and written in one transaction, so on any rejection neither is
        incremented (hitting one cap never consumes budget from the other).

        Args:
            ip_hash: Salted hash of the client IP.
            per_ip_cap: Maximum allowed per IP per day.
            global_cap: Maximum allowed globally per day.
            today: UTC date key override (defaults to the current UTC date).

        Returns:
            A Decision; both counters are incremented only when allowed is True.
        """
        day = today or _utc_date()
        global_ref = self._document(f"global_{day}")
        ip_ref = self._document(f"ip_{day}_{ip_hash}")

        @self._transactional
        def _run(transaction) -> Decision:
            global_count = self._count(global_ref.get(transaction=transaction))
            ip_count = self._count(ip_ref.get(transaction=transaction))
            decision = _decide(global_count, ip_count, per_ip_cap, global_cap)
            if decision.allowed:
                transaction.set(global_ref, {"count": global_count + 1}, merge=True)
                transaction.set(ip_ref, {"count": ip_count + 1}, merge=True)
            return decision

        return _run(self._client.transaction())
