"""Firestore-backed daily request counters for rate limiting.

Tracks two UTC-daily counters in a single Firestore transaction so the per-IP
and global caps are checked and incremented atomically (no race, no overshoot):

    global_<YYYY-MM-DD>        - total queries that day across all clients
    ip_<YYYY-MM-DD>_<ip_hash>  - queries that day from one (hashed) client IP

On any rejection neither counter is incremented, so hitting one cap never
consumes budget from the other.
"""

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class Decision:
    """Outcome of a rate-limit check.

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
    denial is attributed to the global limit.

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

    def check_and_increment(
        self,
        ip_hash: str,
        per_ip_cap: int,
        global_cap: int,
        today: str | None = None,
    ) -> Decision:
        """Atomically check the caps and increment counters when allowed.

        Args:
            ip_hash: Salted hash of the client IP.
            per_ip_cap: Maximum allowed per IP per day.
            global_cap: Maximum allowed globally per day.
            today: UTC date key override (defaults to the current UTC date).

        Returns:
            A Decision; counters are incremented only when allowed is True.
        """
        day = today or _utc_date()
        collection = self._client.collection(self._collection)
        global_ref = collection.document(f"global_{day}")
        ip_ref = collection.document(f"ip_{day}_{ip_hash}")

        @self._transactional
        def _run(transaction) -> Decision:
            global_snap = global_ref.get(transaction=transaction)
            ip_snap = ip_ref.get(transaction=transaction)
            global_count = (
                global_snap.to_dict().get("count", 0) if global_snap.exists else 0
            )
            ip_count = ip_snap.to_dict().get("count", 0) if ip_snap.exists else 0

            decision = _decide(global_count, ip_count, per_ip_cap, global_cap)
            if decision.allowed:
                transaction.set(
                    global_ref, {"count": global_count + 1}, merge=True
                )
                transaction.set(ip_ref, {"count": ip_count + 1}, merge=True)
            return decision

        return _run(self._client.transaction())
