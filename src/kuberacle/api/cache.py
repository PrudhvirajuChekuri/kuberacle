"""Firestore-backed exact-match answer cache.

A repeated question short-circuits the entire billable pipeline (gate, embed,
hybrid retrieval, rerank, generation). Entries are keyed by the exact normalized
question plus the served index version and an answer-config fingerprint, so an
index roll or a generation-config change auto-invalidates stale answers.

Matching is exact on normalized text, never semantic: a hit is provably the same
question, so it can never return a confidently-wrong answer for a merely-similar
one. Entries are written only after a stream completes cleanly server-side and
store the answer text, citations, abstention flags, the terminal outcome, and
the original estimated cost (replayed as the saved-cost estimate on a hit). An
``expires_at`` timestamp drives both a Firestore TTL policy and a read-time
expiry check, so correctness never depends on the policy's deletion lag.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from kuberacle.config import RAGConfig

logger = logging.getLogger(__name__)


def normalize_question(question: str) -> str:
    """Normalize a question for exact-match caching.

    Lowercases, collapses all runs of whitespace to single spaces, and strips
    trailing punctuation so trivial phrasing differences ("What is a Pod?" vs
    "what is a pod") map to the same key.

    Args:
        question: Raw user question.

    Returns:
        The normalized question text.
    """
    text = " ".join(question.lower().split())
    while text and not (text[-1].isalnum() or text[-1] == ")"):
        text = text[:-1]
    return text.strip()


def answer_config_version(config: RAGConfig, prompts: dict[str, dict]) -> str:
    """Fingerprint the config and prompts that determine a question's answer.

    Folds in every runtime input that can change the produced answer for the
    same question and corpus: the generation model and its sampling settings,
    the retrieval and reranker configuration, the citation policy, the relevance
    gate, and the **actual resolved prompt text** the running instance will use.

    The resolved text matters because prompts are fetched from Langfuse by the
    version *label*, which is mutable: a prompt edit under the same label changes
    answers without bumping ``config.prompts.version``. Hashing the real text
    (not just the label) makes such an edit auto-invalidate the cache, so a hit
    is always consistent with the prompts actually in use.

    Args:
        config: Running RAG configuration.
        prompts: Resolved prompt bundles the pipeline is using, keyed by role
            (for example ``{"answer": {...}, "gate": {...}}``).

    Returns:
        A short hex digest of the answer-affecting configuration and prompts.
    """
    retrieval = config.retrieval
    payload = {
        "prompt_version": config.prompts.version,
        "prompts": prompts,
        "generation_model": config.generation.model_id,
        "temperature": config.generation.temperature,
        "max_tokens": config.generation.max_tokens,
        "semantic_top_k": retrieval.semantic_top_k,
        "lexical_top_k": retrieval.lexical_top_k,
        "merged_top_k": retrieval.merged_top_k,
        "final_top_k": retrieval.final_top_k,
        "hybrid_weight_semantic": retrieval.hybrid_weight_semantic,
        "hybrid_weight_lexical": retrieval.hybrid_weight_lexical,
        "min_evidence_score": retrieval.min_evidence_score,
        "min_supporting_chunks": retrieval.min_supporting_chunks,
        "strict_used_only": config.citation.strict_used_only,
        "deduplicate": config.citation.deduplicate,
        "reranker_enabled": config.reranker.enabled,
        "reranker_model": config.reranker.model,
        "reranker_ranking_config": config.reranker.ranking_config,
        "gate_enabled": config.gate.enabled,
        "gate_model": config.gate.model_id,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()[:16]


def answer_cache_key(
    question: str, index_version: str, config_version: str
) -> str:
    """Build the Firestore document ID for a cached answer.

    Args:
        question: Raw user question (normalized internally).
        index_version: Served index version, so an index roll invalidates.
        config_version: Answer-config fingerprint from
            :func:`answer_config_version`.

    Returns:
        Hex sha256 over the normalized question and the two version strings.
    """
    digest = hashlib.sha256()
    digest.update(normalize_question(question).encode("utf-8"))
    digest.update(b"\0")
    digest.update(index_version.encode("utf-8"))
    digest.update(b"\0")
    digest.update(config_version.encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class CachedAnswer:
    """A replayable cached answer.

    Attributes:
        answer: Final answer text.
        citations: Citation payloads (already wire-shaped dicts).
        insufficient_evidence: Whether the answer was ungrounded.
        abstained: Whether the answer is an explicit abstention.
        outcome: Terminal outcome label recorded for the original request.
        cost_usd: Estimated cost of the original (uncached) request, replayed as
            the saved-cost estimate on a hit.
    """

    answer: str
    citations: list[dict]
    insufficient_evidence: bool
    abstained: bool
    outcome: str
    cost_usd: float


class AnswerCache:
    """Stores and retrieves cached answers in Firestore.

    The Firestore client is injectable so the logic can be unit-tested without
    the ``firestore`` package or a live database.
    """

    def __init__(
        self,
        project: str,
        database: str = "(default)",
        collection: str = "answer_cache",
        ttl_days: int = 14,
        client=None,
    ):
        """Initialize the answer cache.

        Args:
            project: GCP project ID (used only when creating a real client).
            database: Firestore database name.
            collection: Firestore collection holding cached answers.
            ttl_days: Days before a written entry expires.
            client: Optional pre-built Firestore client (for tests).
        """
        if client is None:
            from google.cloud import firestore

            client = firestore.Client(project=project, database=database)
        self._client = client
        self._collection = collection
        self._ttl_days = ttl_days

    def get(self, key: str) -> CachedAnswer | None:
        """Look up a cached answer by key.

        Args:
            key: Cache document ID from :func:`answer_cache_key`.

        Returns:
            The cached answer, or None when absent or expired.
        """
        snap = self._client.collection(self._collection).document(key).get()
        if not snap.exists:
            return None
        data = snap.to_dict()
        # Treat a missing expiry as expired: a write always sets ``expires_at``,
        # so its absence means a partial/foreign entry that must not be served.
        expires_at = data.get("expires_at")
        if expires_at is None or _as_utc(expires_at) <= datetime.now(timezone.utc):
            return None
        return CachedAnswer(
            answer=data.get("answer", ""),
            citations=data.get("citations", []),
            insufficient_evidence=data.get("insufficient_evidence", False),
            abstained=data.get("abstained", False),
            outcome=data.get("outcome", ""),
            cost_usd=data.get("cost_usd", 0.0),
        )

    def put(self, key: str, value: CachedAnswer) -> None:
        """Write a cached answer with an expiry timestamp.

        Args:
            key: Cache document ID from :func:`answer_cache_key`.
            value: The answer to cache.
        """
        now = datetime.now(timezone.utc)
        self._client.collection(self._collection).document(key).set(
            {
                "answer": value.answer,
                "citations": value.citations,
                "insufficient_evidence": value.insufficient_evidence,
                "abstained": value.abstained,
                "outcome": value.outcome,
                "cost_usd": value.cost_usd,
                "created_at": now,
                "expires_at": now + timedelta(days=self._ttl_days),
            }
        )


def _as_utc(value: datetime) -> datetime:
    """Coerce a Firestore timestamp to a timezone-aware UTC datetime.

    Args:
        value: A datetime read back from Firestore (tz-aware in production;
            possibly naive in tests).

    Returns:
        The same instant as a UTC-aware datetime.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
