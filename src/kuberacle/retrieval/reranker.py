"""Discovery Engine reranker with ADC bearer token auth."""

import logging
from typing import Any

from kuberacle.domain import RetrievedChunk
from kuberacle.observability import context as obs
from kuberacle.retrieval.constants import DISCOVERY_ENGINE_RANK_URL

logger = logging.getLogger(__name__)

#: HTTP timeout (seconds) for the Discovery Engine rank call, so a hung upstream
#: cannot pin a worker; on timeout the request falls back to hybrid ordering.
_RANK_TIMEOUT = 10.0


class DiscoveryEngineReranker:
    """Rerank query/chunk candidates using the Discovery Engine Ranking API.

    Falls back to incoming ranking if the API call fails.

    Args:
        gcp_project: GCP project ID.
        ranking_config: Discovery Engine ranking config name.
        model: Ranker model string sent in the request body.
        enabled: Whether reranking is active.
        http_session: Optional injected requests.Session for testing.
    """

    def __init__(
        self,
        gcp_project: str,
        ranking_config: str,
        model: str,
        enabled: bool = True,
        http_session: Any = None,
    ) -> None:
        self.gcp_project = gcp_project
        self.ranking_config = ranking_config
        self.model = model
        self.enabled = enabled
        self._session = http_session
        self._credentials: Any = None

    @property
    def session(self) -> Any:
        """Lazily initialize and return the requests session."""
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    def _get_bearer_token(self) -> str:
        """Return a valid ADC bearer token, refreshing only when expired.

        Returns:
            Bearer token string.
        """
        import google.auth
        import google.auth.transport.requests

        if self._credentials is None:
            self._credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        if not self._credentials.valid:
            self._credentials.refresh(google.auth.transport.requests.Request())
        return self._credentials.token

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Rerank chunks by query relevance.

        Args:
            query: User search question.
            chunks: Candidate chunks to rerank.
            top_k: Number of top chunks to return.

        Returns:
            Reranked chunk list, or original order on failure.
        """
        if not self.enabled or not chunks:
            logger.debug("Reranker disabled or no candidates; returning top_%d unchanged", top_k)
            return chunks[:top_k]

        url = DISCOVERY_ENGINE_RANK_URL.format(
            project=self.gcp_project,
            ranking_config=self.ranking_config,
        )
        records = [
            {"id": str(i), "content": chunk.content}
            for i, chunk in enumerate(chunks)
        ]
        payload = {
            "model": self.model,
            "query": query,
            "records": records,
            "topN": top_k,
        }

        logger.debug("Reranking %d candidates for query=%r", len(chunks), query[:80])
        try:
            token = self._get_bearer_token()
            response = self.session.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {token}"},
                timeout=_RANK_TIMEOUT,
            )
            response.raise_for_status()
            results = response.json().get("records", [])
            # A successful rank call is one billable ranking query.
            obs.record_rerank()
        except Exception as exc:
            logger.warning("Reranker API call failed: %s; falling back to hybrid scores", exc)
            return chunks[:top_k]

        if not results:
            return chunks[:top_k]

        reranked: list[RetrievedChunk] = []
        seen_indices: set[int] = set()
        for item in results:
            try:
                idx = int(item["id"])
            except (ValueError, TypeError, KeyError):
                continue
            if idx < 0 or idx >= len(chunks) or idx in seen_indices:
                continue
            seen_indices.add(idx)
            base = chunks[idx]
            reranked.append(
                RetrievedChunk(
                    chunk_id=base.chunk_id,
                    content=base.content,
                    metadata=base.metadata,
                    score=float(item.get("score", base.score)),
                )
            )
        final = reranked[:top_k] if reranked else chunks[:top_k]
        logger.debug("Reranker returned %d results", len(final))
        return final
