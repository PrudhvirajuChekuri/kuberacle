"""Bedrock reranker integration with safe fallback."""

import json
from typing import Any

from k8s_rag.ingestion.schemas import RetrievedChunk


class BedrockReranker:
    """Rerank query/chunk candidates using Bedrock model.

    Falls back to incoming ranking if rerank invocation is unavailable.
    """

    def __init__(
        self,
        model_id: str,
        region_name: str,
        enabled: bool = True,
        bedrock_client: Any = None,
    ) -> None:
        self.model_id = model_id
        self.region_name = region_name
        self.enabled = enabled
        self._client = bedrock_client

    @property
    def client(self) -> Any:
        """Lazily initialize Bedrock runtime client."""
        if self._client is None:
            import boto3

            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self.region_name,
            )
        return self._client

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Rerank chunks by query relevance."""
        if not self.enabled or not chunks:
            return chunks[:top_k]

        try:
            body = json.dumps(
                {
                    "query": query,
                    "documents": [{"text": chunk.content} for chunk in chunks],
                    "top_n": top_k,
                }
            )
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            payload = json.loads(response["body"].read())
            results = payload.get("results", [])
            if not results:
                return chunks[:top_k]

            reranked: list[RetrievedChunk] = []
            for item in results:
                idx = int(item.get("index", -1))
                if idx < 0 or idx >= len(chunks):
                    continue
                base = chunks[idx]
                reranked.append(
                    RetrievedChunk(
                        chunk_id=base.chunk_id,
                        content=base.content,
                        metadata=base.metadata,
                        score=float(item.get("relevance_score", base.score)),
                    )
                )
            return reranked[:top_k] if reranked else chunks[:top_k]
        except Exception:
            # Keep the pipeline resilient; use hybrid scores if reranker fails.
            return chunks[:top_k]
