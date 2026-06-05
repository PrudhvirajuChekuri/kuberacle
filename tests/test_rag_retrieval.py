"""Tests for retrieval and QA orchestration."""

from k8s_rag.ingestion.schemas import RetrievedChunk
from k8s_rag.retrieval.bm25 import BM25Retriever
from k8s_rag.retrieval.generator import VertexAIAnswerGenerator, extract_citation_indices
from k8s_rag.retrieval.hybrid import merge_hybrid_candidates
from k8s_rag.retrieval.qa import RAGQASystem
from k8s_rag.retrieval.reranker import DiscoveryEngineReranker
from k8s_rag.retrieval.retriever import HybridRetriever, SemanticRetriever


class FakeEmbedder:
    """Deterministic embedder test double."""

    def embed_text(self, text):
        return [float(len(text))]


class FakeVectorStore:
    """Deterministic retriever backend test double."""

    def query(self, query_embedding, top_k):
        del query_embedding, top_k
        return [
            RetrievedChunk(
                chunk_id="pods::what-is",
                content="A Pod is the smallest deployable unit in Kubernetes.",
                metadata={"source_url": "https://kubernetes.io/docs/concepts/workloads/pods/"},
                score=0.9,
            )
        ]


class FakeGenAIClient:
    """Fake Gen AI client for generator tests."""

    class _Models:
        def generate_content(self, **kwargs):
            del kwargs

            class _Response:
                text = "Pods run one or more containers [1]."

            return _Response()

    def __init__(self):
        self.models = self._Models()


class EmptyVectorStore:
    """Returns no retrievals to test insufficient evidence branch."""

    def query(self, query_embedding, top_k):
        del query_embedding, top_k
        return []


class FixedAnswerGenerator:
    """Deterministic generator for citation enforcement tests."""

    def __init__(self, answer):
        self.answer = answer

    def generate(self, question, chunks):
        del question, chunks
        return self.answer


class FakeBM25Retriever:
    """Returns a fixed chunk list for BM25 tests."""

    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, query, top_k=None):
        del query
        k = top_k if top_k is not None else len(self.chunks)
        return self.chunks[:k]


class FakeReranker:
    """Passthrough reranker that records calls."""

    def __init__(self):
        self.calls = []

    def rerank(self, query, chunks, top_k):
        self.calls.append({"query": query, "n_candidates": len(chunks), "top_k": top_k})
        return chunks[:top_k]


def _make_chunk(chunk_id: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=f"content of {chunk_id}",
        metadata={"source_url": f"https://kubernetes.io/docs/{chunk_id}"},
        score=score,
    )


def test_semantic_retriever_returns_ranked_chunks():
    """Retriever should return rows from vector store."""
    retriever = SemanticRetriever(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        top_k=5,
    )
    rows = retriever.retrieve("What is a Pod?")
    assert len(rows) == 1
    assert rows[0].chunk_id == "pods::what-is"


def test_hybrid_retriever_merges_and_reranks():
    """HybridRetriever should fuse semantic and lexical results then rerank."""
    semantic_chunks = [_make_chunk("a", 0.9), _make_chunk("b", 0.7)]
    lexical_chunks = [_make_chunk("b", 0.8), _make_chunk("c", 0.6)]
    reranker = FakeReranker()

    retriever = HybridRetriever(
        semantic_retriever=SemanticRetriever(
            embedder=FakeEmbedder(),
            vector_store=FakeVectorStore(),
            top_k=2,
        ),
        bm25_retriever=FakeBM25Retriever(lexical_chunks),
        reranker=reranker,
        semantic_top_k=2,
        lexical_top_k=2,
        merged_top_k=5,
        final_top_k=2,
        semantic_weight=0.6,
        lexical_weight=0.4,
    )
    results = retriever.retrieve("What is a Pod?")

    assert len(results) == 2
    assert reranker.calls[0]["top_k"] == 2


def test_hybrid_retriever_top_k_override_capped_at_merged_pool():
    """top_k override larger than merged pool should be silently capped."""
    lexical_chunks = [_make_chunk("x", 0.5)]
    reranker = FakeReranker()

    retriever = HybridRetriever(
        semantic_retriever=SemanticRetriever(
            embedder=FakeEmbedder(),
            vector_store=FakeVectorStore(),
            top_k=2,
        ),
        bm25_retriever=FakeBM25Retriever(lexical_chunks),
        reranker=reranker,
        semantic_top_k=2,
        lexical_top_k=1,
        merged_top_k=3,
        final_top_k=2,
        semantic_weight=0.6,
        lexical_weight=0.4,
    )
    results = retriever.retrieve("What is a Pod?", top_k=99)

    # merged pool has at most 2 unique chunks (FakeVectorStore returns 1, lexical 1)
    assert len(results) <= reranker.calls[0]["n_candidates"]
    assert reranker.calls[0]["top_k"] <= reranker.calls[0]["n_candidates"]


def test_qa_system_returns_answer_with_citations():
    """QA system should return model answer and citation records."""
    retriever = SemanticRetriever(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        top_k=5,
    )
    generator = VertexAIAnswerGenerator(
        model_id="gemini-2.5-flash-lite",
        gcp_project="test-project",
        gcp_location="us-central1",
        genai_client=FakeGenAIClient(),
    )
    qa = RAGQASystem(retriever=retriever, generator=generator)
    result = qa.ask("What is a Pod?")
    assert "Pods run one or more containers" in result.answer
    assert len(result.citations) == 1
    assert result.citations[0].source_url.startswith("https://kubernetes.io/docs/")


def test_qa_system_handles_insufficient_evidence():
    """QA system should return refusal when no chunks are retrieved."""
    retriever = SemanticRetriever(
        embedder=FakeEmbedder(),
        vector_store=EmptyVectorStore(),
        top_k=5,
    )
    generator = VertexAIAnswerGenerator(
        model_id="gemini-2.5-flash-lite",
        gcp_project="test-project",
        gcp_location="us-central1",
        genai_client=FakeGenAIClient(),
    )
    qa = RAGQASystem(retriever=retriever, generator=generator)
    result = qa.ask("Unknown question?")
    assert result.answer.startswith("INSUFFICIENT_EVIDENCE")
    assert result.citations == []


def test_extract_citation_indices_ordered_unique():
    """Citation parser should return ordered unique indices."""
    assert extract_citation_indices("Fact [2] and [1], again [2].") == [2, 1]


def test_qa_system_outputs_only_used_citations():
    """Citation list should include only indices used by answer."""
    retriever = SemanticRetriever(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        top_k=5,
    )
    generator = FixedAnswerGenerator("Answer text [1].")
    qa = RAGQASystem(
        retriever=retriever,
        generator=generator,
        strict_used_only=True,
        deduplicate_citations=True,
    )
    result = qa.ask("Question?")
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == "pods::what-is"


class TwoChunkVectorStore:
    """Returns two chunks: one above and one below a score threshold."""

    def query(self, query_embedding, top_k):
        del query_embedding, top_k
        return [
            RetrievedChunk("high", "High score chunk.", {"source_url": "u/high"}, score=0.9),
            RetrievedChunk("low", "Low score chunk.", {"source_url": "u/low"}, score=0.1),
        ]


def test_qa_system_citations_exclude_low_score_chunks():
    """Citations should only include chunks that pass min_evidence_score."""
    retriever = SemanticRetriever(
        embedder=FakeEmbedder(),
        vector_store=TwoChunkVectorStore(),
        top_k=5,
    )
    generator = FixedAnswerGenerator("Answer [1] and [2].")
    qa = RAGQASystem(
        retriever=retriever,
        generator=generator,
        strict_used_only=True,
        min_evidence_score=0.5,
        min_supporting_chunks=1,
    )
    result = qa.ask("Question?")
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == "high"


def test_merge_hybrid_candidates_chunk_in_both_lists_scores_higher():
    """A chunk found by both retrievers should outscore one found by only one."""
    shared = _make_chunk("shared", score=0.9)
    semantic_only = _make_chunk("semantic_only", score=0.8)
    lexical_only = _make_chunk("lexical_only", score=0.8)

    merged = merge_hybrid_candidates(
        semantic_chunks=[shared, semantic_only],
        lexical_chunks=[shared, lexical_only],
        semantic_weight=0.6,
        lexical_weight=0.4,
        top_k=3,
    )
    by_id = {c.chunk_id: c.score for c in merged}
    assert by_id["shared"] > by_id["semantic_only"]
    assert by_id["shared"] > by_id["lexical_only"]


def test_merge_hybrid_candidates_one_empty_input():
    """Fusion should work correctly when one retriever returns nothing."""
    chunks = [_make_chunk("a", 0.9), _make_chunk("b", 0.5)]

    merged = merge_hybrid_candidates(
        semantic_chunks=chunks,
        lexical_chunks=[],
        semantic_weight=0.6,
        lexical_weight=0.4,
        top_k=5,
    )
    assert len(merged) == 2
    ids = [c.chunk_id for c in merged]
    assert "a" in ids and "b" in ids


def test_merge_hybrid_candidates_both_empty():
    """Fusion of two empty lists should return empty list."""
    merged = merge_hybrid_candidates(
        semantic_chunks=[],
        lexical_chunks=[],
        semantic_weight=0.6,
        lexical_weight=0.4,
        top_k=5,
    )
    assert merged == []


def test_merge_hybrid_candidates_deduplicates_chunk_ids():
    """Hybrid merge should dedupe matching semantic and lexical ids."""
    semantic = [
        RetrievedChunk("a", "doc a", {"source_url": "u/a"}, 0.9),
        RetrievedChunk("b", "doc b", {"source_url": "u/b"}, 0.8),
    ]
    lexical = [
        RetrievedChunk("a", "doc a", {"source_url": "u/a"}, 12.0),
        RetrievedChunk("c", "doc c", {"source_url": "u/c"}, 11.0),
    ]
    merged = merge_hybrid_candidates(
        semantic_chunks=semantic,
        lexical_chunks=lexical,
        semantic_weight=0.6,
        lexical_weight=0.4,
        top_k=10,
    )
    ids = [chunk.chunk_id for chunk in merged]
    assert len(ids) == len(set(ids))
    assert "a" in ids and "b" in ids and "c" in ids


def test_bm25_retriever_returns_matching_chunks():
    """BM25Retriever should rank chunks by lexical relevance."""
    chunks = [
        _make_chunk("pods"),
        _make_chunk("services"),
        _make_chunk("configmaps"),
    ]
    chunks[0] = RetrievedChunk("pods", "A Pod runs containers.", {}, 0.0)
    chunks[1] = RetrievedChunk("services", "A Service exposes pods.", {}, 0.0)
    chunks[2] = RetrievedChunk("configmaps", "A ConfigMap stores config.", {}, 0.0)

    retriever = BM25Retriever(chunks, top_k=2)
    results = retriever.retrieve("pod containers")

    ids = [r.chunk_id for r in results]
    assert "pods" in ids
    assert all(r.score > 0 for r in results)


def test_bm25_retriever_empty_corpus_returns_empty():
    """BM25Retriever should not crash and return empty list for empty corpus."""
    retriever = BM25Retriever([], top_k=5)
    assert retriever.retrieve("pod") == []


def test_bm25_retriever_no_match_returns_empty():
    """BM25Retriever should return empty list when no chunks match the query."""
    chunks = [RetrievedChunk("a", "Pod runs containers.", {}, 0.0)]
    retriever = BM25Retriever(chunks, top_k=5)
    results = retriever.retrieve("xyzzy_nonexistent_token")
    assert results == []


def test_bm25_retriever_empty_query_returns_empty():
    """BM25Retriever should return empty list for a query with no tokens."""
    chunks = [RetrievedChunk("a", "Pod runs containers.", {}, 0.0)]
    retriever = BM25Retriever(chunks, top_k=5)
    assert retriever.retrieve("!!!") == []


# --- DiscoveryEngineReranker tests ---


class _FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class _FakeSession:
    def __init__(self, response_data):
        self._data = response_data

    def post(self, url, json, headers):
        return _FakeResponse(self._data)


class _FailingSession:
    def post(self, url, json, headers):
        raise RuntimeError("network error")


def _make_reranker(session, enabled=True):
    return DiscoveryEngineReranker(
        gcp_project="proj",
        ranking_config="cfg",
        model="semantic-ranker-default@latest",
        enabled=enabled,
        http_session=session,
    )


def test_reranker_disabled_returns_original_order():
    """Disabled reranker should return first top_k chunks unchanged."""
    chunks = [_make_chunk("a", 0.9), _make_chunk("b", 0.7), _make_chunk("c", 0.5)]
    reranker = _make_reranker(session=None, enabled=False)
    results = reranker.rerank("query", chunks, top_k=2)
    assert [r.chunk_id for r in results] == ["a", "b"]


def test_reranker_happy_path_reorders_and_sets_scores(monkeypatch):
    """Reranker should reorder chunks and apply API scores."""
    chunks = [_make_chunk("a", 0.5), _make_chunk("b", 0.4), _make_chunk("c", 0.3)]
    # API says chunk index 1 is best, then 0
    session = _FakeSession({"records": [
        {"id": "1", "score": 0.95},
        {"id": "0", "score": 0.80},
    ]})
    reranker = _make_reranker(session)
    monkeypatch.setattr(reranker, "_get_bearer_token", lambda: "fake-token")

    results = reranker.rerank("query", chunks, top_k=2)

    assert results[0].chunk_id == "b"
    assert results[0].score == 0.95
    assert results[1].chunk_id == "a"
    assert results[1].score == 0.80


def test_reranker_falls_back_on_network_error(monkeypatch):
    """Reranker should return original top_k order when API call fails."""
    chunks = [_make_chunk("a", 0.9), _make_chunk("b", 0.7), _make_chunk("c", 0.5)]
    reranker = _make_reranker(_FailingSession())
    monkeypatch.setattr(reranker, "_get_bearer_token", lambda: "fake-token")

    results = reranker.rerank("query", chunks, top_k=2)
    assert [r.chunk_id for r in results] == ["a", "b"]


def test_reranker_deduplicates_repeated_api_ids(monkeypatch):
    """Reranker should ignore duplicate IDs in the API response."""
    chunks = [_make_chunk("a", 0.5), _make_chunk("b", 0.4)]
    session = _FakeSession({"records": [
        {"id": "0", "score": 0.9},
        {"id": "0", "score": 0.8},  # duplicate
    ]})
    reranker = _make_reranker(session)
    monkeypatch.setattr(reranker, "_get_bearer_token", lambda: "fake-token")

    results = reranker.rerank("query", chunks, top_k=5)
    assert len(results) == 1
    assert results[0].chunk_id == "a"
