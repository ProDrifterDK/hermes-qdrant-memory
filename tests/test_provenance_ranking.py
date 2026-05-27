from __future__ import annotations

import json

from __init__ import QdrantMemoryProvider
from qdrant_memory.ranking import RankingContext, rank_memory_candidate
from qdrant_memory.retriever import MemoryRetriever


class FakeEmbedding:
    def __init__(self):
        self.queries = []

    def embed_query(self, text):
        self.queries.append(text)
        return [0.1, 0.2]


class FakeQdrant:
    def __init__(self, results):
        self.results = results
        self.searches = []
        self.payload_updates = []

    def search(self, name, vector, limit, filter=None, with_payload=True, with_vector=False):
        self.searches.append(
            {
                "name": name,
                "vector": vector,
                "limit": limit,
                "filter": filter,
                "with_payload": with_payload,
                "with_vector": with_vector,
            }
        )
        return self.results

    def update_payload(self, name, point_id, payload):
        self.payload_updates.append((name, point_id, payload))
        return {"status": "ok"}


def _payload(**overrides):
    payload = {
        "text": "TeamForge MCP binary is teamforge-mcp.",
        "importance": 8,
        "created_at": "2026-01-01T00:00:00+00:00",
        "memory_kind": "assertion",
        "fact_status": "active",
        "canonical": False,
        "stale": False,
        "requires_review": False,
        "source_hash_current": False,
        "derivation_depth": 0,
    }
    payload.update(overrides)
    return payload


def test_policy_penalizes_stale_review_and_inactive_status_by_default():
    context = RankingContext(query="current TeamForge MCP binary")
    clean = rank_memory_candidate(base_score=0.8, vector_score=0.91, payload=_payload(), context=context)

    stale = rank_memory_candidate(base_score=0.8, vector_score=0.91, payload=_payload(stale=True), context=context)
    review = rank_memory_candidate(base_score=0.8, vector_score=0.91, payload=_payload(requires_review=True), context=context)
    status_stale = rank_memory_candidate(base_score=0.8, vector_score=0.91, payload=_payload(fact_status="stale"), context=context)
    status_review = rank_memory_candidate(
        base_score=0.8,
        vector_score=0.91,
        payload=_payload(fact_status="review_required"),
        context=context,
    )
    disputed = rank_memory_candidate(base_score=0.8, vector_score=0.91, payload=_payload(fact_status="disputed"), context=context)
    superseded = rank_memory_candidate(base_score=0.8, vector_score=0.91, payload=_payload(fact_status="superseded"), context=context)
    deprecated = rank_memory_candidate(base_score=0.8, vector_score=0.91, payload=_payload(fact_status="deprecated"), context=context)

    assert stale.score < clean.score
    assert review.score < clean.score
    assert status_stale.score < clean.score
    assert status_review.score < clean.score
    assert disputed.score < clean.score
    assert superseded.score < clean.score
    assert deprecated.score < clean.score
    assert "stale" in stale.debug["penalties"]
    assert "requires_review" in review.debug["penalties"]
    assert "fact_status:stale" in status_stale.debug["penalties"]
    assert "fact_status:review_required" in status_review.debug["penalties"]
    assert "fact_status:disputed" in disputed.debug["penalties"]


def test_policy_does_not_penalize_review_or_history_queries():
    clean = rank_memory_candidate(
        base_score=0.8,
        vector_score=0.91,
        payload=_payload(),
        context=RankingContext(query="current TeamForge MCP binary"),
    )
    explicit_history = rank_memory_candidate(
        base_score=0.8,
        vector_score=0.91,
        payload=_payload(stale=True, requires_review=True, fact_status="disputed"),
        context=RankingContext(query="current TeamForge MCP binary", include_fact_history=True),
    )
    query_history = rank_memory_candidate(
        base_score=0.8,
        vector_score=0.91,
        payload=_payload(stale=True, requires_review=True, fact_status="disputed"),
        context=RankingContext(query="review TeamForge MCP binary history"),
    )

    assert explicit_history.score == clean.score
    assert query_history.score == clean.score
    assert explicit_history.debug["penalties"] == {}
    assert query_history.debug["review_history_requested"] is True

    for fact_status in ("stale", "review_required"):
        explicit_status_history = rank_memory_candidate(
            base_score=0.8,
            vector_score=0.91,
            payload=_payload(fact_status=fact_status),
            context=RankingContext(query="current TeamForge MCP binary", include_fact_history=True),
        )
        query_status_history = rank_memory_candidate(
            base_score=0.8,
            vector_score=0.91,
            payload=_payload(fact_status=fact_status),
            context=RankingContext(query="review TeamForge MCP binary history"),
        )

        assert explicit_status_history.score == clean.score
        assert query_status_history.score == clean.score
        assert explicit_status_history.debug["penalties"] == {}
        assert query_status_history.debug["penalties"] == {}
        assert query_status_history.debug["review_history_requested"] is True


def test_policy_boosts_canonical_fresh_filters_and_exact_fact_matches():
    context = RankingContext(
        query="teamforge.mcp.binary",
        source_filter="ops.md",
        project_path_filter="/repo",
        subject="TeamForge MCP binary",
        fact_key="teamforge.mcp.binary",
    )
    matching = rank_memory_candidate(
        base_score=0.7,
        vector_score=0.88,
        payload=_payload(
            canonical=True,
            source_hash_current=True,
            source="ops.md",
            project_path="/repo",
            subject="TeamForge MCP binary",
            fact_key="teamforge.mcp.binary",
        ),
        context=context,
    )
    nonmatching = rank_memory_candidate(
        base_score=0.7,
        vector_score=0.88,
        payload=_payload(source="other.md", project_path="/other", subject="Other", fact_key="other.fact"),
        context=context,
    )

    assert matching.score > nonmatching.score
    for boost in (
        "canonical",
        "source_hash_current",
        "source_filter",
        "project_path_filter",
        "exact_subject",
        "exact_fact_key",
    ):
        assert boost in matching.debug["boosts"]


def test_policy_prefers_direct_source_backed_assertions_over_long_derivation_chains_when_confidence_equal():
    context = RankingContext(query="TeamForge MCP binary")
    direct = rank_memory_candidate(
        base_score=0.75,
        vector_score=0.9,
        payload=_payload(source_uri="file:///repo/ops.md", content_hash="sha256:abc", derivation_depth=0),
        context=context,
    )
    derived = rank_memory_candidate(
        base_score=0.75,
        vector_score=0.9,
        payload=_payload(derived_from=["p1", "p2", "p3"], derivation_depth=3),
        context=context,
    )

    assert direct.score > derived.score
    assert derived.debug["derivation_depth"] == 3
    assert "derivation_depth" in derived.debug["penalties"]


def test_memory_retriever_applies_policy_and_preserves_raw_vector_score_on_chunk():
    qdrant = FakeQdrant(
        [
            {"id": "derived", "score": 0.82, "payload": _payload(text="Derived answer", derivation_depth=4)},
            {"id": "direct", "score": 0.82, "payload": _payload(text="Direct answer", derivation_depth=0, source_uri="file:///repo/ops.md")},
        ]
    )
    retriever = MemoryRetriever(
        qdrant=qdrant,
        embeddings=FakeEmbedding(),
        collection_name="memory",
        search_candidates=2,
        decay_rate=0.0,
    )

    results = retriever.search("TeamForge MCP binary", top_k=2)

    assert [result.id for result in results] == ["direct", "derived"]
    assert results[0].qdrant_score == 0.82
    assert results[0].ranking_debug["vector_score"] == 0.82
    assert results[0].ranking_debug["base_score"] <= results[0].final_score


def test_search_tool_json_includes_vector_score_and_ranking_debug_for_audit():
    qdrant = FakeQdrant([{"id": "memory-1", "score": 0.87, "payload": _payload(text="Auditable result")}])
    provider = QdrantMemoryProvider()
    provider._retriever = MemoryRetriever(
        qdrant=qdrant,
        embeddings=FakeEmbedding(),
        collection_name="memory",
        search_candidates=1,
        decay_rate=0.0,
    )

    response = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_search",
            {"query": "TeamForge MCP binary", "top_k": 1, "include_metadata": True},
        )
    )

    result = response["results"][0]
    assert result["vector_score"] == 0.87
    assert result["ranking"]["vector_score"] == 0.87
    assert result["score"] == result["ranking"]["score"]
