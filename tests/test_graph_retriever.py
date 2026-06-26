"""Deterministic tests for the read-only graph-aware retriever.

Covers:
- Roadmap acceptance case: graph-connected weak-vector memory promoted above
  unrelated high-vector-score candidate.
- Hard caps (depth, neighbors, total expansion).
- Fact-status filtering of graph edges.
- Empty graph path (no entity refs).
- No mutation safety (zero writes).
- Schema invalid relation_type dropped.
- Graph-distance decay ordering.
"""

from __future__ import annotations

import json
from argparse import Namespace
from typing import Any

import pytest

from qdrant_memory.graph_retriever import (
    GraphExpansionPolicy,
    GraphMemoryRetriever,
    GraphRankWeights,
    GraphSearchResult,
    _extract_entity_refs,
    _get_counterparty,
    _query_alias_matches,
)
from qdrant_memory.graph_schema import make_entity_id, make_edge_id
from qdrant_memory.retriever import MemoryRetriever, RetrievedMemory


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeEmbedding:
    def __init__(self):
        self.queries: list[str] = []

    def embed_query(self, text: str):
        self.queries.append(text)
        return [0.1, 0.2]


class FakeGraphQdrant:
    """In-memory fake Qdrant that supports search, scroll_by_filter, retrieve.

    Tracks all mutation calls (upsert, update_payload, delete) so tests can
    assert zero writes from graph search.
    """

    def __init__(self, search_results=None):
        self.search_results = search_results or []
        self._store: dict[str, dict[str, Any]] = {}  # point_id -> {id, score, payload}
        self.searches: list[dict] = []
        self.scrolls: list[dict] = []
        self.retrieves: list[dict] = []
        self.upserts: list[dict] = []
        self.payload_updates: list[dict] = []
        self.deletes: list[dict] = []
        self.filter_deletes: list[dict] = []

    def add_point(self, point_id: str, payload: dict[str, Any], score: float = 0.5) -> None:
        self._store[point_id] = {"id": point_id, "score": score, "payload": payload}

    def search(self, name, vector, limit, filter=None, with_payload=True, with_vector=False):
        self.searches.append({
            "name": name, "vector": vector, "limit": limit,
            "filter": filter, "with_payload": with_payload, "with_vector": with_vector,
        })
        return list(self.search_results)

    def scroll_by_filter(self, name, filter, *, limit=256, with_payload=True, with_vector=False, max_total=None):
        self.scrolls.append({
            "name": name, "filter": filter, "limit": limit,
            "max_total": max_total,
        })
        # Match points from store against filter
        results = []
        must = filter.get("must", []) if filter else []
        for point_id, point in self._store.items():
            payload = point.get("payload") or {}
            if self._matches_filter(payload, must):
                results.append(point)
        if max_total:
            results = results[:max_total]
        else:
            results = results[:limit]
        return results

    def retrieve(self, name, ids, *, with_payload=True, with_vector=False):
        self.retrieves.append({"name": name, "ids": ids})
        results = []
        for pid in ids:
            if pid in self._store:
                results.append(self._store[pid])
        return results

    def upsert(self, name, points):
        self.upserts.append({"name": name, "points": points})
        return {"status": "ok"}

    def update_payload(self, name, point_id, payload):
        self.payload_updates.append({"name": name, "point_id": point_id, "payload": payload})
        return {"status": "ok"}

    def delete_ids(self, name, ids):
        self.deletes.append({"name": name, "ids": ids})
        return {"status": "ok"}

    def delete_filter(self, name, filter):
        self.filter_deletes.append({"name": name, "filter": filter})
        return {"status": "ok"}

    @staticmethod
    def _matches_filter(payload: dict[str, Any], must: list) -> bool:
        for condition in must:
            key = condition.get("key")
            match = condition.get("match", {})
            if "value" in match:
                if str(payload.get(key)) != str(match["value"]):
                    return False
            elif "any" in match:
                values = {str(v) for v in match["any"]}
                if str(payload.get(key)) not in values:
                    return False
        return True


# ---------------------------------------------------------------------------
# Test 1: Roadmap acceptance case — graph promotion beats high-vector-score
# ---------------------------------------------------------------------------

class TestGraphPromotionAcceptance:
    """A semantically weak but graph-connected memory outranks an unrelated
    higher-vector-score candidate."""

    def test_graph_connected_weak_vector_promoted_above_decoy(self):
        # Setup entity IDs
        deploy_eid = make_entity_id("workflow", "deploy pipeline")
        kubectl_eid = make_entity_id("tool", "kubectl")

        # Seed set: semantic search returns 3 points
        seeds = [
            {
                "id": "seed-strong",
                "score": 0.92,
                "payload": {
                    "text": "deploy script uses kubectl apply",
                    "source_type": "project_doc",
                    "importance": 8,
                    "created_at": "2026-06-20T00:00:00+00:00",
                    "entity_id": deploy_eid,
                },
            },
            {
                "id": "seed-medium",
                "score": 0.80,
                "payload": {
                    "text": "release notes mention rollback procedure",
                    "source_type": "project_doc",
                    "importance": 6,
                    "created_at": "2026-06-20T00:00:00+00:00",
                    "entity_id": deploy_eid,
                },
            },
            {
                "id": "seed-decoy",
                "score": 0.95,
                "payload": {
                    "text": "kubectl is a kubernetes CLI tool",
                    "source_type": "project_doc",
                    "importance": 5,
                    "created_at": "2026-06-20T00:00:00+00:00",
                    # No entity_id — unrelated
                },
            },
        ]

        qdrant = FakeGraphQdrant(search_results=seeds)

        # Add graph entity points
        qdrant.add_point("entity-deploy", {
            "memory_kind": "graph_entity",
            "entity_id": deploy_eid,
            "entity_type": "workflow",
            "label": "deploy pipeline",
            "source_point_ids": ["seed-strong", "seed-medium"],
            "confidence": 0.9,
            "canonical": False,
            "requires_review": True,
            "fact_status": "active",
        })

        qdrant.add_point("entity-kubectl", {
            "memory_kind": "graph_entity",
            "entity_id": kubectl_eid,
            "entity_type": "tool",
            "label": "kubectl",
            "source_point_ids": ["kubectl-notes"],
            "confidence": 0.9,
            "canonical": False,
            "requires_review": True,
            "fact_status": "active",
        })

        # Edge: deploy-pipeline USES_TOOL kubectl
        edge_id = make_edge_id(deploy_eid, kubectl_eid, "USES_TOOL")
        qdrant.add_point(edge_id, {
            "memory_kind": "graph_edge",
            "edge_id": edge_id,
            "source_entity_id": deploy_eid,
            "target_entity_id": kubectl_eid,
            "relation_type": "USES_TOOL",
            "confidence": 0.85,
            "usefulness_weight": 0.7,
            "fact_status": "active",
            "canonical": False,
            "requires_review": True,
        })

        # Target memory point: kubectl-notes (semantically weak but graph-connected)
        qdrant.add_point("kubectl-notes", {
            "text": "kubectl apply --record flags for audit trail",
            "source_type": "project_doc",
            "importance": 7,
            "created_at": "2026-06-20T00:00:00+00:00",
            "fact_status": "active",
        })

        retriever = GraphMemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
            expansion_policy=GraphExpansionPolicy(max_depth=2, max_neighbors_per_node=8, max_total_expansion=64),
            rank_weights=GraphRankWeights(),
        )

        result = retriever.search(
            "how do we run the deploy?",
            top_k=3,
            candidate_seed_top_k=5,
            max_graph_results=5,
            debug=True,
        )

        # Acceptance assertions:
        # 1. kubectl-notes should appear in results (promoted via graph)
        result_ids = [c.point_id for c in result.final]

        # 2. seed-decoy should NOT appear above kubectl-notes
        assert "kubectl-notes" in result_ids, f"kubectl-notes not in results: {result_ids}"

        kubectl_idx = result_ids.index("kubectl-notes")
        decoy_idx = result_ids.index("seed-decoy") if "seed-decoy" in result_ids else len(result_ids)
        assert kubectl_idx < decoy_idx, (
            f"kubectl-notes should rank above seed-decoy: {result_ids}"
        )

        # 3. Debug must be present and transparent
        assert result.debug["algorithm"] == "graph_v1"
        assert result.debug["stages"]["B_entity_extraction"]["matched_from_seeds"] > 0
        assert result.debug["stages"]["C_edge_query"]["depth_1_edges"] > 0

        # 4. Per-candidate debug must explain the score
        kubectl_candidate = next(c for c in result.final if c.point_id == "kubectl-notes")
        comps = kubectl_candidate.debug["component_scores"]
        assert comps["graph_distance"] > 0.0, "graph_distance component must be nonzero"
        assert comps["edge_confidence"] > 0.0, "edge_confidence component must be nonzero"
        assert kubectl_candidate.debug["graph_distance"] == 1


# ---------------------------------------------------------------------------
# Test 2: Hard caps are enforced
# ---------------------------------------------------------------------------

class TestHardCaps:
    def test_neighbor_cap_enforced(self):
        seed_eid = make_entity_id("concept", "hub")
        neighbor_eids = [make_entity_id("concept", f"neighbor-{i}") for i in range(200)]

        seeds = [{
            "id": "seed-1",
            "score": 0.9,
            "payload": {
                "text": "hub entity",
                "importance": 5,
                "created_at": "2026-06-20T00:00:00+00:00",
                "entity_id": seed_eid,
                "fact_status": "active",
            },
        }]

        qdrant = FakeGraphQdrant(search_results=seeds)
        qdrant.add_point("entity-hub", {
            "memory_kind": "graph_entity",
            "entity_id": seed_eid,
            "source_point_ids": [],
            "fact_status": "active",
        })

        # Add 200 edges from hub to 200 neighbors
        for i, nid in enumerate(neighbor_eids):
            edge_id = make_edge_id(seed_eid, nid, "RELATED_TO")
            qdrant.add_point(edge_id, {
                "memory_kind": "graph_edge",
                "edge_id": edge_id,
                "source_entity_id": seed_eid,
                "target_entity_id": nid,
                "relation_type": "RELATED_TO",
                "confidence": 0.8,
                "fact_status": "active",
            })

        # Add neighbor entity points with source_point_ids
        for i, nid in enumerate(neighbor_eids):
            qdrant.add_point(f"entity-neighbor-{i}", {
                "memory_kind": "graph_entity",
                "entity_id": nid,
                "source_point_ids": [f"memory-{i}"],
                "fact_status": "active",
            })
            qdrant.add_point(f"memory-{i}", {
                "text": f"neighbor memory {i}",
                "importance": 5,
                "created_at": "2026-06-20T00:00:00+00:00",
                "fact_status": "active",
            })

        retriever = GraphMemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
            expansion_policy=GraphExpansionPolicy(max_neighbors_per_node=8, max_total_expansion=64),
        )

        result = retriever.search("hub", top_k=5, candidate_seed_top_k=3)

        # Expansions should be capped at max_neighbors_per_node=8 (each neighbor
        # resolves to a memory point)
        assert len(result.expansions) <= 8, f"expansions should be <= 8, got {len(result.expansions)}"
        assert result.debug["hard_caps_hit"]["max_neighbors_per_node"] is True


# ---------------------------------------------------------------------------
# Test 3: Fact-status filter drops stale edges
# ---------------------------------------------------------------------------

class TestFactStatusFilter:
    def test_stale_edges_filtered_by_default(self):
        entity_a = make_entity_id("concept", "Entity A")
        entity_b = make_entity_id("concept", "Entity B")
        entity_c = make_entity_id("concept", "Entity C")

        seeds = [{
            "id": "seed-a",
            "score": 0.9,
            "payload": {
                "text": "entity A",
                "importance": 5,
                "created_at": "2026-06-20T00:00:00+00:00",
                "entity_id": entity_a,
                "fact_status": "active",
            },
        }]

        qdrant = FakeGraphQdrant(search_results=seeds)

        # Entity points
        for eid, mid in [
            (entity_a, "seed-a"),
            (entity_b, "memory-b"),
            (entity_c, "memory-c"),
        ]:
            qdrant.add_point(f"ent-{mid}", {
                "memory_kind": "graph_entity",
                "entity_id": eid,
                "source_point_ids": [mid],
                "fact_status": "active",
            })

        # Stale edge A->B (should be filtered by default)
        stale_edge = make_edge_id(entity_a, entity_b, "REFERENCES")
        qdrant.add_point(stale_edge, {
            "memory_kind": "graph_edge",
            "source_entity_id": entity_a,
            "target_entity_id": entity_b,
            "relation_type": "REFERENCES",
            "confidence": 0.9,
            "fact_status": "deprecated",
        })

        # Active edge A->C (should survive)
        active_edge = make_edge_id(entity_a, entity_c, "REFERENCES")
        qdrant.add_point(active_edge, {
            "memory_kind": "graph_edge",
            "source_entity_id": entity_a,
            "target_entity_id": entity_c,
            "relation_type": "REFERENCES",
            "confidence": 0.4,
            "fact_status": "active",
        })

        # Memory points for B and C
        qdrant.add_point("memory-b", {"text": "entity B content", "importance": 5, "created_at": "2026-06-20T00:00:00+00:00", "fact_status": "active"})
        qdrant.add_point("memory-c", {"text": "entity C content", "importance": 5, "created_at": "2026-06-20T00:00:00+00:00", "fact_status": "active"})

        retriever = GraphMemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
        )

        result = retriever.search("entity A", top_k=5, include_fact_history=False)

        # Only memory-c should be in expansions (active edge survived)
        exp_ids = [e.point_id for e in result.expansions]
        assert "memory-c" in exp_ids, f"memory-c should be in expansions: {exp_ids}"
        assert "memory-b" not in exp_ids, f"memory-b should be filtered (deprecated edge): {exp_ids}"
        assert result.debug["stages"]["C_edge_query"]["filtered_fact_status"] >= 1

        # With include_fact_history, the deprecated edge should also surface
        result2 = retriever.search("entity A", top_k=10, include_fact_history=True)
        exp_ids2 = [e.point_id for e in result2.expansions]
        assert "memory-b" in exp_ids2, f"memory-b should surface with include_fact_history: {exp_ids2}"


# ---------------------------------------------------------------------------
# Test 4: Empty graph path is fine
# ---------------------------------------------------------------------------

class TestEmptyGraphPath:
    def test_no_entity_refs_returns_seeds_only(self):
        seeds = [{
            "id": "seed-plain",
            "score": 0.9,
            "payload": {
                "text": "plain memory with no entity refs",
                "importance": 5,
                "created_at": "2026-06-20T00:00:00+00:00",
                "fact_status": "active",
            },
        }]

        qdrant = FakeGraphQdrant(search_results=seeds)

        retriever = GraphMemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
        )

        result = retriever.search("plain query", top_k=5)

        assert result.expansions == []
        assert len(result.final) == 1
        assert result.final[0].point_id == "seed-plain"
        assert result.debug["stages"]["B_entity_extraction"]["matched_from_seeds"] == 0
        assert result.debug["stages"]["C_edge_query"]["depth_1_edges"] == 0


# ---------------------------------------------------------------------------
# Test 5: No mutation safety
# ---------------------------------------------------------------------------

class TestNoMutation:
    def test_graph_search_does_not_mutate(self):
        deploy_eid = make_entity_id("workflow", "deploy")
        kubectl_eid = make_entity_id("tool", "kubectl")

        seeds = [{
            "id": "seed-1",
            "score": 0.9,
            "payload": {
                "text": "deploy uses kubectl",
                "importance": 5,
                "created_at": "2026-06-20T00:00:00+00:00",
                "entity_id": deploy_eid,
                "fact_status": "active",
            },
        }]

        qdrant = FakeGraphQdrant(search_results=seeds)
        qdrant.add_point("ent-deploy", {
            "memory_kind": "graph_entity",
            "entity_id": deploy_eid,
            "source_point_ids": ["seed-1"],
            "fact_status": "active",
        })
        qdrant.add_point("ent-kubectl", {
            "memory_kind": "graph_entity",
            "entity_id": kubectl_eid,
            "source_point_ids": ["kubectl-mem"],
            "fact_status": "active",
        })
        edge_id = make_edge_id(deploy_eid, kubectl_eid, "USES_TOOL")
        qdrant.add_point(edge_id, {
            "memory_kind": "graph_edge",
            "source_entity_id": deploy_eid,
            "target_entity_id": kubectl_eid,
            "relation_type": "USES_TOOL",
            "confidence": 0.8,
            "fact_status": "active",
        })
        qdrant.add_point("kubectl-mem", {
            "text": "kubectl usage notes",
            "importance": 5,
            "created_at": "2026-06-20T00:00:00+00:00",
            "fact_status": "active",
        })

        retriever = GraphMemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
        )

        result = retriever.search("deploy", top_k=5)

        # Zero mutations
        assert len(qdrant.upserts) == 0, f"upserts should be empty: {qdrant.upserts}"
        assert len(qdrant.payload_updates) == 0, f"payload_updates should be empty: {qdrant.payload_updates}"
        assert len(qdrant.deletes) == 0, f"deletes should be empty: {qdrant.deletes}"
        assert len(qdrant.filter_deletes) == 0, f"filter_deletes should be empty: {qdrant.filter_deletes}"


# ---------------------------------------------------------------------------
# Test 6: Schema invalid relation_type dropped
# ---------------------------------------------------------------------------

class TestInvalidRelationType:
    def test_invalid_relation_type_edge_dropped(self):
        entity_a = make_entity_id("concept", "A")
        entity_b = make_entity_id("concept", "B")

        seeds = [{
            "id": "seed-a",
            "score": 0.9,
            "payload": {
                "text": "entity A",
                "importance": 5,
                "created_at": "2026-06-20T00:00:00+00:00",
                "entity_id": entity_a,
                "fact_status": "active",
            },
        }]

        qdrant = FakeGraphQdrant(search_results=seeds)
        qdrant.add_point("ent-a", {
            "memory_kind": "graph_entity",
            "entity_id": entity_a,
            "source_point_ids": [],
            "fact_status": "active",
        })

        # Invalid relation type edge
        qdrant.add_point("bad-edge", {
            "memory_kind": "graph_edge",
            "source_entity_id": entity_a,
            "target_entity_id": entity_b,
            "relation_type": "BLOWS_UP",  # invalid!
            "confidence": 0.9,
            "fact_status": "active",
        })

        retriever = GraphMemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
        )

        result = retriever.search("entity A", top_k=5)

        # The bad edge should be dropped with a warning
        assert len(result.expansions) == 0
        assert any("invalid relation_type" in w for w in result.debug["warnings"])


# ---------------------------------------------------------------------------
# Test 7: Graph-distance decay ordering
# ---------------------------------------------------------------------------

class TestGraphDistanceDecay:
    def test_closer_candidate_ranks_higher(self):
        """A candidate at graph_distance=1 should get a higher graph_distance
        component than one at graph_distance=2."""
        eid_a = make_entity_id("concept", "A")
        eid_b = make_entity_id("concept", "B")
        eid_c = make_entity_id("concept", "C")

        seeds = [{
            "id": "seed-a",
            "score": 0.5,
            "payload": {
                "text": "entity A",
                "importance": 5,
                "created_at": "2026-06-20T00:00:00+00:00",
                "entity_id": eid_a,
                "fact_status": "active",
            },
        }]

        qdrant = FakeGraphQdrant(search_results=seeds)

        # Entities
        qdrant.add_point("ent-a", {"memory_kind": "graph_entity", "entity_id": eid_a, "source_point_ids": ["seed-a"], "fact_status": "active"})
        qdrant.add_point("ent-b", {"memory_kind": "graph_entity", "entity_id": eid_b, "source_point_ids": ["mem-b"], "fact_status": "active"})
        qdrant.add_point("ent-c", {"memory_kind": "graph_entity", "entity_id": eid_c, "source_point_ids": ["mem-c"], "fact_status": "active"})

        # Edges: A->B (dist 1), B->C (dist 2)
        e1 = make_edge_id(eid_a, eid_b, "REFERENCES")
        qdrant.add_point(e1, {"memory_kind": "graph_edge", "source_entity_id": eid_a, "target_entity_id": eid_b, "relation_type": "REFERENCES", "confidence": 0.8, "fact_status": "active"})
        e2 = make_edge_id(eid_b, eid_c, "REFERENCES")
        qdrant.add_point(e2, {"memory_kind": "graph_edge", "source_entity_id": eid_b, "target_entity_id": eid_c, "relation_type": "REFERENCES", "confidence": 0.8, "fact_status": "active"})

        # Memory points
        qdrant.add_point("mem-b", {"text": "B memory", "importance": 5, "created_at": "2026-06-20T00:00:00+00:00", "fact_status": "active"})
        qdrant.add_point("mem-c", {"text": "C memory", "importance": 5, "created_at": "2026-06-20T00:00:00+00:00", "fact_status": "active"})

        retriever = GraphMemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
            expansion_policy=GraphExpansionPolicy(max_depth=3, max_total_expansion=64),
        )

        result = retriever.search("entity A", top_k=10)

        exp_by_id = {e.point_id: e for e in result.expansions}
        if "mem-b" in exp_by_id and "mem-c" in exp_by_id:
            mem_b = exp_by_id["mem-b"]
            mem_c = exp_by_id["mem-c"]
            assert mem_b.graph_distance <= mem_c.graph_distance, (
                f"mem-b (dist={mem_b.graph_distance}) should be closer than mem-c (dist={mem_c.graph_distance})"
            )


# ---------------------------------------------------------------------------
# Test 8: Helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_extract_entity_refs(self):
        eid = make_entity_id("concept", "Test")
        payload = {
            "entity_id": eid,
            "text": "test",
        }
        refs = _extract_entity_refs(payload)
        assert refs == [eid]

    def test_extract_entity_refs_multiple(self):
        e1 = make_entity_id("concept", "A")
        e2 = make_entity_id("concept", "B")
        payload = {
            "source_entity_id": e1,
            "target_entity_id": e2,
        }
        refs = _extract_entity_refs(payload)
        assert e1 in refs
        assert e2 in refs

    def test_extract_entity_refs_empty(self):
        assert _extract_entity_refs({"text": "no refs"}) == []

    def test_get_counterparty_source(self):
        e1 = make_entity_id("concept", "A")
        e2 = make_entity_id("concept", "B")
        payload = {"source_entity_id": e1, "target_entity_id": e2}
        assert _get_counterparty(payload, e1) == e2

    def test_get_counterparty_target(self):
        e1 = make_entity_id("concept", "A")
        e2 = make_entity_id("concept", "B")
        payload = {"source_entity_id": e1, "target_entity_id": e2}
        assert _get_counterparty(payload, e2) == e1

    def test_query_alias_matches_label(self):
        payload = {"label": "deploy pipeline"}
        assert _query_alias_matches("how do we run the deploy pipeline?", payload) is True

    def test_query_alias_matches_alias(self):
        payload = {"label": "CI/CD", "aliases": ["deploy"]}
        assert _query_alias_matches("tell me about deploy", payload) is True

    def test_query_alias_no_match(self):
        payload = {"label": "database"}
        assert _query_alias_matches("how do we run the deploy?", payload) is False


# ---------------------------------------------------------------------------
# Test 9: Policy clamping
# ---------------------------------------------------------------------------

class TestPolicyClamping:
    def test_depth_clamped_to_max(self):
        policy = GraphExpansionPolicy(max_depth=10)
        assert policy.max_depth == 3

    def test_neighbors_clamped_to_max(self):
        policy = GraphExpansionPolicy(max_neighbors_per_node=100)
        assert policy.max_neighbors_per_node == 16

    def test_total_clamped_to_max(self):
        policy = GraphExpansionPolicy(max_total_expansion=10000)
        assert policy.max_total_expansion == 256

    def test_confidence_floor_clamped(self):
        policy = GraphExpansionPolicy(edge_confidence_floor=5.0)
        assert policy.edge_confidence_floor == 1.0

    def test_defaults(self):
        policy = GraphExpansionPolicy()
        assert policy.max_depth == 2
        assert policy.max_neighbors_per_node == 8
        assert policy.max_total_expansion == 64
        assert policy.edge_confidence_floor == 0.0
        assert policy.require_active_fact_status is True


# ---------------------------------------------------------------------------
# Test 10: Tool schema and CLI build_tool_call
# ---------------------------------------------------------------------------

class TestToolAndCLISurface:
    def test_graph_search_schema_exists(self):
        from qdrant_memory.tools import GRAPH_SEARCH_SCHEMA, TOOL_SCHEMAS

        assert GRAPH_SEARCH_SCHEMA["name"] == "qdrant_memory_graph_search"
        assert GRAPH_SEARCH_SCHEMA in TOOL_SCHEMAS
        params = GRAPH_SEARCH_SCHEMA["parameters"]
        assert "query" in params["required"]
        assert params["additionalProperties"] is False
        props = params["properties"]
        assert "top_k" in props
        assert "max_depth" in props
        assert "entity_types" in props
        assert "relation_types" in props
        assert "debug" in props

    def test_build_tool_call_graph_search(self):
        from qdrant_memory.cli_core import build_tool_call

        args = Namespace(
            qdrant_subcommand="graph-search",
            query="how do we deploy?",
            top_k=5,
            candidate_seed_top_k=20,
            max_graph_results=20,
            max_depth=2,
            include_fact_history=False,
            debug=True,
            collection="memory",
            entity_type=[],
            relation_type=[],
        )
        tool_name, tool_args = build_tool_call(args)
        assert tool_name == "qdrant_memory_graph_search"
        assert tool_args["query"] == "how do we deploy?"
        assert tool_args["top_k"] == 5
        assert tool_args["max_depth"] == 2

    def test_build_tool_call_graph_search_with_filters(self):
        from qdrant_memory.cli_core import build_tool_call

        args = Namespace(
            qdrant_subcommand="graph-search",
            query="test",
            top_k=3,
            candidate_seed_top_k=10,
            max_graph_results=10,
            max_depth=2,
            include_fact_history=True,
            debug=False,
            collection="memory",
            entity_type=["workflow"],
            relation_type=["USES_TOOL", "REFERENCES"],
        )
        tool_name, tool_args = build_tool_call(args)
        assert tool_name == "qdrant_memory_graph_search"
        assert tool_args["entity_types"] == ["workflow"]
        assert tool_args["relation_types"] == ["USES_TOOL", "REFERENCES"]
        assert tool_args["include_fact_history"] is True
        assert tool_args["debug"] is False
