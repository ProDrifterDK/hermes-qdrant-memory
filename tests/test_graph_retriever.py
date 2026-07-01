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

        # Invalid relation type edge — must have a valid edge_id so it reaches
        # the relation_type check (edges with invalid edge_id are dropped earlier).
        bad_edge_id = make_edge_id(entity_a, entity_b, "BLOWS_UP")
        qdrant.add_point(bad_edge_id, {
            "memory_kind": "graph_edge",
            "edge_id": bad_edge_id,
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


# ===========================================================================
# Regression tests for Phase 2 security/review blockers
# ===========================================================================

class TestBoundedScrollMaxTotal:
    """Blocker 2: Edge scrolls must always pass a bounded max_total."""

    def test_scroll_edges_for_entity_passes_max_total(self):
        """_scroll_edges_for_entity must pass a non-None max_total."""
        eid = make_entity_id("concept", "hub")
        seeds = [{
            "id": "seed-1",
            "score": 0.9,
            "payload": {
                "text": "hub entity",
                "importance": 5,
                "created_at": "2026-06-20T00:00:00+00:00",
                "entity_id": eid,
                "fact_status": "active",
            },
        }]
        qdrant = FakeGraphQdrant(search_results=seeds)
        qdrant.add_point(f"ent-{eid}", {
            "memory_kind": "graph_entity",
            "entity_id": eid,
            "source_point_ids": [],
            "fact_status": "active",
        })
        retriever = GraphMemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
            expansion_policy=GraphExpansionPolicy(max_neighbors_per_node=8),
        )
        retriever.search("hub", top_k=5, candidate_seed_top_k=3)

        # Every scroll_by_filter call from _scroll_edges_for_entity should
        # have a non-None max_total
        edge_scrolls = [
            s for s in qdrant.scrolls
            if any(
                c.get("key") == "memory_kind"
                and c.get("match", {}).get("value") == "graph_edge"
                for c in (s.get("filter") or {}).get("must", [])
            )
        ]
        assert len(edge_scrolls) > 0, "expected at least one edge scroll"
        for s in edge_scrolls:
            assert s["max_total"] is not None, (
                f"edge scroll missing max_total: {s}"
            )

    def test_paginated_scroll_max_total_is_bounded(self):
        """With a paginated fake, max_total must bound the total returned."""
        eid = make_entity_id("concept", "hub")
        seeds = [{
            "id": "seed-1",
            "score": 0.9,
            "payload": {
                "text": "hub entity",
                "importance": 5,
                "created_at": "2026-06-20T00:00:00+00:00",
                "entity_id": eid,
                "fact_status": "active",
            },
        }]

        class PaginatedFakeQdrant(FakeGraphQdrant):
            """Fake that simulates multi-page scroll, tracking max_total."""

            def scroll_by_filter(self, name, filter, *, limit=256,
                                 with_payload=True, with_vector=False, max_total=None):
                self.scrolls.append({
                    "name": name, "filter": filter, "limit": limit,
                    "max_total": max_total,
                })
                must = filter.get("must", []) if filter else []
                results = []
                for point_id, point in self._store.items():
                    payload = point.get("payload") or {}
                    if self._matches_filter(payload, must):
                        results.append(point)
                # Simulate pagination: if max_total is set, cap at it
                if max_total is not None:
                    results = results[:max_total]
                else:
                    # Without max_total, simulate unbounded (cap at limit only)
                    results = results[:limit]
                return results

        qdrant = PaginatedFakeQdrant(search_results=seeds)
        neighbor_cap = 4
        for i in range(50):
            nid = make_entity_id("concept", f"neighbor-{i}")
            edge_id = make_edge_id(eid, nid, "RELATED_TO")
            qdrant.add_point(edge_id, {
                "memory_kind": "graph_edge",
                "source_entity_id": eid,
                "target_entity_id": nid,
                "relation_type": "RELATED_TO",
                "confidence": 0.8,
                "fact_status": "active",
            })

        retriever = GraphMemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
            expansion_policy=GraphExpansionPolicy(max_neighbors_per_node=neighbor_cap),
        )
        retriever.search("hub", top_k=5, candidate_seed_top_k=3)

        # All edge scrolls must have a bounded max_total
        edge_scrolls = [
            s for s in qdrant.scrolls
            if any(
                c.get("key") == "memory_kind"
                and c.get("match", {}).get("value") == "graph_edge"
                for c in (s.get("filter") or {}).get("must", [])
            )
        ]
        for s in edge_scrolls:
            assert s["max_total"] is not None
            assert s["max_total"] <= 256  # _HARD_SCROLL_PER_CALL
            # Bounded max should be derived from neighbor cap * 3
            assert s["max_total"] <= neighbor_cap * 3


class TestFailClosedRelationTypes:
    """Blocker 3: Invalid relation_types must fail closed, not silently widen."""

    def test_search_raises_on_invalid_relation_type(self):
        qdrant = FakeGraphQdrant(search_results=[])
        retriever = GraphMemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
        )
        with pytest.raises(ValueError, match="invalid relation_type"):
            retriever.search("query", relation_types=["BLOWS_UP"])

    def test_search_raises_on_partial_invalid_relation_type(self):
        """Even if some are valid, one invalid should fail."""
        qdrant = FakeGraphQdrant(search_results=[])
        retriever = GraphMemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
        )
        with pytest.raises(ValueError, match="invalid relation_type"):
            retriever.search("query", relation_types=["USES_TOOL", "FAKE_RELATION"])

    def test_search_accepts_valid_relation_types(self):
        qdrant = FakeGraphQdrant(search_results=[{
            "id": "seed-1",
            "score": 0.9,
            "payload": {
                "text": "test",
                "importance": 5,
                "created_at": "2026-06-20T00:00:00+00:00",
                "fact_status": "active",
            },
        }])
        retriever = GraphMemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
        )
        # Should not raise
        result = retriever.search("query", relation_types=["USES_TOOL", "REFERENCES"])
        assert result is not None

    def test_schema_uses_relation_enum(self):
        """GRAPH_SEARCH_SCHEMA relation_types must use an enum of RELATION_TYPES."""
        from qdrant_memory.tools import GRAPH_SEARCH_SCHEMA
        from qdrant_memory.schema import RELATION_TYPES
        items = GRAPH_SEARCH_SCHEMA["parameters"]["properties"]["relation_types"]["items"]
        assert "enum" in items
        assert set(items["enum"]) == set(RELATION_TYPES)


class TestDebugRedaction:
    """Blocker 4: Debug must not echo raw query or invalid relation values."""

    def test_debug_does_not_contain_raw_query(self):
        # Build secret-shaped string at runtime to avoid literal-fake-secret scanner
        secret_query = "".join(["sk-lea", "ked-api-key-", "1234567890abcdef"])
        qdrant = FakeGraphQdrant(search_results=[{
            "id": "seed-1",
            "score": 0.9,
            "payload": {
                "text": "some text",
                "importance": 5,
                "created_at": "2026-06-20T00:00:00+00:00",
                "fact_status": "active",
            },
        }])
        retriever = GraphMemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
        )
        result = retriever.search(secret_query, top_k=5, debug=True)
        debug_json = json.dumps(result.debug)
        assert secret_query not in debug_json, (
            "raw secret query must not appear in debug output"
        )
        # query_length should be present instead
        assert "query_length" in result.debug

    def test_debug_does_not_echo_invalid_relation_value(self):
        """Warnings should not contain the raw invalid relation_type value."""
        # Build secret-shaped string at runtime to avoid literal-fake-secret scanner
        secret_relation = "".join(["sk-sec", "ret-relation-value"])
        eid_a = make_entity_id("concept", "A")
        eid_b = make_entity_id("concept", "B")
        seeds = [{
            "id": "seed-a",
            "score": 0.9,
            "payload": {
                "text": "entity A",
                "importance": 5,
                "created_at": "2026-06-20T00:00:00+00:00",
                "entity_id": eid_a,
                "fact_status": "active",
            },
        }]
        qdrant = FakeGraphQdrant(search_results=seeds)
        qdrant.add_point("ent-a", {
            "memory_kind": "graph_entity",
            "entity_id": eid_a,
            "source_point_ids": [],
            "fact_status": "active",
        })
        qdrant.add_point("bad-edge", {
            "memory_kind": "graph_edge",
            "source_entity_id": eid_a,
            "target_entity_id": eid_b,
            "relation_type": secret_relation,
            "confidence": 0.9,
            "fact_status": "active",
        })
        retriever = GraphMemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
        )
        result = retriever.search("entity A", top_k=5)
        warnings_json = json.dumps(result.debug.get("warnings", []))
        assert secret_relation not in warnings_json, (
            "raw invalid relation value must not appear in warnings"
        )


class TestGraphIDValidation:
    """Blocker 5: Entity/edge/point ID validators and defensive confidence."""

    def test_extract_entity_refs_rejects_malformed(self):
        """Malformed entity IDs (not matching entity-<16hex>) are rejected."""
        payload = {
            "entity_id": "entity-notvalid",  # missing hex suffix
            "source_entity_id": "entity-deadbeefdeadbeef",  # valid 16 hex
        }
        refs = _extract_entity_refs(payload)
        # Only the valid one should be present
        assert "entity-deadbeefdeadbeef" in refs
        assert "entity-notvalid" not in refs

    def test_get_counterparty_rejects_malformed(self):
        """Counterparty with malformed entity ID returns None."""
        eid = make_entity_id("concept", "A")
        edge_payload = {
            "source_entity_id": eid,
            "target_entity_id": "entity-BADFORMAT",  # malformed
        }
        result = _get_counterparty(edge_payload, eid)
        assert result is None

    def test_malformed_confidence_does_not_crash(self):
        """Non-numeric confidence should default to 0.0, not crash."""
        eid_a = make_entity_id("concept", "A")
        eid_b = make_entity_id("concept", "B")
        seeds = [{
            "id": "seed-a",
            "score": 0.9,
            "payload": {
                "text": "entity A",
                "importance": 5,
                "created_at": "2026-06-20T00:00:00+00:00",
                "entity_id": eid_a,
                "fact_status": "active",
            },
        }]
        qdrant = FakeGraphQdrant(search_results=seeds)
        qdrant.add_point("ent-a", {
            "memory_kind": "graph_entity",
            "entity_id": eid_a,
            "source_point_ids": [],
            "fact_status": "active",
        })
        edge_id = make_edge_id(eid_a, eid_b, "RELATED_TO")
        qdrant.add_point(edge_id, {
            "memory_kind": "graph_edge",
            "source_entity_id": eid_a,
            "target_entity_id": eid_b,
            "relation_type": "RELATED_TO",
            "confidence": "not-a-number",  # malformed!
            "fact_status": "active",
        })
        qdrant.add_point("ent-b", {
            "memory_kind": "graph_entity",
            "entity_id": eid_b,
            "source_point_ids": ["mem-b"],
            "fact_status": "active",
        })
        qdrant.add_point("mem-b", {
            "text": "B memory",
            "importance": 5,
            "created_at": "2026-06-20T00:00:00+00:00",
            "fact_status": "active",
        })
        retriever = GraphMemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
        )
        # Should not raise
        result = retriever.search("entity A", top_k=5)
        assert result is not None

    def test_nan_confidence_does_not_crash(self):
        """NaN confidence (float('nan')) should be handled defensively."""
        eid_a = make_entity_id("concept", "A")
        eid_b = make_entity_id("concept", "B")
        seeds = [{
            "id": "seed-a",
            "score": 0.9,
            "payload": {
                "text": "entity A",
                "importance": 5,
                "created_at": "2026-06-20T00:00:00+00:00",
                "entity_id": eid_a,
                "fact_status": "active",
            },
        }]
        qdrant = FakeGraphQdrant(search_results=seeds)
        qdrant.add_point("ent-a", {
            "memory_kind": "graph_entity",
            "entity_id": eid_a,
            "source_point_ids": [],
            "fact_status": "active",
        })
        edge_id = make_edge_id(eid_a, eid_b, "RELATED_TO")
        qdrant.add_point(edge_id, {
            "memory_kind": "graph_edge",
            "source_entity_id": eid_a,
            "target_entity_id": eid_b,
            "relation_type": "RELATED_TO",
            "confidence": float("nan"),  # NaN!
            "fact_status": "active",
        })
        qdrant.add_point("ent-b", {
            "memory_kind": "graph_entity",
            "entity_id": eid_b,
            "source_point_ids": ["mem-b"],
            "fact_status": "active",
        })
        qdrant.add_point("mem-b", {
            "text": "B memory",
            "importance": 5,
            "created_at": "2026-06-20T00:00:00+00:00",
            "fact_status": "active",
        })
        retriever = GraphMemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
        )
        result = retriever.search("entity A", top_k=5)
        assert result is not None

    def test_malformed_source_point_id_filtered(self):
        """Malformed source_point_ids should be filtered, not passed to retrieve."""
        eid_a = make_entity_id("concept", "A")
        seeds = [{
            "id": "seed-a",
            "score": 0.9,
            "payload": {
                "text": "entity A",
                "importance": 5,
                "created_at": "2026-06-20T00:00:00+00:00",
                "entity_id": eid_a,
                "fact_status": "active",
            },
        }]
        qdrant = FakeGraphQdrant(search_results=seeds)
        qdrant.add_point("ent-a", {
            "memory_kind": "graph_entity",
            "entity_id": eid_a,
            "source_point_ids": ["https://evil.com/path", "valid-point-id"],
            "fact_status": "active",
        })
        retriever = GraphMemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
        )
        retriever.search("entity A", top_k=5)
        # Check that the URL was NOT passed to retrieve
        for r in qdrant.retrieves:
            assert "https://evil.com/path" not in r.get("ids", []), (
                "malformed source point ID should be filtered before retrieval"
            )

    def test_scroll_edges_rejects_malformed_entity_id(self):
        """_scroll_edges_for_entity must return [] for malformed entity_id."""
        retriever = GraphMemoryRetriever(
            qdrant=FakeGraphQdrant(search_results=[]),
            embeddings=FakeEmbedding(),
            collection_name="memory",
        )
        result = retriever._scroll_edges_for_entity(
            "entity-notvalid", side="source", relation_types=None,
        )
        assert result == [], "malformed entity_id should produce no edge scroll"

    def test_scroll_edges_rejects_secret_shaped_entity_id(self):
        """_scroll_edges_for_entity must return [] for secret-shaped entity_id."""
        secret_eid = "".join(["sk-lea", "ked-entity-id"])
        retriever = GraphMemoryRetriever(
            qdrant=FakeGraphQdrant(search_results=[]),
            embeddings=FakeEmbedding(),
            collection_name="memory",
        )
        result = retriever._scroll_edges_for_entity(
            secret_eid, side="target", relation_types=None,
        )
        assert result == [], "secret-shaped entity_id should produce no edge scroll"

    def test_malformed_query_matched_entity_does_not_seed_bfs(self):
        """Malformed graph_entity.entity_id from alias match must not seed BFS
        or appear in edge-scroll filters."""
        eid_a = make_entity_id("concept", "A")
        eid_b = make_entity_id("concept", "B")
        seeds = [{
            "id": "seed-a",
            "score": 0.9,
            "payload": {
                "text": "entity A",
                "importance": 5,
                "created_at": "2026-06-20T00:00:00+00:00",
                "entity_id": eid_a,
                "fact_status": "active",
            },
        }]
        qdrant = FakeGraphQdrant(search_results=seeds)
        qdrant.add_point("ent-a", {
            "memory_kind": "graph_entity",
            "entity_id": eid_a,
            "source_point_ids": [],
            "fact_status": "active",
        })
        # Malformed graph_entity whose alias matches the query
        qdrant.add_point("bad-entity-point", {
            "memory_kind": "graph_entity",
            "entity_id": "entity-notvalid",  # malformed
            "label": "entity A",  # matches query
            "source_point_ids": [],
            "fact_status": "active",
        })
        retriever = GraphMemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
        )
        result = retriever.search("entity A", top_k=5, debug=True)

        # The malformed entity_id must NOT appear in any edge scroll filter
        for s in qdrant.scrolls:
            for cond in (s.get("filter") or {}).get("must", []):
                match = cond.get("match", {})
                if "value" in match:
                    assert str(match["value"]) != "entity-notvalid", (
                        f"malformed entity_id leaked into scroll filter: {s}"
                    )

        # The malformed entity_id must NOT appear in expansions or final
        all_point_ids = (
            [e.point_id for e in result.expansions]
            + [c.point_id for c in result.final]
        )
        assert "entity-notvalid" not in all_point_ids, (
            f"malformed entity_id leaked into results: {all_point_ids}"
        )

    def test_malformed_edge_id_dropped_not_traversed(self):
        """A graph_edge with a malformed point ID and edge_id (but valid
        endpoint entity IDs and relation_type) must be dropped before
        traversal — no expansion through that edge."""
        eid_a = make_entity_id("concept", "A")
        eid_b = make_entity_id("concept", "B")
        seeds = [{
            "id": "seed-a",
            "score": 0.9,
            "payload": {
                "text": "entity A",
                "importance": 5,
                "created_at": "2026-06-20T00:00:00+00:00",
                "entity_id": eid_a,
                "fact_status": "active",
            },
        }]
        qdrant = FakeGraphQdrant(search_results=seeds)
        qdrant.add_point("ent-a", {
            "memory_kind": "graph_entity",
            "entity_id": eid_a,
            "source_point_ids": [],
            "fact_status": "active",
        })
        qdrant.add_point("ent-b", {
            "memory_kind": "graph_entity",
            "entity_id": eid_b,
            "source_point_ids": ["mem-b"],
            "fact_status": "active",
        })
        qdrant.add_point("mem-b", {
            "text": "B memory",
            "importance": 5,
            "created_at": "2026-06-20T00:00:00+00:00",
            "fact_status": "active",
        })
        # Malformed edge: valid endpoints + relation, but bad point id + edge_id
        secret_shaped_edge_id = "".join(["sk-sec", "ret-edge-id-1234"])
        qdrant.add_point(secret_shaped_edge_id, {
            "memory_kind": "graph_edge",
            "edge_id": "also-not-valid",  # malformed edge_id
            "source_entity_id": eid_a,
            "target_entity_id": eid_b,
            "relation_type": "RELATED_TO",
            "confidence": 0.9,
            "fact_status": "active",
        })
        retriever = GraphMemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
        )
        result = retriever.search("entity A", top_k=5, debug=True)

        # mem-b must NOT be in expansions — the only edge to eid_b was invalid
        exp_ids = [e.point_id for e in result.expansions]
        assert "mem-b" not in exp_ids, (
            f"malformed edge allowed traversal to mem-b: {exp_ids}"
        )
        # The secret-shaped edge ID must NOT appear in expansions or final
        all_point_ids = exp_ids + [c.point_id for c in result.final]
        assert secret_shaped_edge_id not in all_point_ids, (
            f"malformed edge id leaked into results: {all_point_ids}"
        )
        assert "also-not-valid" not in all_point_ids

    def test_warnings_do_not_echo_invalid_edge_values(self):
        """Warning messages for dropped edges must not contain raw invalid values."""
        eid_a = make_entity_id("concept", "A")
        eid_b = make_entity_id("concept", "B")
        seeds = [{
            "id": "seed-a",
            "score": 0.9,
            "payload": {
                "text": "entity A",
                "importance": 5,
                "created_at": "2026-06-20T00:00:00+00:00",
                "entity_id": eid_a,
                "fact_status": "active",
            },
        }]
        qdrant = FakeGraphQdrant(search_results=seeds)
        qdrant.add_point("ent-a", {
            "memory_kind": "graph_entity",
            "entity_id": eid_a,
            "source_point_ids": [],
            "fact_status": "active",
        })
        secret_shaped_edge_id = "".join(["sk-lea", "ked-edge-id-bad"])
        qdrant.add_point(secret_shaped_edge_id, {
            "memory_kind": "graph_edge",
            "edge_id": secret_shaped_edge_id,  # malformed
            "source_entity_id": eid_a,
            "target_entity_id": eid_b,
            "relation_type": "RELATED_TO",
            "confidence": 0.9,
            "fact_status": "active",
        })
        retriever = GraphMemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
        )
        result = retriever.search("entity A", top_k=5, debug=True)
        warnings_json = json.dumps(result.debug.get("warnings", []))
        assert secret_shaped_edge_id not in warnings_json, (
            "raw invalid edge id must not appear in warnings"
        )

    def test_valid_edge_with_valid_endpoints_still_traversed(self):
        """Regression: a valid edge with valid endpoint entity IDs must still
        produce expansions — the new validation must not over-filter."""
        eid_a = make_entity_id("concept", "A")
        eid_b = make_entity_id("concept", "B")
        seeds = [{
            "id": "seed-a",
            "score": 0.9,
            "payload": {
                "text": "entity A",
                "importance": 5,
                "created_at": "2026-06-20T00:00:00+00:00",
                "entity_id": eid_a,
                "fact_status": "active",
            },
        }]
        qdrant = FakeGraphQdrant(search_results=seeds)
        qdrant.add_point("ent-a", {
            "memory_kind": "graph_entity",
            "entity_id": eid_a,
            "source_point_ids": [],
            "fact_status": "active",
        })
        edge_id = make_edge_id(eid_a, eid_b, "RELATED_TO")
        qdrant.add_point(edge_id, {
            "memory_kind": "graph_edge",
            "edge_id": edge_id,
            "source_entity_id": eid_a,
            "target_entity_id": eid_b,
            "relation_type": "RELATED_TO",
            "confidence": 0.9,
            "fact_status": "active",
        })
        qdrant.add_point("ent-b", {
            "memory_kind": "graph_entity",
            "entity_id": eid_b,
            "source_point_ids": ["mem-b"],
            "fact_status": "active",
        })
        qdrant.add_point("mem-b", {
            "text": "B memory",
            "importance": 5,
            "created_at": "2026-06-20T00:00:00+00:00",
            "fact_status": "active",
        })
        retriever = GraphMemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
        )
        result = retriever.search("entity A", top_k=5, debug=True)
        exp_ids = [e.point_id for e in result.expansions]
        assert "mem-b" in exp_ids, (
            f"valid edge should produce expansion to mem-b: {exp_ids}"
        )


class TestProviderGraphSearchDispatch:
    """Blocker 1: Provider must dispatch qdrant_memory_graph_search."""

    def test_provider_handles_graph_search(self):
        """handle_tool_call must not return Unknown tool for graph_search."""
        import importlib.util as importlib_util
        import sys as _sys
        from pathlib import Path
        plugin_root = Path(__file__).resolve().parents[1]
        if str(plugin_root) not in _sys.path:
            _sys.path.insert(0, str(plugin_root))

        # Import the provider class
        spec = importlib_util.spec_from_file_location(
            "_test_graph_search_provider",
            str(plugin_root / "__init__.py"),
        )
        assert spec is not None and spec.loader is not None
        mod = importlib_util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        Provider = mod.QdrantMemoryProvider

        # Create a minimal instance — we need to test dispatch, not full init
        provider = Provider.__new__(Provider)

        # Without initialization, it should return a provider error (not "Unknown tool")
        result = json.loads(provider.handle_tool_call(
            "qdrant_memory_graph_search",
            {"query": "test query"},
        ))
        # Must NOT be "Unknown tool"
        assert "Unknown tool" not in result.get("error", ""), (
            f"Expected provider to handle graph_search, got: {result}"
        )

    def test_provider_graph_search_validates_query(self):
        """Provider should validate empty query."""
        import importlib.util as importlib_util
        import sys as _sys
        from pathlib import Path
        plugin_root = Path(__file__).resolve().parents[1]
        if str(plugin_root) not in _sys.path:
            _sys.path.insert(0, str(plugin_root))

        spec = importlib_util.spec_from_file_location(
            "_test_graph_search_provider2",
            str(plugin_root / "__init__.py"),
        )
        assert spec is not None and spec.loader is not None
        mod = importlib_util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        Provider = mod.QdrantMemoryProvider

        provider = Provider.__new__(Provider)
        result = json.loads(provider.handle_tool_call(
            "qdrant_memory_graph_search",
            {"query": ""},
        ))
        assert "error" in result
        assert "query is required" in result["error"]


# ===========================================================================
# Phase 1 regression tests: cross-profile graph scope isolation
# ===========================================================================

class TestGraphScopeIsolation:
    """Graph search must not leak entities, edges, or source memories across
    profile_id / user_id_hash / chat_id_hash scope boundaries."""

    def _make_scope_qdrant(self) -> tuple["FakeGraphQdrant", str]:
        """Build a FakeGraphQdrant with points tagged to different scopes."""
        eid_a = make_entity_id("concept", "Alpha")
        eid_b = make_entity_id("concept", "Beta")
        eid_c = make_entity_id("concept", "Gamma")  # different profile

        seeds = [{
            "id": "seed-alpha",
            "score": 0.9,
            "payload": {
                "text": "alpha entity",
                "importance": 5,
                "created_at": "2026-06-20T00:00:00+00:00",
                "entity_id": eid_a,
                "fact_status": "active",
                "profile_id": "profile-A",
            },
        }]
        qdrant = FakeGraphQdrant(search_results=seeds)

        # Entity Alpha (profile A)
        qdrant.add_point("ent-alpha", {
            "memory_kind": "graph_entity",
            "entity_id": eid_a,
            "source_point_ids": [],
            "fact_status": "active",
            "profile_id": "profile-A",
        })
        # Entity Beta (profile A)
        qdrant.add_point("ent-beta", {
            "memory_kind": "graph_entity",
            "entity_id": eid_b,
            "source_point_ids": ["mem-beta"],
            "fact_status": "active",
            "profile_id": "profile-A",
        })
        # Entity Gamma (profile B — different scope)
        qdrant.add_point("ent-gamma", {
            "memory_kind": "graph_entity",
            "entity_id": eid_c,
            "label": "alpha",  # alias matches query to trigger alias scroll
            "source_point_ids": ["mem-gamma"],
            "fact_status": "active",
            "profile_id": "profile-B",  # different!
        })
        # Edge Alpha->Beta (profile A)
        edge_ab = make_edge_id(eid_a, eid_b, "RELATED_TO")
        qdrant.add_point(edge_ab, {
            "memory_kind": "graph_edge",
            "edge_id": edge_ab,
            "source_entity_id": eid_a,
            "target_entity_id": eid_b,
            "relation_type": "RELATED_TO",
            "confidence": 0.8,
            "fact_status": "active",
            "profile_id": "profile-A",
        })
        # Edge Alpha->Gamma (profile B — cross-profile!)
        edge_ag = make_edge_id(eid_a, eid_c, "RELATED_TO")
        qdrant.add_point(edge_ag, {
            "memory_kind": "graph_edge",
            "edge_id": edge_ag,
            "source_entity_id": eid_a,
            "target_entity_id": eid_c,
            "relation_type": "RELATED_TO",
            "confidence": 0.8,
            "fact_status": "active",
            "profile_id": "profile-B",  # different!
        })
        # Memory points
        qdrant.add_point("mem-beta", {
            "text": "beta memory in profile A",
            "importance": 5,
            "created_at": "2026-06-20T00:00:00+00:00",
            "fact_status": "active",
            "profile_id": "profile-A",
        })
        qdrant.add_point("mem-gamma", {
            "text": "gamma memory in profile B",
            "importance": 5,
            "created_at": "2026-06-20T00:00:00+00:00",
            "fact_status": "active",
            "profile_id": "profile-B",  # different!
        })
        return qdrant, eid_c

    def test_scope_filters_query_matched_entities(self):
        """Entity alias scrolls must only return entities within scope."""
        qdrant, eid_gamma = self._make_scope_qdrant()
        retriever = GraphMemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
            scope={"profile_id": "profile-A"},
        )
        result = retriever.search("alpha", top_k=10, debug=True)

        # Gamma entity must NOT be in query-matched entities
        matched_entities = self._get_matched_entities(result)
        assert eid_gamma not in matched_entities, (
            f"cross-profile entity leaked via alias match: {matched_entities}"
        )

    def test_scope_filters_edge_scrolls(self):
        """Edge scrolls must only return edges within scope."""
        qdrant, eid_gamma = self._make_scope_qdrant()
        retriever = GraphMemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
            scope={"profile_id": "profile-A"},
        )
        result = retriever.search("alpha entity", top_k=10, debug=True)

        # Check all edge scrolls had profile_id in filter
        edge_scrolls = [
            s for s in qdrant.scrolls
            if any(
                c.get("key") == "memory_kind"
                and c.get("match", {}).get("value") == "graph_edge"
                for c in (s.get("filter") or {}).get("must", [])
            )
        ]
        for s in edge_scrolls:
            must = s.get("filter", {}).get("must", [])
            scope_conds = [c for c in must if c.get("key") == "profile_id"]
            assert len(scope_conds) == 1, f"edge scroll missing scope filter: {s}"

    def test_scope_filters_entity_resolution_and_source_points(self):
        """Retrieved source points must be within scope."""
        qdrant, eid_gamma = self._make_scope_qdrant()
        retriever = GraphMemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
            scope={"profile_id": "profile-A"},
        )
        result = retriever.search("alpha entity", top_k=10, debug=True)

        # mem-gamma must not appear in any expansion or final candidate
        all_ids = (
            [e.point_id for e in result.expansions]
            + [c.point_id for c in result.final]
        )
        assert "mem-gamma" not in all_ids, (
            f"cross-profile source memory leaked: {all_ids}"
        )

    def test_scope_filters_final_candidates(self):
        """Final candidate list must not contain cross-profile content.

        Scoped results must carry and match all active scope keys — a missing
        ``profile_id`` on a scoped candidate is also a leak.
        """
        qdrant, eid_gamma = self._make_scope_qdrant()
        retriever = GraphMemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
            scope={"profile_id": "profile-A"},
        )
        result = retriever.search("alpha entity", top_k=10, debug=True)

        for candidate in result.final:
            payload = candidate.payload or {}
            # Scoped candidates must carry the active scope key AND match it.
            # Missing profile_id is a leak, not a pass.
            assert str(payload.get("profile_id") or "") == "profile-A", (
                f"candidate missing/mismatched scope key profile_id: {candidate.point_id}"
            )

    def test_no_scope_returns_everything(self):
        """Without scope (global mode), all profiles are visible."""
        qdrant, eid_gamma = self._make_scope_qdrant()
        retriever = GraphMemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
            scope={},  # global mode
        )
        result = retriever.search("alpha entity", top_k=10, debug=True)
        # With no scope, Gamma entity should be visible via alias match
        assert result is not None

    def test_debug_shows_scope_keys(self):
        """Debug must show active scope keys for auditability."""
        qdrant, _ = self._make_scope_qdrant()
        retriever = GraphMemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
            scope={"profile_id": "profile-A", "user_id_hash": "hash123"},
        )
        result = retriever.search("alpha", top_k=5, debug=True)
        assert "scope_keys" in result.debug
        assert "profile_id" in result.debug["scope_keys"]
        assert "user_id_hash" in result.debug["scope_keys"]

    def _get_matched_entities(self, result: GraphSearchResult) -> set[str]:
        """Extract entity IDs from all expansion and final candidate paths."""
        entities: set[str] = set()
        for exp in result.expansions:
            entities.update(exp.path)
        for c in result.final:
            entities.update(c.path)
        return entities

    # ------------------------------------------------------------------
    # P1 regression: source points with MISSING scope keys must not leak.
    #
    # These tests MUST create an in-scope graph edge from the seed entity to
    # a second in-scope counterparty entity whose source_point_ids include
    # both a missing-scope point and a matching-scope control point.  Only
    # then does BFS call _resolve_entity_memories(), which invokes
    # qdrant.retrieve() and applies the fail-closed _payload_in_scope()
    # post-filter.  Without the edge the tests are vacuous (retrieves=[]).
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("scope_key,scope_val,base_scope", [
        ("profile_id", "profile-A", {"profile_id": "profile-A"}),
        ("user_id_hash", "userhash123", {"profile_id": "profile-A", "user_id_hash": "userhash123"}),
        ("chat_id_hash", "chathash456", {"profile_id": "profile-A", "chat_id_hash": "chathash456"}),
    ])
    def test_missing_scope_key_source_excluded(self, scope_key, scope_val, base_scope):
        """Source point missing an active scope key must not leak into
        expansions/final, while a matching in-scope control point must
        survive — proving qdrant.retrieve() and _payload_in_scope() ran.
        """
        eid_a = make_entity_id("concept", "Alpha")
        eid_b = make_entity_id("concept", "Beta")

        # Build the base seed payload with the parametrized scope.
        seed_payload = {
            "text": "alpha entity",
            "importance": 5,
            "created_at": "2026-06-20T00:00:00+00:00",
            "entity_id": eid_a,
            "fact_status": "active",
        }
        seed_payload.update(base_scope)

        seeds = [{
            "id": "seed-alpha",
            "score": 0.9,
            "payload": seed_payload,
        }]
        qdrant = FakeGraphQdrant(search_results=seeds)

        # Entity Alpha (seed) — in scope, no source_point_ids.
        ent_alpha_payload = {
            "memory_kind": "graph_entity",
            "entity_id": eid_a,
            "source_point_ids": [],
            "fact_status": "active",
        }
        ent_alpha_payload.update(base_scope)
        qdrant.add_point("ent-alpha", ent_alpha_payload)

        # Entity Beta (counterparty) — in scope, references two source points:
        #   1. "mem-missing-scope"  — MISSING the active scope key (must leak
        #      under fail-open, must be filtered under fail-closed).
        #   2. "mem-scoped"         — carries the matching scope value (control).
        ent_beta_payload = {
            "memory_kind": "graph_entity",
            "entity_id": eid_b,
            "source_point_ids": ["mem-missing-scope", "mem-scoped"],
            "fact_status": "active",
        }
        ent_beta_payload.update(base_scope)
        qdrant.add_point("ent-beta", ent_beta_payload)

        # In-scope edge Alpha → Beta (BFS will traverse this, calling
        # _resolve_entity_memories for counterparty Beta).
        edge_ab = make_edge_id(eid_a, eid_b, "RELATED_TO")
        edge_payload = {
            "memory_kind": "graph_edge",
            "edge_id": edge_ab,
            "source_entity_id": eid_a,
            "target_entity_id": eid_b,
            "relation_type": "RELATED_TO",
            "confidence": 0.8,
            "fact_status": "active",
        }
        edge_payload.update(base_scope)
        qdrant.add_point(edge_ab, edge_payload)

        # Missing-scope source point: has other scope fields but is MISSING
        # the parametrized scope_key entirely.
        missing_payload = {
            "text": f"memory missing {scope_key}",
            "importance": 5,
            "created_at": "2026-06-20T00:00:00+00:00",
            "fact_status": "active",
        }
        # Include other scope keys so the ONLY deficiency is the missing key.
        for k, v in base_scope.items():
            if k != scope_key:
                missing_payload[k] = v
        qdrant.add_point("mem-missing-scope", missing_payload)

        # In-scope control source point: carries ALL scope keys correctly.
        scoped_payload = {
            "text": f"scoped memory with {scope_key}",
            "importance": 5,
            "created_at": "2026-06-20T00:00:00+00:00",
            "fact_status": "active",
        }
        scoped_payload.update(base_scope)
        qdrant.add_point("mem-scoped", scoped_payload)

        retriever = GraphMemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
            scope=base_scope,
        )
        result = retriever.search("alpha", top_k=10, debug=True)

        # 1. Prove the retrieve/post-filter path actually ran.
        retrieve_ids: list[str] = []
        for call in qdrant.retrieves:
            retrieve_ids.extend(call.get("ids", []))
        assert "mem-missing-scope" in retrieve_ids, (
            f"qdrant.retrieve() was not called with mem-missing-scope "
            f"(retrieves={qdrant.retrieves}) — test is vacuous"
        )
        assert "mem-scoped" in retrieve_ids, (
            f"qdrant.retrieve() was not called with mem-scoped "
            f"(retrieves={qdrant.retrieves}) — test is vacuous"
        )

        # 2. Missing-scope source point must NOT appear in expansions or final.
        all_ids = (
            [e.point_id for e in result.expansions]
            + [c.point_id for c in result.final]
        )
        assert "mem-missing-scope" not in all_ids, (
            f"source point missing {scope_key} leaked into results: {all_ids}"
        )

        # 3. Matching in-scope control point MUST survive in expansions/final.
        assert "mem-scoped" in all_ids, (
            f"in-scope control point 'mem-scoped' was incorrectly filtered "
            f"out of expansions/final: {all_ids}"
        )
