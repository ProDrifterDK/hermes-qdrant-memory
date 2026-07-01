"""Read-only graph-aware retrieval for the Qdrant memory plugin.

This module implements Phase 2 of the graph memory roadmap: bounded BFS
expansion over graph entity/edge payloads stored in the same ``hermes_memory``
Qdrant collection, combined with a hybrid reranker that transparently blends
vector score, graph distance, edge confidence, provenance, and usefulness.

Safety invariants:

- **Read-only.** No Qdrant upsert, delete, or payload update. Access metadata
  (``last_accessed``, ``access_count``) is NOT bumped by graph search.
- **No new dependency.** Uses ``dataclasses``, ``math``, ``urllib`` (via
  ``QdrantClient``) and re-uses existing ``MemoryRetriever`` / ``RankingPolicy``.
- **Memory is context, not authority.** Graph edges are never auto-promoted to
  canonical truth. Debug components explain every score contribution.
- **Bounded.** Hard caps on depth, neighbors-per-node, total expansion, and
  final result count prevent runaway scroll.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from .graph_schema import valid_edge_id, valid_entity_id
from .lesson_extractor import contains_secret
from .ranking import RankingContext, RankingPolicy, rank_memory_candidate
from .retriever import MemoryRetriever, RetrievedMemory
from .scoring import normalize_minmax, recency_score
from .schema import RELATION_TYPES, valid_fact_status, valid_point_id_link, valid_relation_type

# ---------------------------------------------------------------------------
# Hard caps (non-negotiable)
# ---------------------------------------------------------------------------

_HARD_MAX_DEPTH = 3
_HARD_MAX_NEIGHBORS = 16
_HARD_MAX_TOTAL_EXPANSION = 256
_HARD_SEED_TOP_K = 50
_HARD_GRAPH_RESULTS = 50
_HARD_SCROLL_PER_CALL = 256

# Fact statuses considered inactive/penalized for graph edges by default.
_INACTIVE_FACT_STATUSES = frozenset({
    "deprecated",
    "superseded",
    "stale",
    "disputed",
    "review_required",
})


# ---------------------------------------------------------------------------
# Policy dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GraphExpansionPolicy:
    """Controls the BFS expansion frontier.

    All values are clamped to hard caps at construction time.
    """

    max_depth: int = 2
    max_neighbors_per_node: int = 8
    max_total_expansion: int = 64
    edge_confidence_floor: float = 0.0
    require_active_fact_status: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_depth", max(1, min(self.max_depth, _HARD_MAX_DEPTH)))
        object.__setattr__(
            self, "max_neighbors_per_node",
            max(1, min(self.max_neighbors_per_node, _HARD_MAX_NEIGHBORS)),
        )
        object.__setattr__(
            self, "max_total_expansion",
            max(1, min(self.max_total_expansion, _HARD_MAX_TOTAL_EXPANSION)),
        )
        object.__setattr__(
            self, "edge_confidence_floor",
            max(0.0, min(float(self.edge_confidence_floor), 1.0)),
        )


@dataclass(frozen=True)
class GraphRankWeights:
    """Weights for the hybrid reranking stage.

    The final score is a weighted sum of component contributions.
    All weights are non-negative floats.

    Graph-distance component is only applied for candidates at graph_distance>0
    (i.e. expansion candidates, not seeds). This ensures the graph boost helps
    non-vector candidates compete with vector-rich seeds without inflating
    seed scores.
    """

    vector: float = 1.0
    graph_distance: float = 0.50
    edge_confidence: float = 0.35
    provenance: float = 0.15
    usefulness: float = 0.15
    staleness_penalty: float = 0.15


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GraphExpandedCandidate:
    """A single candidate in the graph expansion pool."""

    point_id: str
    payload: dict[str, Any]
    vector_score: float  # 0.0 if not a semantic seed
    graph_distance: int  # 0 = seed (vector hit), 1+ = neighbor
    path: list[str]  # entity IDs forming the traversal path
    relation_path: list[str]  # relation types along each hop
    edge_confidences: list[float]  # one per hop
    debug: dict[str, Any] = field(default_factory=dict)
    final_score: float = 0.0


@dataclass
class GraphSearchResult:
    """Complete result of a graph-aware search."""

    query: str
    seeds: list[RetrievedMemory]
    expansions: list[GraphExpandedCandidate]
    final: list[GraphExpandedCandidate]
    debug: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Scope keys that partition memory by profile/user/chat.  These are the same
# keys used by ``retriever._scope_filter``; enumerated here so the graph
# retriever can apply identical conditions to every Qdrant scroll.
_SCOPE_KEYS = ("profile_id", "user_id_hash", "chat_id_hash")


def _scope_filter_conditions(scope: dict[str, str] | None) -> list[dict[str, Any]]:
    """Return Qdrant ``must`` conditions derived from *scope*.

    Only well-formed scope keys (``profile_id``, ``user_id_hash``,
    ``chat_id_hash``) with truthy values are included.  The returned list can
    be appended to any scroll filter's ``must`` array to enforce cross-profile
    isolation.
    """
    if not scope:
        return []
    conditions: list[dict[str, Any]] = []
    for key in _SCOPE_KEYS:
        value = scope.get(key)
        if value:
            conditions.append({"key": key, "match": {"value": str(value)}})
    return conditions


def _payload_in_scope(payload: dict[str, Any], scope: dict[str, str] | None) -> bool:
    """In-memory scope check for retrieved points that bypass scroll filters.

    This is a defensive post-filter used when a Qdrant ``retrieve()`` call does
    not accept a filter (e.g. fetching by point ID).  It verifies that the
    point payload carries the same scope values as *scope*.

    Fail-closed semantics: when *scope* has a truthy value for a key, the
    payload must carry that exact key with a matching value.  A missing or
    empty payload field is rejected, matching Qdrant scroll-filter behavior.
    """
    if not scope:
        return True
    for key in _SCOPE_KEYS:
        expected = scope.get(key)
        if expected:
            actual = str(payload.get(key) or "")
            if actual != str(expected):
                return False
    return True


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(value, hi))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _extract_entity_refs(payload: dict[str, Any]) -> list[str]:
    """Extract entity references from a payload dict.

    Checks ``entity_id``, ``source_entity_id``, ``target_entity_id``, and
    ``entity_ids`` (list) fields. Only well-formed entity IDs pass validation.
    """
    refs: list[str] = []
    seen: set[str] = set()

    def _add(value: Any) -> None:
        validated = valid_entity_id(value)
        if validated and validated not in seen:
            seen.add(validated)
            refs.append(validated)

    _add(payload.get("entity_id"))
    _add(payload.get("source_entity_id"))
    _add(payload.get("target_entity_id"))
    for item in payload.get("entity_ids", []) or []:
        _add(item)
    return refs


def _get_counterparty(edge_payload: dict[str, Any], entity_id: str) -> str | None:
    """Given an edge payload and a known entity on one side, return the other.

    Both source and target IDs are validated; malformed values are dropped.
    """
    src = valid_entity_id(edge_payload.get("source_entity_id"))
    tgt = valid_entity_id(edge_payload.get("target_entity_id"))
    if src == entity_id and tgt and tgt != entity_id:
        return tgt
    if tgt == entity_id and src and src != entity_id:
        return src
    return None


def _query_alias_matches(query_cf: str, payload: dict[str, Any]) -> bool:
    """Check whether query text matches entity label or any alias (casefolded)."""
    label = str(payload.get("label") or payload.get("text") or "").casefold()
    if label and label in query_cf:
        return True
    aliases = payload.get("aliases") or []
    if isinstance(aliases, list):
        for alias in aliases:
            if isinstance(alias, str) and alias.casefold() in query_cf:
                return True
    return False


# ---------------------------------------------------------------------------
# GraphMemoryRetriever
# ---------------------------------------------------------------------------

class GraphMemoryRetriever:
    """Read-only graph-aware memory retriever.

    Combines semantic seed search (via existing ``MemoryRetriever``) with
    bounded BFS expansion over graph entity/edge payloads, then hybrid
    reranks the merged candidate pool.
    """

    def __init__(
        self,
        *,
        qdrant: Any,
        embeddings: Any,
        collection_name: str,
        base_retriever: MemoryRetriever | None = None,
        expansion_policy: GraphExpansionPolicy | None = None,
        rank_weights: GraphRankWeights | None = None,
        ranking_policy: RankingPolicy | None = None,
        scope: dict[str, str] | None = None,
        decay_rate: float = 0.001,
    ) -> None:
        self.qdrant = qdrant
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.expansion_policy = expansion_policy or GraphExpansionPolicy()
        self.rank_weights = rank_weights or GraphRankWeights()
        self.ranking_policy = ranking_policy or RankingPolicy()
        self.scope = scope or {}
        self.decay_rate = decay_rate
        self._base_retriever = base_retriever
        if self._base_retriever is None:
            self._base_retriever = MemoryRetriever(
                qdrant=qdrant,
                embeddings=embeddings,
                collection_name=collection_name,
                scope=scope,
                ranking_policy=self.ranking_policy,
                decay_rate=decay_rate,
            )

    # -- Public API --------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        candidate_seed_top_k: int = 20,
        max_graph_results: int = 20,
        source_type: Any = None,
        tags: list[str] | None = None,
        source: str | None = None,
        file_path: str | None = None,
        project_path: str | None = None,
        since: str | None = None,
        until: str | None = None,
        memory_kind: Any = None,
        fact_status_exclude: Any = None,
        stale: Any = None,
        requires_review: Any = None,
        canonical: Any = None,
        entity_types: list[str] | None = None,
        relation_types: list[str] | None = None,
        include_fact_history: bool = False,
        debug: bool = True,
    ) -> GraphSearchResult:
        """Execute a read-only graph-aware search.

        Returns a ``GraphSearchResult`` with seeds, expansions, and final
        reranked candidates.
        """
        # Clamp user-supplied caps to hard limits
        candidate_seed_top_k = _clamp(candidate_seed_top_k, 1, _HARD_SEED_TOP_K)
        max_graph_results = _clamp(max_graph_results, 1, _HARD_GRAPH_RESULTS)
        top_k = _clamp(top_k, 1, _HARD_GRAPH_RESULTS)

        # Validate relation_types — fail closed if any are invalid
        validated_relation_types: list[str] | None = None
        if relation_types:
            invalid_rts = [rt for rt in relation_types if not valid_relation_type(rt)]
            if invalid_rts:
                raise ValueError(
                    f"invalid relation_type(s): {', '.join(invalid_rts)}. "
                    f"Valid types: {', '.join(RELATION_TYPES)}"
                )
            validated_relation_types = list(dict.fromkeys(
                rt for rt in relation_types if valid_relation_type(rt)
            ))

        # Entity type allowlist (length cap)
        validated_entity_types: list[str] | None = None
        if entity_types:
            validated_entity_types = list(dict.fromkeys(entity_types))[:32]

        warnings: list[str] = []
        hard_caps_hit = {
            "max_depth": False,
            "max_neighbors_per_node": False,
            "max_total_expansion": False,
        }

        # ===================================================================
        # Stage A: Semantic seed search
        # ===================================================================
        seeds = self._base_retriever.search(
            query,
            top_k=candidate_seed_top_k,
            source_type=source_type,
            scope=self.scope or None,
            tags=tags,
            source=source,
            file_path=file_path,
            project_path=project_path,
            since=since,
            until=until,
            memory_kind=memory_kind,
            fact_status_exclude=fact_status_exclude,
            stale=stale,
            requires_review=requires_review,
            canonical=canonical,
            include_fact_history=include_fact_history,
            update_access=False,  # Phase 2: read-only — no access metadata mutation
        )

        stage_a = {
            "requested": candidate_seed_top_k,
            "returned": len(seeds),
            "min_vector_score": min((s.qdrant_score for s in seeds), default=0.0),
        }

        # ===================================================================
        # Stage B: Entity extraction from seeds + query alias matching
        # ===================================================================
        query_cf = " ".join((query or "").split()).casefold()

        # Seed entity refs: entity_id -> seed candidate
        seed_entities: dict[str, GraphExpandedCandidate] = {}
        matched_from_seeds = 0
        for seed in seeds:
            refs = _extract_entity_refs(seed.payload)
            for ref in refs:
                if ref not in seed_entities:
                    seed_entities[ref] = GraphExpandedCandidate(
                        point_id=seed.id,
                        payload=seed.payload,
                        vector_score=seed.final_score,
                        graph_distance=0,
                        path=[ref],
                        relation_path=[],
                        edge_confidences=[],
                    )
                    matched_from_seeds += 1

        # Query alias matching: scroll for graph_entity points whose label/aliases
        # match query tokens. This is the "query-matched entity" seed path.
        matched_from_query = 0
        if query_cf:
            alias_match_entities = self._find_query_matched_entities(
                query_cf,
                validated_entity_types,
                relation_types=validated_relation_types,
            )
            for eid, payload in alias_match_entities:
                if eid not in seed_entities:
                    seed_entities[eid] = GraphExpandedCandidate(
                        point_id=str(payload.get("entity_id") or eid),
                        payload=payload,
                        vector_score=0.0,  # query-match, not vector hit
                        graph_distance=0,
                        path=[eid],
                        relation_path=[],
                        edge_confidences=[],
                    )
                    matched_from_query += 1

        stage_b = {
            "matched_from_seeds": matched_from_seeds,
            "matched_from_query_aliases": matched_from_query,
        }

        # ===================================================================
        # Stage C+D: Bounded BFS expansion over edges
        # ===================================================================
        expansions: list[GraphExpandedCandidate] = []
        visited_entities: set[str] = set()
        frontier: list[tuple[str, int, list[str], list[str], list[float]]] = []

        for eid, candidate in seed_entities.items():
            visited_entities.add(eid)
            frontier.append((eid, 0, candidate.path, candidate.relation_path, candidate.edge_confidences))

        scroll_calls = 0
        depth_edges = {1: 0, 2: 0, 3: 0}
        filtered_fact_status = 0
        filtered_confidence_floor = 0

        max_total = self.expansion_policy.max_total_expansion
        per_node_cap = self.expansion_policy.max_neighbors_per_node
        max_depth = self.expansion_policy.max_depth

        while frontier and len(expansions) < max_total:
            eid, depth, current_path, current_rel_path, current_confidences = frontier.pop(0)

            if depth >= max_depth:
                hard_caps_hit["max_depth"] = True
                continue

            # Scroll for edges where this entity is source OR target
            edges_subject = self._scroll_edges_for_entity(
                eid, side="source", relation_types=validated_relation_types,
            )
            scroll_calls += 1
            edges_object = self._scroll_edges_for_entity(
                eid, side="target", relation_types=validated_relation_types,
            )
            scroll_calls += 1
            all_edges = edges_subject + edges_object

            # Deduplicate edges by validated edge id — drop invalid edge IDs.
            seen_edge_ids: set[str] = set()
            deduped_edges: list[tuple[str, dict[str, Any]]] = []
            dropped_invalid_edge = 0
            for edge in all_edges:
                edge_payload_raw = edge.get("payload") or {}
                # Validate edge id from payload.edge_id or point.id.
                validated_edge_id = valid_edge_id(
                    edge_payload_raw.get("edge_id") or edge.get("id")
                )
                if not validated_edge_id:
                    dropped_invalid_edge += 1
                    continue
                if validated_edge_id not in seen_edge_ids:
                    seen_edge_ids.add(validated_edge_id)
                    deduped_edges.append((validated_edge_id, edge))

            if dropped_invalid_edge:
                warnings.append(
                    f"dropped {dropped_invalid_edge} edge(s) with invalid edge_id"
                )

            neighbor_count = 0
            for validated_edge_id, edge_point in deduped_edges:
                if neighbor_count >= per_node_cap:
                    hard_caps_hit["max_neighbors_per_node"] = True
                    break

                if len(expansions) >= max_total:
                    hard_caps_hit["max_total_expansion"] = True
                    break

                edge_payload = edge_point.get("payload") or {}
                # Validate relation_type — do not echo the raw invalid value
                rel = edge_payload.get("relation_type")
                if rel and not valid_relation_type(rel):
                    warnings.append("dropped edge with invalid relation_type")
                    continue

                # Fact status filter
                edge_fs = valid_fact_status(edge_payload.get("fact_status")) or "active"
                if self.expansion_policy.require_active_fact_status and not include_fact_history:
                    if edge_fs in _INACTIVE_FACT_STATUSES:
                        filtered_fact_status += 1
                        continue

                # Confidence floor — defensive parse (non-numeric/NaN → 0.0)
                raw_conf = edge_payload.get("confidence")
                if raw_conf is None:
                    edge_conf = 0.0
                else:
                    try:
                        edge_conf = float(raw_conf)
                        if not math.isfinite(edge_conf):
                            edge_conf = 0.0
                    except (TypeError, ValueError):
                        edge_conf = 0.0
                if edge_conf < self.expansion_policy.edge_confidence_floor:
                    filtered_confidence_floor += 1
                    continue

                counterparty = _get_counterparty(edge_payload, eid)
                if not counterparty:
                    continue

                # Skip self-loops and already-visited entities to prevent cycles
                if counterparty == eid or counterparty in visited_entities:
                    continue

                neighbor_count += 1
                next_depth = depth + 1
                new_path = current_path + [counterparty]
                new_rel_path = current_rel_path + [str(rel or "")]
                new_confidences = current_confidences + [edge_conf]

                # Resolve the counterparty entity to get its source memory points
                counterparty_memories = self._resolve_entity_memories(
                    counterparty,
                    validated_entity_types,
                    include_fact_history,
                )

                if counterparty_memories:
                    for mem_id, mem_payload in counterparty_memories:
                        if len(expansions) >= max_total:
                            hard_caps_hit["max_total_expansion"] = True
                            break
                        # Skip if memory payload contains secrets
                        mem_text = str(mem_payload.get("text") or "")
                        if contains_secret(mem_text):
                            warnings.append("dropped memory candidate: text triggered secret detection")
                            continue
                        expansions.append(GraphExpandedCandidate(
                            point_id=mem_id,
                            payload=mem_payload,
                            vector_score=0.0,
                            graph_distance=next_depth,
                            path=new_path,
                            relation_path=new_rel_path,
                            edge_confidences=new_confidences,
                        ))
                else:
                    # Even without resolved memory points, include the edge as an
                    # expansion candidate — it may still be useful context.
                    # Use the validated edge id, never the raw point id.
                    expansions.append(GraphExpandedCandidate(
                        point_id=validated_edge_id,
                        payload=edge_payload,
                        vector_score=0.0,
                        graph_distance=next_depth,
                        path=new_path,
                        relation_path=new_rel_path,
                        edge_confidences=new_confidences,
                    ))

                depth_edges[next_depth] = depth_edges.get(next_depth, 0) + 1

                if counterparty not in visited_entities and next_depth < max_depth:
                    visited_entities.add(counterparty)
                    frontier.append((
                        counterparty, next_depth, new_path, new_rel_path, new_confidences,
                    ))

        # Defensive: hard assertion
        if len(expansions) > max_total:
            expansions = expansions[:max_total]
            hard_caps_hit["max_total_expansion"] = True

        stage_c = {
            "depth_1_edges": depth_edges.get(1, 0),
            "depth_2_edges": depth_edges.get(2, 0),
            "depth_3_edges": depth_edges.get(3, 0),
            "scroll_calls": scroll_calls,
            "filtered_fact_status": filtered_fact_status,
            "filtered_confidence_floor": filtered_confidence_floor,
            "total_expansions": len(expansions),
        }

        # ===================================================================
        # Stage E: Hybrid reranking
        # ===================================================================
        # Merge seeds + expansions into the final candidate pool.
        # Seeds are wrapped as candidates at graph_distance=0.
        merged: list[GraphExpandedCandidate] = []

        # Seeds: wrap RetrievedMemory as candidates
        for seed in seeds:
            seed_refs = _extract_entity_refs(seed.payload)
            merged.append(GraphExpandedCandidate(
                point_id=seed.id,
                payload=seed.payload,
                vector_score=seed.final_score,
                graph_distance=0,
                path=seed_refs[:1] if seed_refs else [],
                relation_path=[],
                edge_confidences=[],
            ))

        # Add expansions (deduplicate against seeds by point_id)
        seed_ids = {s.id for s in seeds}
        for exp in expansions:
            if exp.point_id not in seed_ids:
                merged.append(exp)
                seed_ids.add(exp.point_id)

        # Score each candidate
        dropped_zero_score = 0
        for candidate in merged:
            candidate.final_score, candidate.debug = self._score_candidate(
                candidate, query, include_fact_history,
            )
            if candidate.final_score <= 0.0:
                dropped_zero_score += 1

        # Sort by final_score descending, drop zero/negative scores
        scored = [c for c in merged if c.final_score > 0.0]
        scored.sort(key=lambda c: c.final_score, reverse=True)

        # Cap at max_graph_results then top_k
        final = scored[:max_graph_results][:top_k]

        stage_e = {
            "candidates_in": len(merged),
            "candidates_out": len(final),
            "dropped_zero_score": dropped_zero_score,
        }

        result_debug: dict[str, Any] = {}
        if debug:
            result_debug = {
                "algorithm": "graph_v1",
                "query_length": len(query or ""),
                "policy": asdict(self.expansion_policy),
                "weights": asdict(self.rank_weights),
                "scope_keys": [k for k, v in self.scope.items() if v],
                "stages": {
                    "A_seed_search": stage_a,
                    "B_entity_extraction": stage_b,
                    "C_edge_query": stage_c,
                    "E_hybrid_rerank": stage_e,
                },
                "hard_caps_hit": hard_caps_hit,
                "warnings": warnings,
            }

        return GraphSearchResult(
            query=query,
            seeds=seeds,
            expansions=expansions,
            final=final,
            debug=result_debug,
        )

    # -- Private helpers ---------------------------------------------------

    def _score_candidate(
        self,
        candidate: GraphExpandedCandidate,
        query: str,
        include_fact_history: bool,
    ) -> tuple[float, dict[str, Any]]:
        """Compute the hybrid score and debug dict for a single candidate."""
        w = self.rank_weights
        payload = candidate.payload or {}

        # Component 1: Vector score contribution
        vector_component = w.vector * float(candidate.vector_score)

        # Component 2: Graph distance decay (exp(-d)) — only for expansion candidates
        if candidate.graph_distance > 0:
            graph_dist_val = math.exp(-float(candidate.graph_distance))
            graph_dist_component = w.graph_distance * graph_dist_val
        else:
            graph_dist_component = 0.0

        # Component 3: Edge confidence (mean of hop confidences)
        edge_conf_mean = _mean(candidate.edge_confidences)
        edge_conf_component = w.edge_confidence * edge_conf_mean

        # Component 4: Provenance rerank via existing rank_memory_candidate
        recency = recency_score(str(payload.get("created_at") or ""), self.decay_rate)
        base_for_rank = float(candidate.vector_score) if candidate.vector_score else 0.01
        provenance_ranked = rank_memory_candidate(
            base_score=base_for_rank,
            vector_score=float(candidate.vector_score),
            payload=payload,
            context=RankingContext(
                query=query,
                include_fact_history=include_fact_history,
            ),
            policy=self.ranking_policy,
            recency_decay=recency,
        )
        provenance_component = w.provenance * (provenance_ranked.score / max(provenance_ranked.base_score, 0.01))

        # Component 5: Usefulness weight
        usefulness_val = float(payload.get("usefulness_weight") or 0.0)
        usefulness_component = w.usefulness * usefulness_val

        # Component 6: Staleness penalty
        edge_fs = valid_fact_status(payload.get("fact_status")) or "active"
        staleness_indicator = 1.0 if edge_fs in _INACTIVE_FACT_STATUSES else 0.0
        staleness_component = w.staleness_penalty * staleness_indicator

        final_score = (
            vector_component
            + graph_dist_component
            + edge_conf_component
            + provenance_component
            + usefulness_component
            - staleness_component
        )

        debug = {
            "vector_score": candidate.vector_score,
            "graph_distance": candidate.graph_distance,
            "edge_confidences": candidate.edge_confidences,
            "relation_path": candidate.relation_path,
            "component_scores": {
                "vector": round(vector_component, 6),
                "graph_distance": round(graph_dist_component, 6),
                "edge_confidence": round(edge_conf_component, 6),
                "provenance": round(provenance_component, 6),
                "usefulness": round(usefulness_component, 6),
                "staleness_penalty": round(staleness_component, 6),
            },
            "provenance_rerank": provenance_ranked.debug,
            "final_score": round(final_score, 6),
        }

        return final_score, debug

    def _find_query_matched_entities(
        self,
        query_cf: str,
        entity_types: list[str] | None,
        *,
        relation_types: list[str] | None = None,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Scroll for graph_entity points whose label/aliases match query tokens.

        Scope conditions (``profile_id`` / ``user_id_hash`` / ``chat_id_hash``)
        are always appended to prevent cross-profile entity leakage.
        """
        must: list[dict[str, Any]] = [
            {"key": "memory_kind", "match": {"value": "graph_entity"}},
        ]
        must.extend(_scope_filter_conditions(self.scope))
        if entity_types:
            must.append({"key": "entity_type", "match": {"any": entity_types}})

        filt: dict[str, Any] = {"must": must}

        try:
            points = self.qdrant.scroll_by_filter(
                self.collection_name,
                filt,
                limit=_HARD_SCROLL_PER_CALL,
                max_total=min(self.expansion_policy.max_total_expansion, _HARD_SCROLL_PER_CALL),
            )
        except Exception:
            return []

        results: list[tuple[str, dict[str, Any]]] = []
        for point in points:
            payload = point.get("payload") or {}
            if _query_alias_matches(query_cf, payload):
                # Validate entity_id — only accept IDs that pass valid_entity_id().
                # Do not fall back to arbitrary point IDs unless they also validate.
                eid = valid_entity_id(payload.get("entity_id"))
                if eid is None:
                    eid = valid_entity_id(point.get("id"))
                if eid:
                    results.append((eid, payload))
        return results

    def _scroll_edges_for_entity(
        self,
        entity_id: str,
        *,
        side: str,
        relation_types: list[str] | None,
    ) -> list[dict[str, Any]]:
        """Scroll for graph_edge points where the entity is on the given side.

        The scroll is bounded by ``max_total`` derived from the per-node
        neighbor cap to prevent unbounded pagination.
        """
        # Defensive: refuse to scroll with an invalid entity_id.
        if not valid_entity_id(entity_id):
            return []

        key = "source_entity_id" if side == "source" else "target_entity_id"
        must: list[dict[str, Any]] = [
            {"key": "memory_kind", "match": {"value": "graph_edge"}},
            {"key": key, "match": {"value": entity_id}},
        ]
        must.extend(_scope_filter_conditions(self.scope))
        if relation_types:
            must.append({"key": "relation_type", "match": {"any": relation_types}})

        filt: dict[str, Any] = {"must": must}

        # Bound the scroll: allow at most a small overfetch factor above the
        # per-node neighbor cap so dedup/filtering can work without unbounded
        # pagination. Hard cap at _HARD_SCROLL_PER_CALL as a safety ceiling.
        bounded_max = min(
            self.expansion_policy.max_neighbors_per_node * 3,
            _HARD_SCROLL_PER_CALL,
        )

        try:
            return self.qdrant.scroll_by_filter(
                self.collection_name,
                filt,
                limit=min(bounded_max, _HARD_SCROLL_PER_CALL),
                max_total=bounded_max,
            )
        except Exception:
            return []

    def _resolve_entity_memories(
        self,
        entity_id: str,
        entity_types: list[str] | None,
        include_fact_history: bool,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Resolve an entity to its associated memory points via source_point_ids.

        Returns a list of (point_id, payload) tuples for memory points that
        reference this entity.

        Scope conditions are applied to both the entity scroll and as a
        defensive post-filter on retrieved source points to prevent
        cross-profile memory leakage.
        """
        must: list[dict[str, Any]] = [
            {"key": "entity_id", "match": {"value": entity_id}},
        ]
        must.extend(_scope_filter_conditions(self.scope))
        if entity_types:
            must.append({"key": "entity_type", "match": {"any": entity_types}})

        filt: dict[str, Any] = {"must": must}

        try:
            entity_points = self.qdrant.scroll_by_filter(
                self.collection_name,
                filt,
                limit=4,
                max_total=4,
            )
        except Exception:
            entity_points = []

        results: list[tuple[str, dict[str, Any]]] = []
        for ep in entity_points:
            ep_payload = ep.get("payload") or {}
            source_point_ids = ep_payload.get("source_point_ids") or []
            if not source_point_ids:
                continue
            # Validate each source point ID before retrieval
            validated_pids = [
                pid for pid in (valid_point_id_link(p) for p in source_point_ids[:8])
                if pid
            ]
            if not validated_pids:
                continue
            try:
                points = self.qdrant.retrieve(
                    self.collection_name,
                    validated_pids,
                )
            except Exception:
                continue
            for point in points:
                p_payload = point.get("payload") or {}
                # Defensive: reject source points from a different scope even if
                # the entity scroll matched — Qdrant retrieve() does not accept
                # a filter, so we verify scope in-memory.
                if not _payload_in_scope(p_payload, self.scope):
                    continue
                p_fs = valid_fact_status(p_payload.get("fact_status")) or "active"
                if (
                    not include_fact_history
                    and p_fs in _INACTIVE_FACT_STATUSES
                ):
                    continue
                results.append((str(point.get("id") or ""), p_payload))

        return results
