"""Phase 5 RAPTOR search / zoom tests.

These tests verify that:

- parent summary matches surface cited leaves correctly;
- leaf matches surface their parents (root and intermediate);
- evidence mode demotes parents without cited leaves;
- cross-scope child IDs are dropped on retrieve-by-id;
- unsafe / quarantined / secret-bearing / requires_review children are not
  promoted into cited_leaves;
- the searcher is read-only (zero upsert/delete/update_payload on the fake);
- output shape is stable and always carries ``context_not_instruction``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from qdrant_memory.raptor.search import (
    HARD_CONTEXT_CHAR_BUDGET,
    RaptorLeafHit,
    RaptorSearcher,
    RaptorSearchResult,
    RaptorSummaryHit,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeEmbedding:
    def __init__(self):
        self.queries: list[str] = []

    def embed_query(self, text: str):
        self.queries.append(text)
        return [0.1, 0.2]


class FakeRaptorQdrant:
    """In-memory fake Qdrant tracking every mutation for read-only assertions."""

    def __init__(self, dense_seeds=None):
        self.dense_seeds = dense_seeds or []
        self._store: dict[str, dict[str, Any]] = {}
        self.upserts: list = []
        self.update_payloads: list = []
        self.deletes: list = []
        self.delete_filters: list = []

    def add_point(self, point_id: str, payload: dict[str, Any]) -> None:
        self._store[point_id] = {"id": point_id, "payload": payload}

    def search(self, name, vector, limit, filter=None, with_payload=True, with_vector=False):
        return list(self.dense_seeds)

    def retrieve(self, name, ids, *, with_payload=True, with_vector=False):
        out = []
        for pid in ids:
            if pid in self._store:
                out.append(self._store[pid])
        return out

    def update_payload(self, name, point_id, payload):
        self.update_payloads.append((name, point_id, payload))

    def upsert(self, name, points):
        self.upserts.append((name, points))

    def delete_ids(self, name, ids):
        self.deletes.append((name, ids))

    def delete_filter(self, name, flt):
        self.delete_filters.append((name, flt))


class FakeRetriever:
    """Minimal MemoryRetriever-like — only ``search`` is invoked."""

    def __init__(self, dense_seeds):
        self._dense_seeds = dense_seeds
        self.calls: list[dict[str, Any]] = []

    def search(self, query, *, top_k=5, update_access=False, **kwargs):
        self.calls.append({
            "query": query,
            "top_k": top_k,
            "update_access": update_access,
            "kwargs": kwargs,
        })
        return list(self._dense_seeds)


class _RetrievedMemory:
    """Stand-in for :class:`MemoryRetriever`'s output."""

    def __init__(self, pid, text, payload, final_score=0.5):
        self.id = pid
        self.text = text
        self.payload = payload
        self.final_score = final_score
        self.qdrant_score = final_score
        self.ranking_debug = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _raptor_parent_payload(
    *,
    node_id: str,
    text: str = "parent summary text",
    children: list[str] | None = None,
    parents: list[str] | None = None,
    summary_of: list[str] | None = None,
    level: int = 2,
    tree_id: str = "raptor-tree-test",
    root_id: str = "raptor-root-test",
    build_id: str = "raptor-build-test",
    profile_id: str = "default",
    secret: bool = False,
    # Phase 5 fix12 (final10 P2): the parent trust gate now
    # treats ``requires_review=True`` and friends as non-active
    # markers, mirroring ``_raptor_leaf_payload`` style. Tests
    # that want the existing unsafe-parent behaviour must opt in
    # explicitly so the defaults represent an APPROVED / clean
    # parent payload consistent with the production code path.
    requires_review: bool = False,
    raptor_review_status: str = "approved",
    fact_status: str = "",
    stale: bool = False,
    quarantined: bool = False,
    raptor_excluded: bool = False,
    raptor_forgotten: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "raptor_node_id": node_id,
        "raptor_level": int(level),
        "raptor_parent_ids": list(parents or []),
        "raptor_child_ids": list(children or []),
        "raptor_summary_of": list(summary_of or []),
        "raptor_tree_id": tree_id,
        "raptor_root_id": root_id,
        "raptor_build_id": build_id,
        "raptor_cluster_id": f"raptor-cluster-{node_id}",
        "derivation_type": "raptor_summary",
        "memory_kind": "summary",
        "canonical": True,
        "requires_review": bool(requires_review),
        "raptor_review_status": str(raptor_review_status or ""),
        "fact_status": str(fact_status or ""),
        "stale": bool(stale),
        "consolidation_quarantined": bool(quarantined),
        "raptor_excluded": bool(raptor_excluded),
        "raptor_forgotten": bool(raptor_forgotten),
        "text": text,
        "profile_id": profile_id,
        "source_hashes": ["a" * 64],
        "derived_from": [],
    }
    if secret:
        # Inject a bearer-shaped token into the text so contains_secret fires.
        payload["text"] = "".join(["Bearer ", "f" * 24]) + " suffix"
    return payload


def _leaf_payload(
    *,
    point_id: str,
    text: str,
    profile_id: str = "default",
    source_uri: str = "test://source",
    quarantined: bool = False,
    fact_status: str = "",
    requires_review: bool = False,
    stale: bool = False,
    secret: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": point_id,
        "text": text,
        "source_uri": source_uri,
        "source_type": "manual",
        "file_path": "",
        "heading": "",
        "profile_id": profile_id,
        "memory_kind": "memory",
    }
    if quarantined:
        payload["consolidation_quarantined"] = True
    if fact_status:
        payload["fact_status"] = fact_status
    if requires_review:
        payload["requires_review"] = True
    if stale:
        payload["stale"] = True
    if secret:
        # Use a bearer-shaped text that contains_secret trips on.
        payload["text"] = "".join(["Bearer ", "a" * 24]) + " tail"
    return payload


# ---------------------------------------------------------------------------
# Parent → children promotion
# ---------------------------------------------------------------------------


class TestParentToChildren:
    def test_parent_summary_match_returns_cited_leaves(self):
        # Dense seed lands on a RAPTOR parent summary (level 2).
        dense_seeds = [
            _RetrievedMemory(
                "parent-A",
                text="parent summary text",
                payload=_raptor_parent_payload(
                    node_id="parent-A",
                    children=["leaf-1", "leaf-2"],
                    summary_of=["leaf-1", "leaf-2"],
                    level=2,
                ),
                final_score=0.91,
            )
        ]

        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-A", dense_seeds[0].payload)
        qdrant.add_point("leaf-1", _leaf_payload(point_id="leaf-1", text="about deploy"))
        qdrant.add_point("leaf-2", _leaf_payload(point_id="leaf-2", text="about rollback"))

        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )

        result = searcher.search("deploy", top_k=3)

        assert isinstance(result, RaptorSearchResult)
        assert any(s.raptor_node_id == "parent-A" for s in result.summaries)
        promoted_leaf_ids = {l.point_id for l in result.cited_leaves}
        assert promoted_leaf_ids >= {"leaf-1", "leaf-2"}
        assert all(isinstance(item, RaptorLeafHit) for item in result.cited_leaves)


# ---------------------------------------------------------------------------
# Leaf → parent / root promotion
# ---------------------------------------------------------------------------


class TestLeafToParents:
    def test_leaf_match_surfaces_parent_summary(self):
        # Dense seed is a leaf, not a parent. The leaf carries a RAPTOR
        # node id pointing at the same RAPTOR tree whose parent payload we
        # rehydrate via retrieve-by-id.
        dense_seeds = [
            _RetrievedMemory(
                "leaf-X",
                text="leaf content",
                payload={
                    "id": "leaf-X",
                    "text": "leaf content",
                    "raptor_node_id": "leaf-ref-X",
                    "profile_id": "default",
                    "memory_kind": "memory",
                },
                final_score=0.7,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point(
            "leaf-ref-X",
            _raptor_parent_payload(
                node_id="leaf-ref-X",
                children=["leaf-X"],
                summary_of=["leaf-X"],
                level=1,
            ),
        )
        qdrant.add_point("leaf-X", _leaf_payload(point_id="leaf-X", text="leaf content"))
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("leaf", top_k=3)
        assert any(s.raptor_node_id == "leaf-ref-X" for s in result.summaries)

    def test_zoom_to_root_via_parent_chain(self):
        # Two-level RAPTOR: root -> mid -> leaf-ref. The searcher should walk
        # up to the root within max_depth=2.
        dense_seeds = [
            _RetrievedMemory(
                "leaf-mid",
                text="mid leaf content",
                payload={
                    "id": "leaf-mid",
                    "text": "mid leaf content",
                    "raptor_node_id": "mid",
                    "profile_id": "default",
                    "memory_kind": "memory",
                },
                final_score=0.6,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("mid", _raptor_parent_payload(
            node_id="mid", parents=["root"], children=["leaf-mid"], summary_of=["leaf-mid"], level=2,
        ))
        qdrant.add_point("root", _raptor_parent_payload(
            node_id="root", parents=[], children=["mid"], summary_of=[], level=2,
        ))
        qdrant.add_point("leaf-mid", _leaf_payload(point_id="leaf-mid", text="mid leaf content"))
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("any", top_k=5, max_depth=2)
        node_ids = {s.raptor_node_id for s in result.summaries}
        assert {"mid", "root"}.issubset(node_ids)


# ---------------------------------------------------------------------------
# Evidence mode demotion
# ---------------------------------------------------------------------------


class TestEvidenceMode:
    def test_evidence_mode_isolated_to_router(self):
        # Evidence mode lives on the router; the searcher itself doesn't
        # decide promotion. Sanity check that the searcher always returns
        # both summaries and leaves so the router can demote as needed.
        dense_seeds = [
            _RetrievedMemory(
                "p-orphan",
                text="orphan summary",
                payload=_raptor_parent_payload(
                    node_id="p-orphan",
                    children=["leaf-orphan"],
                    summary_of=["leaf-orphan"],
                    level=2,
                ),
                final_score=0.5,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("p-orphan", dense_seeds[0].payload)
        qdrant.add_point("leaf-orphan", _leaf_payload(point_id="leaf-orphan", text="leaf"))
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3)
        # The searcher surfaces both the parent and its cited leaves; the
        # router is what demotes in evidence mode.
        assert any(s.raptor_node_id == "p-orphan" for s in result.summaries)
        assert any(l.point_id == "leaf-orphan" for l in result.cited_leaves)


# ---------------------------------------------------------------------------
# Cross-scope isolation (retrieve-by-id has no filter, so defensive
# in-memory post-filtering is mandatory).
# ---------------------------------------------------------------------------


class TestScopeIsolation:
    def test_cross_profile_child_is_dropped(self):
        # The dense seed lands on parent A with child-foreign. When
        # retrieve-by-id returns child-foreign with profile_id="other" and
        # our scope is profile_id="default", the leaf must NOT be promoted.
        dense_seeds = [
            _RetrievedMemory(
                "parent-A",
                text="parent",
                payload=_raptor_parent_payload(
                    node_id="parent-A",
                    children=["child-local", "child-foreign"],
                    summary_of=["child-local", "child-foreign"],
                    level=2,
                    profile_id="default",
                ),
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-A", dense_seeds[0].payload)
        qdrant.add_point("child-local", _leaf_payload(point_id="child-local", text="local"))
        qdrant.add_point("child-foreign", _leaf_payload(
            point_id="child-foreign", text="foreign", profile_id="other",
        ))

        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
            scope={"profile_id": "default"},
        )

        result = searcher.search("anything", top_k=3)
        promoted = {l.point_id for l in result.cited_leaves}
        assert "child-local" in promoted
        assert "child-foreign" not in promoted


# ---------------------------------------------------------------------------
# Unsafe children are not promoted
# ---------------------------------------------------------------------------


class TestUnsafeChildFiltering:
    def test_quarantined_child_skipped(self):
        dense_seeds = [
            _RetrievedMemory(
                "parent-A",
                text="parent",
                payload=_raptor_parent_payload(
                    node_id="parent-A",
                    children=["leaf-ok", "leaf-q"],
                    summary_of=["leaf-ok", "leaf-q"],
                ),
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-A", dense_seeds[0].payload)
        qdrant.add_point("leaf-ok", _leaf_payload(point_id="leaf-ok", text="ok"))
        qdrant.add_point("leaf-q", _leaf_payload(point_id="leaf-q", text="bad", quarantined=True))
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(qdrant=qdrant, retriever=retriever, collection_name="memory")
        result = searcher.search("anything", top_k=5)
        promoted = {l.point_id for l in result.cited_leaves}
        assert "leaf-ok" in promoted
        assert "leaf-q" not in promoted

    def test_secret_bearing_child_skipped(self):
        dense_seeds = [
            _RetrievedMemory(
                "parent-A",
                text="parent",
                payload=_raptor_parent_payload(
                    node_id="parent-A",
                    children=["leaf-ok", "leaf-secret"],
                    summary_of=["leaf-ok", "leaf-secret"],
                ),
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-A", dense_seeds[0].payload)
        qdrant.add_point("leaf-ok", _leaf_payload(point_id="leaf-ok", text="ok"))
        qdrant.add_point("leaf-secret", _leaf_payload(
            point_id="leaf-secret", text="", secret=True,
        ))
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(qdrant=qdrant, retriever=retriever, collection_name="memory")
        result = searcher.search("anything", top_k=5)
        promoted = {l.point_id for l in result.cited_leaves}
        assert "leaf-ok" in promoted
        assert "leaf-secret" not in promoted

    def test_requires_review_child_demoted(self):
        dense_seeds = [
            _RetrievedMemory(
                "parent-A",
                text="parent",
                payload=_raptor_parent_payload(
                    node_id="parent-A",
                    children=["leaf-r"],
                    summary_of=["leaf-r"],
                ),
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-A", dense_seeds[0].payload)
        qdrant.add_point("leaf-r", _leaf_payload(
            point_id="leaf-r", text="review me", requires_review=True,
        ))
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(qdrant=qdrant, retriever=retriever, collection_name="memory")
        result = searcher.search("anything", top_k=5)
        promoted = {l.point_id for l in result.cited_leaves}
        assert "leaf-r" not in promoted

    def test_stale_fact_status_child_demoted(self):
        dense_seeds = [
            _RetrievedMemory(
                "parent-A",
                text="parent",
                payload=_raptor_parent_payload(
                    node_id="parent-A",
                    children=["leaf-stale"],
                    summary_of=["leaf-stale"],
                ),
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-A", dense_seeds[0].payload)
        qdrant.add_point("leaf-stale", _leaf_payload(
            point_id="leaf-stale", text="stale", fact_status="stale",
        ))
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(qdrant=qdrant, retriever=retriever, collection_name="memory")
        result = searcher.search("anything", top_k=5)
        promoted = {l.point_id for l in result.cited_leaves}
        assert "leaf-stale" not in promoted


# ---------------------------------------------------------------------------
# Read-only safety
# ---------------------------------------------------------------------------


class TestReadOnlyInvariants:
    def test_zero_mutations_called(self):
        dense_seeds = [
            _RetrievedMemory(
                "p",
                text="p",
                payload=_raptor_parent_payload(
                    node_id="p", children=["leaf"], summary_of=["leaf"],
                ),
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("p", dense_seeds[0].payload)
        qdrant.add_point("leaf", _leaf_payload(point_id="leaf", text="leaf text"))
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(qdrant=qdrant, retriever=retriever, collection_name="memory")
        searcher.search("anything", top_k=3)
        assert qdrant.upserts == []
        assert qdrant.update_payloads == []
        assert qdrant.deletes == []
        assert qdrant.delete_filters == []

    def test_update_access_false_is_passed(self):
        dense_seeds = []
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=FakeRaptorQdrant(),
            retriever=retriever,
            collection_name="memory",
        )
        searcher.search("anything", top_k=3)
        assert retriever.calls, "retriever should have been called at least once"
        assert all(c["update_access"] is False for c in retriever.calls)

    def test_warning_does_not_echo_raw_secret_id(self):
        # Build the secret-shaped id at runtime so the scanner doesn't trip
        # on a literal.
        bad_id = "".join(["Bearer ", "z" * 24])
        dense_seeds = [
            _RetrievedMemory(
                bad_id,
                text="parent",
                payload=_raptor_parent_payload(
                    node_id=bad_id, children=["x"], summary_of=["x"],
                ),
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point(bad_id, dense_seeds[0].payload)
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(qdrant=qdrant, retriever=retriever, collection_name="memory")
        result = searcher.search("anything", top_k=3)
        for warning in result.warnings:
            assert bad_id not in warning


# ---------------------------------------------------------------------------
# Budget clamping + output shape
# ---------------------------------------------------------------------------


class TestBudgetsAndShape:
    def test_budgets_clamped_to_hard_caps(self):
        dense_seeds = [
            _RetrievedMemory(
                "p",
                text="p",
                payload=_raptor_parent_payload(
                    node_id="p", children=["leaf"], summary_of=["leaf"],
                ),
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("p", dense_seeds[0].payload)
        qdrant.add_point("leaf", _leaf_payload(point_id="leaf", text="leaf text"))
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(qdrant=qdrant, retriever=retriever, collection_name="memory")
        # Caller asks for absurd caps. Searcher should clamp internally.
        result = searcher.search(
            "anything",
            top_k=1000,
            max_depth=99,
            max_children=999,
            max_source_chars=99999,
        )
        debug = result.debug or {}
        assert debug["top_k"] <= 20
        assert debug["max_depth"] <= 3
        assert debug["max_children"] <= 16
        assert debug["max_source_chars"] <= HARD_CONTEXT_CHAR_BUDGET

    def test_dedupe_by_point_id(self):
        # Two parents share the same child. The search result should
        # contain the child only once in cited_leaves.
        dense_seeds = [
            _RetrievedMemory(
                "p1",
                text="p1",
                payload=_raptor_parent_payload(
                    node_id="p1", children=["shared-leaf"], summary_of=["shared-leaf"], level=1,
                ),
            ),
            _RetrievedMemory(
                "p2",
                text="p2",
                payload=_raptor_parent_payload(
                    node_id="p2", children=["shared-leaf"], summary_of=["shared-leaf"], level=1,
                ),
            ),
        ]
        qdrant = FakeRaptorQdrant()
        for seed in dense_seeds:
            qdrant.add_point(seed.id, seed.payload)
        qdrant.add_point("shared-leaf", _leaf_payload(point_id="shared-leaf", text="shared"))
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(qdrant=qdrant, retriever=retriever, collection_name="memory")
        result = searcher.search("anything", top_k=5)
        leaf_ids = [l.point_id for l in result.cited_leaves]
        assert leaf_ids.count("shared-leaf") == 1

    def test_output_shape(self):
        result = RaptorSearchResult(query="hello")
        d = result.to_dict(include_metadata=False)
        # Phase 5 fix12 (final10 P3): raw ``query`` is NEVER emitted.
        # ``RaptorSearchResult.to_dict`` projects the safe
        # ``_redact_query_metadata`` block instead so a secret-shaped
        # query cannot leak through the JSON envelope.
        assert "query" not in d
        assert d["query_length"] == len("hello")
        assert d["query_redacted"].startswith("[redacted")
        assert isinstance(d["query_digest"], str) and len(d["query_digest"]) == 16
        assert d["summaries"] == []
        assert d["cited_leaves"] == []
        assert d["warnings"] == []
        assert d["unsafe_summary_ids"] == []
        assert d["unsafe_leaf_ids"] == []
        assert isinstance(d["debug"], dict)


# ---------------------------------------------------------------------------
# AST-based read-only check for the raptor.search module
# ---------------------------------------------------------------------------


class TestNoMutationAST:
    def test_raptor_search_module_has_no_mutations(self):
        import ast
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "qdrant_memory" / "raptor" / "search.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        offending: list[str] = []
        forbidden_call_names = {
            "upsert",
            "delete_payload",
            "delete_filter",
            "delete_ids",
            "update_payload",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name: str
                if isinstance(func, ast.Attribute):
                    name = func.attr
                elif isinstance(func, ast.Name):
                    name = func.id
                else:
                    continue
                if name in forbidden_call_names:
                    offending.append(f"{name} at line {node.lineno}")
        assert offending == [], f"forbidden calls: {offending}"


# ---------------------------------------------------------------------------
# Adversarial: secret-shaped IDs never leak through any output channel
# ---------------------------------------------------------------------------


class TestSecretIdRedaction:
    def _build_searcher_with_secret_leaf(self, *, leaf_id: str):
        """Seed a RAPTOR parent that declares a secret-shaped child id.

        The leaf's payload carries the secret-shaped id in
        ``consolidation_quarantined``-adjacent fields so the unsafe
        demotion path emits a warning we can inspect.
        """
        dense_seeds = [
            _RetrievedMemory(
                "parent-A",
                text="parent",
                payload=_raptor_parent_payload(
                    node_id="parent-A",
                    children=[leaf_id],
                    summary_of=[leaf_id],
                ),
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-A", dense_seeds[0].payload)
        # Leaf payload carries the secret-shaped id as the point id;
        # we mark the leaf as ``consolidation_quarantined`` so the
        # searcher's safety assessment demotes it and emits a warning
        # that historically echoed the raw id.
        leaf_payload = _leaf_payload(point_id=leaf_id, text="bad leaf")
        leaf_payload["consolidation_quarantined"] = True
        qdrant.add_point(leaf_id, leaf_payload)
        retriever = FakeRetriever(dense_seeds)
        return RaptorSearcher(qdrant=qdrant, retriever=retriever, collection_name="memory"), qdrant

    def test_secret_id_warning_uses_redacted_handle(self):
        bad_id = "".join(["Bearer ", "r" * 24])
        searcher, _ = self._build_searcher_with_secret_leaf(leaf_id=bad_id)
        result = searcher.search("anything", top_k=5)
        # The leaf is demoted by the safety assessment; the warning
        # channel must use the redacted handle and must not echo the
        # raw secret-shaped id.
        assert any(("demoted" in w.lower() or "skipped" in w.lower())
                   for w in result.warnings)
        for w in result.warnings:
            assert bad_id not in w, f"raw secret-shaped id leaked in warning: {w!r}"
        # unsafe_leaf_ids emitted in the JSON envelope must be redacted
        # handles, never the raw secret-shaped id.
        d = result.to_dict()
        for handle in d["unsafe_leaf_ids"]:
            assert bad_id not in handle

    def test_unsafe_summary_id_warning_redacted(self):
        # The unsafe summary warning channel must use a redacted handle
        # for the raw secret-shaped node id.
        bad_id = "".join(["Bearer ", "q" * 24])
        # Build a parent summary whose payload is secret-bearing so it
        # is demoted by the contains_secret check; the warning channel
        # must use a redacted handle.
        dense_seeds = [
            _RetrievedMemory(
                "parent-clean",
                text="clean",
                payload=_raptor_parent_payload(
                    node_id="parent-clean",
                    children=[],
                    summary_of=[],
                ),
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-clean", dense_seeds[0].payload)
        # Inject a parent payload whose text contains a secret-shaped
        # token so the secret-bearing summary skip path fires.
        secret_payload = _raptor_parent_payload(
            node_id=bad_id, children=[], summary_of=[], secret=True,
        )
        # Add it as a directly-reachable parent via the retriever's
        # dense seed.
        dense_seeds.append(
            _RetrievedMemory(
                bad_id,
                text="secret parent",
                payload=secret_payload,
                final_score=0.7,
            )
        )
        qdrant.add_point(bad_id, secret_payload)
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(qdrant=qdrant, retriever=retriever, collection_name="memory")
        result = searcher.search("anything", top_k=5)
        # The warning channel must not echo the raw secret-shaped id.
        for w in result.warnings:
            assert bad_id not in w, f"raw secret-shaped id leaked in warning: {w!r}"
        # unsafe_summary_ids emitted in the JSON envelope must be
        # redacted handles.
        d = result.to_dict()
        for handle in d["unsafe_summary_ids"]:
            assert bad_id not in handle


# ---------------------------------------------------------------------------
# Adversarial: parent metadata redaction when include_metadata=true
# ---------------------------------------------------------------------------


class TestParentMetadataRedaction:
    def test_secret_bearing_derived_from_redacted(self):
        # derived_from[].source_uri carries a credential-shaped URL.
        secret_uri = "https://user:" + "p" * 24 + "@internal.example/x"
        summary = RaptorSummaryHit(
            point_id="parent-A",
            raptor_node_id="parent-A",
            raptor_root_id="root",
            raptor_level=2,
            raptor_tree_id="tree",
            raptor_build_id="build",
            raptor_cluster_id="cluster",
            text="ok",
            derived_from=[{"source_uri": secret_uri, "kind": "tool"}],
        )
        d = summary.to_dict(include_metadata=True)
        # The raw secret value must NOT appear in the serialised
        # payload, and the metadata_redacted sentinel must be set.
        serialized = json.dumps(d)
        assert secret_uri not in serialized
        assert d["parent_assessment"].get("metadata_redacted") is True
        assert d["derived_from"] == []
        assert d["extra"] == {}

    def test_secret_bearing_extra_redacted(self):
        secret_bearer = "".join(["Bearer ", "s" * 24])
        summary = RaptorSummaryHit(
            point_id="parent-A",
            raptor_node_id="parent-A",
            raptor_root_id="root",
            raptor_level=2,
            raptor_tree_id="tree",
            raptor_build_id="build",
            raptor_cluster_id="cluster",
            text="ok",
            extra={"internal_token": secret_bearer},
        )
        d = summary.to_dict(include_metadata=True)
        serialized = json.dumps(d)
        assert secret_bearer not in serialized
        assert d["parent_assessment"].get("metadata_redacted") is True
        assert d["derived_from"] == []
        assert d["extra"] == {}

    def test_clean_metadata_passes_through(self):
        summary = RaptorSummaryHit(
            point_id="parent-A",
            raptor_node_id="parent-A",
            raptor_root_id="root",
            raptor_level=2,
            raptor_tree_id="tree",
            raptor_build_id="build",
            raptor_cluster_id="cluster",
            text="ok",
            derived_from=[{"source_uri": "https://docs.example/x", "kind": "doc"}],
            extra={"notes": "clean"},
            parent_assessment={"parent_status": "active"},
        )
        d = summary.to_dict(include_metadata=True)
        assert d["derived_from"] == [{"source_uri": "https://docs.example/x", "kind": "doc"}]
        assert d["extra"] == {"notes": "clean"}
        assert d["parent_assessment"] == {"parent_status": "active"}

    def test_default_include_metadata_omits_metadata(self):
        summary = RaptorSummaryHit(
            point_id="parent-A",
            raptor_node_id="parent-A",
            raptor_root_id="root",
            raptor_level=2,
            raptor_tree_id="tree",
            raptor_build_id="build",
            raptor_cluster_id="cluster",
            text="ok",
            derived_from=[{"source_uri": "https://docs.example/x"}],
            extra={"a": "b"},
        )
        d = summary.to_dict(include_metadata=False)
        assert "derived_from" not in d
        assert "extra" not in d
        assert "parent_assessment" not in d


# ---------------------------------------------------------------------------
# Adversarial: parent attribution for disjoint leaves
# ---------------------------------------------------------------------------


class TestLeafParentAttribution:
    def test_two_parents_disjoint_leaves_each_maps_to_own_parent(self):
        dense_seeds = [
            _RetrievedMemory(
                "parent-A",
                text="parent A",
                payload=_raptor_parent_payload(
                    node_id="parent-A",
                    children=["leaf-A"],
                    summary_of=["leaf-A"],
                    level=2,
                ),
            ),
            _RetrievedMemory(
                "parent-B",
                text="parent B",
                payload=_raptor_parent_payload(
                    node_id="parent-B",
                    children=["leaf-B"],
                    summary_of=["leaf-B"],
                    level=2,
                ),
            ),
        ]
        qdrant = FakeRaptorQdrant()
        for seed in dense_seeds:
            qdrant.add_point(seed.id, seed.payload)
        qdrant.add_point("leaf-A", _leaf_payload(point_id="leaf-A", text="leaf A"))
        qdrant.add_point("leaf-B", _leaf_payload(point_id="leaf-B", text="leaf B"))
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(qdrant=qdrant, retriever=retriever, collection_name="memory")
        result = searcher.search("anything", top_k=5)
        leaves_by_id = {leaf.point_id: leaf for leaf in result.cited_leaves}
        assert "leaf-A" in leaves_by_id, "leaf-A must be promoted"
        assert "leaf-B" in leaves_by_id, "leaf-B must be promoted"
        # Each leaf must reference its OWN parent, not the first parent.
        assert leaves_by_id["leaf-A"].parent_raptor_node_id == "parent-A"
        assert leaves_by_id["leaf-B"].parent_raptor_node_id == "parent-B"
        assert leaves_by_id["leaf-A"].parent_point_id == "parent-A"
        assert leaves_by_id["leaf-B"].parent_point_id == "parent-B"

    def test_evidence_mode_keeps_only_parent_with_cited_leaves(self):
        dense_seeds = [
            _RetrievedMemory(
                "parent-A",
                text="parent A",
                payload=_raptor_parent_payload(
                    node_id="parent-A",
                    children=["leaf-A"],
                    summary_of=["leaf-A"],
                    level=2,
                ),
            ),
            _RetrievedMemory(
                "parent-B",
                text="parent B (no children)",
                payload=_raptor_parent_payload(
                    node_id="parent-B",
                    children=[],
                    summary_of=[],
                    level=2,
                ),
            ),
        ]
        qdrant = FakeRaptorQdrant()
        for seed in dense_seeds:
            qdrant.add_point(seed.id, seed.payload)
        qdrant.add_point("leaf-A", _leaf_payload(point_id="leaf-A", text="leaf A"))
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(qdrant=qdrant, retriever=retriever, collection_name="memory")
        # Route through the router so evidence-mode demotion runs.
        from qdrant_memory.hybrid import HybridRouter
        router = HybridRouter(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=FakeRetriever([]),
            raptor_searcher=searcher,
        )
        d = router.retrieve("anything", mode="evidence").to_dict()
        node_ids = {s["raptor_node_id"] for s in d["results"]["summaries"]}
        assert "parent-A" in node_ids
        assert "parent-B" not in node_ids

    def test_orphan_leaf_without_parent_mapping_is_dropped(self):
        # Build a searcher that returns a RaptorSearchResult containing
        # an orphan leaf with no parent attribution. The acceptance
        # criterion is: do NOT mis-attribute the orphan to the first
        # parent. Downstream consumers should be able to see the
        # blank-parent field and act on it without inheriting a wrong
        # parent. We exercise the post-search serializer and the
        # orphan-aware warning channel here.
        from qdrant_memory.raptor.search import (
            RaptorLeafHit,
            RaptorSearchResult,
            RaptorSummaryHit,
        )

        parent = RaptorSummaryHit(
            point_id="parent-A",
            raptor_node_id="parent-A",
            raptor_root_id="root",
            raptor_level=2,
            raptor_tree_id="tree",
            raptor_build_id="build",
            raptor_cluster_id="cluster",
            text="parent",
        )
        orphan = RaptorLeafHit(
            point_id="orphan-leaf",
            parent_raptor_node_id="",
            parent_point_id="",
            text="orphan content",
        )
        result = RaptorSearchResult(
            query="anything",
            summaries=[parent],
            cited_leaves=[orphan],
            warnings=[],
            debug={},
            unsafe_leaf_ids=set(),
        )
        # The serializer must surface the orphan with blank parent fields
        # so callers can detect the absence rather than guessing.
        d = result.to_dict(include_metadata=False)
        assert d["cited_leaves"][0]["point_id"] == "orphan-leaf"
        assert d["cited_leaves"][0]["parent_raptor_node_id"] == ""
        assert d["cited_leaves"][0]["parent_point_id"] == ""
        # The searcher's own emit path (which we just patched) drops
        # orphans with a warning; verify the new behaviour by exercising
        # the searcher in a scenario where the dense seed points to a
        # parent that does NOT declare the leaf in raptor_child_ids.
        # The leaf should be dropped and the warning should carry a
        # redacted handle.
        dense_seeds = [
            _RetrievedMemory(
                "parent-A",
                text="parent A",
                payload=_raptor_parent_payload(
                    node_id="parent-A",
                    children=[],
                    summary_of=[],
                    level=2,
                ),
            ),
            _RetrievedMemory(
                "orphan-seed",
                text="orphan leaf seed",
                payload={
                    "id": "orphan-seed",
                    "text": "orphan leaf seed",
                    "raptor_node_id": "orphan-seed",
                    "profile_id": "default",
                    "memory_kind": "memory",
                },
                final_score=0.4,
            ),
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-A", dense_seeds[0].payload)
        qdrant.add_point(
            "orphan-seed",
            _leaf_payload(point_id="orphan-seed", text="orphan leaf"),
        )
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(qdrant=qdrant, retriever=retriever, collection_name="memory")
        result2 = searcher.search("anything", top_k=5)
        # The orphan-seed leaf has no parent attribution; it must be
        # dropped from cited_leaves and surface as a warning with a
        # redacted handle, NOT a raw secret-shaped id.
        promoted = {l.point_id for l in result2.cited_leaves}
        assert "orphan-seed" not in promoted
        # Warning channel must not echo raw id; serialized envelope
        # must not contain the raw id either.
        serialized = json.dumps(result2.to_dict(), default=str)
        assert "orphan-seed" not in serialized


# ---------------------------------------------------------------------------
# Phase 5 fix3 — adversarial: RAPTOR summary promotion must not leak
# secret-shaped parent IDs / source fields. Fix2 covered unsafe-leaf
# demotion and ``include_metadata=true`` metadata redaction, but the
# dense-seed promotion path still emitted raw ``raptor_node_id`` and
# ``source_hashes`` even when those carried a secret, and one warning
# path interpolated the raw node id.
# ---------------------------------------------------------------------------


class TestRaptorSummaryCoreFieldRedaction:
    def test_no_child_warning_redacted(self):
        """The no-child downgrade warning must use a redacted handle."""
        dense_seeds = [
            _RetrievedMemory(
                "parent-clean",
                text="clean",
                payload=_raptor_parent_payload(
                    node_id="parent-clean",
                    children=[],
                    summary_of=[],
                ),
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-clean", dense_seeds[0].payload)
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(qdrant=qdrant, retriever=retriever, collection_name="memory")
        result = searcher.search("anything", top_k=5)
        # The no-child downgrade warning must be present.
        no_child_warnings = [
            w for w in result.warnings if "no child IDs" in w
        ]
        assert no_child_warnings, "expected a no-child downgrade warning"
        for w in no_child_warnings:
            # The raw node id MUST NOT appear in the warning.
            assert "parent-clean" not in w
            # The redacted handle MUST appear.
            assert "redacted:" in w

    def test_no_child_warning_redacted_for_secret_shaped_node_id(self):
        """A secret-shaped ``raptor_node_id`` must not leak through the warning."""
        bad_id = "".join(["Bearer ", "u" * 24])
        dense_seeds = [
            _RetrievedMemory(
                "parent-clean",
                text="clean",
                payload=_raptor_parent_payload(
                    node_id="parent-clean",
                    children=[],
                    summary_of=[],
                ),
            )
        ]
        # Inject a parent payload whose ``raptor_node_id`` is secret-shaped.
        # The text itself is clean so the existing text-only secret
        # check does not fire — only the new core-field check should
        # catch this.
        secret_payload = _raptor_parent_payload(
            node_id=bad_id,
            text="clean text",
            children=[],
            summary_of=[],
        )
        # ``add_point`` keys the fake store by ``id`` so we use the
        # same key as the node id.
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-clean", dense_seeds[0].payload)
        qdrant.add_point(bad_id, secret_payload)
        # Also add the parent-clean seed pointing at the bad node so
        # the searcher walks to the secret-shaped parent.
        dense_seeds.append(
            _RetrievedMemory(
                bad_id,
                text="clean",
                payload=secret_payload,
                final_score=0.7,
            )
        )
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(qdrant=qdrant, retriever=retriever, collection_name="memory")
        result = searcher.search("anything", top_k=5)
        # The raw secret-shaped id MUST NOT appear anywhere in
        # warnings, debug, or the serialized envelope.
        serialized = json.dumps(result.to_dict(), default=str)
        assert bad_id not in serialized
        for w in result.warnings:
            assert bad_id not in w, f"raw secret-shaped id leaked in warning: {w!r}"
        # unsafe_summary_ids must use the redacted handle.
        d = result.to_dict()
        for handle in d["unsafe_summary_ids"]:
            assert bad_id not in handle

    def test_secret_shaped_node_id_dropped_from_summaries(self):
        """A secret-shaped ``raptor_node_id`` must not be promoted."""
        bad_id = "".join(["Bearer ", "v" * 24])
        # Build a parent whose only secret-bearing field is its node id
        # (the text is clean). The pre-promotion core-field check
        # must demote it to warning-only.
        payload = _raptor_parent_payload(
            node_id=bad_id,
            text="clean text",
            children=[],
            summary_of=[],
        )
        dense_seeds = [
            _RetrievedMemory(
                bad_id,
                text="clean",
                payload=payload,
                final_score=0.7,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point(bad_id, payload)
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(qdrant=qdrant, retriever=retriever, collection_name="memory")
        result = searcher.search("anything", top_k=5)
        # The summary must NOT be promoted.
        promoted_node_ids = {s.raptor_node_id for s in result.summaries}
        assert bad_id not in promoted_node_ids
        # The warning channel must reference the redacted handle.
        skip_warnings = [w for w in result.warnings if "secret-bearing core field" in w]
        assert skip_warnings, "expected a secret-bearing core-field skip warning"
        for w in skip_warnings:
            assert bad_id not in w
            assert "redacted:" in w
        serialized = json.dumps(result.to_dict(), default=str)
        assert bad_id not in serialized

    def test_secret_shaped_source_hash_dropped(self):
        """A credential-shaped entry in ``source_hashes`` must not be promoted."""
        # Use a credential URI that contains a basic-auth user/pass
        # rather than a hex digest. Existing tests still pass with
        # ``"a"*64`` because hex-only strings never trigger
        # ``contains_secret``.
        bad_hash = "https://user:" + ("w" * 24) + "@internal.example/x"
        payload = _raptor_parent_payload(
            node_id="parent-A",
            text="clean text",
            children=[],
            summary_of=[],
        )
        payload["source_hashes"] = [bad_hash]
        dense_seeds = [
            _RetrievedMemory(
                "parent-A",
                text="clean",
                payload=payload,
                final_score=0.7,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-A", payload)
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(qdrant=qdrant, retriever=retriever, collection_name="memory")
        result = searcher.search("anything", top_k=5)
        promoted = {s.raptor_node_id for s in result.summaries}
        # The summary must NOT be promoted (the credential URI in
        # source_hashes trips the core-field scan).
        assert "parent-A" not in promoted
        # The raw secret URI must NOT appear anywhere.
        serialized = json.dumps(result.to_dict(), default=str)
        assert bad_hash not in serialized

    def test_clean_source_hashes_pass_through(self):
        """Hex-only ``source_hashes`` (the canonical digest format) must pass."""
        payload = _raptor_parent_payload(
            node_id="parent-clean",
            text="clean text",
            children=[],
            summary_of=[],
        )
        # The helper already sets source_hashes=["a"*64]; the test
        # simply asserts that this still promotes cleanly.
        dense_seeds = [
            _RetrievedMemory(
                "parent-clean",
                text="clean",
                payload=payload,
                final_score=0.7,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-clean", payload)
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(qdrant=qdrant, retriever=retriever, collection_name="memory")
        result = searcher.search("anything", top_k=5)
        promoted = {s.raptor_node_id for s in result.summaries}
        assert "parent-clean" in promoted
        # The clean digest value must be present in the serialized
        # summary.
        d = result.to_dict()
        parent_summary = next(s for s in d["summaries"] if s["raptor_node_id"] == "parent-clean")
        assert "a" * 64 in parent_summary["source_hashes"]


# ---------------------------------------------------------------------------
# Phase 5 fix3 — adversarial: parent_summary helper unit tests for the
# new pre-promotion core-field scan.
# ---------------------------------------------------------------------------


class TestSummaryCoreFieldHelper:
    def test_clean_payload_passes(self):
        from qdrant_memory.raptor.search import _summary_default_emitted_secret_bearing

        payload = _raptor_parent_payload(
            node_id="parent-clean",
            text="clean",
            children=["leaf-1"],
            summary_of=["leaf-1"],
        )
        point = {"id": "parent-clean", "payload": payload}
        assert _summary_default_emitted_secret_bearing(point, payload) is False

    def test_secret_bearing_node_id_caught(self):
        from qdrant_memory.raptor.search import _summary_default_emitted_secret_bearing

        bad_id = "".join(["Bearer ", "x" * 24])
        payload = _raptor_parent_payload(
            node_id=bad_id,
            text="clean",
            children=[],
            summary_of=[],
        )
        point = {"id": bad_id, "payload": payload}
        assert _summary_default_emitted_secret_bearing(point, payload) is True

    def test_secret_bearing_source_hash_caught(self):
        from qdrant_memory.raptor.search import _summary_default_emitted_secret_bearing

        payload = _raptor_parent_payload(
            node_id="parent-clean",
            text="clean",
            children=[],
            summary_of=[],
        )
        payload["source_hashes"] = ["https://user:" + ("y" * 24) + "@internal.example/z"]
        point = {"id": "parent-clean", "payload": payload}
        assert _summary_default_emitted_secret_bearing(point, payload) is True

    def test_hex_source_hash_not_caught(self):
        """Regression: canonical sha256-hex ``source_hashes`` must not be flagged."""
        from qdrant_memory.raptor.search import _summary_default_emitted_secret_bearing

        payload = _raptor_parent_payload(
            node_id="parent-clean",
            text="clean",
            children=[],
            summary_of=[],
        )
        # _raptor_parent_payload already sets source_hashes=["a"*64].
        point = {"id": "parent-clean", "payload": payload}
        assert _summary_default_emitted_secret_bearing(point, payload) is False


# ---------------------------------------------------------------------------
# Phase 5 fix5 — RAPTOR seed search must NOT invoke ``scroll_by_filter``.
# ---------------------------------------------------------------------------


class TestRaptorSeedSearchDisablesSparseScroll:
    """Regression for finding #2 from the Phase 5 review.

    Phase 5 fix4 wired ``allow_sparse_scroll=False`` into the
    ``HybridRouter.retrieve`` direct dense lane. The RAPTOR seed search
    also calls ``MemoryRetriever.search`` but, until fix5, did NOT
    propagate that flag. Because ``MemoryRetriever.search`` defaults
    ``allow_sparse_scroll`` to ``True``, a strong-signal retrieve
    query (UUID, issue id, route path) could still hit
    ``scroll_by_filter`` through the RAPTOR seed search path, violating
    the read-only invariant.

    These tests assert that:

    * The ``RaptorSearcher.search`` call propagates
      ``allow_sparse_scroll=False`` and ``update_access=False`` to the
      underlying retriever.
    * Wiring the real ``MemoryRetriever`` + a strong-signal UUID query
      through ``HybridRouter`` + ``RaptorSearcher`` does NOT invoke
      ``scroll_by_filter``.
    * The RAPTOR seed search FAILS CLOSED with a warning when the
      underlying retriever does not accept the kwarg (instead of
      silently falling back to the legacy behaviour that would still
      call ``scroll_by_filter``).
    """

    def test_raptor_seed_search_passes_allow_sparse_scroll_false(self):
        # The RAPTOR seed search must always pass
        # ``allow_sparse_scroll=False`` to the underlying retriever
        # so the sparse / scroll-by-filter lane is bypassed.
        dense_seeds = [
            _RetrievedMemory(
                "parent-A",
                text="parent summary text",
                payload=_raptor_parent_payload(
                    node_id="parent-A",
                    children=["leaf-1"],
                    summary_of=["leaf-1"],
                    level=2,
                ),
                final_score=0.91,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-A", dense_seeds[0].payload)
        qdrant.add_point("leaf-1", _leaf_payload(point_id="leaf-1", text="leaf text"))
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        searcher.search("deploy", top_k=3)
        assert retriever.calls
        for call in retriever.calls:
            assert call["kwargs"].get("allow_sparse_scroll") is False
            assert call["update_access"] is False

    def test_raptor_seed_search_keeps_update_access_false(self):
        # The RAPTOR seed search MUST keep ``update_access=False`` so
        # the hybrid retrieve path never bumps access metadata.
        dense_seeds = [
            _RetrievedMemory(
                "parent-A",
                text="parent summary",
                payload=_raptor_parent_payload(
                    node_id="parent-A", children=[], level=2,
                ),
                final_score=0.5,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-A", dense_seeds[0].payload)
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        searcher.search("deploy", top_k=3)
        for call in retriever.calls:
            assert call["update_access"] is False

    def test_hybrid_router_plus_raptor_with_real_memory_retriever_never_scrolls(
        self,
    ):
        # End-to-end wiring test: ``HybridRouter`` + ``RaptorSearcher``
        # + real ``MemoryRetriever`` with a strong-signal UUID query
        # must NOT call ``scroll_by_filter`` on the underlying
        # ``QdrantClient``. The fake ``qdrant`` raises
        # ``AssertionError`` if ``scroll_by_filter`` is invoked, so
        # reaching the assertions below proves the read-only
        # invariant holds for the full HybridRouter -> RaptorSearcher
        # -> MemoryRetriever pipeline.
        from qdrant_memory.retriever import MemoryRetriever

        class _StrictQdrant:
            def __init__(self):
                self.scroll_calls: list = []

            def search(self, name, vector, limit, filter=None, with_payload=True, with_vector=False):
                return [{
                    "id": "parent-real",
                    "score": 0.9,
                    "payload": {
                        "text": "parent summary",
                        "source_type": "manual",
                        "profile_id": "default",
                        "importance": 5,
                        "created_at": "2026-01-15T00:00:00+00:00",
                        "raptor_node_id": "parent-real",
                        "raptor_level": 2,
                        "raptor_child_ids": ["leaf-real"],
                        "raptor_summary_of": ["leaf-real"],
                        "raptor_tree_id": "tree-real",
                        "raptor_root_id": "root-real",
                        "raptor_build_id": "build-real",
                        "raptor_cluster_id": "cluster-real",
                        "derivation_type": "raptor_summary",
                        "memory_kind": "summary",
                        "canonical": False,
                        "requires_review": True,
                    },
                }]

            def retrieve(self, name, ids, *, with_payload=True, with_vector=False):
                out = []
                for pid in ids:
                    if pid == "parent-real":
                        out.append({
                            "id": "parent-real",
                            "payload": self.search("memory", [0.1, 0.2], 1)[0]["payload"],
                        })
                    elif pid == "leaf-real":
                        out.append({
                            "id": "leaf-real",
                            "payload": _leaf_payload(point_id="leaf-real", text="real leaf"),
                        })
                return out

            def scroll_by_filter(self, *args, **kwargs):
                # Sentinel: must NEVER be called from the hybrid
                # retrieve path even when the RAPTOR lane is wired
                # in. Recording lets the test report the exact call
                # if the assertion fires.
                self.scroll_calls.append({"args": args, "kwargs": kwargs})
                raise AssertionError(
                    "scroll_by_filter must not be invoked from the "
                    "Phase 5 HybridRouter + RaptorSearcher path"
                )

            def update_payload(self, *args, **kwargs):
                raise AssertionError(
                    "update_payload must not be invoked from the "
                    "Phase 5 read-only retrieve path"
                )

        qdrant = _StrictQdrant()
        retriever = MemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
            search_candidates=3,
        )
        searcher = RaptorSearcher(
            qdrant=qdrant,
            retriever=retriever,
            collection_name="memory",
        )
        # ``HybridRouter`` only needs a stub qdrant because the
        # raptor lane calls ``qdrant.retrieve`` directly via the
        # searcher's own qdrant reference.
        from qdrant_memory.hybrid import HybridRouter

        class _RouterQdrant:
            pass

        router = HybridRouter(
            qdrant=_RouterQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=retriever,
            raptor_searcher=searcher,
        )

        # Strong-signal query — would normally trigger the sparse
        # lane in MemoryRetriever.search. The fix MUST suppress it.
        d = router.retrieve(
            "550e8400-e29b-41d4-a716-446655440000", top_k=3
        ).to_dict()

        # No scroll_by_filter calls anywhere in the hybrid retrieve
        # pipeline (neither the dense lane nor the RAPTOR seed
        # search).
        assert qdrant.scroll_calls == []
        # No access-metadata mutation.
        # (update_payload is denied above; reaching this line is the
        # proof that no mutation happened.)
        # The seed search MUST have been called with
        # ``allow_sparse_scroll=False``.
        # (MemoryRetriever does not record kwargs on the fake; the
        # scroll_calls==[] assertion above is the strict signal.)

    def test_raptor_seed_search_fails_closed_when_retriever_lacks_kwarg(self):
        # If a custom fake retriever does NOT accept the
        # ``allow_sparse_scroll`` kwarg, the RAPTOR seed search MUST
        # fail closed (empty seeds + warning) rather than silently
        # retrying without the flag and re-enabling the
        # scroll_by-filter lane.
        class _LegacyRetriever:
            """Custom retriever that does not accept the strict kwarg."""

            def __init__(self):
                self.calls: list[dict[str, Any]] = []
                self._seeds = [
                    _RetrievedMemory(
                        "parent-A",
                        text="parent summary",
                        payload=_raptor_parent_payload(
                            node_id="parent-A", children=[], level=2,
                        ),
                        final_score=0.5,
                    )
                ]

            def search(self, query, *, top_k=5, update_access=True, **kwargs):
                # Strict reject: this retriever represents an
                # older MemoryRetriever-like API that predates the
                # ``allow_sparse_scroll`` kwarg. Raise TypeError so
                # ``RaptorSearcher.search`` fails closed instead of
                # silently retrying without the kwarg.
                if "allow_sparse_scroll" in kwargs:
                    raise TypeError(
                        "search() got an unexpected keyword argument "
                        "'allow_sparse_scroll'"
                    )
                self.calls.append({
                    "query": query,
                    "top_k": top_k,
                    "update_access": update_access,
                    "kwargs": kwargs,
                })
                return list(self._seeds)

        qdrant = FakeRaptorQdrant()
        retriever = _LegacyRetriever()
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("deploy", top_k=3)
        # No summaries promoted because the strict kwarg path failed
        # closed — no seeds made it through.
        assert result.summaries == []
        # Warning channel surfaces the failure-closed reason.
        assert any(
            "allow_sparse_scroll" in w and "failing closed" in w
            for w in result.warnings
        )


# ---------------------------------------------------------------------------
# Phase 5 fix6: parent recomputation after child safety demotion
#
# Regression coverage for finding #3 from the final4 reviewer/security
# pass. After ``assess_leaf_safety()`` demotes stale / quarantined /
# review-required / secret-bearing leaves, the parent summary
# ``parent_status`` MUST be recomputed via
# ``assess_parent_status(child_payloads=...)`` instead of remaining
# the original ``active``. Safe-only children still allow active
# parents; mixed unsafe children demote the parent; all-unsafe
# children drop the active parent text entirely.
# ---------------------------------------------------------------------------


class TestParentStatusPostChildSafety:
    def test_mixed_safe_and_stale_children_demotes_parent(self):
        # Parent with two children: one safe, one stale. The safe path
        # classifies both leaves correctly (stale leaf demoted to
        # warning) and the parent MUST be downgraded so the unsafe
        # child evidence cannot promote an unsafe parent summary.
        dense_seeds = [
            _RetrievedMemory(
                "parent-mixed",
                text="parent summary text",
                payload=_raptor_parent_payload(
                    node_id="parent-mixed",
                    children=["leaf-safe", "leaf-stale"],
                    summary_of=["leaf-safe", "leaf-stale"],
                    level=2,
                ),
                final_score=0.91,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-mixed", dense_seeds[0].payload)
        qdrant.add_point(
            "leaf-safe",
            _leaf_payload(point_id="leaf-safe", text="safe leaf content"),
        )
        qdrant.add_point(
            "leaf-stale",
            _leaf_payload(
                point_id="leaf-stale", text="stale leaf content",
                stale=True,
            ),
        )
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3)

        # Find the parent-mixed summary, regardless of where it ended
        # up after the demotion.
        target = next(
            (s for s in result.summaries if s.raptor_node_id == "parent-mixed"),
            None,
        )
        assert target is not None, "parent-mixed should still be tracked"
        # Parent status MUST reflect the post-demotion child review
        # (any unsafe child makes the parent non-active).
        assert target.parent_status != "active"
        # Summary text MUST be cleared so the demoted parent never
        # reaches the caller as active context.
        assert target.text == ""
        # Warning channel MUST surface the demotion with a redacted
        # handle.
        assert any(
            "demoted after child-safety review" in w
            and "parent-mixed" not in w
            for w in result.warnings
        )
        # unsafe_summary_ids MUST carry the parent (via the result
        # envelope) so the calling router can keep it out of the
        # active summaries list.
        assert "parent-mixed" in result.unsafe_summary_ids

    def test_all_unsafe_children_drop_active_parent_text(self):
        # Parent whose children are ALL unsafe (review-required +
        # quarantined mix). The parent summary MUST have zero text
        # (never promoted) and the parent_status MUST be non-active.
        dense_seeds = [
            _RetrievedMemory(
                "parent-all-unsafe",
                text="parent summary text",
                payload=_raptor_parent_payload(
                    node_id="parent-all-unsafe",
                    children=["leaf-review", "leaf-q"],
                    summary_of=["leaf-review", "leaf-q"],
                    level=2,
                ),
                final_score=0.91,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-all-unsafe", dense_seeds[0].payload)
        qdrant.add_point(
            "leaf-review",
            _leaf_payload(
                point_id="leaf-review", text="needs review",
                requires_review=True,
            ),
        )
        qdrant.add_point(
            "leaf-q",
            _leaf_payload(
                point_id="leaf-q", text="quarantined",
                quarantined=True,
            ),
        )
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3)
        target = next(
            (s for s in result.summaries if s.raptor_node_id == "parent-all-unsafe"),
            None,
        )
        assert target is not None
        assert target.parent_status != "active"
        assert target.text == ""
        # No leaf should have been promoted because every child was
        # unsafe; parents with zero cited leaves survive only as a
        # demoted warning entry.
        promoted_leaf_ids = {l.point_id for l in result.cited_leaves}
        assert "leaf-review" not in promoted_leaf_ids
        assert "leaf-q" not in promoted_leaf_ids
        assert "parent-all-unsafe" in result.unsafe_summary_ids

    def test_safe_only_children_keep_parent_active(self):
        # All children are clean → the parent MUST remain active and
        # its text MUST survive untouched so the safe path is not
        # over-cautious.
        original_text = "important parent summary that must survive"
        dense_seeds = [
            _RetrievedMemory(
                "parent-safe",
                text="parent summary text",
                payload=_raptor_parent_payload(
                    node_id="parent-safe",
                    text=original_text,
                    children=["leaf-ok-1", "leaf-ok-2"],
                    summary_of=["leaf-ok-1", "leaf-ok-2"],
                    level=2,
                ),
                final_score=0.92,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-safe", dense_seeds[0].payload)
        qdrant.add_point(
            "leaf-ok-1",
            _leaf_payload(point_id="leaf-ok-1", text="clean 1"),
        )
        qdrant.add_point(
            "leaf-ok-2",
            _leaf_payload(point_id="leaf-ok-2", text="clean 2"),
        )
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3)
        target = next(
            (s for s in result.summaries if s.raptor_node_id == "parent-safe"),
            None,
        )
        assert target is not None
        assert target.parent_status == "active"
        assert target.text == original_text
        # No spurious demotion warning for clean children.
        assert not any(
            "demoted after child-safety review" in w
            for w in result.warnings
        )
        assert "parent-safe" not in result.unsafe_summary_ids


# ---------------------------------------------------------------------------
# Phase 5 fix12 (final10 P2): parent trust gate at construction time.
#
# RAPTOR parents are emitted by the schema/builder with
# ``requires_review=True`` and ``raptor_review_status="review_required"``
# by default. The old code path only honored child-side safety, so a
# parent with a clean child population but trust-flagged payload
# remained ``active`` and emitted raw text. The trust gate runs
# BEFORE the hit is appended to ``summaries`` so a flagged parent
# never reaches the caller as active context.
# ---------------------------------------------------------------------------


class TestParentTrustGateAtConstruction:
    def test_requires_review_parent_with_clean_child_demoted(self):
        # Parent payload carries ``requires_review=True`` but every
        # child is clean. The trust gate must still demote the
        # parent — child safety and parent trust are independent
        # signals.
        dense_seeds = [
            _RetrievedMemory(
                "parent-rr",
                text="parent summary text",
                payload=_raptor_parent_payload(
                    node_id="parent-rr",
                    children=["leaf-ok"],
                    summary_of=["leaf-ok"],
                    level=2,
                    requires_review=True,
                ),
                final_score=0.91,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-rr", dense_seeds[0].payload)
        qdrant.add_point(
            "leaf-ok", _leaf_payload(point_id="leaf-ok", text="clean leaf"),
        )
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3)

        target = next(
            (s for s in result.summaries if s.raptor_node_id == "parent-rr"),
            None,
        )
        assert target is not None, "parent-rr must still be tracked"
        # Trust gate: parent is non-active AND its text is cleared
        # so the caller never sees raw active context from an
        # unsafe-by-trust parent.
        assert target.parent_status == "review_required"
        assert target.text == ""
        # The trust reasons must surface in parent_assessment so
        # an operator can audit *why* the parent was demoted.
        reasons = target.parent_assessment.get("trust_gate_reasons") or []
        assert "requires_review" in reasons
        # The parent MUST be tracked in ``unsafe_summary_ids``.
        assert "parent-rr" in result.unsafe_summary_ids

    def test_raptor_review_status_review_required_parent_demoted(self):
        # Parent payload carries the canonical
        # ``raptor_review_status="review_required"`` marker (set by
        # the RAPTOR schema). The trust gate must demote even when
        # the boolean ``requires_review`` field is False (some
        # payloads use only the string marker).
        dense_seeds = [
            _RetrievedMemory(
                "parent-rstatus",
                text="parent summary text",
                payload=_raptor_parent_payload(
                    node_id="parent-rstatus",
                    children=["leaf-ok"],
                    summary_of=["leaf-ok"],
                    level=2,
                    requires_review=False,
                    raptor_review_status="review_required",
                ),
                final_score=0.91,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-rstatus", dense_seeds[0].payload)
        qdrant.add_point(
            "leaf-ok", _leaf_payload(point_id="leaf-ok", text="clean leaf"),
        )
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3)

        target = next(
            (s for s in result.summaries
             if s.raptor_node_id == "parent-rstatus"),
            None,
        )
        assert target is not None
        assert target.parent_status == "review_required"
        assert target.text == ""
        reasons = target.parent_assessment.get("trust_gate_reasons") or []
        assert any(r.startswith("raptor_review_status:") for r in reasons)
        assert "parent-rstatus" in result.unsafe_summary_ids

    def test_approved_parent_with_clean_child_remains_active(self):
        # Negative control: an approved parent (``requires_review=False``,
        # ``raptor_review_status="approved"``, default clean flags)
        # with all-clean children MUST stay active and keep its
        # original text. The trust gate must NOT over-trigger on
        # the safe path.
        original_text = "approved summary must survive intact"
        dense_seeds = [
            _RetrievedMemory(
                "parent-approved",
                text="parent summary text",
                payload=_raptor_parent_payload(
                    node_id="parent-approved",
                    children=["leaf-ok-1", "leaf-ok-2"],
                    summary_of=["leaf-ok-1", "leaf-ok-2"],
                    level=2,
                    text=original_text,
                ),
                final_score=0.95,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-approved", dense_seeds[0].payload)
        qdrant.add_point(
            "leaf-ok-1", _leaf_payload(point_id="leaf-ok-1", text="clean 1"),
        )
        qdrant.add_point(
            "leaf-ok-2", _leaf_payload(point_id="leaf-ok-2", text="clean 2"),
        )
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3)
        target = next(
            (s for s in result.summaries
             if s.raptor_node_id == "parent-approved"),
            None,
        )
        assert target is not None
        assert target.parent_status == "active"
        assert target.text == original_text
        # No trust-gate warning was emitted for the approved parent.
        assert not any(
            "parent trust gate" in w for w in result.warnings
        ), f"unexpected trust-gate warning: {result.warnings!r}"
        assert "parent-approved" not in result.unsafe_summary_ids
        # Both cited leaves SHOULD be promoted because the parent
        # survived and the children are clean.
        promoted_leaf_ids = {l.point_id for l in result.cited_leaves}
        assert {"leaf-ok-1", "leaf-ok-2"} <= promoted_leaf_ids

    def test_secret_shaped_parent_id_not_leaked_in_warnings_or_json(self):
        # A trust-flagged parent whose ``raptor_node_id`` is
        # secret-shaped must NOT echo the raw id through the
        # warning channel or the serialized JSON envelope. The
        # parent is demoted, but only its redacted handle appears
        # anywhere the caller can read.
        bad_id = "".join(["Bearer ", "w" * 24])
        dense_seeds = [
            _RetrievedMemory(
                bad_id,
                text="parent summary text",
                payload=_raptor_parent_payload(
                    node_id=bad_id,
                    children=["leaf-ok"],
                    summary_of=["leaf-ok"],
                    level=2,
                    requires_review=True,
                ),
                final_score=0.92,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point(bad_id, dense_seeds[0].payload)
        qdrant.add_point(
            "leaf-ok", _leaf_payload(point_id="leaf-ok", text="clean leaf"),
        )
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3)

        # The raw secret-shaped id must not appear in any warning.
        for w in result.warnings:
            assert bad_id not in w, (
                f"raw secret-shaped parent id leaked in warning: {w!r}"
            )
        # The serialized JSON envelope must also refuse to leak
        # the raw id via ``unsafe_summary_ids`` or
        # ``summaries[*].raptor_node_id`` — the redacted handle
        # only.
        d = result.to_dict()
        for handle in d["unsafe_summary_ids"]:
            assert bad_id not in handle
        serialized = json.dumps(d)
        assert bad_id not in serialized, (
            "raw secret-shaped parent id leaked in serialized "
            "RaptorSearchResult.to_dict() output"
        )


# ---------------------------------------------------------------------------
# Phase 5 fix13 (final11 P2 regression): trust-gated parents still enqueue
# their children.
#
# The trust gate at promotion time (fix12) demotes a parent whose payload
# itself carries ``requires_review=True`` / ``raptor_review_status=
# "review_required"`` / ``stale=True`` / ``consolidation_quarantined`` /
# ``raptor_excluded`` / ``raptor_forgotten`` / unsafe ``fact_status`` so
# its raw text never reaches the caller as active context. But the
# parent's *referenced* children are independently source-backed
# evidence: a clean child must remain retrievable / citable for
# evidence-mode traces and downstream prompts. The fix13 contract is
# that the trust-gated parent still flows through the same cap-bounded
# child enqueue / accounting block as an active parent, so a clean
# child leaf is added to ``cited_leaves`` (and survives the demoted
# parent text/status).
#
# The complementary rule (unchanged) is that an UNSAFE child (secret-
# bearing, requires_review, quarantined, stale, etc.) must NOT become
# active, and the parent's own text/status must still be demoted.
# ---------------------------------------------------------------------------


class TestTrustGateStillEnqueuesChildren:
    def test_review_required_parent_cites_clean_child(self):
        # review_required=True parent with a clean child. Parent
        # must be demoted (text "", parent_status="review_required")
        # AND the clean child must appear in cited_leaves so
        # evidence-mode traces and downstream prompts can still
        # surface the source-backed evidence.
        dense_seeds = [
            _RetrievedMemory(
                "parent-rr",
                text="parent summary text",
                payload=_raptor_parent_payload(
                    node_id="parent-rr",
                    children=["leaf-ok"],
                    summary_of=["leaf-ok"],
                    level=2,
                    requires_review=True,
                ),
                final_score=0.91,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-rr", dense_seeds[0].payload)
        qdrant.add_point(
            "leaf-ok", _leaf_payload(point_id="leaf-ok", text="clean leaf")
        )
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3)

        target = next(
            (s for s in result.summaries if s.raptor_node_id == "parent-rr"),
            None,
        )
        assert target is not None
        # Trust gate: parent stays demoted.
        assert target.parent_status == "review_required"
        assert target.text == ""
        reasons = target.parent_assessment.get("trust_gate_reasons") or []
        assert "requires_review" in reasons
        assert "parent-rr" in result.unsafe_summary_ids
        # Fix13: the clean child is now actually retrievable / citable.
        promoted_leaf_ids = {l.point_id for l in result.cited_leaves}
        assert "leaf-ok" in promoted_leaf_ids
        # The clean child is NOT tracked as unsafe.
        assert "leaf-ok" not in result.unsafe_leaf_ids

    def test_raptor_review_status_review_required_parent_cites_clean_child(self):
        # The canonical Phase 3 marker
        # ``raptor_review_status="review_required"`` (set by
        # the RAPTOR schema) must also preserve clean child
        # citation while keeping the parent demoted.
        dense_seeds = [
            _RetrievedMemory(
                "parent-rstatus",
                text="parent summary text",
                payload=_raptor_parent_payload(
                    node_id="parent-rstatus",
                    children=["leaf-ok"],
                    summary_of=["leaf-ok"],
                    level=2,
                    requires_review=False,
                    raptor_review_status="review_required",
                ),
                final_score=0.91,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-rstatus", dense_seeds[0].payload)
        qdrant.add_point(
            "leaf-ok", _leaf_payload(point_id="leaf-ok", text="clean leaf")
        )
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3)

        target = next(
            (s for s in result.summaries
             if s.raptor_node_id == "parent-rstatus"),
            None,
        )
        assert target is not None
        assert target.parent_status == "review_required"
        assert target.text == ""
        reasons = target.parent_assessment.get("trust_gate_reasons") or []
        assert any(r.startswith("raptor_review_status:") for r in reasons)
        # Clean child still surfaced.
        assert "leaf-ok" in {l.point_id for l in result.cited_leaves}

    def test_unsafe_child_of_trust_gated_parent_still_dropped(self):
        # An UNSAFE child of a trust-gated parent must NOT become
        # active. The parent's demotion is independent of the
        # child's safety, and the unsafe child must still be
        # demoted to a warning-only entry.
        dense_seeds = [
            _RetrievedMemory(
                "parent-rr",
                text="parent summary text",
                payload=_raptor_parent_payload(
                    node_id="parent-rr",
                    children=["leaf-unsafe", "leaf-ok"],
                    summary_of=["leaf-unsafe", "leaf-ok"],
                    level=2,
                    requires_review=True,
                ),
                final_score=0.91,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-rr", dense_seeds[0].payload)
        qdrant.add_point(
            "leaf-unsafe",
            _leaf_payload(point_id="leaf-unsafe", text="quarantined leaf", quarantined=True),
        )
        qdrant.add_point(
            "leaf-ok", _leaf_payload(point_id="leaf-ok", text="clean leaf"),
        )
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3)

        # Parent still demoted.
        target = next(
            (s for s in result.summaries if s.raptor_node_id == "parent-rr"),
            None,
        )
        assert target is not None
        assert target.parent_status == "review_required"
        assert target.text == ""
        # The unsafe child is NOT cited; the clean child IS cited.
        promoted_leaf_ids = {l.point_id for l in result.cited_leaves}
        assert "leaf-unsafe" not in promoted_leaf_ids
        assert "leaf-ok" in promoted_leaf_ids
        assert "leaf-unsafe" in result.unsafe_leaf_ids
        assert "leaf-ok" not in result.unsafe_leaf_ids

    def test_secret_bearing_child_of_trust_gated_parent_not_cited(self):
        # A secret-bearing child of a trust-gated parent must NOT
        # be promoted and must not leak through the warning
        # channel. The parent is demoted for trust reasons AND
        # the secret-bearing child is dropped.
        dense_seeds = [
            _RetrievedMemory(
                "parent-rr",
                text="parent summary text",
                payload=_raptor_parent_payload(
                    node_id="parent-rr",
                    children=["leaf-ok", "leaf-secret"],
                    summary_of=["leaf-ok", "leaf-secret"],
                    level=2,
                    requires_review=True,
                ),
                final_score=0.91,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-rr", dense_seeds[0].payload)
        qdrant.add_point(
            "leaf-ok", _leaf_payload(point_id="leaf-ok", text="clean leaf"),
        )
        qdrant.add_point(
            "leaf-secret", _leaf_payload(point_id="leaf-secret", text="placeholder", secret=True),
        )
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3)

        # Parent demoted; clean child still cited.
        target = next(
            (s for s in result.summaries if s.raptor_node_id == "parent-rr"),
            None,
        )
        assert target is not None
        assert target.parent_status == "review_required"
        assert target.text == ""
        promoted_leaf_ids = {l.point_id for l in result.cited_leaves}
        assert "leaf-ok" in promoted_leaf_ids
        assert "leaf-secret" not in promoted_leaf_ids
        assert "leaf-secret" in result.unsafe_leaf_ids
        # The serialized envelope must not echo the raw
        # secret-bearing child text. The fake uses a bearer-
        # shaped token built from ``"f" * 24`` in
        # ``_leaf_payload(..., secret=True)``; it must not
        # appear in the cited_leaves payload projection or
        # warnings. The redacted sentinel replaces the text
        # via ``_redact_leaf_text`` in the searcher.
        d = result.to_dict()
        for entry in d["cited_leaves"]:
            assert entry.get("point_id") != "leaf-secret", (
                "secret-bearing child must not be projected into "
                "cited_leaves at all"
            )
        serialized = json.dumps(d)
        # The raw 24-character repeated-char token used to
        # build the fake bearer-shaped child text must not
        # appear in the serialized envelope.
        assert ("f" * 24) not in serialized, (
            "raw secret-bearing child text leaked through "
            "RaptorSearchResult.to_dict()"
        )
        # And no warning echoes the raw token either.
        for w in result.warnings:
            assert ("f" * 24) not in w, (
                f"raw secret-bearing child text leaked in warning: {w!r}"
            )

    def test_trust_gated_parent_assessment_parent_status_aligned(self):
        # Phase 5 fix14: in the post-demotion recompute loop's
        # trust-gate replay branch, ``hit.parent_assessment`` is
        # re-assigned from ``assess_parent_status(assessment_children)``
        # BEFORE the trust-gate check fires. When every surviving
        # child is clean, that assessment returns
        # ``{"parent_status": "active", ...}`` — but the trust gate
        # forces the dataclass field ``hit.parent_status`` to
        # ``"review_required"``. Without an explicit sync, the
        # serialized envelope under ``include_metadata=True`` would
        # disagree: ``summary["parent_status"] == "review_required"``
        # alongside ``summary["parent_assessment"]
        # ["parent_status"] == "active"``.
        #
        # This regression asserts the contract:
        #   1. ``hit.parent_status == "review_required"`` (unchanged).
        #   2. ``hit.text == ""`` (unchanged).
        #   3. ``hit.parent_assessment["parent_status"] ==
        #       "review_required"`` (new fix14 invariant).
        #   4. ``trust_gate_reasons`` is preserved on the
        #      ``parent_assessment`` dict.
        #   5. The clean child is still cited (fix13 invariant
        #      preserved through the fix14 sync).
        #   6. The serialized envelope is internally consistent
        #      (the projection through ``to_dict(include_metadata=True)``
        #      also agrees).
        dense_seeds = [
            _RetrievedMemory(
                "parent-rr-clean",
                text="parent summary text",
                payload=_raptor_parent_payload(
                    node_id="parent-rr-clean",
                    children=["leaf-ok"],
                    summary_of=["leaf-ok"],
                    level=2,
                    requires_review=True,
                ),
                final_score=0.91,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-rr-clean", dense_seeds[0].payload)
        qdrant.add_point(
            "leaf-ok", _leaf_payload(point_id="leaf-ok", text="clean leaf")
        )
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3)

        target = next(
            (s for s in result.summaries
             if s.raptor_node_id == "parent-rr-clean"),
            None,
        )
        assert target is not None
        # (1) dataclass-level parent_status forced to review_required
        assert target.parent_status == "review_required"
        # (2) text cleared
        assert target.text == ""
        # (3) parent_assessment parent_status aligned to the
        #     demoted value — this is the fix14 invariant that
        #     would have failed before the sync.
        assert target.parent_assessment.get("parent_status") == "review_required"
        # (4) trust_gate_reasons must still be carried so
        #     operators can audit WHY the parent was demoted
        #     even when every child is clean.
        reasons = target.parent_assessment.get("trust_gate_reasons") or []
        assert "requires_review" in reasons
        # The trust-gated parent is tracked as unsafe.
        assert "parent-rr-clean" in result.unsafe_summary_ids
        # (5) Fix13 invariant: the clean child is still cited.
        promoted_leaf_ids = {l.point_id for l in result.cited_leaves}
        assert "leaf-ok" in promoted_leaf_ids
        # (6) Serialized envelope is internally consistent.
        d = result.to_dict(include_metadata=True)
        summary = next(
            (s for s in d.get("summaries", [])
             if s.get("raptor_node_id") == "parent-rr-clean"),
            None,
        )
        assert summary is not None
        assert summary.get("parent_status") == "review_required"
        # The nested parent_assessment must agree with the
        # top-level parent_status — that is the whole point of
        # fix14. Anything else means the inconsistency the
        # user reported is back.
        assert (
            summary.get("parent_assessment", {}).get("parent_status")
            == "review_required"
        ), (
            "summary.parent_assessment.parent_status must agree "
            "with summary.parent_status for trust-gated parents"
        )

    def test_secret_shaped_trust_gated_parent_id_redaction_intact(self):
        # The fix13 child-citation change must NOT regress the
        # earlier secret-core-field redaction. A parent whose
        # ``raptor_node_id`` itself is secret-shaped (e.g. a
        # ``Bearer ...`` token) is dropped by the
        # ``_summary_default_emitted_secret_bearing`` gate BEFORE
        # the trust gate runs, so neither parent text nor child
        # citation happens. The raw id must still not leak
        # through the warning channel or the serialized JSON
        # envelope. (The trust-gate path is exercised by the
        # tests above; here we only re-assert that the
        # secret-core-field gate keeps its invariants under the
        # fix13 refactor.)
        bad_id = "".join(["Bearer ", "z" * 24])
        dense_seeds = [
            _RetrievedMemory(
                bad_id,
                text="parent summary text",
                payload=_raptor_parent_payload(
                    node_id=bad_id,
                    children=["leaf-ok"],
                    summary_of=["leaf-ok"],
                    level=2,
                    requires_review=True,
                ),
                final_score=0.92,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point(bad_id, dense_seeds[0].payload)
        qdrant.add_point(
            "leaf-ok", _leaf_payload(point_id="leaf-ok", text="clean leaf"),
        )
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3)

        # No warning leaks the raw secret-shaped id.
        for w in result.warnings:
            assert bad_id not in w, (
                f"raw secret-shaped parent id leaked in warning: {w!r}"
            )
        # The serialized JSON envelope uses the redacted handle
        # for the unsafe summary id and does NOT echo the raw
        # bearer-shaped token.
        d = result.to_dict()
        for handle in d["unsafe_summary_ids"]:
            assert bad_id not in handle
        serialized = json.dumps(d)
        assert bad_id not in serialized, (
            "raw secret-shaped parent id leaked in serialized "
            "RaptorSearchResult.to_dict() output"
        )


# ---------------------------------------------------------------------------
# Phase 5 fix12 (final10 P3): ``RaptorSearchResult.to_dict`` query redaction.
#
# The standalone public RAPTOR API used to echo ``self.query`` verbatim
# into the JSON envelope. The acceptance criterion says it must mirror
# ``HybridRouteResult`` (safe ``query_length`` / ``query_digest`` /
# ``query_redacted`` block) so a secret-shaped query the user
# accidentally pasted in can never leak through the LLM context
# downstream.
# ---------------------------------------------------------------------------


class TestRaptorSearchResultQueryRedaction:
    def test_secret_shaped_query_not_in_serialized_output(self):
        # Construct the secret-shaped query at runtime so the
        # secret-bearing literal never appears in the source file.
        bad_query = "".join(["Bearer ", "x" * 24])
        result = RaptorSearchResult(query=bad_query)
        d = result.to_dict()
        # Raw ``query`` key is gone.
        assert "query" not in d
        # Secret-shaped query must NOT appear anywhere in the
        # serialized output.
        serialized = json.dumps(d)
        assert bad_query not in serialized, (
            "raw secret-shaped query leaked through "
            "RaptorSearchResult.to_dict()"
        )
        # Safe metadata block is present and well-formed.
        assert d["query_length"] == len(bad_query)
        assert d["query_redacted"].startswith("[redacted")
        assert isinstance(d["query_digest"], str) and len(d["query_digest"]) == 16


# ---------------------------------------------------------------------------
# Phase 5 fix6: ``_safe_retrieve`` warning cannot leak secret-shaped ids
#
# Regression coverage for finding #4 from the final4 reviewer/security
# pass. A backend ``retrieve()`` exception whose ``__str__`` echoes
# the requested point ids must not surface those raw ids through
# ``result.warnings`` (or anything else in the JSON envelope).
# ---------------------------------------------------------------------------


class TestSafeRetrieveWarningNoSecretIdLeak:
    def test_backend_exception_echoing_secret_id_not_leaked(self):
        # Construct the secret-shaped point id at runtime so the
        # scanner doesn't trip on a literal in the source file.
        bad_id = "".join(["Bearer ", "q" * 24])

        class _EchoingQdrant:
            """Fake Qdrant whose ``retrieve`` echoes requested ids."""

            def retrieve(self, name, ids, *, with_payload=True, with_vector=False):
                # Echo the full requested id list inside the
                # exception so ``str(exc)`` carries the secret-shaped
                # value verbatim.
                raise RuntimeError(
                    "backend refused request ids=" + ",".join(ids)
                )

        retriever = FakeRetriever(dense_seeds=[])
        searcher = RaptorSearcher(
            qdrant=_EchoingQdrant(),
            retriever=retriever,
            collection_name="memory",
        )
        # Reach into ``_safe_retrieve`` with the bad id directly so we
        # exercise the exception-warning path without needing a
        # populated dense seed.
        warnings: list[str] = []
        out = searcher._safe_retrieve([bad_id], warnings)
        # No payload promoted because the backend raised.
        assert out == []
        # The warning channel must record the failure but the raw
        # secret-shaped id MUST NOT appear in any warning.
        assert warnings
        for warning in warnings:
            assert bad_id not in warning, (
                "safe_retrieve warning leaked the requested secret-shaped id"
            )
        # The serialized form of the warnings list must also be free
        # of the raw id.
        import json as _json
        for warning in warnings:
            assert bad_id not in _json.dumps([warning])

    def test_clean_qdrant_no_warning_emitted(self):
        # Sanity: when the backend works normally (no exception),
        # no spurious warning is added even for ids that would
        # otherwise have been secret-shaped.
        bad_id = "".join(["Bearer ", "r" * 24])
        qdrant = FakeRaptorQdrant()
        # No payload registered; should not raise, just return [].
        warnings: list[str] = []
        out = searcher._safe_retrieve([bad_id], warnings) if False else (
            searcher_with_qdrant(qdrant)._safe_retrieve([bad_id], warnings)
        )
        assert out == []
        # No failure warning because the backend did not raise.
        assert warnings == []


def searcher_with_qdrant(qdrant) -> RaptorSearcher:
    """Tiny helper to bind a searcher to a given qdrant fake."""
    return RaptorSearcher(
        qdrant=qdrant,
        retriever=FakeRetriever(dense_seeds=[]),
        collection_name="memory",
    )


# ---------------------------------------------------------------------------
# Phase 5 fix7: RAPTOR seed-search warnings must NOT leak raw exception text
#
# Regression coverage for finding #1 from the final5 reviewer/security
# pass. When the dense+sparse seed retriever raises (TypeError because a
# legacy retriever lacks the ``allow_sparse_scroll`` kwarg, or a generic
# RuntimeError because the retriever backend is misbehaving), the
# resulting ``result.warnings`` entry MUST NOT interpolate ``str(exc)``.
# Backend exception ``__str__`` can echo the requested query string
# (which may carry a secret-shaped token) or other raw backend strings
# into the JSON envelope via HybridRouter.
# ---------------------------------------------------------------------------


class TestSeedSearchWarningNoRawExceptionLeak:
    def test_type_error_warning_no_raw_exception(self):
        # Construct the secret-shaped query at runtime so the
        # scanner doesn't trip on a literal in the source file.
        bad_query = "".join(["Bearer ", "x" * 24])

        class _RejectsKwargRetriever:
            """Custom retriever that raises a TypeError that echoes the query."""

            def __init__(self):
                self.calls: list[dict[str, Any]] = []

            def search(self, query, *, top_k=5, update_access=True, **kwargs):
                # Reject the strict kwarg like a legacy retriever
                # would, then attach the secret-shaped query to the
                # exception ``__str__`` so we can prove the warning
                # channel does NOT carry it.
                if "allow_sparse_scroll" in kwargs:
                    raise TypeError(
                        "search() got an unexpected keyword argument "
                        "'allow_sparse_scroll' from query=" + query
                    )
                return []

        qdrant = FakeRaptorQdrant()
        retriever = _RejectsKwargRetriever()
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search(bad_query, top_k=3)

        # No seeds promoted (fail-closed).
        assert result.summaries == []
        # The warning MUST exist and MUST NOT echo the raw secret-
        # shaped query (or any other component from str(exc)).
        assert result.warnings
        seed_warnings = [w for w in result.warnings if "allow_sparse_scroll" in w]
        assert seed_warnings
        for w in seed_warnings:
            assert bad_query not in w, (
                "seed-search TypeError warning leaked the requested query"
            )
            assert "no raw exception leaked" in w
            # ``allow_sparse_scroll`` TypeError may still mention
            # that kwarg by name (it is the public API surface, not
            # raw backend text). Just check that the secret-shaped
            # query isn't echoed through it.
        # The debug envelope MUST NOT echo the raw secret-shaped
        # query either (e.g. via debug.stages).
        import json as _json
        debug_serialized = _json.dumps(
            result.debug, default=str
        )
        assert bad_query not in debug_serialized
        # debug.stages.seed_search carries a stable error code so
        # operators can correlate without leaking str(exc).
        assert (
            result.debug.get("stages", {}).get("seed_search", {}).get("error")
            == "type_error"
        )

    def test_generic_exception_warning_no_raw_exception(self):
        # Construct the secret-shaped query at runtime so the
        # scanner doesn't trip on a literal in the source file.
        bad_query = "".join(["Bearer ", "y" * 24])

        class _EchoingRuntimeErrorRetriever:
            """Custom retriever that raises a generic RuntimeError that
            echoes the query verbatim.
            """

            def __init__(self):
                self.calls: list[dict[str, Any]] = []

            def search(self, query, *, top_k=5, update_access=True, **kwargs):
                # Echo the full query into the exception ``__str__``
                # so a leak via ``f"...{exc}"`` would surface the
                # secret-shaped token into the warning channel.
                raise RuntimeError(
                    "backend refused to process query=" + repr(query)
                )

        qdrant = FakeRaptorQdrant()
        retriever = _EchoingRuntimeErrorRetriever()
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search(bad_query, top_k=3)

        assert result.summaries == []
        # The generic-exception arm fires: at least one warning MUST
        # exist and MUST NOT echo the raw secret-shaped query.
        seed_warnings = [
            w for w in result.warnings
            if "raptor seed search failed" in w
        ]
        assert seed_warnings, (
            "expected the generic-exception seed-search warning to fire"
        )
        for w in seed_warnings:
            assert bad_query not in w, (
                "seed-search RuntimeError warning leaked the requested query"
            )
            assert "no raw exception leaked" in w
        # The debug envelope MUST NOT echo the raw secret-shaped
        # query either (e.g. via debug.stages).
        import json as _json
        debug_serialized = _json.dumps(
            result.debug, default=str
        )
        assert bad_query not in debug_serialized
        # debug.stages.seed_search carries a stable error code so
        # operators can correlate without leaking str(exc).
        assert (
            result.debug.get("stages", {}).get("seed_search", {}).get("error")
            == "exception"
        )


# ---------------------------------------------------------------------------
# Phase 5 fix7: parent must NOT remain active when referenced children are
# missing / deleted / scope-filtered
#
# Regression coverage for finding #2 from the final5 reviewer/security
# pass. fix6 only incremented the per-parent unsafe-child counter for
# children that were returned AND demoted for safety reasons. If a
# parent references a child that is missing/deleted/scope-filtered
# (i.e. ``_safe_retrieve`` returns no payload for it), both the
# safe-payloads list and the unsafe counter stayed empty and the parent
# remained ``active`` with no cited leaves and no warning. We now track
# the per-parent *referenced* child count so any parent whose children
# were silently dropped by the backend demotes accordingly.
# ---------------------------------------------------------------------------


class TestParentStatusMissingChildrenDemotion:
    def test_one_missing_child_demotes_parent(self):
        # Parent references two children; only one is actually
        # retrievable on the backend (the other is missing). The
        # parent MUST demote to a non-active status and emit a
        # redacted demotion warning; the surviving safe child does
        # NOT keep the parent active when there is a missing child.
        dense_seeds = [
            _RetrievedMemory(
                "parent-one-missing",
                text="parent summary text",
                payload=_raptor_parent_payload(
                    node_id="parent-one-missing",
                    children=["leaf-safe", "leaf-missing"],
                    summary_of=["leaf-safe", "leaf-missing"],
                    level=2,
                ),
                final_score=0.91,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-one-missing", dense_seeds[0].payload)
        # Deliberately do NOT add leaf-missing to the fake store:
        # ``_safe_retrieve`` will return no payload for it, exactly
        # mimicking a missing/deleted/scope-filtered child.
        qdrant.add_point(
            "leaf-safe",
            _leaf_payload(point_id="leaf-safe", text="safe leaf content"),
        )
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3)
        target = next(
            (s for s in result.summaries if s.raptor_node_id == "parent-one-missing"),
            None,
        )
        assert target is not None, "parent-one-missing should still be tracked"
        # The parent MUST not remain ``active`` because a referenced
        # child is missing.
        assert target.parent_status != "active"
        # Summary text MUST be cleared so the demoted parent never
        # reaches the caller as active context.
        assert target.text == ""
        # Warning channel MUST surface the demotion with a redacted
        # handle (no raw missing child id, no raw parent id).
        assert any(
            "demoted after child-safety review" in w
            and "leaf-missing" not in w
            and "parent-one-missing" not in w
            for w in result.warnings
        )
        assert "parent-one-missing" in result.unsafe_summary_ids

    def test_all_missing_children_demote_parent(self):
        # Parent references two children but BOTH are missing on the
        # backend; no children retrieved at all. The parent MUST
        # demote, its text MUST be cleared, and no cited leaves
        # should survive for it.
        dense_seeds = [
            _RetrievedMemory(
                "parent-all-missing",
                text="parent summary text",
                payload=_raptor_parent_payload(
                    node_id="parent-all-missing",
                    children=["leaf-gone-1", "leaf-gone-2"],
                    summary_of=["leaf-gone-1", "leaf-gone-2"],
                    level=2,
                ),
                final_score=0.91,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-all-missing", dense_seeds[0].payload)
        # Deliberately add neither child to the fake store.
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3)
        target = next(
            (s for s in result.summaries if s.raptor_node_id == "parent-all-missing"),
            None,
        )
        assert target is not None
        assert target.parent_status != "active"
        assert target.text == ""
        # No leaf promoted (none was retrieved).
        promoted_leaf_ids = {l.point_id for l in result.cited_leaves}
        assert "leaf-gone-1" not in promoted_leaf_ids
        assert "leaf-gone-2" not in promoted_leaf_ids
        # Warning uses redacted handle only — no raw leaf ids leaked.
        assert any(
            "demoted after child-safety review" in w
            and "leaf-gone-1" not in w
            and "leaf-gone-2" not in w
            for w in result.warnings
        )
        assert "parent-all-missing" in result.unsafe_summary_ids

    def test_safe_existing_children_keep_parent_active(self):
        # All referenced children are present + clean: parent MUST
        # stay active and its text MUST survive. This is the safe
        # baseline that proves we have NOT over-demoted.
        original_text = "important parent summary that must survive"
        dense_seeds = [
            _RetrievedMemory(
                "parent-safe-existing",
                text="parent summary text",
                payload=_raptor_parent_payload(
                    node_id="parent-safe-existing",
                    text=original_text,
                    children=["leaf-ok-1", "leaf-ok-2"],
                    summary_of=["leaf-ok-1", "leaf-ok-2"],
                    level=2,
                ),
                final_score=0.92,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-safe-existing", dense_seeds[0].payload)
        qdrant.add_point(
            "leaf-ok-1",
            _leaf_payload(point_id="leaf-ok-1", text="clean 1"),
        )
        qdrant.add_point(
            "leaf-ok-2",
            _leaf_payload(point_id="leaf-ok-2", text="clean 2"),
        )
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3)
        target = next(
            (s for s in result.summaries if s.raptor_node_id == "parent-safe-existing"),
            None,
        )
        assert target is not None
        assert target.parent_status == "active"
        assert target.text == original_text
        assert not any(
            "demoted after child-safety review" in w
            for w in result.warnings
        )
        assert "parent-safe-existing" not in result.unsafe_summary_ids

    def test_secret_shaped_missing_child_id_not_leaked_in_warning(self):
        # When the missing child id is itself secret-shaped, the
        # warning MUST NOT interpolate it. Only the redacted parent
        # handle is allowed.
        bad_missing = "".join(["Bearer ", "z" * 24])
        dense_seeds = [
            _RetrievedMemory(
                "parent-secret-missing",
                text="parent summary text",
                payload=_raptor_parent_payload(
                    node_id="parent-secret-missing",
                    children=["leaf-clean", bad_missing],
                    summary_of=["leaf-clean", bad_missing],
                    level=2,
                ),
                final_score=0.91,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-secret-missing", dense_seeds[0].payload)
        qdrant.add_point(
            "leaf-clean",
            _leaf_payload(point_id="leaf-clean", text="clean leaf"),
        )
        # Note: bad_missing is intentionally NOT added.
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3)
        # Parent demotes because the secret-shaped child is missing.
        assert "parent-secret-missing" in result.unsafe_summary_ids
        # Warning channel: no raw secret-shaped child id anywhere.
        for w in result.warnings:
            assert bad_missing not in w, (
                "missing-child warning leaked the raw secret-shaped child id"
            )
        # The serialized envelope is secret-shaped-id free.
        import json as _json
        serialized = _json.dumps(result.to_dict(), default=str)
        assert bad_missing not in serialized


# ---------------------------------------------------------------------------
# Phase 5 fix8 (final6 finding #2): shared unsafe / missing child must
# demote EVERY parent that referenced it, not just the first parent
# encountered. Previously, retrieval-pass dedupe (a single global
# ``seen_leaf_ids`` set combined with ``setdefault`` attribution) made
# the second parent's referenced/retrieved counts diverge silently, so
# a parent that shared its only child with another parent could remain
# ``active`` while its evidence had been demoted. We now track
# per-parent referenced children AND apply safety accounting to every
# parent that depended on a shared child.
# ---------------------------------------------------------------------------


class TestSharedUnsafeChildDemotesEveryParent:
    """Two parents reference the same child. When that child is unsafe
    (e.g. ``stale=True``), BOTH parents must demote and clear their
    text. Pre-fix8 only the first parent encountered was demoted.
    """

    def test_two_parents_share_stale_child_both_demote(self):
        shared_leaf = "shared-stale-leaf"
        dense_seeds = [
            _RetrievedMemory(
                "parent-A",
                text="parent A summary",
                payload=_raptor_parent_payload(
                    node_id="parent-A",
                    children=[shared_leaf],
                    summary_of=[shared_leaf],
                    level=2,
                ),
                final_score=0.91,
            ),
            _RetrievedMemory(
                "parent-B",
                text="parent B summary",
                payload=_raptor_parent_payload(
                    node_id="parent-B",
                    children=[shared_leaf],
                    summary_of=[shared_leaf],
                    level=2,
                ),
                final_score=0.90,
            ),
        ]
        qdrant = FakeRaptorQdrant()
        for seed in dense_seeds:
            qdrant.add_point(seed.id, seed.payload)
        # The shared child carries ``stale=True`` so ``assess_leaf_safety``
        # flags it as unsafe.
        qdrant.add_point(
            shared_leaf,
            _leaf_payload(point_id=shared_leaf, text="stale shared", stale=True),
        )
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3)
        # Both parents MUST be demoted.
        assert "parent-A" in result.unsafe_summary_ids, (
            "parent-A (first-seen) should demote because the shared child "
            "is unsafe"
        )
        assert "parent-B" in result.unsafe_summary_ids, (
            "parent-B (second-seen, sharing only the stale child with parent-A) "
            "must also demote — pre-fix8 only parent-A was demoted because "
            "retrieval-pass dedupe stripped parent-B's reference to the shared "
            "child"
        )
        # Both parents' text MUST be cleared.
        for parent_id in ("parent-A", "parent-B"):
            target = next(
                (s for s in result.summaries if s.raptor_node_id == parent_id),
                None,
            )
            assert target is not None
            assert target.text == "", (
                f"{parent_id} text must be cleared after demotion; "
                f"pre-fix8 parent-B retained its text because the shared "
                f"unsafe child was attributed to parent-A only"
            )
            assert target.parent_status != "active"

    def test_two_parents_share_missing_child_both_demote(self):
        shared_leaf = "shared-missing-leaf"
        dense_seeds = [
            _RetrievedMemory(
                "parent-X",
                text="parent X summary",
                payload=_raptor_parent_payload(
                    node_id="parent-X",
                    children=[shared_leaf],
                    summary_of=[shared_leaf],
                    level=2,
                ),
                final_score=0.91,
            ),
            _RetrievedMemory(
                "parent-Y",
                text="parent Y summary",
                payload=_raptor_parent_payload(
                    node_id="parent-Y",
                    children=[shared_leaf],
                    summary_of=[shared_leaf],
                    level=2,
                ),
                final_score=0.90,
            ),
        ]
        qdrant = FakeRaptorQdrant()
        for seed in dense_seeds:
            qdrant.add_point(seed.id, seed.payload)
        # Note: shared_leaf is intentionally NOT added — it is
        # missing from the backend so it must be classified as a
        # missing-child for both parents.
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3)
        # Both parents MUST be demoted.
        assert "parent-X" in result.unsafe_summary_ids, (
            "parent-X should demote because its shared child is missing"
        )
        assert "parent-Y" in result.unsafe_summary_ids, (
            "parent-Y (sharing only the missing child with parent-X) must "
            "also demote — pre-fix8 only parent-X was demoted because "
            "retrieval-pass dedupe stripped parent-Y's reference"
        )
        # Both parents' text MUST be cleared.
        for parent_id in ("parent-X", "parent-Y"):
            target = next(
                (s for s in result.summaries if s.raptor_node_id == parent_id),
                None,
            )
            assert target is not None
            assert target.text == "", (
                f"{parent_id} text must be cleared because its only "
                f"referenced child is missing"
            )
            assert target.parent_status != "active"
        # Missing-child warning uses the redacted parent handle only.
        demote_warnings = [
            w for w in result.warnings
            if "demoted after child-safety review" in w
        ]
        assert len(demote_warnings) >= 2, (
            "expected at least one demotion warning per parent"
        )

    def test_two_parents_share_safe_child_both_active(self):
        # When the shared child is clean and present in the backend,
        # BOTH parents must remain ``active`` and their text must
        # survive untouched. Pre-fix8 this already worked for
        # parent-A, but parent-B's safety accounting depended on
        # the global ``seen_leaf_ids`` set; we now verify the full
        # shared-safe path still keeps both parents promoted.
        original_a = "parent A summary"
        original_b = "parent B summary"
        shared_leaf = "shared-safe-leaf"
        dense_seeds = [
            _RetrievedMemory(
                "parent-A",
                text=original_a,
                payload=_raptor_parent_payload(
                    node_id="parent-A",
                    text=original_a,
                    children=[shared_leaf],
                    summary_of=[shared_leaf],
                    level=2,
                ),
                final_score=0.92,
            ),
            _RetrievedMemory(
                "parent-B",
                text=original_b,
                payload=_raptor_parent_payload(
                    node_id="parent-B",
                    text=original_b,
                    children=[shared_leaf],
                    summary_of=[shared_leaf],
                    level=2,
                ),
                final_score=0.91,
            ),
        ]
        qdrant = FakeRaptorQdrant()
        for seed in dense_seeds:
            qdrant.add_point(seed.id, seed.payload)
        qdrant.add_point(
            shared_leaf,
            _leaf_payload(point_id=shared_leaf, text="clean shared"),
        )
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3)
        # Both parents MUST stay active with their original text.
        for parent_id, original in (("parent-A", original_a), ("parent-B", original_b)):
            target = next(
                (s for s in result.summaries if s.raptor_node_id == parent_id),
                None,
            )
            assert target is not None
            assert target.parent_status == "active", (
                f"{parent_id} must remain active when the shared child is safe"
            )
            assert target.text == original, (
                f"{parent_id} text must survive when the shared child is safe"
            )
        # No spurious demotion warnings for either parent.
        demote_warnings = [
            w for w in result.warnings
            if "demoted after child-safety review" in w
        ]
        assert not any(
            "parent-A" in w or "parent-B" in w for w in demote_warnings
        ), (
            "no spurious demotion warnings expected for shared safe child; "
            f"got {demote_warnings!r}"
        )

    def test_three_parents_share_secret_shaped_stale_child(self):
        # Stress test: three parents share a single child whose
        # ``point_id`` is a bearer-shaped token. The leaf appears in
        # each parent's ``raptor_child_ids`` /
        # ``raptor_summary_of``, so the parent-summary pre-promotion
        # secret-bearing check (which scans default-emitted core
        # fields including ``raptor_child_ids``) drops the parent
        # from ``summaries`` entirely. The acceptance criterion is:
        # the raw secret-shaped shared child id MUST NOT leak
        # through the warning channel or the serialized JSON
        # envelope. Only the redacted parent handle is allowed.
        #
        # ``unsafe_summary_ids`` for parents that were dropped by
        # the secret-bearing summary check is still populated (see
        # ``_summary_default_emitted_secret_bearing``), so we also
        # assert the redacted warnings carry only safe handles.
        shared_leaf = "".join(["Bearer ", "s" * 24])
        dense_seeds = []
        for parent_id in ("parent-1", "parent-2", "parent-3"):
            dense_seeds.append(
                _RetrievedMemory(
                    parent_id,
                    text=f"{parent_id} summary",
                    payload=_raptor_parent_payload(
                        node_id=parent_id,
                        children=[shared_leaf],
                        summary_of=[shared_leaf],
                        level=2,
                    ),
                    final_score=0.91,
                )
            )
        qdrant = FakeRaptorQdrant()
        for seed in dense_seeds:
            qdrant.add_point(seed.id, seed.payload)
        # Benign text body — the secret-shaped token lives in the
        # leaf's ``point_id`` (the field under test for
        # leak-prevention). Putting the bearer in the text body
        # would also trip the secret detector (it should — that's
        # part of the defence-in-depth) but obscures whether the
        # warning channel truly uses redacted handles.
        qdrant.add_point(
            shared_leaf,
            _leaf_payload(point_id=shared_leaf, text="stale shared child", stale=True),
        )
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3)
        # Warning channel MUST NOT leak the raw secret-shaped shared
        # child id anywhere (only the redacted parent handle is
        # allowed).
        for w in result.warnings:
            assert shared_leaf not in w, (
                "warning channel leaked the raw secret-shaped shared child id: "
                + w
            )
        # The serialized envelope must remain secret-shaped-id
        # free; the parent summaries may not appear in ``summaries``
        # at all (they were dropped by the secret-bearing summary
        # check), but the unsafe_summary_ids / debug / warnings
        # channels must use redacted handles only.
        import json as _json
        serialized = _json.dumps(result.to_dict(), default=str)
        assert shared_leaf not in serialized, (
            "serialized envelope leaked the raw secret-shaped shared child id"
        )
        # The unsafe_summary_ids envelope (which always emits
        # redacted handles) must be a list of redacted handles — no
        # raw secret-shaped values.
        unsafe_ids_dict = result.to_dict().get("unsafe_summary_ids") or []
        assert isinstance(unsafe_ids_dict, list)
        for handle in unsafe_ids_dict:
            assert shared_leaf not in handle, (
                "unsafe_summary_ids handle leaked raw secret-shaped id: "
                + handle
            )
        # Every warning that mentions ``raptor summary`` MUST use a
        # redacted parent handle, never the raw secret-shaped
        # shared child id.
        summary_warnings = [
            w for w in result.warnings if "raptor summary" in w
        ]
        assert summary_warnings, (
            "expected at least one raptor-summary warning so the "
            "redaction contract is exercised"
        )
        for w in summary_warnings:
            assert shared_leaf not in w
            assert "handle=" in w
            assert "redacted:" in w


# ---------------------------------------------------------------------------
# Phase 5 fix8 (final6 finding #1): real HybridRouter + real
# GraphMemoryRetriever + real MemoryRetriever + UUID/strong-signal
# query + strict fake Qdrant sentinel proving ZERO ``scroll_by_filter``
# calls and no access metadata updates.
# ---------------------------------------------------------------------------


class TestHybridRouterNoScrollByFilterUnderStrongSignal:
    """End-to-end regression for the Phase 5 hybrid retrieve contract.

    A strong-signal UUID-shaped query previously triggered
    ``scroll_by_filter`` from inside the graph lane (because the
    ``MemoryRetriever`` sparse lane ran on the dense seed and the
    graph BFS expansion always scrolled for graph_entity /
    graph_edge payloads). Phase 5 fix8 adds two flags —
    ``allow_sparse_scroll`` and ``allow_graph_scroll`` — and the
    ``HybridRouter`` propagates them with ``False`` so the new
    ``qdrant_memory_retrieve`` path is guaranteed to invoke zero
    ``scroll_by_filter`` calls and zero ``update_payload`` mutations.
    """

    def _build_qdrant_sentinel(self) -> Any:
        """Strict fake Qdrant that records every scroll / mutation."""

        class _StrictFakeQdrant:
            def __init__(self):
                self.search_calls: list = []
                self.scroll_calls: list = []
                self.retrieve_calls: list = []
                self.upserts: list = []
                self.update_payloads: list = []
                self.deletes: list = []
                self.delete_filters: list = []

            def search(self, *args, **kwargs):
                self.search_calls.append((args, kwargs))
                return []

            def scroll_by_filter(self, *args, **kwargs):
                self.scroll_calls.append((args, kwargs))
                return []

            def retrieve(self, *args, **kwargs):
                self.retrieve_calls.append((args, kwargs))
                return []

            def upsert(self, *args, **kwargs):
                self.upserts.append((args, kwargs))
                raise AssertionError("upsert must not be called")

            def update_payload(self, *args, **kwargs):
                self.update_payloads.append((args, kwargs))
                raise AssertionError(
                    "update_payload must not be called from read-only retrieve"
                )

            def delete_ids(self, *args, **kwargs):
                self.deletes.append((args, kwargs))
                raise AssertionError("delete_ids must not be called")

            def delete_filter(self, *args, **kwargs):
                self.delete_filters.append((args, kwargs))
                raise AssertionError("delete_filter must not be called")

        return _StrictFakeQdrant()

    def test_hybrid_router_zero_scroll_by_filter_and_zero_updates(self):
        # Real ``MemoryRetriever``, real ``GraphMemoryRetriever``,
        # real ``RaptorSearcher``, strict fake Qdrant. The query is a
        # strong-signal UUID-shaped literal so the sparse lane would
        # normally fire ``scroll_by_filter``.
        from qdrant_memory.graph_retriever import GraphMemoryRetriever
        from qdrant_memory.retriever import MemoryRetriever
        from qdrant_memory.raptor.search import RaptorSearcher

        qdrant = self._build_qdrant_sentinel()
        # Real MemoryRetriever — wired against the strict fake Qdrant.
        memory_retriever = MemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
        )
        # Real GraphMemoryRetriever (which itself wraps a
        # MemoryRetriever). We force the inner retriever to share
        # the strict fake Qdrant so any scroll would surface as a
        # recorded scroll call.
        graph_retriever = GraphMemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
        )
        # Minimal RaptorSearcher against the strict fake Qdrant.
        raptor_searcher = RaptorSearcher(
            qdrant=qdrant,
            retriever=memory_retriever,
            collection_name="memory",
        )

        from qdrant_memory.hybrid import HybridRouter

        router = HybridRouter(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=memory_retriever,
            graph_retriever=graph_retriever,
            raptor_searcher=raptor_searcher,
        )
        # Strong-signal UUID-shaped query that previously triggered
        # the sparse lane's ``scroll_by_filter``.
        strong_signal = "550e8400-e29b-41d4-a716-446655440000"
        d = router.retrieve(strong_signal, top_k=3).to_dict()

        # Contract 1: zero ``scroll_by_filter`` calls anywhere in
        # the entire retrieve pipeline (dense lane, graph lane,
        # raptor lane).
        assert qdrant.scroll_calls == [], (
            "HybridRouter.retrieve triggered "
            f"{len(qdrant.scroll_calls)} scroll_by_filter call(s); "
            "the Phase 5 contract forbids any scroll under the read-only "
            "retrieve path. Recorded calls: "
            f"{qdrant.scroll_calls!r}"
        )
        # Contract 2: zero ``update_payload`` calls — no access
        # metadata mutation under any lane.
        assert qdrant.update_payloads == [], (
            "HybridRouter.retrieve triggered "
            f"{len(qdrant.update_payloads)} update_payload call(s); "
            "the Phase 5 contract forbids any access-metadata mutation. "
            f"Recorded calls: {qdrant.update_payloads!r}"
        )
        # Contract 3: zero ``upsert`` / ``delete_ids`` / ``delete_filter``
        # mutations (the fake raises on any of them; reaching here is
        # the proof).
        # Contract 4: read-only invariant envelope present.
        assert d["read_only"] is True if "read_only" in d else d["debug"]["read_only"] is True
        # Contract 5: debug envelope records the graph stage as
        # successful (no scroll). The new contract uses
        # ``scroll_suppressed`` in the graph sub-debug envelope.
        graph_stage = d["debug"]["stages"].get("graph", {})
        assert graph_stage.get("skipped") is None
        assert graph_stage.get("error") is None
        # Contract 6: dense seed lane ran via the qdrant.search path
        # (vector search) — verify the strong-signal query reached
        # the dense lane at least once so the test would catch a
        # regression where the router silently swallowed the call.
        assert qdrant.search_calls, (
            "expected at least one qdrant.search call (vector dense seed)"
        )

    def test_graph_retriever_short_circuit_does_not_scroll(self):
        # Standalone ``GraphMemoryRetriever.search`` with
        # ``allow_graph_scroll=False`` MUST short-circuit BEFORE
        # any ``scroll_by_filter`` call. The sparse-lane
        # ``scroll_by_filter`` from the dense seed must also be
        # suppressed when ``allow_sparse_scroll=False`` is passed.
        from qdrant_memory.graph_retriever import GraphMemoryRetriever

        qdrant = self._build_qdrant_sentinel()
        graph_retriever = GraphMemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
        )
        result = graph_retriever.search(
            "550e8400-e29b-41d4-a716-446655440000",  # strong-signal
            top_k=3,
            allow_sparse_scroll=False,
            allow_graph_scroll=False,
        )
        # The graph BFS expansion MUST NOT have been entered.
        assert qdrant.scroll_calls == [], (
            "GraphMemoryRetriever.search with allow_graph_scroll=False "
            f"triggered {len(qdrant.scroll_calls)} scroll_by_filter call(s)"
        )
        # The result is empty (no seeds, no final).
        assert result.final == []
        assert result.seeds == []
        # Debug envelope reports scroll suppression.
        assert result.debug.get("scroll_suppressed") is True

    def test_graph_retriever_default_behaviour_unchanged(self):
        # Standalone ``GraphMemoryRetriever.search`` with the
        # default ``allow_graph_scroll=True`` keeps its existing
        # behaviour. We do not assert on ``scroll_calls`` (the fake
        # Qdrant returns empty) but we DO assert that the
        # ``scroll_suppressed`` debug flag is absent (the gate was
        # not engaged) and that the graph stage-A seed search ran
        # via ``qdrant.search`` rather than being short-circuited.
        from qdrant_memory.graph_retriever import GraphMemoryRetriever

        qdrant = self._build_qdrant_sentinel()
        graph_retriever = GraphMemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
        )
        result = graph_retriever.search("plain natural language", top_k=3)
        # Standalone defaults MUST NOT short-circuit.
        assert "scroll_suppressed" not in (result.debug or {}), (
            "default GraphMemoryRetriever.search must NOT short-circuit; "
            "scroll_suppressed should be absent"
        )
        # Stage-A seed search ran (vector search on the fake Qdrant).
        assert qdrant.search_calls, (
            "expected qdrant.search call for stage-A seed search"
        )


# ---------------------------------------------------------------------------
# Phase 5 fix9 (final7 finding #1): RAPTOR child fanout cap must NOT be
# counted as missing evidence. Children beyond ``max_children`` are
# intentionally not retrieved because of the fanout budget; they are
# not missing/deleted/scope-filtered evidence. Pre-fix9 the per-parent
# referenced set recorded every child in ``raptor_child_ids`` /
# ``raptor_summary_of`` BEFORE the cap was applied, so a parent with
# >max_children all-safe children was demoted to stale with empty text.
# ---------------------------------------------------------------------------


class TestFanoutCapNotCountedAsMissing:
    """Regression for final7 finding #1.

    Children beyond ``safe_max_children`` are budget-skipped, not
    missing. The parent-status recomputation must derive the
    missing-count from the cap-bounded referenced set, not the
    full ``raptor_child_ids`` / ``raptor_summary_of`` payload.
    """

    def test_parent_with_more_than_max_children_keeps_active_text(self):
        # 20 children, all safe, max_children=8. Only 8 should be
        # retrieved; the other 12 are budget-skipped, not missing.
        # The parent MUST remain ``active`` and keep its text.
        n_children = 20
        max_children = 8
        original_text = "important parent summary that must survive cap"
        children = [f"leaf-{i:02d}" for i in range(n_children)]
        dense_seeds = [
            _RetrievedMemory(
                "parent-cap-safe",
                text="parent summary text",
                payload=_raptor_parent_payload(
                    node_id="parent-cap-safe",
                    text=original_text,
                    children=children,
                    summary_of=children,
                    level=2,
                ),
                final_score=0.93,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-cap-safe", dense_seeds[0].payload)
        # All 20 children are present on the backend AND safe.
        for i, cid in enumerate(children):
            qdrant.add_point(
                cid,
                _leaf_payload(point_id=cid, text=f"clean child {i}"),
            )
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search(
            "anything", top_k=3, max_children=max_children,
        )
        target = next(
            (s for s in result.summaries if s.raptor_node_id == "parent-cap-safe"),
            None,
        )
        assert target is not None
        # The parent MUST stay active — no children are missing
        # within the cap, so the missing-count must be zero.
        assert target.parent_status == "active", (
            f"parent with all-safe children was demoted to "
            f"{target.parent_status!r}; final7 finding #1 regression: "
            "children beyond max_children must NOT count as missing"
        )
        # The parent text MUST survive the cap.
        assert target.text == original_text, (
            "parent text was cleared despite all retrieved children being safe"
        )
        # The cited-leaves output MUST be bounded by the cap
        # (max_children) — fanout cap must actually take effect.
        promoted = [
            leaf for leaf in result.cited_leaves
            if leaf.parent_raptor_node_id == "parent-cap-safe"
        ]
        assert len(promoted) <= max_children, (
            f"parent-cap-safe cited {len(promoted)} leaves but cap was "
            f"{max_children}; the cap must bound actual promotion"
        )
        # The parent MUST NOT be in unsafe_summary_ids.
        assert "parent-cap-safe" not in result.unsafe_summary_ids
        # No demotion warning for the parent.
        assert not any(
            "parent-cap-safe" in w
            and "demoted after child-safety review" in w
            for w in result.warnings
        ), (
            "spurious demotion warning for an all-safe parent beyond the cap"
        )

    def test_parent_with_missing_child_inside_cap_demotes(self):
        # 3 children, all referenced, max_children=8 (cap never
        # reached). One of the 3 is missing on the backend. The
        # parent MUST demote because the missing child is within
        # the cap and therefore the searcher was responsible for
        # retrieving it.
        children = ["leaf-ok-1", "leaf-ok-2", "leaf-gone"]
        original_text = "parent with one missing child"
        dense_seeds = [
            _RetrievedMemory(
                "parent-one-gone-in-cap",
                text="parent summary text",
                payload=_raptor_parent_payload(
                    node_id="parent-one-gone-in-cap",
                    text=original_text,
                    children=children,
                    summary_of=children,
                    level=2,
                ),
                final_score=0.93,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-one-gone-in-cap", dense_seeds[0].payload)
        # Only 2 of 3 children are present; the third is missing.
        for cid in ("leaf-ok-1", "leaf-ok-2"):
            qdrant.add_point(
                cid, _leaf_payload(point_id=cid, text=f"clean {cid}"),
            )
        # Note: leaf-gone is intentionally NOT added.
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3, max_children=8)
        target = next(
            (s for s in result.summaries
             if s.raptor_node_id == "parent-one-gone-in-cap"),
            None,
        )
        assert target is not None
        # Missing child inside the cap MUST demote the parent.
        assert target.parent_status != "active", (
            "parent with a missing child inside the fanout cap should demote; "
            "final7 finding #1 acceptance: a genuinely missing child within "
            "the cap must still demote the parent"
        )
        assert target.text == ""
        assert "parent-one-gone-in-cap" in result.unsafe_summary_ids
        # Warning channel uses redacted handle only.
        assert any(
            "demoted after child-safety review" in w
            and "parent-one-gone-in-cap" not in w
            and "leaf-gone" not in w
            for w in result.warnings
        )

    def test_parent_with_all_safe_children_beyond_cap_no_demotion_warning(self):
        # 20 safe children, max_children=4. Parent MUST stay active
        # AND no per-child "demoted" or "missing" warning should be
        # emitted for the 16 budget-skipped children. Only the
        # actual retrieved leaves should appear in cited_leaves.
        n_children = 20
        max_children = 4
        children = [f"safe-child-{i:02d}" for i in range(n_children)]
        original_text = "fanout-capped safe parent"
        dense_seeds = [
            _RetrievedMemory(
                "parent-fanout-safe",
                text="parent summary text",
                payload=_raptor_parent_payload(
                    node_id="parent-fanout-safe",
                    text=original_text,
                    children=children,
                    summary_of=children,
                    level=2,
                ),
                final_score=0.93,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-fanout-safe", dense_seeds[0].payload)
        for i, cid in enumerate(children):
            qdrant.add_point(
                cid, _leaf_payload(point_id=cid, text=f"clean {i}"),
            )
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search(
            "anything", top_k=3, max_children=max_children,
        )
        target = next(
            (s for s in result.summaries
             if s.raptor_node_id == "parent-fanout-safe"),
            None,
        )
        assert target is not None
        assert target.parent_status == "active"
        assert target.text == original_text
        # cited_leaves for this parent must be bounded by cap.
        promoted_pids = {
            leaf.point_id
            for leaf in result.cited_leaves
            if leaf.parent_raptor_node_id == "parent-fanout-safe"
        }
        assert len(promoted_pids) <= max_children
        # No budget-skipped child should surface as "missing" or
        # trigger a demotion-related warning.
        for skipped in children:
            if skipped in promoted_pids:
                continue
            # The skipped id MUST NOT be in unsafe_leaf_ids.
            assert skipped not in result.unsafe_leaf_ids, (
                f"budget-skipped child {skipped} was marked unsafe/missing"
            )
        # No demotion warnings for the parent at all.
        demote_warnings = [
            w for w in result.warnings
            if "parent-fanout-safe" in w
            and "demoted after child-safety review" in w
        ]
        assert not demote_warnings, (
            f"spurious demotion warning for all-safe fanout-capped parent: "
            f"{demote_warnings!r}"
        )

    def test_shared_child_safety_accounting_intact_under_cap(self):
        # fix8 contract regression under the new cap-aware accounting:
        # when a shared child is unsafe, BOTH parents must still
        # demote. The cap branch must not break the shared-child
        # safety path.
        shared_leaf = "shared-unsafe-leaf"
        dense_seeds = [
            _RetrievedMemory(
                "share-parent-A",
                text="parent A summary",
                payload=_raptor_parent_payload(
                    node_id="share-parent-A",
                    children=[shared_leaf],
                    summary_of=[shared_leaf],
                    level=2,
                ),
                final_score=0.92,
            ),
            _RetrievedMemory(
                "share-parent-B",
                text="parent B summary",
                payload=_raptor_parent_payload(
                    node_id="share-parent-B",
                    children=[shared_leaf],
                    summary_of=[shared_leaf],
                    level=2,
                ),
                final_score=0.91,
            ),
        ]
        qdrant = FakeRaptorQdrant()
        for seed in dense_seeds:
            qdrant.add_point(seed.id, seed.payload)
        qdrant.add_point(
            shared_leaf,
            _leaf_payload(point_id=shared_leaf, text="stale shared", stale=True),
        )
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3, max_children=8)
        # Both parents must demote — fix8 contract preserved.
        assert "share-parent-A" in result.unsafe_summary_ids
        assert "share-parent-B" in result.unsafe_summary_ids
        for parent_id in ("share-parent-A", "share-parent-B"):
            target = next(
                (s for s in result.summaries if s.raptor_node_id == parent_id),
                None,
            )
            assert target is not None
            assert target.text == "", (
                f"{parent_id} text must be cleared when the shared child is "
                f"unsafe; final7 finding #1 must not regress fix8 finding #2"
            )
            assert target.parent_status != "active"


# ---------------------------------------------------------------------------
# Phase 5 fix15 (final12 P2 finding): childless RAPTOR parent summaries
# MUST fail closed by default — final output must be non-active/excluded,
# text cleared, parent node ID in unsafe_summary_ids, parent_assessment[
# "parent_status"] aligned to the demoted status, and warning only
# sanitized / redacted. The earlier "no child IDs; downgraded" warning
# at promotion time was lost when the post-demotion recompute called
# assess_parent_status([]) and got back "active". These tests pin the
# fix in place so the regression cannot return.
# ---------------------------------------------------------------------------


class TestChildlessParentFailClosed:
    def test_childless_parent_final_status_is_excluded(self):
        # A parent with NO source-backed child refs (raptor_child_ids
        # and raptor_summary_of both empty) must end up with
        # parent_status != "active" — preferred value "excluded"
        # to match the original detection-time status. Pre-fix15
        # this test failed because the post-demotion recompute
        # forced ``hit.parent_status = "active"`` after
        # ``assess_parent_status([])`` returned "active".
        original_text = "childless parent summary"
        dense_seeds = [
            _RetrievedMemory(
                "parent-childless",
                text="parent summary text",
                payload=_raptor_parent_payload(
                    node_id="parent-childless",
                    children=[],
                    summary_of=[],
                    level=2,
                    text=original_text,
                ),
                final_score=0.85,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-childless", dense_seeds[0].payload)
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3)

        # The hit is still surfaced (so callers can audit *which*
        # parent was unsafe and why) but its effective status and
        # text must reflect the demotion, not the original raw
        # payload projection.
        target = next(
            (s for s in result.summaries
             if s.raptor_node_id == "parent-childless"),
            None,
        )
        assert target is not None, (
            "childless RAPTOR parent must still surface as a non-active "
            "audit summary"
        )
        assert target.parent_status != "active", (
            "final12 P2 regression: childless RAPTOR parent must NOT stay "
            "active after recompute; expected 'excluded' (or other "
            "non-active value) because the post-demotion recompute must "
            "honor the original no-child downgrade"
        )
        assert target.parent_status == "excluded", (
            "final12 P2 acceptance: prefer 'excluded' to match the original "
            "detection-time status"
        )
        assert target.text == "", (
            "final12 P2 acceptance: text must be cleared so the childless "
            "summary text cannot reach the caller as active context"
        )
        # The node ID MUST be tracked in unsafe_summary_ids so the
        # dense envelope can render it through the same redacted
        # handle discipline as every other unsafe summary.
        assert "parent-childless" in result.unsafe_summary_ids, (
            "final12 P2 acceptance: unsafe_summary_ids must contain the "
            "childless parent node id"
        )

    def test_childless_parent_assessment_aligned_to_excluded(self):
        # The nested ``parent_assessment["parent_status"]`` MUST
        # agree with the dataclass-level ``hit.parent_status`` so
        # ``include_metadata=True`` callers see a consistent
        # projection (mirroring fix14's trust-gate invariant).
        dense_seeds = [
            _RetrievedMemory(
                "parent-childless-2",
                text="text",
                payload=_raptor_parent_payload(
                    node_id="parent-childless-2",
                    children=[],
                    summary_of=[],
                    level=2,
                ),
                final_score=0.8,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-childless-2", dense_seeds[0].payload)
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3)
        target = next(
            (s for s in result.summaries
             if s.raptor_node_id == "parent-childless-2"),
            None,
        )
        assert target is not None
        d = target.to_dict(include_metadata=True)
        # parent_assessment must surface (the metadata is clean)
        # and must agree with hit.parent_status (= "excluded").
        assert d["parent_assessment"].get("parent_status") == "excluded", (
            "final12 P2 acceptance: parent_assessment['parent_status'] "
            "must be aligned to the demoted 'excluded' value, not the "
            "default 'active' value that assess_parent_status([]) returns"
        )
        # The marker that drives the replay must be present so the
        # recompute path can keep the parent demoted.
        assert d["parent_assessment"].get("no_child_refs") is True, (
            "final12 P2 acceptance: parent_assessment must carry the "
            "no_child_refs marker captured at construction time so the "
            "post-demotion recompute can keep the parent excluded"
        )
        missing = d["parent_assessment"].get("missing_child_reasons") or []
        assert "no_child_refs" in missing, (
            "final12 P2 acceptance: missing_child_reasons must include "
            "the no_child_refs token"
        )

    def test_childless_warning_redacted_handle_no_raw_leak(self):
        # The no-child warning must use the redacted handle, not
        # the raw node id, for a CLEAN (non-secret-shaped) node id.
        # Phase 5 fix15 preserves the original promotion-time
        # "no child IDs; downgraded" warning AND adds a
        # recompute-time "no source-backed child refs" warning
        # when the recompute replays the marker — both must use
        # the redacted handle discipline so a secret-shaped node
        # id can never leak through the no-child warning channel.
        #
        # We exercise BOTH branches separately:
        #
        #   1) clean-id scenario: the promotion-time warning fires
        #      and uses the redacted handle for the clean node id.
        #   2) secret-shaped-id scenario: the earlier fix2/fix3
        #      secret-bearing summary skip path takes precedence
        #      over the no-child check (this is intentional — a
        #      secret-bearing payload must NEVER reach the
        #      no-child marker so the "excluded" demotion cannot
        #      be misinterpreted as "secret-bearing but still
        #      auditable"). The JSON envelope must still not echo
        #      the raw id (only a redacted handle).
        #
        # Build any secret-shaped ids at runtime to avoid tripping
        # the scanner on a literal.
        # Part 1: clean node id, verify the no-child warning uses
        # the redacted handle for the clean node id, not the raw id.
        dense_seeds = [
            _RetrievedMemory(
                "parent-clean-nochild",
                text="parent",
                payload=_raptor_parent_payload(
                    node_id="parent-clean-nochild",
                    children=[], summary_of=[],
                ),
                final_score=0.7,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-clean-nochild", dense_seeds[0].payload)
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3)
        no_child_warnings = [
            w for w in result.warnings
            if "no child IDs" in w or "no source-backed child refs" in w
        ]
        assert no_child_warnings, (
            "expected at least one no-child warning in the warning channel"
        )
        for w in no_child_warnings:
            # The raw id MUST NOT echo the raw node id; the
            # redacted handle (or "redacted:" sentinel) MUST appear
            # so operators can correlate without leaking the raw id.
            assert "parent-clean-nochild" not in w, (
                f"raw node id leaked in no-child warning: {w!r}"
            )
            assert "redacted:" in w, (
                f"redacted handle missing in no-child warning: {w!r}"
            )

        # Part 2: secret-shaped node id, verify the JSON envelope
        # never echoes the raw id even when the secret-bearing
        # skip path drops the summary entirely.
        bad_id = "".join(["Bearer ", "v" * 24])
        dense_seeds_bad = [
            _RetrievedMemory(
                bad_id,
                text="parent",
                payload=_raptor_parent_payload(
                    node_id=bad_id, children=[], summary_of=[],
                ),
                final_score=0.7,
            )
        ]
        qdrant2 = FakeRaptorQdrant()
        qdrant2.add_point(bad_id, dense_seeds_bad[0].payload)
        retriever2 = FakeRetriever(dense_seeds_bad)
        searcher2 = RaptorSearcher(
            qdrant=qdrant2,
            retriever=retriever2,
            collection_name="memory",
        )
        result2 = searcher2.search("anything", top_k=3)
        # The secret-bearing summary skip path is intentional and
        # runs before the no-child check. We do NOT require the
        # no-child warning to appear here because the secret-bearing
        # path is the authoritative drop. We DO require that the
        # JSON envelope never echoes the raw id.
        serialized = json.dumps(result2.to_dict(), default=str)
        assert bad_id not in serialized, (
            "raw secret-shaped id leaked in the serialised envelope "
            "for the childless + secret-shaped scenario"
        )
        # And the warning channel must use the redacted handle too.
        for w in result2.warnings:
            assert bad_id not in w, (
                f"raw secret-shaped id leaked in warning channel: {w!r}"
            )

    def test_childless_parent_no_cited_leaves_and_zero_text(self):
        # Defense-in-depth: a childless parent must not adopt any
        # cited leaves either (no children means no children to
        # cite). The cited_leaves projection must remain empty for
        # the childless parent.
        dense_seeds = [
            _RetrievedMemory(
                "parent-orphan",
                text="orphan parent text",
                payload=_raptor_parent_payload(
                    node_id="parent-orphan",
                    children=[],
                    summary_of=[],
                    level=2,
                ),
                final_score=0.8,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-orphan", dense_seeds[0].payload)
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3)
        target = next(
            (s for s in result.summaries
             if s.raptor_node_id == "parent-orphan"),
            None,
        )
        assert target is not None
        assert target.text == ""
        # No leaves should have been attributed to this parent.
        attributed = [
            l for l in result.cited_leaves
            if l.parent_raptor_node_id == "parent-orphan"
        ]
        assert attributed == [], (
            "childless RAPTOR parent must not be the parent of any "
            "promoted leaf — there are no children to cite"
        )

    def test_approved_parent_with_clean_child_remains_active_regression(self):
        # Regression: existing fix13/fix14 behavior must NOT regress.
        # An approved parent (``requires_review=False``,
        # ``raptor_review_status="approved"``, default clean
        # flags) with a clean child must:
        #   - stay ``active`` (parent_status unchanged),
        #   - keep its original text,
        #   - cite its clean child,
        #   - NOT be in unsafe_summary_ids.
        # The fix15 path is strictly additive: only parents whose
        # construction-time ``parent_assessment`` carries the
        # ``no_child_refs`` / ``missing_child_reasons`` marker
        # take the new replay branch. Approved clean parents
        # follow the legacy recompute path unchanged.
        original_text = "approved parent must survive"
        dense_seeds = [
            _RetrievedMemory(
                "parent-clean-approved",
                text="parent summary text",
                payload=_raptor_parent_payload(
                    node_id="parent-clean-approved",
                    children=["leaf-clean"],
                    summary_of=["leaf-clean"],
                    level=2,
                    text=original_text,
                ),
                final_score=0.92,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-clean-approved", dense_seeds[0].payload)
        qdrant.add_point(
            "leaf-clean",
            _leaf_payload(point_id="leaf-clean", text="clean child"),
        )
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3)
        target = next(
            (s for s in result.summaries
             if s.raptor_node_id == "parent-clean-approved"),
            None,
        )
        assert target is not None
        assert target.parent_status == "active"
        assert target.text == original_text
        assert "parent-clean-approved" not in result.unsafe_summary_ids
        promoted = {l.point_id for l in result.cited_leaves}
        assert "leaf-clean" in promoted, (
            "approved parent with clean child must still cite it "
            "(fix13 / fix14 behavior must not regress)"
        )

    def test_trust_gated_parent_with_clean_child_still_cites_child_regression(self):
        # Regression: existing fix13/fix14 behavior must NOT regress.
        # A trust-gated parent (requires_review=True) with a clean
        # child must:
        #   - stay ``review_required`` (parent_status unchanged),
        #   - have text cleared,
        #   - STILL cite its clean child for evidence-mode traces,
        #   - have the trust_gate_reasons marker preserved in
        #     parent_assessment.
        # The fix15 path runs AFTER the trust-gate branch and is
        # only entered when the no_child_refs / missing_child_reasons
        # marker is present, so trust-gated parents with children
        # take the legacy fix14 replay unchanged.
        dense_seeds = [
            _RetrievedMemory(
                "parent-rr-clean",
                text="parent summary text",
                payload=_raptor_parent_payload(
                    node_id="parent-rr-clean",
                    children=["leaf-ok"],
                    summary_of=["leaf-ok"],
                    level=2,
                    requires_review=True,
                ),
                final_score=0.91,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-rr-clean", dense_seeds[0].payload)
        qdrant.add_point(
            "leaf-ok", _leaf_payload(point_id="leaf-ok", text="clean"),
        )
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3)
        target = next(
            (s for s in result.summaries
             if s.raptor_node_id == "parent-rr-clean"),
            None,
        )
        assert target is not None
        # Trust gate must still win over the new no-child branch
        # (the trust gate reason list is populated independently
        # of the no_child_refs marker because the parent has a
        # child, so the new fix15 branch is NOT entered).
        assert target.parent_status == "review_required"
        assert target.text == ""
        assert "parent-rr-clean" in result.unsafe_summary_ids
        # The trust_gate_reasons marker must be preserved (not
        # replaced by the new no_child_refs marker).
        reasons = target.parent_assessment.get("trust_gate_reasons") or []
        assert reasons, "trust_gate_reasons must survive fix15"
        # Cited child must survive for evidence-mode traces
        # (fix13 contract regression).
        promoted = {l.point_id for l in result.cited_leaves}
        assert "leaf-ok" in promoted, (
            "trust-gated parent with clean child must still cite it "
            "(fix13 contract must not regress on the fix15 path)"
        )

    def test_existing_missing_child_demotion_intact_regression(self):
        # Regression: existing fix7 / fix8 missing-child demotion
        # semantics must NOT change. A parent with a missing child
        # inside the fanout cap must still demote (final7 finding
        # #1 acceptance).
        children = ["leaf-a", "leaf-b", "leaf-gone"]
        dense_seeds = [
            _RetrievedMemory(
                "parent-mix",
                text="text",
                payload=_raptor_parent_payload(
                    node_id="parent-mix",
                    children=children,
                    summary_of=children,
                    level=2,
                ),
                final_score=0.9,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-mix", dense_seeds[0].payload)
        for cid in ("leaf-a", "leaf-b"):
            qdrant.add_point(cid, _leaf_payload(point_id=cid, text=cid))
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3, max_children=8)
        target = next(
            (s for s in result.summaries
             if s.raptor_node_id == "parent-mix"),
            None,
        )
        assert target is not None
        assert target.parent_status != "active", (
            "existing missing-child demotion regression: a parent with "
            "a missing child inside the fanout cap must still demote "
            "(final7 finding #1 acceptance must not regress)"
        )
        assert target.text == ""
        assert "parent-mix" in result.unsafe_summary_ids


# ---------------------------------------------------------------------------
# Phase 5 fix16 (final13 P2 finding): a childless RAPTOR parent that
# ALSO carries the production-shaped trust markers
# (``requires_review=True`` and / or
# ``raptor_review_status="review_required"``) must still resolve to
# ``"excluded"`` rather than ``"review_required"``. The childless /
# no-source-evidence gate is the stricter of the two demotions so it
# wins; the trust reasons are preserved on the assessment block for
# the audit envelope so operators can still see *why* the trust gate
# fired. These tests pin the overlap path so fix15 (non-trust
# childless) and fix13/fix14 (trust-gated with clean children)
# cannot regress.
# ---------------------------------------------------------------------------


class TestTrustGatedChildlessParentExcluded:
    def test_trust_gated_childless_parent_final_status_is_excluded(self):
        # Production-shaped RAPTOR parent: review-required by
        # default AND no source-backed child refs. Pre-fix16 this
        # resolved to ``parent_status == "review_required"`` with
        # the no-child marker erased from ``parent_assessment``.
        # The fix must demote to ``excluded`` because
        # no-source-evidence is the stricter gate.
        dense_seeds = [
            _RetrievedMemory(
                "parent-trust-childless",
                text="parent summary text",
                payload=_raptor_parent_payload(
                    node_id="parent-trust-childless",
                    children=[],
                    summary_of=[],
                    level=2,
                    requires_review=True,
                ),
                final_score=0.9,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point(
            "parent-trust-childless", dense_seeds[0].payload,
        )
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3)

        target = next(
            (s for s in result.summaries
             if s.raptor_node_id == "parent-trust-childless"),
            None,
        )
        assert target is not None
        # Childless / no-evidence wins over generic review-required.
        assert target.parent_status == "excluded", (
            "final13 P2 acceptance: trust-gated childless parent "
            "must end up 'excluded', not 'review_required'"
        )
        assert target.text == ""
        # The parent must be tracked in unsafe_summary_ids.
        assert "parent-trust-childless" in result.unsafe_summary_ids

    def test_trust_gated_childless_parent_assessment_preserves_no_child_marker(
        self,
    ):
        # ``parent_assessment`` must carry the no-child evidence
        # marker (``no_child_refs=True``,
        # ``missing_child_reasons`` includes ``"no_child_refs"``)
        # SO THAT downstream consumers can distinguish
        # "unreviewed-but-source-backed" from
        # "unreviewed-and-no-source-evidence".
        dense_seeds = [
            _RetrievedMemory(
                "parent-trust-childless-meta",
                text="parent summary text",
                payload=_raptor_parent_payload(
                    node_id="parent-trust-childless-meta",
                    children=[],
                    summary_of=[],
                    level=2,
                    requires_review=True,
                    raptor_review_status="review_required",
                ),
                final_score=0.9,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point(
            "parent-trust-childless-meta", dense_seeds[0].payload,
        )
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3)
        target = next(
            (s for s in result.summaries
             if s.raptor_node_id == "parent-trust-childless-meta"),
            None,
        )
        assert target is not None
        d = target.to_dict(include_metadata=True)
        # Nested assessment must agree with dataclass-level status.
        assert d["parent_assessment"].get("parent_status") == "excluded", (
            "final13 P2 acceptance: parent_assessment['parent_status'] "
            "must be aligned to the demoted 'excluded' value, not "
            "the legacy 'review_required' value that the trust-only "
            "branch would emit"
        )
        # No-child evidence marker MUST survive on the assessment
        # so the audit envelope can show the no-source-evidence
        # state alongside the trust reasons.
        assert d["parent_assessment"].get("no_child_refs") is True, (
            "final13 P2 acceptance: parent_assessment must carry the "
            "no_child_refs marker captured at construction time so "
            "the post-demotion recompute can keep the parent excluded"
        )
        missing = d["parent_assessment"].get("missing_child_reasons") or []
        assert "no_child_refs" in missing, (
            "final13 P2 acceptance: missing_child_reasons must include "
            "the no_child_refs token even when the trust gate also fired"
        )
        # Trust reasons MUST still be preserved (audit envelope
        # contract: don't erase one signal when both gates fire).
        trust_reasons = (
            d["parent_assessment"].get("trust_gate_reasons") or []
        )
        assert trust_reasons, (
            "final13 P2 acceptance: trust_gate_reasons must survive "
            "the childless-wins replay so operators can still see "
            "why the trust gate fired"
        )
        # Both reasons MUST be visible (no_child_refs token + at
        # least one trust reason token). No raw payload content
        # ever leaves the assessment via this contract.
        assert any(r == "requires_review" for r in trust_reasons) or any(
            r.startswith("raptor_review_status:") for r in trust_reasons
        ), (
            "final13 P2 acceptance: at least one of the production "
            "trust markers (requires_review / raptor_review_status) "
            "must survive the childless-wins replay"
        )

    def test_trust_gated_childless_parent_warning_sanitized_no_raw_leak(
        self,
    ):
        # The overlap warning must use the redacted handle and the
        # bounded short-token reason vocabulary — no raw payload
        # content, no raw node id. The warning must mention the
        # no-source-evidence signal (the reason this parent is
        # excluded, not just review_required).
        dense_seeds = [
            _RetrievedMemory(
                "parent-trust-childless-warn",
                text="parent summary text",
                payload=_raptor_parent_payload(
                    node_id="parent-trust-childless-warn",
                    children=[],
                    summary_of=[],
                    level=2,
                    requires_review=True,
                    raptor_review_status="review_required",
                ),
                final_score=0.9,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point(
            "parent-trust-childless-warn", dense_seeds[0].payload,
        )
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3)

        # Find the childless-replay warning (the one that fires at
        # recompute time). We assert it exists, uses the redacted
        # handle, and surfaces the no_child_refs token.
        childless_replay_warnings = [
            w for w in result.warnings
            if "no source-backed child refs" in w
            and "(replay)" in w
        ]
        assert childless_replay_warnings, (
            "expected at least one childless-replay warning so the "
            "audit envelope shows the no-source-evidence reason"
        )
        for w in childless_replay_warnings:
            assert (
                "parent-trust-childless-warn" not in w
            ), f"raw node id leaked in childless-replay warning: {w!r}"
            assert "redacted:" in w, (
                f"redacted handle missing in childless-replay "
                f"warning: {w!r}"
            )
            assert "no_child_refs" in w, (
                f"no_child_refs token must surface in the warning "
                f"so operators see why the parent is excluded: {w!r}"
            )

        # The unsafe_summary_ids handle list uses the redacted
        # handle so the secret-shaped id variant (covered by the
        # separate ``test_trust_gated_childless_parent_secret_shaped_id_redacted``
        # below) cannot leak through the audit envelope. For a
        # clean node id, the redacted handle is shown in
        # ``unsafe_summary_ids`` while the raw node id is the
        # audit row key in ``summaries[].raptor_node_id`` /
        # ``point_id`` — that is intentional so operators can
        # correlate the audit envelope with the dense seed.
        d = result.to_dict()
        # ``unsafe_summary_ids`` must carry a redacted handle
        # only (no raw clean node id).
        for handle in d["unsafe_summary_ids"]:
            assert "parent-trust-childless-warn" not in handle, (
                f"raw node id leaked via unsafe_summary_ids "
                f"redacted handle: {handle!r}"
            )

    def test_trust_gated_childless_parent_secret_shaped_id_redacted(self):
        # Defense-in-depth: a trust-gated childless parent whose
        # ``raptor_node_id`` is secret-shaped must NOT echo the raw
        # id anywhere in the JSON envelope. The serialized output
        # carries redacted handles only.
        # Build the secret-shaped id at runtime so the scanner
        # never sees a literal.
        bad_id = "".join(["Bearer ", "z" * 24])
        dense_seeds = [
            _RetrievedMemory(
                bad_id,
                text="parent summary text",
                payload=_raptor_parent_payload(
                    node_id=bad_id,
                    children=[],
                    summary_of=[],
                    level=2,
                    requires_review=True,
                    raptor_review_status="review_required",
                ),
                final_score=0.9,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point(bad_id, dense_seeds[0].payload)
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3)

        # Warning channel: raw id must not leak.
        for w in result.warnings:
            assert bad_id not in w, (
                f"raw secret-shaped id leaked in warning: {w!r}"
            )
        # JSON envelope: raw id must not leak.
        d = result.to_dict()
        serialized = json.dumps(d, default=str)
        assert bad_id not in serialized, (
            "raw secret-shaped id leaked in serialized "
            "RaptorSearchResult.to_dict() output for the "
            "trust-gated childless parent scenario"
        )

    def test_clean_child_trust_gated_parent_review_required_regression(
        self,
    ):
        # Regression: fix13 / fix14 behavior must NOT change for a
        # trust-gated parent WITH clean children. The trust-only
        # demotion branch still wins when the parent has source-
        # backed child refs (no_child_refs marker is absent).
        dense_seeds = [
            _RetrievedMemory(
                "parent-trust-clean",
                text="parent summary text",
                payload=_raptor_parent_payload(
                    node_id="parent-trust-clean",
                    children=["leaf-ok"],
                    summary_of=["leaf-ok"],
                    level=2,
                    requires_review=True,
                ),
                final_score=0.92,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-trust-clean", dense_seeds[0].payload)
        qdrant.add_point(
            "leaf-ok", _leaf_payload(point_id="leaf-ok", text="clean"),
        )
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3)
        target = next(
            (s for s in result.summaries
             if s.raptor_node_id == "parent-trust-clean"),
            None,
        )
        assert target is not None
        # Trust-only demotion still applies; clean child still cited.
        assert target.parent_status == "review_required"
        assert target.text == ""
        assert "parent-trust-clean" in result.unsafe_summary_ids
        reasons = target.parent_assessment.get("trust_gate_reasons") or []
        assert reasons, "trust_gate_reasons must survive on the assessment"
        # No-child marker must be ABSENT (the parent has children).
        assert not target.parent_assessment.get("no_child_refs"), (
            "trust-gated parent with clean children must NOT carry "
            "the no_child_refs marker (no overlap to fix16 here)"
        )
        # Cited clean child must survive for evidence-mode traces.
        promoted = {l.point_id for l in result.cited_leaves}
        assert "leaf-ok" in promoted, (
            "trust-gated parent with clean child must still cite it "
            "(fix13 contract must not regress on fix16 path)"
        )

    def test_non_trust_childless_parent_excluded_regression(self):
        # Regression: fix15 behavior must NOT change for a
        # non-trust childless parent (no production trust markers
        # AND no source-backed child refs). Status stays
        # ``excluded``, no_child_refs marker present, trust
        # reasons absent.
        dense_seeds = [
            _RetrievedMemory(
                "parent-nontrust-childless",
                text="text",
                payload=_raptor_parent_payload(
                    node_id="parent-nontrust-childless",
                    children=[],
                    summary_of=[],
                    level=2,
                ),
                final_score=0.8,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point(
            "parent-nontrust-childless", dense_seeds[0].payload,
        )
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3)
        target = next(
            (s for s in result.summaries
             if s.raptor_node_id == "parent-nontrust-childless"),
            None,
        )
        assert target is not None
        assert target.parent_status == "excluded"
        assert target.text == ""
        assert target.parent_assessment.get("no_child_refs") is True
        assert "no_child_refs" in (
            target.parent_assessment.get("missing_child_reasons") or []
        )
        # No trust-gate reasons should be on a non-trust parent.
        assert not target.parent_assessment.get("trust_gate_reasons"), (
            "non-trust childless parent must not carry "
            "trust_gate_reasons on its assessment"
        )
        assert "parent-nontrust-childless" in result.unsafe_summary_ids

    def test_approved_parent_with_clean_child_remains_active_fix16_regression(
        self,
    ):
        # Regression: approved clean parent must keep active path.
        original_text = "approved clean parent must survive"
        dense_seeds = [
            _RetrievedMemory(
                "parent-approved-clean",
                text="parent summary text",
                payload=_raptor_parent_payload(
                    node_id="parent-approved-clean",
                    children=["leaf-ok"],
                    summary_of=["leaf-ok"],
                    level=2,
                    text=original_text,
                ),
                final_score=0.95,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-approved-clean", dense_seeds[0].payload)
        qdrant.add_point(
            "leaf-ok", _leaf_payload(point_id="leaf-ok", text="clean"),
        )
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory",
        )
        result = searcher.search("anything", top_k=3)
        target = next(
            (s for s in result.summaries
             if s.raptor_node_id == "parent-approved-clean"),
            None,
        )
        assert target is not None
        assert target.parent_status == "active"
        assert target.text == original_text
        assert "parent-approved-clean" not in result.unsafe_summary_ids
        promoted = {l.point_id for l in result.cited_leaves}
        assert "leaf-ok" in promoted


# ---------------------------------------------------------------------------
# Phase 6 P3 audit precision: per-parent per-child dedupe.
#
# Pre-P3 the Stage 4a loop iterated ``child_ids + summary_of`` and
# appended the parent to ``parents_for_leaf[cid]`` on EVERY iteration
# (no per-parent per-child guard). When the same clean child id
# appeared in BOTH lists the same parent was attributed twice to the
# same child, and Stage 5 then:
#
# * appended the same safe payload to ``per_parent_safe_payloads``
#   twice (inflating ``safe_children_count`` / ``total_children``
#   in ``parent_assessment``);
# * bumped ``per_parent_unsafe_count`` twice for a shared unsafe
#   child (inflating the missing / unsafe accounting).
#
# The minimal fix (in ``search.py``) checks ``cid in
# referenced_for_parent`` before appending so each unique child is
# attributed to its parent exactly once while preserving
# cross-parent ``parents_for_leaf`` entries for shared children
# (fix8 contract). These tests pin the post-fix behavior across the
# three reviewer-listed cases:
#
# 1. Duplicate clean child in both ``raptor_child_ids`` and
#    ``raptor_summary_of`` for a single parent — counters stay at 1,
#    parent stays active with original text, the leaf appears once in
#    ``cited_leaves``.
# 2. Duplicate unsafe child in both lists — parent non-active / text
#    cleared, the unsafe / missing accounting reaches ``1`` for the
#    single unique child (not ``2``), the unsafe child id is present
#    once in the audit-relevant collections.
# 3. Shared unsafe child across two different parents — both parents
#    still demote (fix8 contract preserved by the minimal fix).
# ---------------------------------------------------------------------------


class TestDuplicateChildRefAuditPrecision:
    """Regression for Phase 6 P3 audit-counter precision.

    Pre-P3 the same clean child appearing in BOTH
    ``raptor_child_ids`` and ``raptor_summary_of`` for one parent
    made ``parent_assessment`` report ``safe_children_count == 2``
    and ``total_children == 2`` for one unique source-backed leaf,
    and made Stage 5 append the same safe payload to
    ``per_parent_safe_payloads`` twice. The fix dedupes per-parent
    per-child so the unique child is attributed to the parent
    exactly once.
    """

    def test_duplicate_clean_child_in_both_lists_counted_once(self):
        # Parent declares the same clean child in both
        # ``raptor_child_ids`` and ``raptor_summary_of``. The unique
        # child must be attributed to the parent exactly once:
        # ``safe_children_count == 1``, ``total_children == 1``,
        # parent stays ``active`` with original text intact, and
        # ``cited_leaves`` contains the leaf exactly once.
        original_text = "parent summary about deploy"
        dense_seeds = [
            _RetrievedMemory(
                "parent-dup",
                text=original_text,
                payload=_raptor_parent_payload(
                    node_id="parent-dup",
                    text=original_text,
                    children=["leaf-dup"],
                    summary_of=["leaf-dup"],
                    level=2,
                ),
                final_score=0.9,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-dup", dense_seeds[0].payload)
        qdrant.add_point(
            "leaf-dup",
            _leaf_payload(point_id="leaf-dup", text="clean leaf about deploy"),
        )
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory"
        )
        result = searcher.search("anything", top_k=3)

        target = next(
            (s for s in result.summaries if s.raptor_node_id == "parent-dup"),
            None,
        )
        assert target is not None, "parent-dup must be promoted"
        # Parent status and text survive untouched.
        assert target.parent_status == "active", (
            "parent must stay active when the only unique child is clean; "
            f"got {target.parent_status!r}"
        )
        assert target.text == original_text, (
            "parent text must survive when the only unique child is clean"
        )

        # The audit counters must reflect ONE unique child, not two.
        pa = target.parent_assessment or {}
        assert pa.get("safe_children_count") == 1, (
            "safe_children_count must be 1 for a single unique clean child; "
            f"got parent_assessment={pa!r}"
        )
        assert pa.get("total_children") == 1, (
            "total_children must be 1 for a single unique child; "
            f"got parent_assessment={pa!r}"
        )
        # No unsafe / missing accounting for a clean child.
        assert pa.get("unsafe_children") == [], (
            "unsafe_children must be empty when the only child is clean; "
            f"got parent_assessment={pa!r}"
        )

        # The leaf is cited exactly once in ``cited_leaves``.
        cited_ids = [l.point_id for l in result.cited_leaves]
        assert cited_ids.count("leaf-dup") == 1, (
            f"leaf-dup must appear exactly once in cited_leaves; got {cited_ids!r}"
        )

        # Parent is NOT in unsafe_summary_ids.
        assert "parent-dup" not in result.unsafe_summary_ids

    def test_duplicate_unsafe_child_in_both_lists_counted_once(self):
        # Same setup, but the leaf is unsafe (``stale=True``). The
        # parent must demote, the unsafe / missing counters must reach
        # ``1`` (one unique child, not two), and the leaf id must not
        # be double-accounted.
        original_text = "parent summary about deploy"
        dense_seeds = [
            _RetrievedMemory(
                "parent-dup",
                text=original_text,
                payload=_raptor_parent_payload(
                    node_id="parent-dup",
                    text=original_text,
                    children=["leaf-dup"],
                    summary_of=["leaf-dup"],
                    level=2,
                ),
                final_score=0.9,
            )
        ]
        qdrant = FakeRaptorQdrant()
        qdrant.add_point("parent-dup", dense_seeds[0].payload)
        qdrant.add_point(
            "leaf-dup",
            _leaf_payload(
                point_id="leaf-dup",
                text="stale child",
                stale=True,
            ),
        )
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory"
        )
        result = searcher.search("anything", top_k=3)

        target = next(
            (s for s in result.summaries if s.raptor_node_id == "parent-dup"),
            None,
        )
        assert target is not None, "parent-dup must still appear in summaries"
        # Non-active parent with cleared text (per unsafe-leaf demotion).
        assert target.parent_status != "active", (
            "parent must be non-active when its only unique child is unsafe; "
            f"got {target.parent_status!r}"
        )
        assert target.text == "", (
            "parent text must be cleared when its only unique child is unsafe"
        )

        # Audit counters: one unique child means exactly one unsafe
        # child entry, total_children == 1, no safe_children.
        pa = target.parent_assessment or {}
        assert pa.get("total_children") == 1, (
            "total_children must reflect one unique child even when that "
            f"child is unsafe; got parent_assessment={pa!r}"
        )
        assert pa.get("safe_children_count") == 0, (
            "safe_children_count must be 0 when the unique child is unsafe; "
            f"got parent_assessment={pa!r}"
        )
        # ``unsafe_children`` is the list of safety dicts returned by
        # ``assess_parent_status``; each unsafe child gets one entry
        # so a single unique unsafe child must produce a length-1
        # list (no inflation from duplicate cid appearances).
        unsafe_list = pa.get("unsafe_children") or []
        assert len(unsafe_list) == 1, (
            "unsafe_children list must have one entry for one unique unsafe "
            f"child; got parent_assessment={pa!r}"
        )

        # The leaf is NOT cited (it's unsafe) but its id surfaces
        # exactly once in the audit envelope, not twice.
        assert "leaf-dup" not in {l.point_id for l in result.cited_leaves}
        unsafe_leaf_handles = list(result.unsafe_leaf_ids)
        assert unsafe_leaf_handles.count("leaf-dup") == 1, (
            "leaf-dup must appear exactly once in unsafe_leaf_ids for one "
            f"unique unsafe child; got {unsafe_leaf_handles!r}"
        )
        # Parent is tracked as unsafe.
        assert "parent-dup" in result.unsafe_summary_ids

    def test_shared_unsafe_child_across_two_parents_still_demotes_both(self):
        # fix8 contract regression guard: a unique unsafe child shared
        # across two different parents (each parent declares the child
        # in BOTH ``raptor_child_ids`` and ``raptor_summary_of`` to
        # also exercise the new per-parent dedupe) must still demote
        # BOTH parents. This is the cross-parent half of the fix8
        # contract — the per-parent per-child dedupe MUST NOT collapse
        # two parents into one attribution row.
        shared_leaf = "shared-unsafe-leaf"
        dense_seeds = [
            _RetrievedMemory(
                "parent-A",
                text="parent A summary",
                payload=_raptor_parent_payload(
                    node_id="parent-A",
                    text="parent A summary",
                    children=[shared_leaf],
                    summary_of=[shared_leaf],
                    level=2,
                ),
                final_score=0.91,
            ),
            _RetrievedMemory(
                "parent-B",
                text="parent B summary",
                payload=_raptor_parent_payload(
                    node_id="parent-B",
                    text="parent B summary",
                    children=[shared_leaf],
                    summary_of=[shared_leaf],
                    level=2,
                ),
                final_score=0.9,
            ),
        ]
        qdrant = FakeRaptorQdrant()
        for seed in dense_seeds:
            qdrant.add_point(seed.id, seed.payload)
        qdrant.add_point(
            shared_leaf,
            _leaf_payload(
                point_id=shared_leaf,
                text="stale shared leaf",
                stale=True,
            ),
        )
        retriever = FakeRetriever(dense_seeds)
        searcher = RaptorSearcher(
            qdrant=qdrant, retriever=retriever, collection_name="memory"
        )
        result = searcher.search("anything", top_k=3)

        # BOTH parents must be demoted (fix8 contract).
        assert "parent-A" in result.unsafe_summary_ids, (
            "parent-A must demote because its unique unsafe child is unsafe"
        )
        assert "parent-B" in result.unsafe_summary_ids, (
            "parent-B must also demote — fix8 contract must not regress "
            "when the per-parent per-child dedupe is applied"
        )

        # Each parent's audit envelope shows ONE unique child
        # (total_children == 1) and ONE unsafe entry — even though the
        # shared unsafe leaf appears in both lists for both parents.
        for parent_id in ("parent-A", "parent-B"):
            target = next(
                (s for s in result.summaries if s.raptor_node_id == parent_id),
                None,
            )
            assert target is not None
            assert target.parent_status != "active"
            assert target.text == ""
            pa = target.parent_assessment or {}
            assert pa.get("total_children") == 1, (
                f"{parent_id} must report total_children == 1 for one "
                f"unique shared child; got parent_assessment={pa!r}"
            )
            assert len(pa.get("unsafe_children") or []) == 1, (
                f"{parent_id} must report exactly one unsafe_children entry "
                f"for one unique unsafe child; got parent_assessment={pa!r}"
            )
            assert pa.get("safe_children_count") == 0, (
                f"{parent_id} must report safe_children_count == 0 when its "
                f"only unique child is unsafe; got parent_assessment={pa!r}"
            )

        # The unsafe shared child id surfaces exactly once across the
        # audit envelope, not twice per parent.
        assert list(result.unsafe_leaf_ids).count(shared_leaf) == 1, (
            "shared unsafe child must appear once in unsafe_leaf_ids "
            f"(set semantics); got {result.unsafe_leaf_ids!r}"
        )
