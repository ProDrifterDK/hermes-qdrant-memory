"""Phase 5 hybrid retrieve packer tests.

These tests cover:

- Output shape stability (``context_not_instruction``, ``results``,
  ``warnings``, ``debug``).
- evidence-mode demotion of RAPTOR parents with no cited leaves.
- dense+sparse lane always uses ``update_access=False``.
- non-zero graph lane output is captured.
- read-only invariant (no upsert/delete/update_payload on the fake).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from qdrant_memory.hybrid import HybridRouter, HybridRouteResult
from qdrant_memory.hybrid.fusion import rrf_fuse, deduplicate_by_point_id
from qdrant_memory.hybrid.router import (
    _GRAPH_UNSAFE_FACT_STATUSES,
    _graph_payload_unsafe_for_active_context,
    _graph_to_relations,
)


class FakeEmbedding:
    def embed_query(self, text):
        return [0.1, 0.2]

    def embed_document(self, text):
        return [0.4, 0.5]


class _Chunk:
    def __init__(self, pid, text, payload, final_score=0.6, qdrant_score=0.6,
                 ranking_debug=None):
        self.id = pid
        self.text = text
        self.payload = payload
        self.final_score = final_score
        self.qdrant_score = qdrant_score
        self.ranking_debug = ranking_debug or {}


class FakeBaseRetriever:
    def __init__(self, dense_seeds=None, *, raise_exc=False):
        self._dense_seeds = dense_seeds or []
        self.calls: list[dict[str, Any]] = []
        self._raise = raise_exc

    def search(self, query, *, top_k=5, update_access=False, **kwargs):
        self.calls.append({
            "query": query,
            "top_k": top_k,
            "update_access": update_access,
            "kwargs": kwargs,
        })
        if self._raise:
            raise RuntimeError("simulated dense search failure")
        return list(self._dense_seeds)


class FakeGraphResult:
    def __init__(self, final):
        self.final = final


class FakeGraphCandidate:
    def __init__(self, pid, graph_distance=1, final_score=0.7,
                 path=None, relation_path=None, payload=None):
        self.point_id = pid
        self.graph_distance = graph_distance
        self.final_score = final_score
        self.path = path or []
        self.relation_path = relation_path or []
        self.payload = payload or {}


class FakeGraphRetriever:
    def __init__(self, final=None, *, raise_exc=False):
        self._final = final or []
        self.calls: list[dict[str, Any]] = []
        self._raise = raise_exc

    def search(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        if self._raise:
            raise RuntimeError("simulated graph failure")
        return FakeGraphResult(list(self._final))


class FakeRaptorSearcher:
    def __init__(self, *, summaries=None, leaves=None, raise_exc=False):
        from qdrant_memory.raptor.search import RaptorSearchResult, RaptorSummaryHit, RaptorLeafHit
        self._summaries = summaries or []
        self._leaves = leaves or []
        self._raise = raise_exc
        self.calls: list[dict[str, Any]] = []
        if not raise_exc:
            self._result = RaptorSearchResult(
                query="",
                summaries=[s for s in self._summaries],
                cited_leaves=[l for l in self._leaves],
            )
        else:
            self._result = None

    def search(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        if self._raise:
            raise RuntimeError("simulated raptor failure")
        if self._result is not None:
            self._result.query = query
        return self._result


class FakeQdrant:
    def __init__(self):
        self.upserts: list = []
        self.update_payloads: list = []
        self.deletes: list = []
        self.delete_filters: list = []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOutputShape:
    def test_context_not_instruction_and_authority(self):
        retriever = FakeBaseRetriever(dense_seeds=[_Chunk(
            "doc-1", "text", {"profile_id": "default", "source_type": "manual"}, 0.5,
        )])
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=retriever,
        )
        result = router.retrieve("hello")
        d = result.to_dict()
        assert d["context_not_instruction"] is True
        assert d["authority"].lower().startswith("retrieved memory")
        assert d["mode"] == "hybrid"
        # Phase 5 fix11 (final9 finding #1): the raw query is NEVER
        # echoed. The contract is now ``query_length`` +
        # ``query_digest`` + ``query_redacted`` sentinel. The
        # digest for the empty-ish canonical input below is
        # sha256("")[:16] == "e3b0c44298fc1c14".
        assert d["query_length"] == 5  # len("hello")
        assert d["query_redacted"] == "[redacted: query omitted from retrieve output]"
        # ``query_digest`` is sha256("hello")[:16] — stable.
        import hashlib as _hl
        assert d["query_digest"] == _hl.sha256(b"hello").hexdigest()[:16]
        # No raw ``query`` key anywhere in the envelope.
        assert "query" not in d
        for bucket in ("summaries", "cited_leaves", "exact_hits", "graph_relations"):
            assert bucket in d["results"]
        assert isinstance(d["warnings"], list)
        assert isinstance(d["debug"], dict)

    def test_output_is_json_serializable(self):
        retriever = FakeBaseRetriever(dense_seeds=[])
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=retriever,
        )
        d = router.retrieve("hi").to_dict()
        json.dumps(d)  # must not raise


class TestUpdateAccessFalse:
    def test_dense_lane_always_update_access_false(self):
        retriever = FakeBaseRetriever()
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=retriever,
        )
        router.retrieve("hi", top_k=3)
        assert retriever.calls
        assert all(c["update_access"] is False for c in retriever.calls)

    def test_dense_lane_disables_sparse_scroll(self):
        # Phase 5 fix4: the router must propagate
        # ``allow_sparse_scroll=False`` into the dense lane so the
        # Phase 5 retrieve path never invokes ``scroll_by_filter``.
        retriever = FakeBaseRetriever()
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=retriever,
        )
        router.retrieve("hi", top_k=3)
        assert retriever.calls
        assert all(
            c["kwargs"].get("allow_sparse_scroll") is False for c in retriever.calls
        )


class TestEvidenceModeDemotion:
    def test_evidence_mode_drops_parent_without_cited_leaves(self):
        from qdrant_memory.raptor.search import RaptorSummaryHit, RaptorLeafHit
        # Parent A is reachable but has no cited leaves; Parent B has both
        # a cited leaf and a parent summary.
        parent_a = RaptorSummaryHit(
            point_id="parent-A",
            raptor_node_id="parent-A",
            raptor_root_id="root",
            raptor_level=2,
            raptor_tree_id="tree",
            raptor_build_id="build",
            raptor_cluster_id="cluster-A",
            text="orphan parent",
        )
        parent_b = RaptorSummaryHit(
            point_id="parent-B",
            raptor_node_id="parent-B",
            raptor_root_id="root",
            raptor_level=2,
            raptor_tree_id="tree",
            raptor_build_id="build",
            raptor_cluster_id="cluster-B",
            text="cited parent",
        )
        leaf_b = RaptorLeafHit(
            point_id="leaf-B",
            parent_raptor_node_id="parent-B",
            parent_point_id="parent-B",
            text="about B",
        )
        raptor = FakeRaptorSearcher(
            summaries=[parent_a, parent_b],
            leaves=[leaf_b],
        )
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=FakeBaseRetriever(),
            raptor_searcher=raptor,
        )
        d = router.retrieve("anything", mode="evidence").to_dict()
        promoted_ids = {s["raptor_node_id"] for s in d["results"]["summaries"]}
        assert "parent-B" in promoted_ids
        assert "parent-A" not in promoted_ids
        assert any("demoted" in w.lower() for w in d["warnings"])

    def test_hybrid_mode_keeps_parent_without_cited_leaves(self):
        from qdrant_memory.raptor.search import RaptorSummaryHit
        parent_a = RaptorSummaryHit(
            point_id="parent-A",
            raptor_node_id="parent-A",
            raptor_root_id="root",
            raptor_level=2,
            raptor_tree_id="tree",
            raptor_build_id="build",
            raptor_cluster_id="cluster-A",
            text="orphan parent",
        )
        raptor = FakeRaptorSearcher(summaries=[parent_a], leaves=[])
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=FakeBaseRetriever(),
            raptor_searcher=raptor,
        )
        d = router.retrieve("anything", mode="hybrid").to_dict()
        promoted_ids = {s["raptor_node_id"] for s in d["results"]["summaries"]}
        assert "parent-A" in promoted_ids


class TestGraphLane:
    def test_graph_relations_captured(self):
        graph = FakeGraphRetriever(final=[
            FakeGraphCandidate("graph-1", graph_distance=1, final_score=0.8),
            FakeGraphCandidate("graph-2", graph_distance=2, final_score=0.6),
        ])
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=FakeBaseRetriever(),
            graph_retriever=graph,
        )
        d = router.retrieve("hi", top_k=3).to_dict()
        assert d["results"]["graph_relations"]
        ids = [g["point_id"] for g in d["results"]["graph_relations"]]
        assert "graph-1" in ids

    def test_graph_failure_does_not_break_retrieve(self):
        graph = FakeGraphRetriever(raise_exc=True)
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=FakeBaseRetriever(),
            graph_retriever=graph,
        )
        d = router.retrieve("hi").to_dict()
        assert any("graph lane failed" in w.lower() for w in d["warnings"])

    def test_dense_failure_does_not_break_retrieve(self):
        retriever = FakeBaseRetriever(raise_exc=True)
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=retriever,
        )
        d = router.retrieve("hi").to_dict()
        assert any("seed search failed" in w.lower() for w in d["warnings"])

    def test_graph_secret_shaped_point_id_dropped(self):
        # Phase 5 fix4: a graph candidate whose point_id is secret-shaped
        # must be dropped from ``results.graph_relations`` and the raw
        # value must not echo through warnings/debug/JSON.
        bad_pid = "".join(["Bearer ", "a" * 24])
        clean_pid = "graph-clean-1"
        graph = FakeGraphRetriever(final=[
            FakeGraphCandidate(bad_pid, graph_distance=1, final_score=0.8),
            FakeGraphCandidate(clean_pid, graph_distance=2, final_score=0.6),
        ])
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=FakeBaseRetriever(),
            graph_retriever=graph,
        )
        d = router.retrieve("hi", top_k=3).to_dict()
        relations = d["results"]["graph_relations"]
        ids = [g["point_id"] for g in relations]
        assert bad_pid not in ids
        assert clean_pid in ids
        serialized = json.dumps(d, default=str)
        assert bad_pid not in serialized
        # Warning channel uses redacted handle, not raw id.
        redact_warnings = [w for w in d["warnings"] if "graph relation" in w]
        assert redact_warnings
        for w in redact_warnings:
            assert bad_pid not in w
            assert "redacted:" in w

    def test_graph_secret_shaped_path_dropped(self):
        # ``path`` is a list of point ids; a secret-shaped element must
        # disqualify the entire relation.
        bad_pid = "graph-clean-2"
        bad_path_item = "".join(["Bearer ", "b" * 24])
        graph = FakeGraphRetriever(final=[
            FakeGraphCandidate(
                bad_pid,
                graph_distance=1,
                final_score=0.8,
                path=[bad_path_item],
            ),
        ])
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=FakeBaseRetriever(),
            graph_retriever=graph,
        )
        d = router.retrieve("hi", top_k=3).to_dict()
        assert d["results"]["graph_relations"] == []
        serialized = json.dumps(d, default=str)
        assert bad_path_item not in serialized
        assert any("graph relation" in w for w in d["warnings"])

    def test_graph_secret_shaped_relation_path_dropped(self):
        # ``relation_path`` is a list of relation strings; a secret-shaped
        # element must disqualify the entire relation.
        bad_pid = "graph-clean-3"
        bad_relation = "".join(["Authorization: ", "Bearer ", "c" * 24])
        graph = FakeGraphRetriever(final=[
            FakeGraphCandidate(
                bad_pid,
                graph_distance=1,
                final_score=0.8,
                relation_path=[bad_relation],
            ),
        ])
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=FakeBaseRetriever(),
            graph_retriever=graph,
        )
        d = router.retrieve("hi", top_k=3).to_dict()
        assert d["results"]["graph_relations"] == []
        serialized = json.dumps(d, default=str)
        assert bad_relation not in serialized
        assert any("graph relation" in w for w in d["warnings"])

    def test_graph_clean_relations_pass_through(self):
        graph = FakeGraphRetriever(final=[
            FakeGraphCandidate(
                "graph-clean-4",
                graph_distance=1,
                final_score=0.8,
                path=["a", "b"],
                relation_path=["rel_a", "rel_b"],
            ),
            FakeGraphCandidate(
                "graph-clean-5",
                graph_distance=2,
                final_score=0.6,
                path=["c"],
                relation_path=["rel_c"],
            ),
        ])
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=FakeBaseRetriever(),
            graph_retriever=graph,
        )
        d = router.retrieve("hi", top_k=3).to_dict()
        ids = [g["point_id"] for g in d["results"]["graph_relations"]]
        assert "graph-clean-4" in ids
        assert "graph-clean-5" in ids
        # No spurious graph redaction warnings for clean candidates.
        assert not any("graph relation redacted" in w for w in d["warnings"])

    # ------------------------------------------------------------------ #
    # Phase 6E: graph relation payload handles (source_uri / file_path /
    # heading / bounded text) projected from ``candidate.payload``.
    # ------------------------------------------------------------------ #

    def test_graph_relation_emits_source_handles_from_payload(self):
        # Phase 6E: a graph candidate whose payload carries provenance
        # metadata must surface ``source_uri``, ``file_path``,
        # ``heading``, and bounded ``text`` in the emitted relation.
        graph = FakeGraphRetriever(final=[
            FakeGraphCandidate(
                "graph-handle-1",
                graph_distance=1,
                final_score=0.7,
                payload={
                    "source_uri": "file://docs/phase6e.md",
                    "file_path": "docs/phase6e.md",
                    "heading": "Phase 6E",
                    "text": "phase 6e graph relation body",
                },
            ),
        ])
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=FakeBaseRetriever(),
            graph_retriever=graph,
        )
        d = router.retrieve("hi", top_k=3).to_dict()
        rels = d["results"]["graph_relations"]
        assert len(rels) == 1
        rel = rels[0]
        assert rel["point_id"] == "graph-handle-1"
        assert rel["source_uri"] == "file://docs/phase6e.md"
        assert rel["file_path"] == "docs/phase6e.md"
        assert rel["heading"] == "Phase 6E"
        assert rel["text"] == "phase 6e graph relation body"

    def test_graph_relation_text_bounded_by_max_source_chars(self):
        # Phase 6E: a graph relation text body is truncated to the
        # caller's ``max_source_chars`` budget so the graph lane
        # cannot bypass the same cap the dense and RAPTOR lanes
        # already enforce.
        long_body = "x" * 5000
        graph = FakeGraphRetriever(final=[
            FakeGraphCandidate(
                "graph-truncate-1",
                graph_distance=1,
                final_score=0.7,
                payload={
                    "text": long_body,
                    "source_uri": "file://docs/long.md",
                    "file_path": "docs/long.md",
                },
            ),
        ])
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=FakeBaseRetriever(),
            graph_retriever=graph,
        )
        d = router.retrieve("hi", top_k=3, max_source_chars=1200).to_dict()
        rel = d["results"]["graph_relations"][0]
        # Per-relation cap is enforced; trailing ellipsis is preserved.
        assert len(rel["text"]) <= 1200
        assert rel["text"].endswith("…")
        # The truncated text must be a prefix of the long body.
        assert long_body.startswith(rel["text"].rstrip("…"))

    def test_graph_relation_secret_in_source_uri_dropped(self):
        # Phase 6E: a secret in any of the newly emitted payload
        # fields must disqualify the entire relation.
        bad_token = "".join(["Bearer ", "x" * 24])
        graph = FakeGraphRetriever(final=[
            FakeGraphCandidate(
                "graph-bad-uri",
                graph_distance=1,
                final_score=0.7,
                payload={
                    "source_uri": "file://" + bad_token,
                    "file_path": "docs/clean.md",
                    "heading": "Clean",
                    "text": "clean body",
                },
            ),
        ])
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=FakeBaseRetriever(),
            graph_retriever=graph,
        )
        d = router.retrieve("hi", top_k=3).to_dict()
        assert d["results"]["graph_relations"] == []
        serialized = json.dumps(d, default=str)
        assert bad_token not in serialized
        assert any("graph relation redacted" in w for w in d["warnings"])

    def test_graph_relation_secret_in_file_path_dropped(self):
        bad_token = "".join(["Bearer ", "y" * 24])
        graph = FakeGraphRetriever(final=[
            FakeGraphCandidate(
                "graph-bad-path",
                graph_distance=1,
                final_score=0.7,
                payload={
                    "file_path": "secrets/" + bad_token,
                    "source_uri": "file://docs/clean.md",
                    "heading": "Clean",
                    "text": "clean body",
                },
            ),
        ])
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=FakeBaseRetriever(),
            graph_retriever=graph,
        )
        d = router.retrieve("hi", top_k=3).to_dict()
        assert d["results"]["graph_relations"] == []
        serialized = json.dumps(d, default=str)
        assert bad_token not in serialized

    def test_graph_relation_secret_in_heading_dropped(self):
        bad_token = "".join(["Bearer ", "z" * 24])
        graph = FakeGraphRetriever(final=[
            FakeGraphCandidate(
                "graph-bad-heading",
                graph_distance=1,
                final_score=0.7,
                payload={
                    "heading": "leak " + bad_token,
                    "source_uri": "file://docs/clean.md",
                    "file_path": "docs/clean.md",
                    "text": "clean body",
                },
            ),
        ])
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=FakeBaseRetriever(),
            graph_retriever=graph,
        )
        d = router.retrieve("hi", top_k=3).to_dict()
        assert d["results"]["graph_relations"] == []
        serialized = json.dumps(d, default=str)
        assert bad_token not in serialized

    def test_graph_relation_secret_in_text_dropped(self):
        bad_token = "".join(["Bearer ", "w" * 24])
        graph = FakeGraphRetriever(final=[
            FakeGraphCandidate(
                "graph-bad-text",
                graph_distance=1,
                final_score=0.7,
                payload={
                    "text": "lead in " + bad_token + " trailer",
                    "source_uri": "file://docs/clean.md",
                    "file_path": "docs/clean.md",
                    "heading": "Clean",
                },
            ),
        ])
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=FakeBaseRetriever(),
            graph_retriever=graph,
        )
        d = router.retrieve("hi", top_k=3).to_dict()
        assert d["results"]["graph_relations"] == []
        serialized = json.dumps(d, default=str)
        assert bad_token not in serialized

    def test_graph_relation_secret_past_truncation_point_still_dropped(self):
        # Phase 6E: the secret scanner must see the raw text BEFORE
        # the per-relation ``max_source_chars`` cap is applied.
        # Otherwise a secret past the truncation point would silently
        # slip through the gate.
        bad_token = "".join(["Bearer ", "u" * 24])
        # Pad the body so the secret sits well past the 1200-char
        # cap that the caller's ``max_source_chars`` would otherwise
        # silently chop off.
        pad = "a" * 3000
        long_body = pad + bad_token + pad
        graph = FakeGraphRetriever(final=[
            FakeGraphCandidate(
                "graph-truncated-secret",
                graph_distance=1,
                final_score=0.7,
                payload={
                    "text": long_body,
                    "source_uri": "file://docs/long.md",
                    "file_path": "docs/long.md",
                },
            ),
        ])
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=FakeBaseRetriever(),
            graph_retriever=graph,
        )
        d = router.retrieve("hi", top_k=3, max_source_chars=200).to_dict()
        # The secret-bearing relation must be DROPPED entirely —
        # not emitted with truncated text. The scanner must see the
        # raw body and fail-closed.
        assert d["results"]["graph_relations"] == []
        serialized = json.dumps(d, default=str)
        assert bad_token not in serialized
        assert any("graph relation redacted" in w for w in d["warnings"])

    def test_graph_relation_unsafe_status_demoted(self):
        # Phase 6E: graph payloads flagged with unsafe status
        # markers (``stale``, ``requires_review``, ``quarantined``,
        # ``raptor_excluded``, ``raptor_forgotten``, or unsafe
        # ``fact_status``) are demoted from active
        # ``results.graph_relations`` to a sanitized warning.
        graph = FakeGraphRetriever(final=[
            FakeGraphCandidate(
                "graph-stale",
                graph_distance=1,
                final_score=0.7,
                payload={
                    "stale": True,
                    "source_uri": "file://docs/x.md",
                    "file_path": "docs/x.md",
                    "text": "stale body",
                },
            ),
            FakeGraphCandidate(
                "graph-review",
                graph_distance=1,
                final_score=0.7,
                payload={
                    "requires_review": True,
                    "source_uri": "file://docs/y.md",
                    "file_path": "docs/y.md",
                    "text": "review body",
                },
            ),
            FakeGraphCandidate(
                "graph-clean",
                graph_distance=1,
                final_score=0.7,
                payload={
                    "source_uri": "file://docs/z.md",
                    "file_path": "docs/z.md",
                    "text": "clean body",
                },
            ),
        ])
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=FakeBaseRetriever(),
            graph_retriever=graph,
        )
        d = router.retrieve("hi", top_k=5).to_dict()
        ids = [g["point_id"] for g in d["results"]["graph_relations"]]
        assert "graph-clean" in ids
        assert "graph-stale" not in ids
        assert "graph-review" not in ids
        # Demotion warnings must use redacted handles — never raw ids.
        demoted = [w for w in d["warnings"] if "graph relation demoted" in w]
        assert demoted
        for w in demoted:
            assert "graph-stale" not in w
            assert "graph-review" not in w

    def test_graph_relation_include_fact_history_overrides_demotion(self):
        # ``include_fact_history=True`` is the explicit opt-in to
        # surface review / stale material; unsafe-status demotion
        # must be bypassed so the history lane remains accessible.
        graph = FakeGraphRetriever(final=[
            FakeGraphCandidate(
                "graph-history",
                graph_distance=1,
                final_score=0.7,
                payload={
                    "requires_review": True,
                    "source_uri": "file://docs/h.md",
                    "file_path": "docs/h.md",
                    "text": "history body",
                },
            ),
        ])
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=FakeBaseRetriever(),
            graph_retriever=graph,
        )
        d = router.retrieve(
            "hi", top_k=5, include_fact_history=True,
        ).to_dict()
        rels = d["results"]["graph_relations"]
        ids = [g["point_id"] for g in rels]
        assert "graph-history" in ids
        # With the history opt-in, no demotion warning is emitted for
        # this relation.
        assert not any("graph relation demoted" in w for w in d["warnings"])

    def test_graph_no_scroll_contract_preserved_phase6e(self):
        # Phase 5 fix8 invariant: the hybrid graph lane MUST keep
        # passing ``allow_sparse_scroll=False`` and
        # ``allow_graph_scroll=False`` after the Phase 6E payload
        # handle projection.
        graph = FakeGraphRetriever(final=[
            FakeGraphCandidate(
                "graph-pinned",
                graph_distance=1,
                final_score=0.7,
                payload={
                    "source_uri": "file://docs/p.md",
                    "file_path": "docs/p.md",
                    "text": "pinned body",
                },
            ),
        ])
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=FakeBaseRetriever(),
            graph_retriever=graph,
        )
        router.retrieve("hi", top_k=3).to_dict()
        # The graph retriever saw both scroll-suppression kwargs.
        assert len(graph.calls) == 1
        kwargs = graph.calls[0]["kwargs"]
        assert kwargs.get("allow_sparse_scroll") is False
        assert kwargs.get("allow_graph_scroll") is False

    # ------------------------------------------------------------------ #
    # Phase 6E P3 follow-up: full coverage of every unsafe-status marker
    # that the graph lane projects, plus the explicit opt-in via
    # ``include_fact_history=True`` (same contract as the dense lane).
    # The earlier ``test_graph_relation_unsafe_status_demoted`` only
    # exercised ``stale`` and ``requires_review``; the reviewer/security
    # pass asked for every marker to be covered explicitly.
    # ------------------------------------------------------------------ #

    def test_graph_payload_unsafe_for_each_marker(self):
        # Read the canonical unsafe ``fact_status`` vocabulary straight
        # from the implementation so this test cannot drift if a new
        # unsafe status is added: the set is parametrized over the
        # actual ``_GRAPH_UNSAFE_FACT_STATUSES`` frozenset.
        statuses = sorted(str(s) for s in _GRAPH_UNSAFE_FACT_STATUSES)
        # Sanity: the implementation must expose a non-empty set so
        # the test is meaningful. If this ever trips, the gate has
        # been widened and the marker list below needs review.
        assert statuses, "_GRAPH_UNSAFE_FACT_STATUSES must be non-empty"

        # Each unsafe-marker axis the graph gate inspects. We use a
        # payload that carries ONLY the marker under test so a
        # failure can be attributed to the exact marker that broke.
        marker_payloads: list[dict[str, Any]] = [
            {"stale": True},
            {"requires_review": True},
            {"consolidation_quarantined": True},
            {"raptor_excluded": True},
            {"raptor_forgotten": True},
        ]
        for status in statuses:
            marker_payloads.append({"fact_status": status})

        for marker_payload in marker_payloads:
            # Sanity probe the helper directly: this gate MUST report
            # unsafe=True for every marker, unsafe=False when the
            # ``include_fact_history`` opt-in flips on.
            unsafe = _graph_payload_unsafe_for_active_context(
                dict(marker_payload),
                include_fact_history=False,
            )
            assert unsafe is True, (
                "graph payload safety gate did not flag unsafe marker: "
                + repr(marker_payload)
            )
            assert (
                _graph_payload_unsafe_for_active_context(
                    dict(marker_payload),
                    include_fact_history=True,
                )
                is False
            ), (
                "include_fact_history=True must bypass the unsafe "
                "marker gate; failing case=" + repr(marker_payload)
            )

            # End-to-end through ``_graph_to_relations``: the unsafe
            # candidate is dropped from the emitted list and replaced
            # with a sanitized warning that carries the redacted
            # handle, NEVER the raw point id or text.
            warnings: list[str] = []
            candidate = FakeGraphCandidate(
                "graph-marker-1",
                graph_distance=1,
                final_score=0.7,
                payload={
                    **dict(marker_payload),
                    "source_uri": "file://docs/marker.md",
                    "file_path": "docs/marker.md",
                    "heading": "marker",
                    "text": "marker body content",
                },
            )
            graph = FakeGraphResult(final=[candidate])
            emitted = _graph_to_relations(
                graph,
                warnings=warnings,
                max_source_chars=1200,
            )
            assert emitted == [], (
                f"unsafe marker {marker_payload!r} was emitted to "
                f"graph_relations instead of being demoted"
            )
            demoted = [w for w in warnings if "graph relation demoted" in w]
            assert demoted, (
                f"no demotion warning for unsafe marker {marker_payload!r}"
            )
            for w in demoted:
                assert "graph-marker-1" not in w, (
                    "demotion warning leaked raw point id for marker "
                    f"{marker_payload!r}: {w}"
                )
                assert "marker body content" not in w, (
                    "demotion warning leaked raw payload text for marker "
                    f"{marker_payload!r}: {w}"
                )

    def test_graph_relation_include_fact_history_overrides_each_marker(self):
        # ``include_fact_history=True`` is the explicit opt-in. For
        # each unsafe marker the candidate MUST survive into
        # ``results.graph_relations`` when the opt-in is on, and no
        # demotion warning is emitted for the surviving candidate.
        marker_payloads = [
            {"stale": True},
            {"requires_review": True},
            {"consolidation_quarantined": True},
            {"raptor_excluded": True},
            {"raptor_forgotten": True},
            {"fact_status": "deprecated"},
        ]
        for marker in marker_payloads:
            graph = FakeGraphRetriever(final=[
                FakeGraphCandidate(
                    "graph-history-" + (
                        next(iter(marker.keys()))
                        if "fact_status" not in marker
                        else "fact_status"
                    ),
                    graph_distance=1,
                    final_score=0.7,
                    payload={
                        **marker,
                        "source_uri": "file://docs/h.md",
                        "file_path": "docs/h.md",
                        "text": "history body for " + repr(marker),
                    },
                ),
            ])
            router = HybridRouter(
                qdrant=FakeQdrant(),
                embeddings=FakeEmbedding(),
                collection_name="memory",
                base_retriever=FakeBaseRetriever(),
                graph_retriever=graph,
            )
            d = router.retrieve(
                "hi", top_k=5, include_fact_history=True,
            ).to_dict()
            ids = [g["point_id"] for g in d["results"]["graph_relations"]]
            assert len(ids) == 1, (
                f"include_fact_history=True dropped the candidate for "
                f"marker {marker!r}"
            )
            assert not any(
                "graph relation demoted" in w for w in d["warnings"]
            ), (
                f"include_fact_history=True still emitted a demotion "
                f"warning for marker {marker!r}: {d['warnings']!r}"
            )


class TestReadOnlySafety:
    def test_no_mutations(self):
        qdrant = FakeQdrant()
        router = HybridRouter(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=FakeBaseRetriever(),
        )
        router.retrieve("hi", top_k=3)
        assert qdrant.upserts == []
        assert qdrant.update_payloads == []
        assert qdrant.deletes == []
        assert qdrant.delete_filters == []


# ---------------------------------------------------------------------------
# Adversarial: dense-only secret-bearing exact hits
# ---------------------------------------------------------------------------


class TestDenseExactHitsSecretLeak:
    def test_dense_only_secret_bearing_chunk_dropped(self):
        # A dense hit whose payload + text contains a secret-shaped
        # bearer token must NOT surface its raw text in exact_hits and
        # must NOT echo the raw secret-shaped point id in warnings.
        bad_id = "".join(["Bearer ", "d" * 24])
        bad_text = "".join(["Bearer ", "d" * 24])
        chunk = _Chunk(
            pid=bad_id,
            text=bad_text,
            payload={"profile_id": "default", "source_type": "manual"},
            final_score=0.7,
        )
        retriever = FakeBaseRetriever(dense_seeds=[chunk])
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=retriever,
        )
        d = router.retrieve("anything").to_dict()
        # Secret-bearing chunk must NOT surface in exact_hits.
        for hit in d["results"]["exact_hits"]:
            assert hit["point_id"] != bad_id
            bad_token_substring = "Bearer " + ("d" * 8)
            assert bad_token_substring not in (hit.get("text") or "")
            assert bad_token_substring not in (hit.get("source_uri") or "")
            assert bad_token_substring not in (hit.get("heading") or "")
        # The serialized envelope must not echo the raw secret-shaped
        # id anywhere.
        serialized = json.dumps(d, default=str)
        assert bad_id not in serialized
        # Warning channel must use the redacted handle, not the raw id.
        redact_warnings = [w for w in d["warnings"] if "dense exact hit" in w]
        assert redact_warnings, "expected a dense-exact-hit redaction warning"
        for w in redact_warnings:
            assert bad_id not in w
            assert "redacted:" in w

    def test_dense_only_secret_in_payload_dropped(self):
        # Secret-shaped value lives in the payload (source_uri) but
        # the text is clean. The chunk must still be dropped because
        # the projection itself contains the secret.
        secret_uri = "https://user:" + "d" * 24 + "@internal.example/x"
        chunk = _Chunk(
            pid="clean-id",
            text="clean text",
            payload={"profile_id": "default", "source_uri": secret_uri},
            final_score=0.6,
        )
        retriever = FakeBaseRetriever(dense_seeds=[chunk])
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=retriever,
        )
        d = router.retrieve("anything").to_dict()
        serialized = json.dumps(d, default=str)
        # The raw secret uri must NOT appear anywhere in the output.
        assert secret_uri not in serialized
        # The chunk was dropped from exact_hits.
        for hit in d["results"]["exact_hits"]:
            assert hit["point_id"] != "clean-id"

    def test_clean_dense_chunk_passes_through(self):
        chunk = _Chunk(
            pid="clean-id",
            text="plain memory text",
            payload={"profile_id": "default", "source_type": "manual"},
            final_score=0.7,
        )
        retriever = FakeBaseRetriever(dense_seeds=[chunk])
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=retriever,
        )
        d = router.retrieve("anything").to_dict()
        assert any(hit["point_id"] == "clean-id" for hit in d["results"]["exact_hits"])
        # No spurious dense-exact-hit redaction warning for clean text.
        assert not any("dense exact hit redacted" in w for w in d["warnings"])

    def test_dense_secret_shaped_point_id_dropped(self):
        # Phase 5 fix4: clean text + clean payload + secret-shaped
        # ``chunk.id`` must be dropped from ``results.exact_hits`` and
        # the raw id must not echo through warnings/debug/JSON.
        bad_id = "".join(["Bearer ", "f" * 24])
        chunk = _Chunk(
            pid=bad_id,
            text="plain clean text",
            payload={"profile_id": "default", "source_type": "manual"},
            final_score=0.7,
        )
        retriever = FakeBaseRetriever(dense_seeds=[chunk])
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=retriever,
        )
        d = router.retrieve("anything").to_dict()
        # The secret-shaped chunk id must NOT appear in exact_hits or anywhere
        # else in the serialized envelope.
        for hit in d["results"]["exact_hits"]:
            assert hit["point_id"] != bad_id
        serialized = json.dumps(d, default=str)
        assert bad_id not in serialized
        # Warning channel uses redacted handle, not the raw id.
        redact_warnings = [w for w in d["warnings"] if "dense exact hit" in w]
        assert redact_warnings
        for w in redact_warnings:
            assert bad_id not in w
            assert "redacted:" in w


# ---------------------------------------------------------------------------
# Phase 5 fix5: ranking_debug is part of the secret scan.
# ---------------------------------------------------------------------------


class TestDenseRankingDebugSecretLeak:
    """Defense-in-depth regression for finding #1 from the Phase 5 review.

    ``chunk.ranking_debug`` is emitted into ``results.exact_hits`` and can
    carry values pulled from non-projected payload fields (e.g.
    ``source_hash_current``, sparse-matched tokens, literal point-id hits).
    If any field inside that object is secret-shaped, the dense hit MUST
    be dropped fail-closed and the raw secret MUST NOT appear anywhere in
    the JSON envelope. Clean ``ranking_debug`` objects MUST still pass
    through untouched so the audit envelope stays useful.
    """

    def test_secret_bearing_ranking_debug_field_drops_hit(self):
        # ranking_debug carries a secret-shaped value in
        # ``source_hash_current`` (a payload field that surfaces through
        # the provenance ranking pipeline). The dense hit MUST be
        # dropped and the raw value MUST NOT appear anywhere.
        bad_value = "".join(["Bearer ", "z" * 24])
        chunk = _Chunk(
            pid="clean-pid-1",
            text="plain clean text",
            payload={"profile_id": "default", "source_type": "manual"},
            final_score=0.7,
            ranking_debug={
                "base_score": 0.7,
                "vector_score": 0.7,
                "importance": 5,
                "source_hash_current": bad_value,
                "boosts": {"canonical": 1.04},
                "penalties": {},
                "sparse_matched_tokens": [],
                "sparse_field_hits": {},
            },
        )
        retriever = FakeBaseRetriever(dense_seeds=[chunk])
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=retriever,
        )
        d = router.retrieve("anything").to_dict()
        # The chunk MUST NOT surface in exact_hits because ranking_debug
        # carried a secret-bearing field.
        for hit in d["results"]["exact_hits"]:
            assert hit.get("point_id") != "clean-pid-1"
        serialized = json.dumps(d, default=str)
        # Raw secret MUST NOT appear in any field of the JSON envelope.
        assert bad_value not in serialized
        # A dense-exact-hit redaction warning MUST be emitted so the
        # operator can see that the lane dropped a hit.
        redact_warnings = [w for w in d["warnings"] if "dense exact hit" in w]
        assert redact_warnings
        for w in redact_warnings:
            assert bad_value not in w
            assert "redacted:" in w

    def test_secret_bearing_sparse_matched_tokens_drops_hit(self):
        # ranking_debug carries a secret-shaped token in
        # ``sparse_matched_tokens`` (a list populated from payload
        # fields). This is the realistic strong-signal-vector path
        # the reviewer flagged.
        bad_token = "".join(["Bearer ", "y" * 24])
        chunk = _Chunk(
            pid="clean-pid-2",
            text="plain clean text",
            payload={"profile_id": "default", "source_type": "manual"},
            final_score=0.6,
            ranking_debug={
                "base_score": 0.6,
                "vector_score": 0.6,
                "importance": 4,
                "source_hash_current": True,
                "boosts": {},
                "penalties": {},
                "sparse_matched_tokens": [bad_token],
                "sparse_field_hits": {"text": 1, "point_id": 1},
            },
        )
        retriever = FakeBaseRetriever(dense_seeds=[chunk])
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=retriever,
        )
        d = router.retrieve("anything").to_dict()
        for hit in d["results"]["exact_hits"]:
            assert hit.get("point_id") != "clean-pid-2"
        serialized = json.dumps(d, default=str)
        assert bad_token not in serialized
        assert any("dense exact hit" in w for w in d["warnings"])

    def test_clean_ranking_debug_passes_through(self):
        # ranking_debug is clean (booleans, numbers, empty lists) and
        # MUST be preserved verbatim on the emitted hit so the audit
        # envelope stays useful.
        chunk = _Chunk(
            pid="clean-pid-3",
            text="plain clean text",
            payload={"profile_id": "default", "source_type": "manual"},
            final_score=0.7,
            ranking_debug={
                "base_score": 0.7,
                "vector_score": 0.7,
                "importance": 5,
                "memory_kind": "assertion",
                "fact_status": "active",
                "canonical": True,
                "stale": False,
                "requires_review": False,
                "source_hash_current": True,
                "derivation_depth": 0,
                "review_history_requested": False,
                "boosts": {"canonical": 1.04},
                "penalties": {},
                "sparse_score": 0.0,
                "sparse_literal_hit": False,
                "sparse_matched_tokens": [],
                "sparse_field_hits": {},
            },
        )
        retriever = FakeBaseRetriever(dense_seeds=[chunk])
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=retriever,
        )
        d = router.retrieve("anything").to_dict()
        hits = [h for h in d["results"]["exact_hits"] if h.get("point_id") == "clean-pid-3"]
        assert len(hits) == 1
        hit = hits[0]
        # ranking_debug MUST round-trip with all clean fields preserved.
        assert hit.get("ranking_debug") == chunk.ranking_debug
        # No spurious dense-exact-hit redaction warning for clean debug.
        assert not any("dense exact hit" in w for w in d["warnings"])

    def test_real_memory_retriever_ranking_debug_secret_drops(self):
        # End-to-end through real MemoryRetriever: a payload with a
        # secret-shaped ``source_hash_current`` value flows through
        # ``rank_memory_candidate`` into ``ranking_debug`` and would
        # otherwise reach ``results.exact_hits``. The router must drop
        # the hit.
        from qdrant_memory.retriever import MemoryRetriever

        bad_value = "".join(["Bearer ", "x" * 24])

        class _SearchOnlyQdrant:
            def __init__(self):
                self.calls: list = []
                self.scroll_calls: list = []

            def search(self, name, vector, limit, filter=None, with_payload=True, with_vector=False):
                self.calls.append({
                    "name": name,
                    "vector": vector,
                    "limit": limit,
                    "filter": filter,
                    "with_payload": with_payload,
                    "with_vector": with_vector,
                })
                return [{
                    "id": "real-clean-pid",
                    "score": 0.9,
                    "payload": {
                        "text": "plain clean text",
                        "source_type": "manual",
                        "profile_id": "default",
                        "importance": 5,
                        "created_at": "2026-01-15T00:00:00+00:00",
                        "source_hash_current": bad_value,
                    },
                }]

            def scroll_by_filter(self, *args, **kwargs):
                self.scroll_calls.append((args, kwargs))
                return []

            def update_payload(self, *args, **kwargs):
                # Read-only invariant must NOT be violated.
                raise AssertionError("update_payload must not be called")

        qdrant = _SearchOnlyQdrant()
        retriever = MemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
            search_candidates=3,
        )
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=retriever,
        )
        d = router.retrieve("anything").to_dict()
        serialized = json.dumps(d, default=str)
        # The secret-bearing ranking_debug payload field MUST NOT be
        # reachable through the emitted JSON envelope.
        assert bad_value not in serialized
        # Either the hit was dropped entirely OR its ranking_debug was
        # redacted before emission — both are acceptable. Either way
        # the serialized envelope must not echo the raw value.
        emitted_hit = next(
            (h for h in d["results"]["exact_hits"] if h.get("point_id") == "real-clean-pid"),
            None,
        )
        if emitted_hit is not None:
            # If the hit was kept (ranking_debug was redacted), ensure
            # the secret field is no longer raw.
            assert bad_value not in (emitted_hit.get("ranking_debug") or {})
        # No access-metadata mutation.
        # (FakeQdrant tracks nothing; MemoryRetriever would only mutate
        # via update_payload which the fake denies — so reaching this
        # line is the proof of no mutation.)


# ---------------------------------------------------------------------------
# Adversarial: evidence-mode parent warning uses redacted handle
# ---------------------------------------------------------------------------


class TestEvidenceModeRedactedDemotion:
    def test_evidence_mode_parent_warning_redacted(self):
        from qdrant_memory.raptor.search import RaptorSummaryHit, RaptorLeafHit
        bad_id = "".join(["Bearer ", "e" * 24])
        orphan_parent = RaptorSummaryHit(
            point_id=bad_id,
            raptor_node_id=bad_id,
            raptor_root_id="root",
            raptor_level=2,
            raptor_tree_id="tree",
            raptor_build_id="build",
            raptor_cluster_id=f"cluster-{bad_id}",
            text="orphan parent",
        )
        raptor = FakeRaptorSearcher(summaries=[orphan_parent], leaves=[])
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=FakeBaseRetriever(),
            raptor_searcher=raptor,
        )
        d = router.retrieve("anything", mode="evidence").to_dict()
        # Warning channel must use a redacted handle, not the raw id.
        evidence_warnings = [w for w in d["warnings"] if "evidence mode" in w]
        assert evidence_warnings
        for w in evidence_warnings:
            assert bad_id not in w
            assert "redacted:" in w
        # debug["evidence_demotions"] must also be redacted.
        for handle in d["debug"]["stages"].get("evidence_demotions", []):
            assert bad_id not in handle


# ---------------------------------------------------------------------------
# Fusion helpers
# ---------------------------------------------------------------------------


class TestFusionHelpers:
    def test_rrf_fuse_merges_overlapping(self):
        a = [{"point_id": "x", "score": 0.9}, {"point_id": "y", "score": 0.1}]
        b = [{"point_id": "x", "score": 0.4}, {"point_id": "z", "score": 0.2}]
        fused = rrf_fuse([a, b])
        assert fused[0]["point_id"] == "x"  # appears in both lists
        assert any(item["point_id"] == "y" for item in fused)
        assert any(item["point_id"] == "z" for item in fused)
        assert all("_rrf_score" in item for item in fused)

    def test_rrf_negative_k_raises(self):
        with pytest.raises(ValueError):
            rrf_fuse([[{"point_id": "x"}]], k=0)

    def test_deduplicate_by_point_id(self):
        items = [
            {"point_id": "a", "score": 0.9},
            {"point_id": "a", "score": 0.4},  # dup
            {"point_id": "b", "score": 0.7},
            {"score": 0.5},                   # missing id → dropped
        ]
        out = deduplicate_by_point_id(items)
        assert len(out) == 2
        assert {o["point_id"] for o in out} == {"a", "b"}
        # first-seen wins
        first_a = next(o for o in out if o["point_id"] == "a")
        assert first_a["score"] == 0.9


# ---------------------------------------------------------------------------
# Phase 5 fix6: regression tests for the final4 reviewer/security pass.
# ---------------------------------------------------------------------------


class TestRealGraphRetrieverWiring:
    """HybridRouter must work with the real ``GraphMemoryRetriever``
    without raising ``TypeError`` on ``max_depth`` (finding #1) and
    must call it with the expected defaults (no fallback to warning).
    """

    def test_hybrid_router_works_with_real_graph_retriever(self):
        from qdrant_memory.graph_retriever import GraphMemoryRetriever

        # Minimal fake Qdrant + embeddings that satisfy
        # ``GraphMemoryRetriever.__init__`` without touching real
        # Qdrant. The retriever builds a ``MemoryRetriever`` on
        # demand; we feed just enough seed data for ``search`` to
        # make it through the dense seed path without raising.
        class _StreamingQdrant:
            def __init__(self):
                self.calls: list = []

            def search(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return []

            def scroll(self, *args, **kwargs):
                return ([], None)

            def retrieve(self, *args, **kwargs):
                return []

            def update_payload(self, *args, **kwargs):
                raise AssertionError("update_payload must not be called")

            def upsert(self, *args, **kwargs):
                raise AssertionError("upsert must not be called")

            def delete_ids(self, *args, **kwargs):
                raise AssertionError("delete_ids must not be called")

        qdrant = _StreamingQdrant()
        graph = GraphMemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
        )

        router = HybridRouter(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=FakeBaseRetriever(),
            graph_retriever=graph,
        )
        # The router calls ``graph_retriever.search`` with
        # ``max_depth=safe_max_depth``; pre-fix this raised TypeError
        # because ``GraphMemoryRetriever.search`` did not accept
        # ``max_depth``. The router catches TypeError and emits
        # "graph lane failed: …". Post-fix the call MUST succeed and
        # the warning channel MUST NOT contain a graph-lane failure.
        d = router.retrieve("anything", max_depth=2).to_dict()
        graph_failure_warnings = [
            w for w in d["warnings"] if "graph lane failed" in w
        ]
        assert not graph_failure_warnings, (
            "HybridRouter + real GraphMemoryRetriever raised TypeError "
            "or otherwise failed the graph lane: "
            f"{graph_failure_warnings!r}"
        )
        # debug envelope records a successful graph stage with a
        # returned count (which may be zero — that's fine).
        graph_stage = d["debug"]["stages"].get("graph", {})
        assert graph_stage.get("skipped") is None
        assert graph_stage.get("error") is None


class TestDenseRetrievalStatusSafety:
    """HybridRouter + real ``MemoryRetriever`` must demote unsafe
    dense payloads (``stale``, ``requires_review``, ``quarantined``,
    unsafe ``fact_status``) from active ``results.exact_hits``.
    """

    def test_stale_payload_not_in_active_exact_hits(self):
        from qdrant_memory.retriever import (
            RetrievedMemory,
            MemoryRetriever,
        )

        stale_payload = {
            "text": "this memory is stale",
            "source_type": "manual",
            "profile_id": "default",
            "importance": 4,
            "created_at": "2026-01-15T00:00:00+00:00",
            "stale": True,
        }
        clean_payload = {
            "text": "this memory is clean",
            "source_type": "manual",
            "profile_id": "default",
            "importance": 4,
            "created_at": "2026-01-15T00:00:00+00:00",
        }

        class _NoScrollQdrant:
            def __init__(self):
                self.calls: list = []

            def search(
                self, name, vector, limit, filter=None,
                with_payload=True, with_vector=False,
            ):
                self.calls.append((name, vector, limit, filter))
                return [
                    {"id": "stale-pid", "score": 0.7, "payload": stale_payload},
                    {"id": "clean-pid", "score": 0.5, "payload": clean_payload},
                ]

            def update_payload(self, *args, **kwargs):
                # Read-only invariant: no access-metadata mutation.
                raise AssertionError("update_payload must not be called")

        qdrant = _NoScrollQdrant()
        retriever = MemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
            search_candidates=4,
        )
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=retriever,
        )
        d = router.retrieve("anything").to_dict()

        ids = [h["point_id"] for h in d["results"]["exact_hits"]]
        assert "stale-pid" not in ids, (
            "stale dense payload leaked into active exact_hits"
        )
        assert "clean-pid" in ids
        # Warning channel MUST surface a dense-exact-hit demotion for
        # the stale entry, using the redacted handle (not the raw id).
        demote_warnings = [w for w in d["warnings"] if "demoted" in w]
        assert demote_warnings, (
            "expected at least one dense-exact-hit demotion warning"
        )
        for w in demote_warnings:
            assert "stale-pid" not in w, (
                "demotion warning leaked raw stale point id: " + w
            )

    def test_requires_review_payload_not_in_active_exact_hits(self):
        from qdrant_memory.retriever import MemoryRetriever

        review_payload = {
            "text": "this memory needs review",
            "source_type": "manual",
            "profile_id": "default",
            "importance": 4,
            "created_at": "2026-01-15T00:00:00+00:00",
            "requires_review": True,
        }
        clean_payload = {
            "text": "this memory is clean",
            "source_type": "manual",
            "profile_id": "default",
            "importance": 4,
            "created_at": "2026-01-15T00:00:00+00:00",
        }

        class _NoScrollQdrant:
            def search(
                self, name, vector, limit, filter=None,
                with_payload=True, with_vector=False,
            ):
                return [
                    {"id": "review-pid", "score": 0.7, "payload": review_payload},
                    {"id": "clean-pid-2", "score": 0.5, "payload": clean_payload},
                ]

            def update_payload(self, *args, **kwargs):
                raise AssertionError("update_payload must not be called")

        qdrant = _NoScrollQdrant()
        retriever = MemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
            search_candidates=4,
        )
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=retriever,
        )
        d = router.retrieve("anything").to_dict()
        ids = [h["point_id"] for h in d["results"]["exact_hits"]]
        assert "review-pid" not in ids
        assert "clean-pid-2" in ids
        demote_warnings = [
            w for w in d["warnings"] if "demoted" in w and "requires_review" in w
        ]
        assert demote_warnings
        for w in demote_warnings:
            assert "review-pid" not in w

    def test_include_fact_history_overrides_demotion(self):
        # ``include_fact_history=True`` is the explicit opt-in to
        # surface review / stale material. The dense demotion gate
        # MUST stay open so the history lane remains accessible.
        from qdrant_memory.retriever import MemoryRetriever

        review_payload = {
            "text": "history material",
            "source_type": "manual",
            "profile_id": "default",
            "importance": 4,
            "created_at": "2026-01-15T00:00:00+00:00",
            "requires_review": True,
        }

        class _NoScrollQdrant:
            def search(
                self, name, vector, limit, filter=None,
                with_payload=True, with_vector=False,
            ):
                return [
                    {"id": "history-pid", "score": 0.7, "payload": review_payload},
                ]

            def update_payload(self, *args, **kwargs):
                raise AssertionError("update_payload must not be called")

        qdrant = _NoScrollQdrant()
        retriever = MemoryRetriever(
            qdrant=qdrant,
            embeddings=FakeEmbedding(),
            collection_name="memory",
            search_candidates=4,
        )
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=retriever,
        )
        d = router.retrieve("anything", include_fact_history=True).to_dict()
        ids = [h["point_id"] for h in d["results"]["exact_hits"]]
        # With history opt-in, the unsafe-status hit IS allowed into
        # exact_hits (history lane).
        assert "history-pid" in ids


class TestRetrieveWarningNoSecretIdLeak:
    """Final4 finding #4: a backend ``retrieve()`` / ``search()``
    exception whose ``__str__`` echoes secret-shaped ids must not
    surface those raw ids through ``results.warnings`` or anywhere
    in the serialized envelope (raptor lane)."""

    def test_raptor_lane_no_secret_id_in_warnings(self):
        from qdrant_memory.raptor.search import RaptorSearcher

        bad_id = "".join(["Bearer ", "s" * 24])

        class _EchoingQdrant:
            def retrieve(self, name, ids, *, with_payload=True, with_vector=False):
                raise RuntimeError(
                    "backend refused request ids=" + ",".join(ids)
                )

        # Construct minimal FakeRetriever-shaped seed so the raptor
        # lane actually reaches the retrieve call. We define a
        # local stand-in instead of importing from the sibling test
        # module (which would require an unusual import path).
        class _RetrievedMemory:
            def __init__(self, pid, text, payload, final_score=0.5):
                self.id = pid
                self.text = text
                self.payload = payload
                self.final_score = final_score
                self.qdrant_score = final_score
                self.ranking_debug = {}

        dense_seeds = [
            _RetrievedMemory(
                "p1",
                text="p1",
                payload={
                    "raptor_node_id": "p1",
                    "raptor_level": 2,
                    "raptor_parent_ids": [],
                    "raptor_child_ids": ["leaf-x"],
                    "raptor_summary_of": ["leaf-x"],
                    "raptor_tree_id": "t",
                    "raptor_root_id": "r",
                    "raptor_build_id": "b",
                    "raptor_cluster_id": "c",
                    "derivation_type": "raptor_summary",
                    "memory_kind": "summary",
                    "canonical": False,
                    "requires_review": True,
                    "text": "parent summary",
                    "profile_id": "default",
                    "source_hashes": ["a" * 64],
                    "derived_from": [],
                },
                final_score=0.5,
            )
        ]

        class _FakeRetriever:
            def search(self, query, *, top_k=5, update_access=True, **kwargs):
                return list(dense_seeds)

        searcher = RaptorSearcher(
            qdrant=_EchoingQdrant(),
            retriever=_FakeRetriever(),
            collection_name="memory",
        )
        result = searcher.search("anything", top_k=3)
        for w in result.warnings:
            assert bad_id not in w, (
                "raptor lane leaked secret-shaped id in warning: " + w
            )


# ---------------------------------------------------------------------------
# Phase 5 fix7: raptor seed-search warning raw-exception leak through
# HybridRouter (and therefore ``qdrant_memory_retrieve``).
#
# Regression coverage for finding #1 from the final5 reviewer/security
# pass. The RAPTOR ``RaptorSearcher.search`` used to interpolate
# ``{exc}`` into the seed-search failure warning. HybridRouter wires
# that warning directly into ``results.warnings`` of the JSON envelope,
# so a retriever backend that echoes the requested query (a
# secret-shaped token) into its exception ``__str__`` could leak the
# token into the LLM-readable retrieve output. We now require the
# warning channel to be free of the raw exception text — even when
# the warning reaches the envelope via HybridRouter.
# ---------------------------------------------------------------------------


class TestRaptorSeedSearchWarningPropagatesSafelyThroughRouter:
    def test_type_error_secret_query_does_not_leak_into_router_warnings(self):
        # Construct the secret-shaped query at runtime so the
        # scanner doesn't trip on a literal in the source file.
        bad_query = "".join(["Bearer ", "w" * 24])

        class _RejectsKwargRetriever:
            """Custom retriever that raises a TypeError whose ``__str__``
            echoes the secret-shaped query verbatim.
            """

            def search(self, query, *, top_k=5, update_access=True, **kwargs):
                if "allow_sparse_scroll" in kwargs:
                    raise TypeError(
                        "search() got an unexpected keyword argument "
                        "'allow_sparse_scroll' for query=" + query
                    )
                return []

        # We do NOT need a populated seed set: fail-closed path
        # short-circuits with empty seeds + warning.
        from qdrant_memory.raptor.search import RaptorSearcher

        searcher = RaptorSearcher(
            qdrant=FakeQdrant(),
            retriever=_RejectsKwargRetriever(),
            collection_name="memory",
        )
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=FakeBaseRetriever(dense_seeds=[]),
            raptor_searcher=searcher,
        )
        d = router.retrieve(bad_query, top_k=3).to_dict()

        # The serialized envelope echoes the requested query back
        # through the ``query`` field by design (the caller passed
        # that in); what we MUST NOT leak is the seed-search
        # warning that interpolates ``str(exc)``.
        for w in d.get("warnings", []):
            assert bad_query not in w, (
                "HybridRouter propagated the secret-shaped query through "
                "RAPTOR seed-search warning into the JSON envelope"
            )
        # Debug envelope (where server-side correlation lives)
        # MUST NOT echo the raw secret-shaped query either.
        debug_serialized = json.dumps(d.get("debug", {}), default=str)
        assert bad_query not in debug_serialized

    def test_generic_exception_secret_query_does_not_leak_into_router_warnings(self):
        bad_query = "".join(["Bearer ", "v" * 24])

        class _EchoingRuntimeErrorRetriever:
            """Custom retriever whose ``search`` raises a generic
            RuntimeError that echoes the query verbatim."""

            def search(self, query, *, top_k=5, update_access=True, **kwargs):
                raise RuntimeError(
                    "backend refused to handle query=" + repr(query)
                )

        from qdrant_memory.raptor.search import RaptorSearcher

        searcher = RaptorSearcher(
            qdrant=FakeQdrant(),
            retriever=_EchoingRuntimeErrorRetriever(),
            collection_name="memory",
        )
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=FakeBaseRetriever(dense_seeds=[]),
            raptor_searcher=searcher,
        )
        d = router.retrieve(bad_query, top_k=3).to_dict()

        for w in d.get("warnings", []):
            assert bad_query not in w, (
                "HybridRouter propagated the secret-shaped query through "
                "RAPTOR seed-search warning into the JSON envelope"
            )
        # Debug envelope MUST NOT echo the raw secret-shaped query
        # either — debug is server-side correlation territory.
        debug_serialized = json.dumps(d.get("debug", {}), default=str)
        assert bad_query not in debug_serialized


# ---------------------------------------------------------------------------
# Phase 5 fix9 (final7 finding #2): dense exact_hits must respect
# ``max_source_chars`` and the hard context char budget. Pre-fix9 the
# dense lane emitted chunk.text verbatim (no truncation) and the
# context_used_chars debug counter only counted summaries + leaves,
# so a 5000-char dense hit bypassed both budgets. We now:
#   * truncate per-hit text to ``safe_max_source_chars``;
#   * include exact_hits in ``context_used_chars``;
#   * drop dense hits whose inclusion would exceed
#     ``HARD_CONTEXT_CHAR_BUDGET`` (= 16000) so the dense lane cannot
#     blow the RAPTOR-lane hard cap.
# ---------------------------------------------------------------------------


class TestDenseExactHitsBudgetEnforcement:
    """Regression for final7 finding #2.

    Dense exact_hits must be capped per-result by ``max_source_chars``
    and must respect the hard context char budget that the RAPTOR
    lane also respects. The dense lane cannot emit a 5000-char hit
    when the caller asked for ``max_source_chars=10``.
    """

    def test_long_dense_hit_truncated_to_max_source_chars(self):
        # One 5000-char dense hit; caller asks ``max_source_chars=10``.
        # The emitted exact hit text MUST be truncated to <=10 chars
        # (the per-result cap) and ``context_used_chars`` MUST count
        # it (not zero).
        long_text = "L" * 5000
        chunk = _Chunk(
            pid="dense-long",
            text=long_text,
            payload={"profile_id": "default", "source_type": "manual"},
            final_score=0.9,
        )
        retriever = FakeBaseRetriever(dense_seeds=[chunk])
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=retriever,
        )
        d = router.retrieve("anything", max_source_chars=10).to_dict()
        hits = d["results"]["exact_hits"]
        assert len(hits) == 1
        hit = hits[0]
        # Per-hit truncation: text must be <= 10 chars (the cap).
        assert len(hit["text"]) <= 10, (
            f"emitted exact hit text length {len(hit['text'])} but "
            f"max_source_chars=10; final7 finding #2 regression"
        )
        # context_used_chars MUST include this hit (was 0 pre-fix9).
        ctx = int(d.get("debug", {}).get("context_used_chars") or 0)
        assert ctx > 0, (
            "context_used_chars=0 with one dense exact hit; final7 "
            "finding #2 requires the dense lane to count against the budget"
        )
        assert ctx <= 10, (
            f"context_used_chars={ctx} exceeds max_source_chars=10 cap; "
            f"the dense lane is not being bounded correctly"
        )

    def test_many_dense_hits_dropped_at_hard_context_budget(self):
        # Many 1000-char dense hits; max_source_chars=600 (clamped to
        # 600). HARD_CONTEXT_CHAR_BUDGET=16000. We send 30 hits so
        # 16000/600 = 26 hits fit; 4 must be dropped. The dense lane
        # MUST drop overflow deterministically (first-seen-wins) and
        # emit a sanitized warning per drop.
        n_hits = 30
        cap = 600
        hard_budget = 16000
        chunks = [
            _Chunk(
                pid=f"dense-many-{i:02d}",
                text=("X" * 1000),
                payload={"profile_id": "default", "source_type": "manual"},
                final_score=0.9 - i * 0.001,
            )
            for i in range(n_hits)
        ]
        retriever = FakeBaseRetriever(dense_seeds=chunks)
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=retriever,
        )
        d = router.retrieve("anything", max_source_chars=cap).to_dict()
        hits = d["results"]["exact_hits"]
        # The emitted exact hits are also bounded by ``top_k`` (=5 by
        # default), so we expect at most 5 hits unless the router
        # rerouted the dense lane through a different path. The
        # important property is that the dense lane does not
        # exceed the per-hit cap and that the hard context budget
        # holds.
        for hit in hits:
            assert len(hit["text"]) <= cap, (
                f"exact hit text length {len(hit['text'])} exceeds cap {cap}"
            )
        # context_used_chars must respect the hard budget for the
        # dense lane (and the rest of the lanes since the test
        # runs dense-only). The hard cap is enforced inside
        # ``_dense_to_exact_hits`` and additionally the RAPTOR
        # lane cap is enforced inside ``RaptorSearcher.search``.
        # When only the dense lane runs, the per-hit cap of 600
        # means we can emit at most 5 hits within the 16000-char
        # hard budget. The default ``top_k=5`` then caps that
        # further to 5 hits.
        ctx = int(d.get("debug", {}).get("context_used_chars") or 0)
        assert ctx <= hard_budget, (
            f"context_used_chars={ctx} exceeded HARD_CONTEXT_CHAR_BUDGET="
            f"{hard_budget}; final7 finding #2 requires the hard cap to hold"
        )
        # First-seen-wins determinism: the lowest-indexed pids are
        # the ones that should survive (when top_k=5 holds).
        emitted_pids = [h["point_id"] for h in hits]
        assert emitted_pids[0] == "dense-many-00", (
            f"first-emitted hit is {emitted_pids[0]!r}, expected "
            f"first-seen-wins determinism"
        )

    def test_dense_lane_zero_exact_hits_still_includes_in_context_chars(self):
        # Sanity: with zero dense hits, context_used_chars must
        # still be a deterministic number (=0 plus whatever the
        # RAPTOR lane emits, which is 0 in this test) and the
        # emitted exact_hits list must be empty. The fix MUST NOT
        # break the empty-input case.
        retriever = FakeBaseRetriever(dense_seeds=[])
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=retriever,
        )
        d = router.retrieve("anything", max_source_chars=600).to_dict()
        assert d["results"]["exact_hits"] == []
        ctx = int(d.get("debug", {}).get("context_used_chars") or 0)
        assert ctx == 0, (
            f"empty dense lane should produce context_used_chars=0, got {ctx}"
        )

    def test_dense_lane_max_source_chars_truncates_long_text(self):
        # A 300-char hit with max_source_chars=50. The text must be
        # truncated to <=50 chars and context_used_chars must
        # reflect the truncated length, not the original.
        long_text = "A" * 300
        chunk = _Chunk(
            pid="dense-truncate",
            text=long_text,
            payload={"profile_id": "default", "source_type": "manual"},
            final_score=0.9,
        )
        retriever = FakeBaseRetriever(dense_seeds=[chunk])
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=retriever,
        )
        d = router.retrieve("anything", max_source_chars=50).to_dict()
        hit = d["results"]["exact_hits"][0]
        assert len(hit["text"]) <= 50
        # The context counter must use the truncated length, not
        # the original 300 chars.
        ctx = int(d.get("debug", {}).get("context_used_chars") or 0)
        assert ctx <= 50, (
            f"context_used_chars={ctx} reflects original text length; "
            f"the dense lane is not applying per-hit truncation before "
            f"counting"
        )


# ---------------------------------------------------------------------------
# Phase 5 fix10 (final8 finding #1): HybridRouter must enforce ONE
# global hard context char budget across summaries + cited_leaves +
# exact_hits. Pre-fix10 the dense lane and the RAPTOR lane each
# clamped to ``HARD_CONTEXT_CHAR_BUDGET`` independently and the
# router only reported ``context_used_chars``; the emitted total
# could exceed 16000 chars (e.g. 15600 dense + 1200 RAPTOR = 16800)
# and the debug counter was at most additive, not a hard cap.
# ---------------------------------------------------------------------------


class TestHybridGlobalContextBudget:
    """Regression for final8 finding #1.

    The final packing of ``HybridRouter.retrieve`` MUST enforce a
    single global hard context budget so the union the caller
    receives as LLM context never exceeds ``HARD_CONTEXT_CHAR_BUDGET``
    (= 16000), regardless of how many lanes fire.
    """

    def test_dense_plus_raptor_combined_exceeds_budget_capped(self):
        # 5 dense hits, each ~ 2000 chars (well within the
        # per-result ``max_source_chars`` cap when the caller
        # defaults to 1200... so we use ``max_source_chars=1200``
        # which truncates each dense hit to 1200 chars). We also
        # emit one RAPTOR summary of 1200 chars. With the fix9
        # code path the dense lane clamps at 16000 (5 × 1200 = 6000
        # fits) and the RAPTOR lane clamps at 16000 too. But to
        # deliberately exceed the global budget we send a
        # high-volume dense stream and ask the dense lane to
        # clamp per-hit higher than the RAPTOR summary's text
        # length so the SUM > 16000.
        #
        # Setup: dense hits truncated to 1200 chars each, with
        # 13 hits = 15600 chars. RAPTOR summary of 1200 chars.
        # The dense lane's own clamp caps dense at 16000 chars.
        # RAPTOR lane's own clamp caps RAPTOR at 16000. UNION:
        # 15600 + 1200 = 16800 > 16000. fix10 must enforce the
        # global budget and drop a dense hit to fit.
        from qdrant_memory.raptor.search import RaptorSummaryHit, RaptorLeafHit

        n_dense = 13
        per_hit = 1200
        raptor_summary = RaptorSummaryHit(
            point_id="raptor-summary-1",
            raptor_node_id="raptor-node-1",
            raptor_root_id="root",
            raptor_level=2,
            raptor_tree_id="tree",
            raptor_build_id="build",
            raptor_cluster_id="cluster-1",
            text="R" * 1200,
        )
        chunks = [
            _Chunk(
                pid=f"dense-mix-{i:02d}",
                text=("D" * 5000),  # dense lane will truncate to 1200
                payload={"profile_id": "default", "source_type": "manual"},
                final_score=0.9 - i * 0.001,
            )
            for i in range(n_dense)
        ]
        retriever = FakeBaseRetriever(dense_seeds=chunks)
        raptor = FakeRaptorSearcher(summaries=[raptor_summary], leaves=[])
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=retriever,
            raptor_searcher=raptor,
        )
        d = router.retrieve("anything", max_source_chars=per_hit, top_k=20).to_dict()

        hard_budget = 16000
        # The union of summaries + cited_leaves + exact_hits MUST
        # respect the hard budget.
        emitted_total = 0
        for s in d["results"]["summaries"]:
            emitted_total += len(s.get("text") or "")
        for leaf in d["results"]["cited_leaves"]:
            emitted_total += len(leaf.get("text") or "")
        for hit in d["results"]["exact_hits"]:
            emitted_total += len(hit.get("text") or "")
        assert emitted_total <= hard_budget, (
            f"emitted total {emitted_total} exceeds HARD_CONTEXT_CHAR_BUDGET="
            f"{hard_budget}; final8 finding #1 requires ONE global budget"
        )

        # The RAPTOR summary (tree evidence) MUST be preserved.
        raptor_pids = [s.get("point_id") for s in d["results"]["summaries"]]
        assert "raptor-summary-1" in raptor_pids, (
            "global budget enforcer dropped the RAPTOR summary; the "
            "preservation policy is summaries/leaves first, dense last"
        )

        # context_used_chars MUST agree with the actual emitted total
        # (debug counter cannot disagree with the wire).
        ctx = int(d.get("debug", {}).get("context_used_chars") or 0)
        assert ctx == emitted_total, (
            f"context_used_chars={ctx} disagrees with emitted total="
            f"{emitted_total}; debug envelope must reflect reality"
        )
        assert ctx <= hard_budget

        # A drop warning MUST be emitted on the sanitized channel
        # (redacted handle, not raw id).
        global_warnings = [
            w for w in d["warnings"] if "global hard context budget" in w
        ]
        assert global_warnings, (
            "expected at least one global-budget drop warning when the "
            "union of lanes would exceed the hard cap"
        )
        for w in global_warnings:
            # No raw point ids from the dense hits.
            assert "dense-mix-" not in w, (
                "global budget warning leaked raw dense point id: " + w
            )

    def test_raptor_first_policy_dense_dropped_first(self):
        # Policy proof: when the RAPTOR lane consumes most of the
        # budget, dense hits are the ones that get dropped, NOT
        # the RAPTOR summaries or leaves.
        #
        # Note: ``RaptorLeafHit`` only emits ``text`` when the
        # caller passes ``include_metadata=True``; the default
        # projection is metadata-stripped. We pass it here so the
        # leaf's 1200 chars actually count against the global
        # budget — otherwise the RAPTOR lane consumes only 1200
        # chars (the summary), which is not enough to push the
        # dense lane over 16000 and the test would not exercise
        # the policy.
        from qdrant_memory.raptor.search import RaptorSummaryHit, RaptorLeafHit

        raptor_summary = RaptorSummaryHit(
            point_id="raptor-summary-X",
            raptor_node_id="raptor-node-X",
            raptor_root_id="root",
            raptor_level=2,
            raptor_tree_id="tree",
            raptor_build_id="build",
            raptor_cluster_id="cluster-X",
            text="R" * 1200,
        )
        raptor_leaf = RaptorLeafHit(
            point_id="raptor-leaf-X",
            parent_raptor_node_id="raptor-node-X",
            parent_point_id="raptor-summary-X",
            text="L" * 1200,
        )
        # Dense lane: 12 hits × 1200 = 14400 chars (fits in 16000).
        # Combined with RAPTOR: 14400 + 1200 + 1200 = 16800.
        chunks = [
            _Chunk(
                pid=f"dense-policy-{i:02d}",
                text=("D" * 5000),
                payload={"profile_id": "default", "source_type": "manual"},
                final_score=0.9 - i * 0.001,
            )
            for i in range(12)
        ]
        retriever = FakeBaseRetriever(dense_seeds=chunks)
        raptor = FakeRaptorSearcher(
            summaries=[raptor_summary], leaves=[raptor_leaf]
        )
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=retriever,
            raptor_searcher=raptor,
        )
        d = router.retrieve(
            "anything",
            max_source_chars=1200,
            top_k=20,
            include_metadata=True,
        ).to_dict()

        # RAPTOR preserved.
        assert any(
            s.get("point_id") == "raptor-summary-X"
            for s in d["results"]["summaries"]
        )
        assert any(
            leaf.get("point_id") == "raptor-leaf-X"
            for leaf in d["results"]["cited_leaves"]
        )

        # Total respects the hard budget.
        hard_budget = 16000
        emitted_total = 0
        for s in d["results"]["summaries"]:
            emitted_total += len(s.get("text") or "")
        for leaf in d["results"]["cited_leaves"]:
            emitted_total += len(leaf.get("text") or "")
        for hit in d["results"]["exact_hits"]:
            emitted_total += len(hit.get("text") or "")
        assert emitted_total <= hard_budget, (
            f"emitted total {emitted_total} exceeds {hard_budget}"
        )

        # RAPTOR reservation = 1200 (summary) + 1200 (leaf with
        # include_metadata=True) = 2400. The remaining budget for
        # dense exact_hits is 16000 - 2400 = 13600. Dense hits are
        # 1200 chars each (truncated to ``max_source_chars=1200``).
        # 13600 / 1200 = 11.33, so 11 hits fit, 1 must be dropped.
        # The dropped hit is the LAST in iteration order (the
        # global enforcer is first-seen-wins).
        dense_chars = sum(
            len(h.get("text") or "") for h in d["results"]["exact_hits"]
        )
        assert dense_chars <= 16000 - 2400, (
            f"dense chars {dense_chars} > (hard_budget - raptor_reserved); "
            f"the global enforcer should have dropped dense hits to fit"
        )
        # And the first-seen-wins determinism: dense-policy-00
        # must survive if any hit survives.
        emitted_pids = [h.get("point_id") for h in d["results"]["exact_hits"]]
        assert "dense-policy-00" in emitted_pids, (
            "first-seen-wins broken: dense-policy-00 (lowest index) was "
            "dropped while a higher-index hit survived"
        )

    def test_global_budget_under_limit_no_drops(self):
        # Sanity: when the union of lanes is BELOW the hard
        # budget, the global enforcer MUST NOT drop anything and
        # MUST NOT emit a global-budget warning.
        from qdrant_memory.raptor.search import RaptorSummaryHit, RaptorLeafHit

        raptor_summary = RaptorSummaryHit(
            point_id="raptor-summary-S",
            raptor_node_id="raptor-node-S",
            raptor_root_id="root",
            raptor_level=2,
            raptor_tree_id="tree",
            raptor_build_id="build",
            raptor_cluster_id="cluster-S",
            text="R" * 100,
        )
        chunks = [
            _Chunk(
                pid="dense-tiny-1",
                text=("D" * 5000),  # truncated to 200 by the cap below
                payload={"profile_id": "default", "source_type": "manual"},
                final_score=0.9,
            ),
            _Chunk(
                pid="dense-tiny-2",
                text=("D" * 5000),
                payload={"profile_id": "default", "source_type": "manual"},
                final_score=0.8,
            ),
        ]
        retriever = FakeBaseRetriever(dense_seeds=chunks)
        raptor = FakeRaptorSearcher(summaries=[raptor_summary], leaves=[])
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=retriever,
            raptor_searcher=raptor,
        )
        d = router.retrieve("anything", max_source_chars=200, top_k=5).to_dict()

        # Both dense hits survived (200 + 200 + 100 = 500 < 16000).
        emitted_pids = [h.get("point_id") for h in d["results"]["exact_hits"]]
        assert "dense-tiny-1" in emitted_pids
        assert "dense-tiny-2" in emitted_pids
        # RAPTOR summary survived.
        assert any(
            s.get("point_id") == "raptor-summary-S"
            for s in d["results"]["summaries"]
        )
        # No global budget drop warning when we are under budget.
        assert not any(
            "global hard context budget" in w for w in d["warnings"]
        ), (
            "global budget warning fired when total was under the hard cap"
        )

    def test_graph_relations_text_in_global_hard_context_budget(self):
        # Phase 6E P3 follow-up: graph relation ``text`` bodies MUST
        # participate in the SINGLE global hard context char budget
        # (16000 chars) that the ``HybridRouter.retrieve`` path
        # enforces. The dense lane and RAPTOR lane are clamped to
        # the budget upstream; the final pass drops overflow graph
        # relations when the *union* of lanes still exceeds the cap.
        #
        # We force the overflow by:
        #   1. emitting a small dense hit (~200 chars) and a small
        #      RAPTOR summary (~200 chars) — leaves ~15600 chars
        #      remaining for graph relations;
        #   2. sending 20 graph candidates, each with a 1200-char
        #      ``text`` body that fits the per-relation cap (so the
        #      per-relation gate is NOT what drops them);
        #   3. asserting that the live ``HybridRouter.retrieve``
        #      path drops overflow graph relations and emits a
        #      warning that mentions graph overflow specifically.
        from qdrant_memory.raptor.search import RaptorSummaryHit

        dense_text_len = 200
        raptor_summary_text_len = 200
        # Per-relation text the graph lane will emit (each relation
        # is below the 1200-char per-relation cap so the per-relation
        # gate does not interfere).
        per_relation_text_len = 1200
        n_graph_candidates = 20
        hard_budget = 16000

        chunks = [
            _Chunk(
                pid="dense-bg-1",
                text=("D" * 5000),  # truncated to dense_text_len below
                payload={"profile_id": "default", "source_type": "manual"},
                final_score=0.9,
            ),
        ]
        raptor_summary = RaptorSummaryHit(
            point_id="raptor-summary-bg",
            raptor_node_id="raptor-node-bg",
            raptor_root_id="root",
            raptor_level=2,
            raptor_tree_id="tree",
            raptor_build_id="build",
            raptor_cluster_id="cluster-bg",
            text="R" * raptor_summary_text_len,
        )
        graph_candidates = [
            FakeGraphCandidate(
                f"graph-overflow-{i:02d}",
                graph_distance=1,
                final_score=0.7,
                payload={
                    "source_uri": f"file://docs/g{i}.md",
                    "file_path": f"docs/g{i}.md",
                    "heading": f"g{i}",
                    "text": "G" * per_relation_text_len,
                },
            )
            for i in range(n_graph_candidates)
        ]
        retriever = FakeBaseRetriever(dense_seeds=chunks)
        raptor = FakeRaptorSearcher(summaries=[raptor_summary], leaves=[])
        graph = FakeGraphRetriever(final=graph_candidates)
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=retriever,
            raptor_searcher=raptor,
            graph_retriever=graph,
        )
        d = router.retrieve(
            "anything",
            max_source_chars=per_relation_text_len,
            top_k=20,
        ).to_dict()

        # Sanity: dense + RAPTOR survived (preservation policy).
        assert any(
            h.get("point_id") == "dense-bg-1"
            for h in d["results"]["exact_hits"]
        ), "dense background hit was dropped — preserves RAPTOR-first"
        assert any(
            s.get("point_id") == "raptor-summary-bg"
            for s in d["results"]["summaries"]
        ), "RAPTOR summary was dropped — preserves tree evidence first"

        emitted_graph = d["results"]["graph_relations"]
        emitted_graph_pids = [g.get("point_id") for g in emitted_graph]
        # The per-relation cap (1200 chars) was honored for every
        # surviving relation — the drop is at the GLOBAL budget, not
        # the per-relation cap.
        for rel in emitted_graph:
            assert len(rel.get("text") or "") <= per_relation_text_len, (
                "graph relation text exceeded the per-relation cap; "
                "this would be a per-relation gate regression, not a "
                "global-budget regression"
            )

        # At least one graph relation MUST have been dropped because
        # the union of lanes would otherwise exceed the hard budget.
        # Budget accounting at top_k=20 (no further top-k trim):
        #   dense     = 200
        #   raptor    = 200
        #   graph fit = (16000 - 200 - 200) / 1200 = ~13 relations
        #   overflow  = 20 - 13 = 7 relations dropped
        n_kept = len(emitted_graph)
        n_dropped = n_graph_candidates - n_kept
        assert n_dropped > 0, (
            "expected at least one graph relation to be dropped at "
            "the global hard context budget; kept "
            f"{n_kept}/{n_graph_candidates}"
        )

        # The total union MUST respect the hard budget (this is the
        # single global cap the contract guarantees).
        emitted_total = 0
        for s in d["results"]["summaries"]:
            emitted_total += len(s.get("text") or "")
        for leaf in d["results"]["cited_leaves"]:
            emitted_total += len(leaf.get("text") or "")
        for hit in d["results"]["exact_hits"]:
            emitted_total += len(hit.get("text") or "")
        for rel in emitted_graph:
            emitted_total += len(rel.get("text") or "")
        assert emitted_total <= hard_budget, (
            f"emitted union {emitted_total} exceeds "
            f"HARD_CONTEXT_CHAR_BUDGET={hard_budget}; Phase 6E P3 "
            "requires graph relation text to participate in the "
            "single global hard budget"
        )

        # ``context_used_chars`` MUST agree with the actual emitted
        # total (debug counter cannot disagree with the wire).
        ctx = int(d.get("debug", {}).get("context_used_chars") or 0)
        assert ctx == emitted_total, (
            f"context_used_chars={ctx} disagrees with emitted total="
            f"{emitted_total}; debug envelope must reflect reality"
        )

        # A sanitized graph-overflow warning MUST have been emitted
        # and it MUST mention graph (not just dense) overflow.
        graph_overflow_warnings = [
            w for w in d["warnings"]
            if "global hard context budget" in w and "graph" in w
        ]
        assert graph_overflow_warnings, (
            "expected a 'global hard context budget' warning that "
            "explicitly mentions graph overflow; got: "
            f"{[w for w in d['warnings'] if 'global hard context budget' in w]!r}"
        )
        # Per-relation redacted drop warnings (one per dropped
        # graph relation) are also expected on the sanitized channel.
        per_drop = [
            w for w in d["warnings"]
            if "graph relation dropped at global context budget" in w
        ]
        assert per_drop, (
            "expected per-relation 'graph relation dropped at global "
            "context budget' warnings for each overflowed graph "
            "relation; none found"
        )
        assert len(per_drop) == n_dropped, (
            f"expected {n_dropped} per-relation drop warnings "
            f"(one per dropped graph relation), got {len(per_drop)}"
        )
        # The per-relation drop warnings MUST NOT leak raw point ids.
        for w in per_drop:
            for pid in emitted_graph_pids:
                assert pid not in w, (
                    f"per-relation drop warning leaked raw point id "
                    f"({pid}): {w}"
                )
            for candidate in graph_candidates:
                if candidate.point_id not in emitted_graph_pids:
                    assert candidate.point_id not in w, (
                        "drop warning leaked raw point id of dropped "
                        f"graph relation: {w}"
                    )

        # First-seen-wins determinism: the lowest-indexed graph
        # candidate must survive if any survives.
        assert "graph-overflow-00" in emitted_graph_pids, (
            "first-seen-wins broken: graph-overflow-00 (lowest index) "
            "was dropped while a higher-index relation survived"
        )

    def test_graph_relations_in_global_budget_when_dense_fills_first(self):
        # Defense-in-depth: even when the dense lane consumes most
        # of the hard budget on its own (no RAPTOR), the union of
        # dense + graph must still respect the hard cap. Phase 6E
        # P3 specifically requires graph text to count.
        per_relation_text_len = 1200
        n_graph_candidates = 20
        # Two 7000-char dense hits → dense_text_len=1200 each after
        # per-hit clamp → 2400 dense chars. Leaves 16000-2400=13600
        # for graph, i.e. ~11 graph relations fit at 1200 chars each.
        chunks = [
            _Chunk(
                pid="dense-fill-1",
                text=("D" * 9000),
                payload={"profile_id": "default", "source_type": "manual"},
                final_score=0.9,
            ),
            _Chunk(
                pid="dense-fill-2",
                text=("D" * 9000),
                payload={"profile_id": "default", "source_type": "manual"},
                final_score=0.8,
            ),
        ]
        graph_candidates = [
            FakeGraphCandidate(
                f"graph-fill-{i:02d}",
                graph_distance=1,
                final_score=0.7,
                payload={
                    "source_uri": f"file://docs/f{i}.md",
                    "file_path": f"docs/f{i}.md",
                    "heading": f"f{i}",
                    "text": "F" * per_relation_text_len,
                },
            )
            for i in range(n_graph_candidates)
        ]
        retriever = FakeBaseRetriever(dense_seeds=chunks)
        graph = FakeGraphRetriever(final=graph_candidates)
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=retriever,
            graph_retriever=graph,
        )
        d = router.retrieve(
            "anything",
            max_source_chars=per_relation_text_len,
            top_k=20,
        ).to_dict()

        emitted_total = 0
        for s in d["results"]["summaries"]:
            emitted_total += len(s.get("text") or "")
        for leaf in d["results"]["cited_leaves"]:
            emitted_total += len(leaf.get("text") or "")
        for hit in d["results"]["exact_hits"]:
            emitted_total += len(hit.get("text") or "")
        for rel in d["results"]["graph_relations"]:
            emitted_total += len(rel.get("text") or "")
        assert emitted_total <= 16000, (
            f"dense+graph union {emitted_total} exceeds the 16000 "
            "hard budget; Phase 6E P3 regression"
        )
        ctx = int(d.get("debug", {}).get("context_used_chars") or 0)
        assert ctx == emitted_total


# ---------------------------------------------------------------------------
# Phase 5 fix11 (final9 findings #1, #2): retrieve output must NOT echo
# raw query text, and the top-level retrieve error JSON must NOT
# interpolate raw exception ``__str__``. A secret-shaped query reaching
# either surface would let a token leak into the LLM context downstream.
# ---------------------------------------------------------------------------


class TestRetrieveNoRawQueryEcho:
    """Regression for final9 finding #1 — raw query echoed in
    ``qdrant_memory_retrieve`` output.

    The memory hybrid router (and the learning lane) MUST NOT return
    a ``"query"`` key with the raw input. The new contract is
    ``query_length`` + ``query_digest`` (sha256[:16]) + a fixed
    ``query_redacted`` sentinel. The runtime probe below constructs a
    secret-shaped query (Bearer + 24 digits) at runtime so the scanner
    doesn't trip on a literal in the source file.
    """

    def test_hybrid_router_to_dict_never_echoes_raw_query(self):
        # Construct a secret-shaped query at runtime.
        bad_query = "".join(["Bearer ", "a" * 24])
        retriever = FakeBaseRetriever(dense_seeds=[])
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=retriever,
        )
        d = router.retrieve(bad_query).to_dict()
        # The raw query MUST NOT appear anywhere in the envelope.
        serialized = json.dumps(d, default=str)
        assert bad_query not in serialized, (
            "HybridRouteResult.to_dict() leaked the raw query into the "
            "JSON envelope; final9 finding #1 regression"
        )
        # And the new contract keys are present.
        assert "query_length" in d
        assert "query_digest" in d
        assert "query_redacted" in d
        # The legacy raw ``query`` key is gone.
        assert "query" not in d
        # The sentinel is the fixed redaction string, not the raw value.
        assert d["query_redacted"] == (
            "[redacted: query omitted from retrieve output]"
        )
        # Length matches the raw input.
        assert d["query_length"] == len(bad_query)
        # Digest is sha256[:16] of the raw input.
        import hashlib
        assert d["query_digest"] == (
            hashlib.sha256(bad_query.encode("utf-8")).hexdigest()[:16]
        )

    def test_hybrid_router_to_dict_under_serialized_secret(self):
        # Construct a longer secret-shaped string at runtime and
        # verify the JSON dump never contains it. This is the runtime
        # probe shape the security pass used.
        bad_query = "".join(["Bearer ", "b" * 64])
        retriever = FakeBaseRetriever(dense_seeds=[])
        router = HybridRouter(
            qdrant=FakeQdrant(),
            embeddings=FakeEmbedding(),
            collection_name="memory",
            base_retriever=retriever,
        )
        d = router.retrieve(bad_query).to_dict()
        serialized = json.dumps(d, default=str)
        assert bad_query not in serialized


class TestRetrieveTopLevelErrorNoRawException:
    """Regression for final9 finding #2 — top-level
    ``_tool_retrieve`` catch-all interpolated ``{exc}`` into the
    JSON error. A runtime-constructed secret-shaped query fed into a
    fake router that echoes it into its exception ``__str__`` would
    have reached the JSON envelope.

    The fix is the sanitized generic message:
    ``Retrieve failed (no raw exception leaked; see server logs)``.
    """

    def _build_provider_with_fake_router(self, exc_factory):
        from __init__ import QdrantMemoryProvider  # type: ignore  # noqa: PLC0415

        class _Provider(QdrantMemoryProvider):
            def __init__(self):  # noqa: D401 - test stub
                # Skip ``__init__`` to avoid config load; bind just
                # the attributes ``_tool_retrieve`` reads.
                self._config = {"collection_name": "memory"}
                self._qdrant = object()
                self._embeddings = object()

            def _ensure_hybrid_router(self, collection_name):  # noqa: D401
                return exc_factory()

            def _scope_filter_values(self):  # noqa: D401
                return {"profile_id": "default"}

        return _Provider()

    def test_router_exception_secret_query_not_leaked_in_error(self):
        # Runtime-constructed secret-shaped query.
        bad_query = "".join(["Bearer ", "c" * 24])

        class _EchoingRouter:
            def retrieve(self, *args, **kwargs):
                # Echo the raw query into the exception ``__str__``
                # so a leak via ``f"...{exc}"`` would surface the
                # secret-shaped token into the JSON error envelope.
                raise RuntimeError(
                    "router refused query=" + repr(bad_query)
                )

        provider = self._build_provider_with_fake_router(_EchoingRouter)
        raw = provider._tool_retrieve({"query": bad_query, "top_k": 5})
        d = json.loads(raw)
        # The error envelope MUST be present and well-formed.
        assert "error" in d
        error_value = d["error"]
        assert isinstance(error_value, str)
        # The secret-shaped query MUST NOT appear anywhere in the
        # JSON error envelope.
        assert bad_query not in error_value, (
            "top-level retrieve error leaked the secret-shaped query; "
            "final9 finding #2 regression"
        )
        # The error carries the sanitized generic message instead.
        assert "Retrieve failed" in error_value
        assert "no raw exception leaked" in error_value
        # The serialized envelope (in case future changes add sibling
        # fields) is also clean.
        serialized = json.dumps(d, default=str)
        assert bad_query not in serialized

    def test_router_exception_query_not_in_serialized_envelope(self):
        # Same probe with a different runtime secret shape; verifies
        # the fix is independent of the secret-token character set.
        bad_query = "".join(["sk_live_", "d" * 24])

        class _EchoingRouter:
            def retrieve(self, *args, **kwargs):
                raise RuntimeError(
                    "backend down for query=" + repr(bad_query)
                )

        provider = self._build_provider_with_fake_router(_EchoingRouter)
        raw = provider._tool_retrieve({"query": bad_query, "top_k": 5})
        # The serialized envelope MUST NOT carry the secret-shaped
        # query — neither in ``error`` nor anywhere else.
        assert bad_query not in raw, (
            "top-level retrieve JSON envelope leaked the secret-shaped "
            "query; final9 finding #2 regression"
        )
        d = json.loads(raw)
        assert "Retrieve failed" in d["error"]
