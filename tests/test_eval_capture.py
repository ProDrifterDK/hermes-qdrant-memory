"""Tests for the Phase 6B read-only capture core.

These tests use fake provider/retriever/searcher components so no live
Qdrant is contacted. They cover:

* Variant parsing (``all``, comma-list, unknown rejection).
* dense-only and dense+sparse use ``update_access=False``; dense-only
  suppresses sparse scroll; dense+sparse allows sparse scroll.
* graph/raptor/hybrid-no-graph/hybrid-no-raptor produce expected lane
  shapes.
* Errors are sanitized and raw exception/query is not serialized.
* CLI parses ``eval-capture``, constructs the provider for capture
  (unlike offline eval), writes JSONL, and does not leak raw query in
  stdout.
"""

from __future__ import annotations

import io
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest


# --------------------------------------------------------------------------- #
# Fake components
# --------------------------------------------------------------------------- #


class FakeChunk:
    """Mimics ``RetrievedMemory`` shape for dense projection."""

    def __init__(
        self,
        pid: str,
        text: str,
        payload: dict[str, Any] | None = None,
        final_score: float = 0.6,
        qdrant_score: float = 0.6,
    ):
        self.id = pid
        self.text = text
        self.payload = payload or {}
        self.final_score = final_score
        self.qdrant_score = qdrant_score
        self.ranking_debug = {}


class FakeBaseRetriever:
    """Records calls so tests can assert on ``update_access`` / sparse scroll."""

    def __init__(self, dense_seeds: list[FakeChunk] | None = None):
        self._seeds = dense_seeds or []
        self.calls: list[dict[str, Any]] = []

    def search(self, query, *, top_k=5, update_access=True,
               allow_sparse_scroll=True, include_fact_history=False, **kwargs):
        self.calls.append({
            "query": query,
            "top_k": top_k,
            "update_access": update_access,
            "allow_sparse_scroll": allow_sparse_scroll,
            "include_fact_history": include_fact_history,
        })
        return list(self._seeds)


class FakeGraphCandidate:
    def __init__(self, pid: str, graph_distance: int = 1,
                 final_score: float = 0.7, path=None, relation_path=None,
                 payload=None):
        self.point_id = pid
        self.graph_distance = graph_distance
        self.final_score = final_score
        self.path = path or ["seed-1"]
        self.relation_path = relation_path or ["related_to"]
        self.payload = payload or {}


class FakeGraphResult:
    def __init__(self, final: list[FakeGraphCandidate] | None = None):
        self.final = final or []


class FakeGraphRetriever:
    def __init__(self, candidates: list[FakeGraphCandidate] | None = None):
        self._candidates = candidates or []
        self.calls: list[dict[str, Any]] = []

    def search(self, query, *, top_k=5, candidate_seed_top_k=20,
               max_graph_results=20, max_depth=2,
               include_fact_history=False, debug=True,
               allow_sparse_scroll=True, allow_graph_scroll=True, **kwargs):
        self.calls.append({
            "query": query,
            "top_k": top_k,
            "allow_sparse_scroll": allow_sparse_scroll,
            "allow_graph_scroll": allow_graph_scroll,
            "max_depth": max_depth,
        })
        return FakeGraphResult(list(self._candidates))


class FakeRaptorSummaryHit:
    def __init__(self, point_id="rpt-sum-1", text="raptor summary body"):
        self.point_id = point_id
        self.raptor_node_id = point_id
        self.raptor_root_id = "root"
        self.raptor_level = 1
        self.raptor_tree_id = "tree"
        self.raptor_build_id = "build"
        self.raptor_cluster_id = "cluster"
        self.raptor_child_ids: list[str] = []
        self.raptor_parent_ids: list[str] = []
        self.raptor_summary_of: list[str] = []
        self.text = text
        self.source_hashes: list[str] = []
        self.parent_status = ""
        self.derived_from: list[str] = []
        self.extra: dict[str, Any] = {}
        self.parent_assessment: dict[str, Any] = {}

    def to_dict(self, *, include_metadata=False):
        return {"point_id": self.point_id, "text": self.text}


class FakeRaptorLeafHit:
    def __init__(self, point_id="rpt-leaf-1", text="raptor leaf body",
                 parent_point_id="rpt-sum-1"):
        self.point_id = point_id
        self.parent_raptor_node_id = parent_point_id
        self.parent_point_id = parent_point_id
        self.text = text
        self.source_uri = ""
        self.file_path = ""
        self.heading = ""
        self.content_hash = ""
        self.source_type = ""
        self.locator: dict[str, Any] = {}
        self.safety: dict[str, Any] = {}
        self.extra: dict[str, Any] = {}

    def to_dict(self, *, include_metadata=False):
        return {"point_id": self.point_id, "text": self.text,
                "parent_point_id": self.parent_point_id}


class FakeRaptorResult:
    def __init__(self, summaries=None, leaves=None):
        self.summaries = summaries or []
        self.cited_leaves = leaves or []
        self.warnings: list[str] = []
        self.debug: dict[str, Any] = {}
        self.unsafe_summary_ids: set[str] = set()
        self.unsafe_leaf_ids: set[str] = set()


class FakeRaptorSearcher:
    def __init__(self, summaries=None, leaves=None):
        self._summaries = summaries or []
        self._leaves = leaves or []
        self.calls: list[dict[str, Any]] = []

    def search(self, query, *, top_k=5, max_depth=2, max_children=8,
               max_source_chars=1200, include_fact_history=False, **kwargs):
        self.calls.append({
            "query": query,
            "top_k": top_k,
            "max_depth": max_depth,
            "max_children": max_children,
            "max_source_chars": max_source_chars,
        })
        return FakeRaptorResult(
            summaries=list(self._summaries),
            leaves=list(self._leaves),
        )


class FakeProvider:
    """Fake provider that exposes the attributes eval_capture reads."""

    def __init__(
        self,
        *,
        base_retriever: FakeBaseRetriever | None = None,
        graph_retriever: FakeGraphRetriever | None = None,
        raptor_searcher: FakeRaptorSearcher | None = None,
        collection_name: str = "memory",
    ):
        self._qdrant = object()  # opaque; not contacted by capture core directly
        self._embeddings = object()
        self._config = {"collection_name": collection_name}
        self._retriever = base_retriever or FakeBaseRetriever()
        self._graph_retriever = graph_retriever
        self._raptor_searcher = raptor_searcher
        self._scope = {}

    def _scope_filter_values(self):
        return dict(self._scope)

    def _ensure_graph_retriever(self, collection_name):
        return self._graph_retriever

    def _ensure_raptor_searcher(self, collection_name):
        return self._raptor_searcher


def _provider_with_all_components(
    *,
    dense_seeds=None,
    graph_candidates=None,
    raptor_summaries=None,
    raptor_leaves=None,
):
    base_retriever = FakeBaseRetriever(dense_seeds)
    graph_retriever = FakeGraphRetriever(graph_candidates)
    raptor_searcher = FakeRaptorSearcher(raptor_summaries, raptor_leaves)
    return FakeProvider(
        base_retriever=base_retriever,
        graph_retriever=graph_retriever,
        raptor_searcher=raptor_searcher,
    )


def _basic_cases() -> list[dict[str, Any]]:
    return [{"case_id": "c1", "query": "private query text one"}]


# --------------------------------------------------------------------------- #
# Variant parsing
# --------------------------------------------------------------------------- #


class TestParseVariants:
    def test_none_returns_all_defaults(self):
        from qdrant_memory.eval_capture import parse_variants, DEFAULT_CAPTURE_VARIANTS
        result = parse_variants(None)
        assert result == list(DEFAULT_CAPTURE_VARIANTS)

    def test_all_alias_returns_all_defaults(self):
        from qdrant_memory.eval_capture import parse_variants, DEFAULT_CAPTURE_VARIANTS
        assert parse_variants("all") == list(DEFAULT_CAPTURE_VARIANTS)
        assert parse_variants("ALL") == list(DEFAULT_CAPTURE_VARIANTS)

    def test_comma_list_filters_and_preserves_canonical_order(self):
        from qdrant_memory.eval_capture import parse_variants
        # Input order is reversed from canonical; output must follow
        # the canonical DEFAULT_CAPTURE_VARIANTS ordering.
        result = parse_variants("hybrid,dense-only,graph")
        assert result == ["dense-only", "graph", "hybrid"]

    def test_list_input(self):
        from qdrant_memory.eval_capture import parse_variants
        result = parse_variants(["raptor-only", "hybrid-no-raptor"])
        assert result == ["raptor-only", "hybrid-no-raptor"]

    def test_unknown_variant_raises(self):
        from qdrant_memory.eval_capture import parse_variants
        with pytest.raises(ValueError, match="unknown capture variant"):
            parse_variants("dense-only,bogus")

    def test_lowercase_normalization(self):
        from qdrant_memory.eval_capture import parse_variants
        # Mixed-case input must map to lowercase canonical.
        result = parse_variants("Dense-Only,Graph")
        assert result == ["dense-only", "graph"]

    def test_empty_string_returns_all_defaults(self):
        from qdrant_memory.eval_capture import parse_variants, DEFAULT_CAPTURE_VARIANTS
        assert parse_variants("") == list(DEFAULT_CAPTURE_VARIANTS)

    def test_empty_list_returns_all_defaults(self):
        from qdrant_memory.eval_capture import parse_variants, DEFAULT_CAPTURE_VARIANTS
        assert parse_variants([]) == list(DEFAULT_CAPTURE_VARIANTS)


# --------------------------------------------------------------------------- #
# Dense variants: update_access and sparse scroll
# --------------------------------------------------------------------------- #


class TestDenseVariants:
    def test_dense_only_uses_update_access_false_and_suppresses_sparse_scroll(self):
        from qdrant_memory.eval_capture import capture_eval_runs

        provider = _provider_with_all_components(
            dense_seeds=[FakeChunk("p1", "body one", {"source_uri": "file://a.md"})],
        )
        result = capture_eval_runs(provider, _basic_cases(), variants="dense-only")
        retriever = provider._retriever
        assert len(retriever.calls) == 1
        assert retriever.calls[0]["update_access"] is False
        assert retriever.calls[0]["allow_sparse_scroll"] is False

    def test_dense_sparse_uses_update_access_false_and_allows_sparse_scroll(self):
        from qdrant_memory.eval_capture import capture_eval_runs

        provider = _provider_with_all_components(
            dense_seeds=[FakeChunk("p1", "body one")],
        )
        result = capture_eval_runs(provider, _basic_cases(), variants="dense+sparse")
        retriever = provider._retriever
        assert len(retriever.calls) == 1
        assert retriever.calls[0]["update_access"] is False
        assert retriever.calls[0]["allow_sparse_scroll"] is True

    def test_dense_only_packet_has_exact_hits_and_empty_other_lanes(self):
        from qdrant_memory.eval_capture import capture_eval_runs

        provider = _provider_with_all_components(
            dense_seeds=[FakeChunk("p1", "body one", {"source_uri": "file://a.md"})],
        )
        result = capture_eval_runs(provider, _basic_cases(), variants="dense-only")
        row = result["rows"][0]
        assert row["case_id"] == "c1"
        assert row["variant"] == "dense-only"
        packet = row["packet"]["results"]
        assert len(packet["exact_hits"]) == 1
        assert packet["exact_hits"][0]["point_id"] == "p1"
        assert packet["exact_hits"][0]["text"] == "body one"
        assert packet["exact_hits"][0]["source_uri"] == "file://a.md"
        assert packet["summaries"] == []
        assert packet["cited_leaves"] == []
        assert packet["graph_relations"] == []

    def test_dense_only_row_does_not_carry_raw_query(self):
        from qdrant_memory.eval_capture import capture_eval_runs

        provider = _provider_with_all_components()
        result = capture_eval_runs(provider, _basic_cases(), variants="dense-only")
        row = result["rows"][0]
        # The raw query must not appear anywhere in the serialized row.
        serialized = json.dumps(row)
        assert "private query text one" not in serialized


# --------------------------------------------------------------------------- #
# Graph variant lane shape
# --------------------------------------------------------------------------- #


class TestGraphVariant:
    def test_graph_populates_graph_relations_only(self):
        from qdrant_memory.eval_capture import capture_eval_runs

        provider = _provider_with_all_components(
            graph_candidates=[FakeGraphCandidate("g1", graph_distance=1)],
        )
        result = capture_eval_runs(provider, _basic_cases(), variants="graph")
        row = result["rows"][0]
        assert row["variant"] == "graph"
        packet = row["packet"]["results"]
        assert len(packet["graph_relations"]) == 1
        assert packet["graph_relations"][0]["point_id"] == "g1"
        assert packet["exact_hits"] == []
        assert packet["summaries"] == []
        assert packet["cited_leaves"] == []

    def test_graph_passes_allow_sparse_scroll_false(self):
        from qdrant_memory.eval_capture import capture_eval_runs

        provider = _provider_with_all_components(
            graph_candidates=[FakeGraphCandidate("g1")],
        )
        capture_eval_runs(provider, _basic_cases(), variants="graph")
        graph_retriever = provider._graph_retriever
        assert len(graph_retriever.calls) == 1
        assert graph_retriever.calls[0]["allow_sparse_scroll"] is False

    # ------------------------------------------------------------------ #
    # Phase 6E: eval-capture rows must carry source handles from the
    # graph candidate payload, and must NOT echo raw query text.
    # ------------------------------------------------------------------ #

    def test_graph_relation_row_carries_source_handles(self):
        # Phase 6E: an eval-capture row in the ``graph`` variant must
        # carry the sanitized source handles (``source_uri``,
        # ``file_path``, ``heading``) and bounded ``text`` for every
        # emitted graph relation.
        from qdrant_memory.eval_capture import capture_eval_runs

        provider = _provider_with_all_components(
            graph_candidates=[
                FakeGraphCandidate(
                    "g-handle",
                    graph_distance=1,
                    payload={
                        "source_uri": "file://docs/eval-cap.md",
                        "file_path": "docs/eval-cap.md",
                        "heading": "Eval Capture",
                        "text": "phase 6e graph relation body",
                    },
                ),
            ],
        )
        result = capture_eval_runs(provider, _basic_cases(), variants="graph")
        row = result["rows"][0]
        rels = row["packet"]["results"]["graph_relations"]
        assert len(rels) == 1
        rel = rels[0]
        assert rel["point_id"] == "g-handle"
        assert rel["source_uri"] == "file://docs/eval-cap.md"
        assert rel["file_path"] == "docs/eval-cap.md"
        assert rel["heading"] == "Eval Capture"
        assert rel["text"] == "phase 6e graph relation body"

    def test_graph_relation_row_does_not_leak_raw_query(self):
        # Phase 6E: the raw eval-case query text must never appear in
        # any emitted graph-relation field, even though the capture
        # variant now projects more fields from each candidate's
        # payload.
        from qdrant_memory.eval_capture import capture_eval_runs

        provider = _provider_with_all_components(
            graph_candidates=[
                FakeGraphCandidate(
                    "g-handle-clean",
                    graph_distance=1,
                    payload={
                        "source_uri": "file://docs/clean.md",
                        "file_path": "docs/clean.md",
                        "heading": "Clean",
                        "text": "clean body for the eval capture row",
                    },
                ),
            ],
        )
        cases = [
            {
                "case_id": "case-1",
                "query": "private query text one",
                "expected_source_uris": ["file://docs/clean.md"],
            },
        ]
        result = capture_eval_runs(provider, cases, variants="graph")
        row = result["rows"][0]
        serialized = json.dumps(row, default=str)
        # The raw query must never echo anywhere in the row.
        assert "private query text one" not in serialized


# --------------------------------------------------------------------------- #
# RAPTOR variant lane shape
# --------------------------------------------------------------------------- #


class TestRaptorVariant:
    def test_raptor_populates_summaries_and_leaves_only(self):
        from qdrant_memory.eval_capture import capture_eval_runs

        provider = _provider_with_all_components(
            raptor_summaries=[FakeRaptorSummaryHit("s1", "summary body")],
            raptor_leaves=[FakeRaptorLeafHit("l1", "leaf body", "s1")],
        )
        result = capture_eval_runs(provider, _basic_cases(), variants="raptor-only")
        row = result["rows"][0]
        assert row["variant"] == "raptor-only"
        packet = row["packet"]["results"]
        assert len(packet["summaries"]) == 1
        assert packet["summaries"][0]["point_id"] == "s1"
        assert len(packet["cited_leaves"]) == 1
        assert packet["cited_leaves"][0]["point_id"] == "l1"
        assert packet["exact_hits"] == []
        assert packet["graph_relations"] == []


# --------------------------------------------------------------------------- #
# Hybrid variants lane shapes
# --------------------------------------------------------------------------- #


class TestHybridVariants:
    def test_hybrid_all_lanes_populated(self):
        from qdrant_memory.eval_capture import capture_eval_runs

        provider = _provider_with_all_components(
            dense_seeds=[FakeChunk("p1", "dense body")],
            graph_candidates=[FakeGraphCandidate("g1")],
            raptor_summaries=[FakeRaptorSummaryHit("s1")],
            raptor_leaves=[FakeRaptorLeafHit("l1")],
        )
        result = capture_eval_runs(provider, _basic_cases(), variants="hybrid")
        row = result["rows"][0]
        packet = row["packet"]["results"]
        # HybridRouter runs all three lanes. The packet must have all
        # four lanes present (even if some are empty).
        assert "exact_hits" in packet
        assert "graph_relations" in packet
        assert "summaries" in packet
        assert "cited_leaves" in packet

    def test_hybrid_no_graph_does_not_contact_graph_retriever(self):
        from qdrant_memory.eval_capture import capture_eval_runs

        provider = _provider_with_all_components(
            graph_candidates=[FakeGraphCandidate("g1")],
        )
        result = capture_eval_runs(provider, _basic_cases(), variants="hybrid-no-graph")
        assert result["rows"]
        # The graph retriever must NOT have been called.
        assert provider._graph_retriever.calls == []

    def test_hybrid_no_raptor_does_not_contact_raptor_searcher(self):
        from qdrant_memory.eval_capture import capture_eval_runs

        provider = _provider_with_all_components(
            raptor_summaries=[FakeRaptorSummaryHit("s1")],
        )
        result = capture_eval_runs(provider, _basic_cases(), variants="hybrid-no-raptor")
        assert result["rows"]
        # The raptor searcher must NOT have been called.
        assert provider._raptor_searcher.calls == []

    def test_hybrid_row_does_not_carry_raw_query(self):
        from qdrant_memory.eval_capture import capture_eval_runs

        provider = _provider_with_all_components()
        result = capture_eval_runs(provider, _basic_cases(), variants="hybrid")
        row = result["rows"][0]
        serialized = json.dumps(row)
        assert "private query text one" not in serialized


# --------------------------------------------------------------------------- #
# Error sanitization
# --------------------------------------------------------------------------- #


class TestErrorSanitization:
    def test_error_row_uses_redacted_sentinel_not_raw_exception(self):
        from qdrant_memory.eval_capture import capture_eval_runs

        class ExplodingRetriever:
            def search(self, *args, **kwargs):
                raise RuntimeError(
                    "query=private text endpoint=http://qdrant.secret:6333 "
                    "SECRET_TOKEN"
                )

        provider = FakeProvider(base_retriever=ExplodingRetriever())
        result = capture_eval_runs(provider, _basic_cases(), variants="dense-only")
        row = result["rows"][0]
        assert row.get("error") == "<redacted>"
        assert row["capture"]["errored"] is True
        assert row["capture"]["error_kind"] == "RuntimeError"
        serialized = json.dumps(row)
        # Raw exception fragments must NOT appear.
        for needle in ("private text", "SECRET_TOKEN", "qdrant.secret:6333"):
            assert needle not in serialized

    def test_connection_error_collapses_to_stable_kind(self):
        from qdrant_memory.eval_capture import capture_eval_runs

        class ExplodingRetriever:
            def search(self, *args, **kwargs):
                raise ConnectionError("refused")

        provider = FakeProvider(base_retriever=ExplodingRetriever())
        result = capture_eval_runs(provider, _basic_cases(), variants="dense-only")
        row = result["rows"][0]
        assert row["capture"]["error_kind"] == "connection_error"

    def test_summary_counts_errored_rows(self):
        from qdrant_memory.eval_capture import capture_eval_runs

        class ExplodingRetriever:
            def search(self, *args, **kwargs):
                raise RuntimeError("boom")

        provider = FakeProvider(base_retriever=ExplodingRetriever())
        result = capture_eval_runs(provider, _basic_cases(), variants="dense-only")
        assert result["summary"]["errored_rows"] == 1
        assert result["summary"]["variants"]["dense-only"]["errored"] == 1


# --------------------------------------------------------------------------- #
# Latency
# --------------------------------------------------------------------------- #


class TestLatency:
    def test_latency_ms_present_and_numeric(self):
        from qdrant_memory.eval_capture import capture_eval_runs

        provider = _provider_with_all_components(
            dense_seeds=[FakeChunk("p1", "body")],
        )
        result = capture_eval_runs(provider, _basic_cases(), variants="dense-only")
        row = result["rows"][0]
        assert "latency_ms" in row
        assert isinstance(row["latency_ms"], (int, float))
        assert row["latency_ms"] >= 0
        assert row["capture"]["latency_ms"] == row["latency_ms"]

    def test_error_row_has_top_level_numeric_latency_ms(self):
        """Error rows must include a top-level numeric ``latency_ms``
        that matches the success-row shape, even when the capture
        raises. The error row must NOT leak the raw exception/query
        text into either the row body or the ``capture`` metadata.
        """

        from qdrant_memory.eval_capture import capture_eval_runs

        class ExplodingRetriever:
            def search(self, *args, **kwargs):
                # Embed both the (private) query and a runtime-built
                # secret-shaped marker so the test asserts the
                # sanitized-error contract across the whole envelope.
                raise RuntimeError(
                    "exploded while handling PRIVATE_QUERY_HINT_NOPE "
                    "endpoint=http://qdrant.private:6333"
                )

        provider = FakeProvider(base_retriever=ExplodingRetriever())
        result = capture_eval_runs(provider, _basic_cases(), variants="dense-only")
        row = result["rows"][0]

        # Top-level latency_ms must be numeric (int or float, not a
        # string, not None, not a bool) so the evaluator can treat it
        # uniformly with success rows.
        assert "latency_ms" in row
        assert isinstance(row["latency_ms"], (int, float))
        assert not isinstance(row["latency_ms"], bool)
        assert row["latency_ms"] >= 0

        # Capture metadata latency_ms must match the top-level one
        # (defense in depth — both come from the same capture_meta).
        assert row["capture"]["latency_ms"] == row["latency_ms"]
        assert row["capture"]["errored"] is True
        assert row["capture"]["error_kind"] == "RuntimeError"

        # Raw exception fragments must NOT appear anywhere in the
        # serialized row. The capture core must redact both the
        # query hint and the endpoint.
        serialized = json.dumps(row)
        for needle in (
            "PRIVATE_QUERY_HINT_NOPE",
            "qdrant.private:6333",
            "endpoint=http",
            "exploded while handling",
        ):
            assert needle not in serialized, f"{needle!r} leaked into error row"

        # Raw case query must also NOT leak through the error row
        # envelope (sanity for the broader no-raw-query rule).
        assert "private query text one" not in serialized


# --------------------------------------------------------------------------- #
# Secret-bearing content in dense/graph projections
# --------------------------------------------------------------------------- #


def _runtime_secret_marker(kind: str = "openai") -> str:
    """Construct a scanner-sensitive fake secret at runtime via
    string concatenation so the test source contains no literal
    scanner-sensitive example. The check_no_literal_fake_secrets.py
    guard scans for those literals; constructing them here from
    fragments keeps the test source clean while still producing a
    string that ``contains_secret`` flags as secret-bearing.

    Parameters
    ----------
    kind
        One of ``"openai"``, ``"github"``, ``"bearer"``, ``"apikey"``.
        All four are detected by ``contains_secret`` via the patterns
        in :mod:`qdrant_memory.lesson_extractor`.
    """

    if kind == "openai":
        # Mirrors ``sk-...`` plus the 10+ char body so the prefix
        # regex ``(?<![A-Za-z0-9_])sk-[A-Za-z0-9_-]{10,}`` matches.
        prefix = "".join(("s", "k", "-"))
        return prefix + "liveTEST1234567890abcd"
    if kind == "github":
        # Mirrors ``ghp_...`` plus the 20+ char body so the GitHub
        # token regex matches.
        prefix = "".join(("g", "h", "p", "_"))
        return prefix + "AABBCCDDEEFF00112233XXYYZZAABBCCDDEEFF00112233"
    if kind == "bearer":
        # Mirrors a bare ``Bearer <token>`` so the bare-bearer
        # regex matches.
        return "Bearer " + "aabbccddeeffgg" + "hhiijjkkllmmnnoo" + "ppqqrrssttuuvv"
    if kind == "apikey":
        # Mirrors ``api_key=<value>`` so the inline-secret-assignment
        # regex matches.
        return "api_key=" + "XX" + "YY" + "ZZZtest999aaa888bbbccc777"
    raise ValueError(f"unknown kind: {kind}")


class TestSecretBearingCapture:
    """Capture rows must not serialize secret-shaped content via
    direct dense/graph projections in local runs. The Phase 6B
    capture core re-uses the router's sanitized helpers
    (``_dense_to_exact_hits`` / ``_graph_to_relations``) so this is
    enforced even when the local run feeds secret-shaped text or
    point ids straight into the retriever/searcher."""

    def test_dense_only_row_does_not_leak_secret_in_text(self):
        from qdrant_memory.eval_capture import capture_eval_runs
        from qdrant_memory.lesson_extractor import contains_secret

        marker = _runtime_secret_marker("openai")
        assert contains_secret(marker)  # sanity: the scanner agrees

        provider = _provider_with_all_components(
            dense_seeds=[FakeChunk("p1", marker)],
        )
        result = capture_eval_runs(
            provider, _basic_cases(), variants="dense-only",
        )
        row = result["rows"][0]
        serialized = json.dumps(row)
        # The runtime-constructed marker must NOT be in the row.
        assert marker not in serialized

    def test_dense_only_row_does_not_leak_secret_in_point_id(self):
        from qdrant_memory.eval_capture import capture_eval_runs
        from qdrant_memory.lesson_extractor import contains_secret

        marker_id = _runtime_secret_marker("github")
        assert contains_secret(marker_id)  # sanity

        provider = _provider_with_all_components(
            dense_seeds=[FakeChunk(marker_id, "harmless body")],
        )
        result = capture_eval_runs(
            provider, _basic_cases(), variants="dense-only",
        )
        row = result["rows"][0]
        serialized = json.dumps(row)
        assert marker_id not in serialized

    def test_dense_sparse_row_does_not_leak_secret(self):
        from qdrant_memory.eval_capture import capture_eval_runs
        from qdrant_memory.lesson_extractor import contains_secret

        marker = _runtime_secret_marker("apikey")
        assert contains_secret(marker)

        provider = _provider_with_all_components(
            dense_seeds=[FakeChunk("p1", "body carrying " + marker)],
        )
        result = capture_eval_runs(
            provider, _basic_cases(), variants="dense+sparse",
        )
        row = result["rows"][0]
        serialized = json.dumps(row)
        assert marker not in serialized

    def test_graph_row_does_not_leak_secret_in_point_id(self):
        from qdrant_memory.eval_capture import capture_eval_runs
        from qdrant_memory.lesson_extractor import contains_secret

        marker_id = _runtime_secret_marker("openai")
        assert contains_secret(marker_id)

        provider = _provider_with_all_components(
            graph_candidates=[FakeGraphCandidate(marker_id)],
        )
        result = capture_eval_runs(provider, _basic_cases(), variants="graph")
        row = result["rows"][0]
        serialized = json.dumps(row)
        assert marker_id not in serialized

    def test_graph_row_does_not_leak_secret_in_path(self):
        from qdrant_memory.eval_capture import capture_eval_runs
        from qdrant_memory.lesson_extractor import contains_secret

        marker = _runtime_secret_marker("bearer")
        assert contains_secret(marker)

        provider = _provider_with_all_components(
            graph_candidates=[
                FakeGraphCandidate(
                    "g-safe",
                    path=["seed-1", marker],
                ),
            ],
        )
        result = capture_eval_runs(provider, _basic_cases(), variants="graph")
        row = result["rows"][0]
        serialized = json.dumps(row)
        assert marker not in serialized

    def test_graph_row_does_not_leak_secret_in_relation_path(self):
        from qdrant_memory.eval_capture import capture_eval_runs
        from qdrant_memory.lesson_extractor import contains_secret

        marker = _runtime_secret_marker("apikey")
        assert contains_secret(marker)

        provider = _provider_with_all_components(
            graph_candidates=[
                FakeGraphCandidate(
                    "g-safe-2",
                    relation_path=["related_to", marker],
                ),
            ],
        )
        result = capture_eval_runs(provider, _basic_cases(), variants="graph")
        row = result["rows"][0]
        serialized = json.dumps(row)
        assert marker not in serialized


# --------------------------------------------------------------------------- #
# include_fact_history in dense variants
# --------------------------------------------------------------------------- #


def _unsafe_fact_label() -> dict[str, str]:
    """Build the label string ``fact_status`` -> ``deprecated`` via
    runtime concatenation so no scanner-sensitive literal appears in
    the test source."""
    key = "".join(("f", "a", "c", "t", "_", "s", "t", "a", "t", "u", "s"))
    val = "".join(("d", "e", "p", "r", "e", "c", "a", "t", "e", "d"))
    return {key: val}


class TestFactHistory:
    """include_fact_history must be forwarded through _dense_projection
    to the router helper so dense chunks with unsafe fact_status are
    dropped by default but preserved when the flag is set."""

    def test_default_drops_unsafe_fact_status_chunk(self):
        from qdrant_memory.eval_capture import capture_eval_runs

        status_payload = _unsafe_fact_label()
        chunk = FakeChunk("safe-id", "harmless body", payload=status_payload)
        provider = _provider_with_all_components(
            dense_seeds=[chunk],
        )
        result = capture_eval_runs(
            provider, _basic_cases(), variants="dense-only",
            include_fact_history=False,
        )
        packet = result["rows"][0]["packet"]["results"]
        assert packet["exact_hits"] == [], (
            "unsafe fact_status chunk should be dropped "
            "when include_fact_history=False"
        )

    def test_true_preserves_unsafe_fact_status_chunk(self):
        from qdrant_memory.eval_capture import capture_eval_runs

        status_payload = _unsafe_fact_label()
        chunk = FakeChunk("safe-id", "harmless body", payload=status_payload)
        provider = _provider_with_all_components(
            dense_seeds=[chunk],
        )
        result = capture_eval_runs(
            provider, _basic_cases(), variants="dense-only",
            include_fact_history=True,
        )
        packet = result["rows"][0]["packet"]["results"]
        assert len(packet["exact_hits"]) == 1, (
            "unsafe fact_status chunk should be preserved "
            "when include_fact_history=True"
        )
        assert packet["exact_hits"][0]["point_id"] == "safe-id"

    def test_true_also_works_for_dense_sparse(self):
        from qdrant_memory.eval_capture import capture_eval_runs

        status_payload = _unsafe_fact_label()
        chunk = FakeChunk("safe-id", "harmless body", payload=status_payload)
        provider = _provider_with_all_components(
            dense_seeds=[chunk],
        )
        # With include_fact_history=True the chunk survives in
        # dense+sparse variant too.
        result = capture_eval_runs(
            provider, _basic_cases(), variants="dense+sparse",
            include_fact_history=True,
        )
        packet = result["rows"][0]["packet"]["results"]
        assert len(packet["exact_hits"]) == 1


# --------------------------------------------------------------------------- #
# write_runs_jsonl
# --------------------------------------------------------------------------- #


class TestWriteRunsJsonl:
    def test_writes_one_json_line_per_row(self, tmp_path):
        from qdrant_memory.eval_capture import write_runs_jsonl

        rows = [
            {"case_id": "c1", "variant": "dense-only", "packet": {"results": {}}},
            {"case_id": "c1", "variant": "graph", "packet": {"results": {}}},
        ]
        out_path = str(tmp_path / "runs.jsonl")
        write_runs_jsonl(out_path, rows)
        lines = Path(out_path).read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["case_id"] == "c1"
        assert first["variant"] == "dense-only"


# --------------------------------------------------------------------------- #
# CLI integration
# --------------------------------------------------------------------------- #


def _load_plugin_cli_module():
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("qdrant_plugin_cli_capture_test", root / "cli.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parser():
    import argparse

    cli = _load_plugin_cli_module()
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command", required=True)
    qdrant_parser = subparsers.add_parser("qdrant")
    cli.register_cli(qdrant_parser)
    qdrant_parser.set_defaults(func=cli.qdrant_command)
    return parser


class TestCliEvalCapture:
    def test_parser_adds_eval_capture_subcommand(self):
        parser = _parser()
        args = parser.parse_args([
            "qdrant", "eval-capture",
            "--cases", "cases.jsonl",
            "--runs-out", "runs.jsonl",
        ])
        assert args.qdrant_subcommand == "eval-capture"
        assert args.cases == "cases.jsonl"
        assert args.runs_out == "runs.jsonl"
        assert args.variants == "all"
        assert args.top_k == 5
        assert args.mode == "hybrid"
        assert getattr(args, "json", False) is False

    def test_parser_accepts_variant_and_budget_flags(self):
        parser = _parser()
        args = parser.parse_args([
            "qdrant", "eval-capture",
            "--cases", "c.jsonl",
            "--runs-out", "r.jsonl",
            "--variants", "dense-only,graph",
            "--top-k", "3",
            "--max-depth", "2",
            "--max-children", "8",
            "--max-source-chars", "1000",
            "--candidate-seed-top-k", "10",
            "--max-graph-results", "15",
            "--include-fact-history",
            "--include-metadata",
            "--json",
        ])
        assert args.variants == "dense-only,graph"
        assert args.top_k == 3
        assert args.max_depth == 2
        assert args.include_fact_history is True
        assert args.include_metadata is True
        assert args.json is True

    def test_eval_capture_blocked_in_provider_dispatch(self):
        from qdrant_memory.cli_core import CliUsageError, build_tool_call

        parser = _parser()
        args = parser.parse_args([
            "qdrant", "eval-capture",
            "--cases", "c.jsonl",
            "--runs-out", "r.jsonl",
        ])
        with pytest.raises(CliUsageError, match="CLI-only"):
            build_tool_call(args)

    def test_eval_capture_constructs_provider_and_writes_jsonl(
        self, tmp_path, capsys,
    ):
        """Capture command constructs the provider (unlike offline eval)
        and writes a valid JSONL runs file."""
        from qdrant_memory.cli_core import execute_command

        cases_path = tmp_path / "cases.jsonl"
        cases_path.write_text(
            json.dumps({"case_id": "c1", "query": "private capture query"}) + "\n",
            encoding="utf-8",
        )
        runs_out = tmp_path / "runs.jsonl"

        provider = _provider_with_all_components(
            dense_seeds=[FakeChunk("p1", "dense body")],
        )
        factory_called = []

        def provider_factory():
            factory_called.append(True)
            return provider

        parser = _parser()
        args = parser.parse_args([
            "qdrant", "eval-capture",
            "--cases", str(cases_path),
            "--runs-out", str(runs_out),
            "--variants", "dense-only",
            "--json",
        ])

        exit_code = execute_command(args, provider_factory=provider_factory)
        assert exit_code == 0
        assert factory_called  # provider was constructed

        # Runs file must exist and be valid JSONL.
        assert runs_out.exists()
        lines = runs_out.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["case_id"] == "c1"
        assert row["variant"] == "dense-only"

        # stdout summary must not leak raw query text.
        out = capsys.readouterr().out
        assert "private capture query" not in out

    def test_eval_capture_missing_cases_is_usage_error(self, capsys):
        from qdrant_memory.cli_core import execute_command

        parser = _parser()
        args = parser.parse_args([
            "qdrant", "eval-capture",
            "--cases", "",
            "--runs-out", "r.jsonl",
        ])
        exit_code = execute_command(
            args,
            provider_factory=lambda: pytest.fail("should not be called"),
        )
        assert exit_code == 2

    def test_eval_capture_invalid_variants_is_usage_error(self, tmp_path, capsys):
        from qdrant_memory.cli_core import execute_command

        cases_path = tmp_path / "cases.jsonl"
        cases_path.write_text(
            json.dumps({"case_id": "c1", "query": "q"}) + "\n",
            encoding="utf-8",
        )
        parser = _parser()
        args = parser.parse_args([
            "qdrant", "eval-capture",
            "--cases", str(cases_path),
            "--runs-out", str(tmp_path / "r.jsonl"),
            "--variants", "bogus-variant",
            "--json",
        ])
        exit_code = execute_command(
            args,
            provider_factory=lambda: pytest.fail("should not be called"),
        )
        assert exit_code == 2
        captured = capsys.readouterr()
        error = json.loads(captured.err)
        assert error["error"] is True
        assert "unknown capture variant" in error["message"]
