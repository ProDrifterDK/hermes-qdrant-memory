from __future__ import annotations

from typing import Any

import pytest

from qdrant_memory.retriever import MemoryRetriever
from qdrant_memory.sparse_search import (
    SparseScore,
    combine_score,
    extract_signals,
    fetch_sparse_candidates,
    has_strong_signal,
    is_payload_visible,
    merge_candidates,
    score_candidates,
)


# ---------------------------------------------------------------------------
# Tokenizer behavior
# ---------------------------------------------------------------------------


def test_extract_signals_recognizes_uuid_issue_route_and_dotted_symbol():
    query = "see 550e8400-e29b-41d4-a716-446655440000 in /api/v1/projects and SMDFS-455 (pkg.mod.Class)"
    signals = extract_signals(query)
    tokens = set(signals.tokens)
    assert "550e8400-e29b-41d4-a716-446655440000" in tokens
    assert "SMDFS-455" in tokens
    assert any(tok.startswith("/api/") for tok in tokens)
    assert "pkg.mod.Class" in tokens


def test_extract_signals_recognizes_error_literal_and_status_code():
    signals = extract_signals("TypeError: connection refused (HTTP 503)")
    tokens = set(signals.tokens)
    assert any("TypeError" in tok for tok in tokens)
    assert "HTTP 503" in tokens


def test_extract_signals_drops_generic_stopwords():
    signals = extract_signals("the quick brown fox")
    # No exact-signal patterns should be emitted from pure stopwords.
    for tok in signals.tokens:
        assert tok.lower() not in {"the", "quick", "brown", "fox"}


def test_has_strong_signal_only_for_literal_queries():
    assert has_strong_signal("recall SMDFS-455")
    assert has_strong_signal("see /api/v1/projects endpoint")
    assert has_strong_signal("TypeError: foo")
    assert has_strong_signal("look up 550e8400-e29b-41d4-a716-446655440000")
    assert has_strong_signal("pkg.mod.Class is broken")
    assert not has_strong_signal("how do we deploy this thing?")
    assert not has_strong_signal("")
    assert not has_strong_signal("general semantic question about history")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_score_candidates_ranks_exact_uuid_hit_above_decoy():
    candidates = [
        {
            "id": "decoy-1",
            "payload": {
                "text": "unrelated notes about deployment",
                "source_type": "project_doc",
                "importance": 5,
            },
        },
        {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "payload": {
                "text": "incident ticket points to error 503",
                "source_type": "incident",
                "importance": 7,
            },
        },
    ]
    scores = score_candidates("recall 550e8400-e29b-41d4-a716-446655440000", candidates)
    by_id = {score.point_id: score for score in scores}
    assert by_id["550e8400-e29b-41d4-a716-446655440000"].score > by_id["decoy-1"].score
    assert by_id["550e8400-e29b-41d4-a716-446655440000"].literal_hit is True
    assert by_id["decoy-1"].score == 0.0


def test_score_candidates_recognizes_api_route_hit_in_file_path():
    candidates = [
        {
            "id": "wrong-route",
            "payload": {
                "text": "misc",
                "file_path": "/repo/src/other/route.py",
            },
        },
        {
            "id": "matching-route",
            "payload": {
                "text": "v1 docs",
                "file_path": "/repo/docs/api/v1/projects.md",
            },
        },
    ]
    scores = score_candidates("see /api/v1/projects", candidates)
    by_id = {score.point_id: score for score in scores}
    assert by_id["matching-route"].score > 0
    assert by_id["wrong-route"].score == 0


def test_score_candidates_drops_quarantined_payload():
    candidates = [
        {
            "id": "q-1",
            "payload": {"text": "important looking notes", "consolidation_quarantined": True},
        },
    ]
    scores = score_candidates("recall SMDFS-455", candidates)
    assert len(scores) == 1
    assert scores[0].quarantined is True
    assert scores[0].score == 0.0
    assert scores[0].payload_invisible is True


def test_score_candidates_drops_secret_bearing_payload():
    candidates = [
        {
            "id": "secret-1",
            "payload": {"text": "see Authorization: Bearer <REDACTED_BEARER> here"},
        },
        {
            "id": "clean-1",
            "payload": {"text": "this is a clean note about SMDFS-455"},
        },
    ]
    scores = score_candidates("recall SMDFS-455", candidates)
    by_id = {score.point_id: score for score in scores}
    assert by_id["secret-1"].secret_blocked is True
    assert by_id["secret-1"].score == 0.0
    assert by_id["clean-1"].score > 0


def test_score_candidates_returns_zero_scores_for_natural_language_query():
    candidates = [
        {"id": "p1", "payload": {"text": "notes about deploys"}},
    ]
    scores = score_candidates("how do we deploy this thing?", candidates)
    # No exact-signal tokens → no positive scoring even if query passes the
    # strong-signal gate in the retriever (this test calls score_candidates
    # directly with a plain query).
    assert all(score.score == 0.0 for score in scores)


def test_score_candidates_aggregates_multiple_field_hits():
    candidates = [
        {
            "id": "multi-1",
            "payload": {
                "text": "SMDFS-455 fix landed",
                "heading": "SMDFS-455",
                "file_path": "/repo/issues/SMDFS-455.md",
            },
        }
    ]
    scores = score_candidates("SMDFS-455 fix", candidates)
    assert scores[0].score > 0
    field_hits = scores[0].field_hits
    # The point id is not a literal hit here, but multiple compact fields match.
    assert sum(field_hits.values()) >= 2


# ---------------------------------------------------------------------------
# fetch_sparse_candidates defensive scroll wrapper
# ---------------------------------------------------------------------------


class _NoScrollQdrant:
    """Qdrant stand-in that does NOT expose scroll_by_filter."""


class _ScrollQdrant:
    def __init__(self, points):
        self.points = points
        self.calls = []

    def scroll_by_filter(self, name, flt, *, limit=256, with_payload=True, with_vector=False, max_total=None):
        self.calls.append({"name": name, "flt": flt, "limit": limit, "max_total": max_total})
        cap = max_total if max_total is not None else limit
        return list(self.points)[:cap]


def test_fetch_sparse_candidates_returns_empty_when_scroll_missing():
    qdrant = _NoScrollQdrant()
    assert fetch_sparse_candidates(qdrant, collection_name="memory", flt={"must": []}) == []


def test_fetch_sparse_candidates_refuses_empty_filter():
    qdrant = _ScrollQdrant([{"id": "x", "payload": {"text": "hi"}}])
    assert fetch_sparse_candidates(qdrant, collection_name="memory", flt={}) == []
    assert qdrant.calls == []


def test_fetch_sparse_candidates_caps_total():
    points = [{"id": f"p-{i}", "payload": {"text": "SMDFS-455"}} for i in range(20)]
    qdrant = _ScrollQdrant(points)
    out = fetch_sparse_candidates(qdrant, collection_name="memory", flt={"must": [{"key": "k", "match": {"value": "v"}}]}, candidate_cap=5)
    assert len(out) == 5
    assert qdrant.calls[-1]["max_total"] == 5


def test_fetch_sparse_candidates_swallows_exceptions():
    class BoomQdrant:
        def scroll_by_filter(self, *args, **kwargs):
            raise RuntimeError("kaboom")

    assert fetch_sparse_candidates(BoomQdrant(), collection_name="memory", flt={"must": [{"key": "k", "match": {"value": "v"}}]}) == []


# ---------------------------------------------------------------------------
# merge_candidates / combine_score
# ---------------------------------------------------------------------------


def test_merge_candidates_dense_only_when_no_sparse():
    dense = [
        {"id": "d1", "score": 0.9, "payload": {"text": "alpha"}},
        {"id": "d2", "score": 0.7, "payload": {"text": "beta"}},
    ]
    merged = merge_candidates(
        dense=dense,
        sparse_scores=[],
        sparse_points_by_id={},
    )
    assert [c.point_id for c in merged] == ["d1", "d2"]
    assert all(c.sparse_score == 0.0 for c in merged)


def test_merge_candidates_lifts_sparse_literal_hit_above_dense_decoy():
    dense = [
        {"id": "decoy", "score": 0.95, "payload": {"text": "generic deploy notes"}},
    ]
    sparse_scores = [
        SparseScore(point_id="target", score=3.0, literal_hit=True, matched_tokens=["SMDFS-455"]),
        SparseScore(point_id="decoy", score=0.0),
    ]
    sparse_points_by_id = {
        "target": {"id": "target", "payload": {"text": "SMDFS-455 fix notes"}},
        "decoy": {"id": "decoy", "payload": {"text": "generic deploy notes"}},
    }
    merged = merge_candidates(
        dense=dense,
        sparse_scores=sparse_scores,
        sparse_points_by_id=sparse_points_by_id,
    )
    by_id = {c.point_id: c for c in merged}
    # The sparse-only target must outrank the dense-only decoy when combined.
    assert combine_score(by_id["target"]) > combine_score(by_id["decoy"])


def test_merge_candidates_drops_quarantined_and_secret_blocked():
    dense = [
        {"id": "ok", "score": 0.9, "payload": {"text": "clean"}},
        {"id": "secret-1", "score": 0.8, "payload": {"text": "Authorization: Bearer <REDACTED_BEARER>"}},
    ]
    sparse_scores = [
        SparseScore(point_id="quarantined", score=2.0, quarantined=True, payload_invisible=True),
        SparseScore(point_id="secret-1", score=2.0, secret_blocked=True, payload_invisible=True),
    ]
    sparse_points_by_id = {
        "quarantined": {"id": "quarantined", "payload": {"text": "x", "consolidation_quarantined": True}},
        "secret-1": dense[1],
    }
    merged = merge_candidates(
        dense=dense,
        sparse_scores=sparse_scores,
        sparse_points_by_id=sparse_points_by_id,
    )
    by_id = {c.point_id: c for c in merged}
    assert by_id["quarantined"].quarantined is True
    assert by_id["secret-1"].secret_blocked is True
    assert combine_score(by_id["quarantined"]) == 0.0
    assert combine_score(by_id["secret-1"]) == 0.0


def test_is_payload_visible_respects_quarantine_key():
    assert is_payload_visible({"consolidation_quarantined": True}) is False
    assert is_payload_visible({"other": 1}) is True
    assert is_payload_visible(None) is True


# ---------------------------------------------------------------------------
# End-to-end retriever integration
# ---------------------------------------------------------------------------


class _Embedding:
    def __init__(self):
        self.queries = []

    def embed_query(self, text):
        self.queries.append(text)
        return [0.1, 0.2]


class _SparseAwareQdrant:
    """Qdrant fake that supports both search and scroll_by_filter.

    scroll_by_filter honors the filter must-conditions so scope / fact_status /
    quarantine tests can verify that wrong-scope points never reach the
    scorer.
    """

    def __init__(self, *, dense_results, scroll_points):
        self.dense_results = dense_results
        self.scroll_points = scroll_points
        self.search_calls = []
        self.scroll_calls = []
        self.payload_updates = []

    def search(self, name, vector, limit, filter=None, with_payload=True, with_vector=False):
        self.search_calls.append({"name": name, "filter": filter, "limit": limit})
        return list(self.dense_results)

    def scroll_by_filter(self, name, flt, *, limit=256, with_payload=True, with_vector=False, max_total=None):
        self.scroll_calls.append({"name": name, "filter": flt, "limit": limit, "max_total": max_total})
        must = (flt or {}).get("must") or []
        must_not = (flt or {}).get("must_not") or []
        out: list[dict[str, Any]] = []
        for point in self.scroll_points:
            payload = point.get("payload") or {}
            if not self._matches_must(payload, must):
                continue
            if self._matches_must_not(payload, must_not):
                continue
            out.append(point)
        cap = max_total if max_total is not None else limit
        return out[:cap]

    @staticmethod
    def _matches_must(payload: dict[str, Any], must: list) -> bool:
        for cond in must:
            key = cond.get("key")
            match = cond.get("match", {})
            if not key:
                return False
            value = payload.get(key)
            if "value" in match:
                expected = str(match["value"])
                if isinstance(value, (list, tuple, set)):
                    if not any(str(item) == expected for item in value):
                        return False
                else:
                    if str(value) != expected:
                        return False
            elif "any" in match:
                values = {str(v) for v in match["any"]}
                if isinstance(value, (list, tuple, set)):
                    if not any(str(item) in values for item in value):
                        return False
                else:
                    if str(value) not in values:
                        return False
            elif "range" in match:
                rng = match["range"]
                text_value = str(value or "")
                if "gte" in rng and text_value < str(rng["gte"]):
                    return False
                if "lte" in rng and text_value > str(rng["lte"]):
                    return False
        return True

    @staticmethod
    def _matches_must_not(payload: dict[str, Any], must_not: list) -> bool:
        for cond in must_not:
            key = cond.get("key")
            match = cond.get("match", {})
            if not key:
                continue
            if "value" in match:
                if str(payload.get(key)) == str(match["value"]):
                    return True
        return False

    def update_payload(self, name, point_id, payload):
        self.payload_updates.append((name, point_id, payload))
        return {"status": "ok"}


def _payload_with_scope(pid, scope, **extra):
    base = {
        "text": "memory",
        "source_type": "project_doc",
        "importance": 7,
        "created_at": "2026-06-20T00:00:00+00:00",
        "profile_id": scope.get("profile_id", "default"),
        "user_id_hash": scope.get("user_id_hash", "u"),
        "chat_id_hash": scope.get("chat_id_hash", "c"),
        "fact_status": "active",
    }
    base.update(extra)
    return {"id": pid, "payload": base}


def test_sparse_lane_recovers_exact_uuid_point_dense_missed():
    target_id = "550e8400-e29b-41d4-a716-446655440000"
    scope = {"profile_id": "p1", "user_id_hash": "u1", "chat_id_hash": "c1"}
    # Dense only returns a decoy with high similarity.
    dense_results = [
        _payload_with_scope("decoy", scope, text="general deploy notes about clusters"),
    ]
    # Sparse scroll returns the actual target point from the same scope.
    scroll_points = [
        _payload_with_scope(target_id, scope, text="incident notes about 550e8400-e29b-41d4-a716-446655440000"),
        _payload_with_scope("decoy", scope, text="general deploy notes about clusters"),
    ]
    qdrant = _SparseAwareQdrant(dense_results=dense_results, scroll_points=scroll_points)
    retriever = MemoryRetriever(
        qdrant=qdrant,
        embeddings=_Embedding(),
        collection_name="memory",
        scope=scope,
        search_candidates=5,
    )

    results = retriever.search(f"recall {target_id}", top_k=5)
    assert any(chunk.id == target_id for chunk in results), (
        f"UUID literal hit should be in results: {[c.id for c in results]}"
    )


def test_sparse_lane_does_not_promote_wrong_scope_point():
    target_id = "550e8400-e29b-41d4-a716-446655440000"
    active_scope = {"profile_id": "p1", "user_id_hash": "u1", "chat_id_hash": "c1"}
    wrong_scope = {"profile_id": "p2", "user_id_hash": "u2", "chat_id_hash": "c2"}

    dense_results = []
    scroll_points = [
        _payload_with_scope(target_id, wrong_scope, text=f"incident {target_id}"),
    ]
    qdrant = _SparseAwareQdrant(dense_results=dense_results, scroll_points=scroll_points)
    retriever = MemoryRetriever(
        qdrant=qdrant,
        embeddings=_Embedding(),
        collection_name="memory",
        scope=active_scope,
        search_candidates=5,
    )

    results = retriever.search(f"recall {target_id}", top_k=5)
    assert not any(chunk.id == target_id for chunk in results)
    # The scroll call must carry the active scope filter so the wrong-scope
    # point never reaches the scorer.
    flt = qdrant.scroll_calls[-1]["filter"]
    must = flt.get("must", [])
    for key, value in active_scope.items():
        assert {"key": key, "match": {"value": value}} in must


def test_sparse_lane_hides_deprecated_and_superseded_points_by_default():
    target_id = "550e8400-e29b-41d4-a716-446655440000"
    scope = {"profile_id": "p1"}
    dense_results = []
    scroll_points = [
        _payload_with_scope(target_id, scope, text=f"incident {target_id}", fact_status="deprecated"),
        _payload_with_scope("sup-id", scope, text="SMDFS-455 fix", fact_status="superseded"),
        _payload_with_scope("ok-id", scope, text="SMDFS-455 fix", fact_status="active"),
    ]
    qdrant = _SparseAwareQdrant(dense_results=dense_results, scroll_points=scroll_points)
    retriever = MemoryRetriever(
        qdrant=qdrant,
        embeddings=_Embedding(),
        collection_name="memory",
        scope=scope,
        search_candidates=5,
    )

    default_results = retriever.search("recall SMDFS-455", top_k=5)
    assert {chunk.id for chunk in default_results} == {"ok-id"}

    history_results = retriever.search("recall SMDFS-455", top_k=5, include_fact_history=True)
    ids = {chunk.id for chunk in history_results}
    assert "ok-id" in ids


def test_sparse_lane_respects_quarantine_marker():
    target_id = "550e8400-e29b-41d4-a716-446655440000"
    scope = {"profile_id": "p1"}
    dense_results = []
    scroll_points = [
        _payload_with_scope(target_id, scope, text=f"incident {target_id}", consolidation_quarantined=True),
        _payload_with_scope("clean-id", scope, text="SMDFS-455 fix"),
    ]
    qdrant = _SparseAwareQdrant(dense_results=dense_results, scroll_points=scroll_points)
    retriever = MemoryRetriever(
        qdrant=qdrant,
        embeddings=_Embedding(),
        collection_name="memory",
        scope=scope,
        search_candidates=5,
    )

    results = retriever.search("recall SMDFS-455", top_k=5)
    ids = {chunk.id for chunk in results}
    assert "clean-id" in ids
    assert target_id not in ids


def test_sparse_lane_does_not_promote_secret_bearing_payload():
    target_id = "550e8400-e29b-41d4-a716-446655440000"
    scope = {"profile_id": "p1"}
    dense_results = []
    scroll_points = [
        _payload_with_scope(target_id, scope, text="SMDFS-455 fix landed; Authorization: Bearer <REDACTED_BEARER>"),
        _payload_with_scope("clean-id", scope, text="SMDFS-455 clean fix"),
    ]
    qdrant = _SparseAwareQdrant(dense_results=dense_results, scroll_points=scroll_points)
    retriever = MemoryRetriever(
        qdrant=qdrant,
        embeddings=_Embedding(),
        collection_name="memory",
        scope=scope,
        search_candidates=5,
    )

    results = retriever.search("recall SMDFS-455", top_k=5)
    ids = {chunk.id for chunk in results}
    assert "clean-id" in ids
    assert target_id not in ids


def test_sparse_lane_falls_back_to_dense_when_scroll_missing():
    scope = {"profile_id": "p1"}
    dense_results = [_payload_with_scope("d1", scope, text="general deploy notes")]

    class _NoScroll:
        def __init__(self):
            self.search_calls = []
            self.payload_updates = []

        def search(self, name, vector, limit, filter=None, with_payload=True, with_vector=False):
            self.search_calls.append({"name": name, "limit": limit})
            return list(dense_results)

        def update_payload(self, name, point_id, payload):
            self.payload_updates.append((name, point_id, payload))
            return {"status": "ok"}

    qdrant = _NoScroll()
    retriever = MemoryRetriever(
        qdrant=qdrant,
        embeddings=_Embedding(),
        collection_name="memory",
        scope=scope,
        search_candidates=5,
    )

    results = retriever.search("recall SMDFS-455", top_k=5)
    assert [chunk.id for chunk in results] == ["d1"]


def test_sparse_lane_skips_natural_language_queries():
    scope = {"profile_id": "p1"}
    dense_results = [_payload_with_scope("d1", scope, text="general deploy notes")]
    scroll_points = [
        _payload_with_scope("decoy", scope, text="deploy deploy deploy"),
    ]

    qdrant = _SparseAwareQdrant(dense_results=dense_results, scroll_points=scroll_points)
    retriever = MemoryRetriever(
        qdrant=qdrant,
        embeddings=_Embedding(),
        collection_name="memory",
        scope=scope,
        search_candidates=5,
    )

    results = retriever.search("how do we deploy this thing?", top_k=5)
    # Sparse scroll must NOT be invoked for plain natural language.
    assert qdrant.scroll_calls == []
    assert [chunk.id for chunk in results] == ["d1"]


def test_sparse_lane_updates_access_only_for_selected_chunks():
    scope = {"profile_id": "p1"}
    target_id = "550e8400-e29b-41d4-a716-446655440000"
    dense_results = []
    scroll_points = [
        _payload_with_scope(target_id, scope, text=f"incident {target_id}"),
        _payload_with_scope("other", scope, text="some other SMDFS-455 note"),
    ]
    qdrant = _SparseAwareQdrant(dense_results=dense_results, scroll_points=scroll_points)
    retriever = MemoryRetriever(
        qdrant=qdrant,
        embeddings=_Embedding(),
        collection_name="memory",
        scope=scope,
        search_candidates=5,
    )

    retriever.search(f"recall {target_id}", top_k=1)
    # update_access is called only for selected chunks, not for every sparse
    # candidate that the scorer saw.
    assert len(qdrant.payload_updates) <= 1
    if qdrant.payload_updates:
        assert qdrant.payload_updates[0][1] == target_id


def test_sparse_lane_can_be_disabled_per_retriever():
    scope = {"profile_id": "p1"}
    target_id = "550e8400-e29b-41d4-a716-446655440000"
    dense_results = []
    scroll_points = [
        _payload_with_scope(target_id, scope, text=f"incident {target_id}"),
    ]
    qdrant = _SparseAwareQdrant(dense_results=dense_results, scroll_points=scroll_points)
    retriever = MemoryRetriever(
        qdrant=qdrant,
        embeddings=_Embedding(),
        collection_name="memory",
        scope=scope,
        search_candidates=5,
        sparse_enabled=False,
    )

    results = retriever.search(f"recall {target_id}", top_k=5)
    # With sparse disabled, only dense results surface (none in this fixture).
    assert results == []
    assert qdrant.scroll_calls == []


def test_sparse_lane_respects_stale_and_requires_review_filters():
    scope = {"profile_id": "p1"}
    target_id = "550e8400-e29b-41d4-a716-446655440000"
    dense_results = []
    scroll_points = [
        _payload_with_scope(target_id, scope, text="SMDFS-455 fix", stale=True),
        _payload_with_scope("clean-id", scope, text="SMDFS-455 fix"),
    ]
    qdrant = _SparseAwareQdrant(dense_results=dense_results, scroll_points=scroll_points)
    retriever = MemoryRetriever(
        qdrant=qdrant,
        embeddings=_Embedding(),
        collection_name="memory",
        scope=scope,
        search_candidates=5,
    )

    # Explicit stale=False filters out stale=True points.
    results = retriever.search("recall SMDFS-455", top_k=5, stale=False)
    assert {chunk.id for chunk in results} == {"clean-id"}

    # Explicit stale=True returns only the stale=True point.
    results_stale = retriever.search("recall SMDFS-455", top_k=5, stale=True)
    assert {chunk.id for chunk in results_stale} == {target_id}

    # requires_review filter behaves analogously.
    scroll_points_rr = [
        _payload_with_scope(target_id, scope, text="SMDFS-455 fix", requires_review=True),
        _payload_with_scope("clean-id", scope, text="SMDFS-455 fix"),
    ]
    qdrant_rr = _SparseAwareQdrant(dense_results=dense_results, scroll_points=scroll_points_rr)
    retriever_rr = MemoryRetriever(
        qdrant=qdrant_rr,
        embeddings=_Embedding(),
        collection_name="memory",
        scope=scope,
        search_candidates=5,
    )
    results_rr = retriever_rr.search("recall SMDFS-455", top_k=5, requires_review=False)
    assert {chunk.id for chunk in results_rr} == {"clean-id"}


def test_sparse_lane_respects_canonical_filter():
    scope = {"profile_id": "p1"}
    target_id = "550e8400-e29b-41d4-a716-446655440000"
    dense_results = []
    scroll_points = [
        _payload_with_scope(target_id, scope, text="SMDFS-455 fix", canonical=True),
        _payload_with_scope("unc-id", scope, text="SMDFS-455 fix"),
    ]
    qdrant = _SparseAwareQdrant(dense_results=dense_results, scroll_points=scroll_points)
    retriever = MemoryRetriever(
        qdrant=qdrant,
        embeddings=_Embedding(),
        collection_name="memory",
        scope=scope,
        search_candidates=5,
    )
    results = retriever.search("recall SMDFS-455", top_k=5, canonical=True)
    assert {chunk.id for chunk in results} == {target_id}


def test_sparse_lane_respects_memory_kind_filter():
    scope = {"profile_id": "p1"}
    target_id = "550e8400-e29b-41d4-a716-446655440000"
    dense_results = []
    scroll_points = [
        _payload_with_scope(target_id, scope, text="SMDFS-455 fix", memory_kind="assertion"),
        _payload_with_scope("note-id", scope, text="SMDFS-455 fix", memory_kind="manual_fact"),
    ]
    qdrant = _SparseAwareQdrant(dense_results=dense_results, scroll_points=scroll_points)
    retriever = MemoryRetriever(
        qdrant=qdrant,
        embeddings=_Embedding(),
        collection_name="memory",
        scope=scope,
        search_candidates=5,
    )
    results = retriever.search("recall SMDFS-455", top_k=5, memory_kind="assertion")
    assert {chunk.id for chunk in results} == {target_id}


def test_sparse_lane_respects_tag_source_file_path_filters():
    scope = {"profile_id": "p1"}
    target_id = "550e8400-e29b-41d4-a716-446655440000"
    dense_results = []
    scroll_points = [
        _payload_with_scope(
            target_id,
            scope,
            text="SMDFS-455 fix",
            tags=["runbook", "oncall"],
            source="runbook.md",
            file_path="/repo/runbook.md",
            project_path="/repo",
        ),
        _payload_with_scope("other", scope, text="SMDFS-455 fix"),
    ]
    qdrant = _SparseAwareQdrant(dense_results=dense_results, scroll_points=scroll_points)
    retriever = MemoryRetriever(
        qdrant=qdrant,
        embeddings=_Embedding(),
        collection_name="memory",
        scope=scope,
        search_candidates=5,
    )
    results = retriever.search(
        "recall SMDFS-455",
        top_k=5,
        tags=["runbook"],
        source="runbook.md",
        file_path="/repo/runbook.md",
        project_path="/repo",
    )
    assert {chunk.id for chunk in results} == {target_id}


def test_sparse_lane_secret_scan_does_not_log_payload_text():
    """The retriever must never log or surface secret-bearing payload text."""
    scope = {"profile_id": "p1"}
    target_id = "550e8400-e29b-41d4-a716-446655440000"
    dense_results = []
    bearer_line = "Authorization: Bearer <REDACTED_BEARER>"
    scroll_points = [
        _payload_with_scope(target_id, scope, text=bearer_line),
    ]
    qdrant = _SparseAwareQdrant(dense_results=dense_results, scroll_points=scroll_points)
    retriever = MemoryRetriever(
        qdrant=qdrant,
        embeddings=_Embedding(),
        collection_name="memory",
        scope=scope,
        search_candidates=5,
    )
    results = retriever.search("recall SMDFS-455", top_k=5)
    # The bearer token must NOT have leaked into any returned chunk text.
    for chunk in results:
        assert "<REDACTED_BEARER>" not in chunk.text
        assert "Bearer" not in chunk.text or "Authorization" not in chunk.text


# ---------------------------------------------------------------------------
# P1 regression: zero-score sparse scroll candidates must not be returned or
# access-updated. Sparse-only candidates must be merged/promoted only when the
# sparse score is positive AND there is a real matched literal/token signal.
# ---------------------------------------------------------------------------


def test_merge_candidates_drops_sparse_zero_score_no_match():
    """Sparse-only zero-score scroll candidates with no matched literals/tokens
    must NOT be merged into the candidate pool. Otherwise the retriever would
    surface them (default ``min_final_score=0.0``) and trigger
    ``update_access_metadata`` on them.
    """
    dense: list[dict[str, Any]] = []
    sparse_scores = [
        SparseScore(point_id="decoy-0", score=0.0, matched_tokens=[], field_hits={}),
        SparseScore(point_id="decoy-1", score=0.0, matched_tokens=[], field_hits={}),
    ]
    sparse_points_by_id = {
        "decoy-0": {"id": "decoy-0", "payload": {"text": "unrelated note"}},
        "decoy-1": {"id": "decoy-1", "payload": {"text": "another unrelated note"}},
    }
    merged = merge_candidates(
        dense=dense,
        sparse_scores=sparse_scores,
        sparse_points_by_id=sparse_points_by_id,
    )
    assert [c.point_id for c in merged] == []


def test_merge_candidates_drops_sparse_no_match_dense_kept():
    """Dense candidates remain dense candidates even when sparse was no-match.

    Only the sparse lane's no-match follow-up must not be merged/promoted.
    """
    dense = [
        {"id": "d-1", "score": 0.9, "payload": {"text": "alpha"}},
    ]
    sparse_scores = [
        SparseScore(point_id="decoy", score=0.0, matched_tokens=[], field_hits={}),
    ]
    sparse_points_by_id = {
        "decoy": {"id": "decoy", "payload": {"text": "unrelated"}},
    }
    merged = merge_candidates(
        dense=dense,
        sparse_scores=sparse_scores,
        sparse_points_by_id=sparse_points_by_id,
    )
    by_id = {c.point_id: c for c in merged}
    assert "d-1" in by_id
    assert "decoy" not in by_id


def test_sparse_lane_dense_miss_no_scroll_match_returns_empty_and_no_payload_update():
    """P1 regression: strong-signal query + empty dense + same-scope scroll
    points with NO token match must return ``[]`` AND ``payload_updates == []``
    under the default ``update_access=True``.

    Read-only reproduction described by both reviewer + security-reviewer:
    a query for ``550e8400-e29b-41d4-a716-446655440000`` with empty dense and
    scroll returning five decoys used to surface the decoys (zero score but
    default ``min_final_score=0.0``) and trigger ``update_access_metadata``
    on each. With the fix, the sparse lane must drop those candidates.
    """
    target_id = "550e8400-e29b-41d4-a716-446655440000"
    scope = {"profile_id": "p1", "user_id_hash": "u1", "chat_id_hash": "c1"}
    dense_results: list[dict[str, Any]] = []
    scroll_points = [
        _payload_with_scope(f"decoy-{i}", scope, text="generic deploy notes about clusters")
        for i in range(5)
    ]
    qdrant = _SparseAwareQdrant(dense_results=dense_results, scroll_points=scroll_points)
    retriever = MemoryRetriever(
        qdrant=qdrant,
        embeddings=_Embedding(),
        collection_name="memory",
        scope=scope,
        search_candidates=5,
    )
    # Default ``update_access=True`` — explicitly asserted later.
    results = retriever.search(f"recall {target_id}", top_k=5)
    assert results == [], (
        f"expected no results from no-match scroll, got {[c.id for c in results]}"
    )
    # update_access is on by default; the fix must result in ZERO payload
    # updates when no sparse candidate qualified for promotion.
    assert qdrant.payload_updates == [], (
        f"expected no payload updates for no-match scroll, got {qdrant.payload_updates}"
    )


# ---------------------------------------------------------------------------
# P1 regression: sparse secret blocker must recursively scan the full payload
# (nested dicts, lists, and non-indexed metadata fields), not only the compact
# indexed text fields.
# ---------------------------------------------------------------------------


def test_score_candidates_secret_blocked_in_nested_dict():
    """Secret buried in a nested dict (non-indexed metadata) must block."""
    bearer_frag = " ".join(["Authorization:", "Bearer", "".join(["x"] * 20)])
    candidates = [
        {
            "id": "nested-secret",
            "payload": {
                "text": "SMDFS-455 clean notes about the fix",
                "metadata": {"auth_header": bearer_frag},
            },
        }
    ]
    scores = score_candidates("recall SMDFS-455", candidates)
    assert scores[0].secret_blocked is True
    assert scores[0].score == 0.0
    assert scores[0].payload_invisible is True


def test_score_candidates_secret_blocked_in_list():
    """Secret buried inside a list (non-indexed metadata) must block."""
    token_pair = "".join(["alpha", "beta", "gamma", "delta"])
    secret_line = " ".join(["token", token_pair])
    candidates = [
        {
            "id": "list-secret",
            "payload": {
                "text": "SMDFS-455 clean notes",
                "history": [
                    "regular entry",
                    secret_line,
                ],
            },
        }
    ]
    scores = score_candidates("recall SMDFS-455", candidates)
    assert scores[0].secret_blocked is True
    assert scores[0].score == 0.0


def test_score_candidates_secret_blocked_in_non_indexed_metadata_key():
    """A credential-like string inside a non-indexed payload key must block,
    even when the compact indexed fields are completely clean.
    """
    key_token = "".join(["alphabeta", "gammadelt", "epsilonz"])
    secret_line = " ".join(["api_key", key_token])
    candidates = [
        {
            "id": "non-indexed-secret",
            "payload": {
                "text": "SMDFS-455 fix",
                "file_path": "/repo/notes.md",
                "importance": 7,
                "source_type": "project_doc",
                "internal_notes": secret_line,
            },
        }
    ]
    scores = score_candidates("recall SMDFS-455", candidates)
    assert scores[0].secret_blocked is True
    assert scores[0].score == 0.0


def test_score_candidates_secret_blocked_in_deeply_nested_payload():
    """Secret buried several layers deep must still block."""
    password_value = "".join(["hunter", "2hunter", "2hunter2"])
    secret_value = " ".join(["password", password_value])
    candidates = [
        {
            "id": "deep-secret",
            "payload": {
                "text": "SMDFS-455 fix",
                "context": {
                    "session": {
                        "headers": [
                            {"name": "x-other", "value": "ok"},
                            {"name": "x-auth", "value": secret_value},
                        ]
                    }
                },
            },
        }
    ]
    scores = score_candidates("recall SMDFS-455", candidates)
    assert scores[0].secret_blocked is True
    assert scores[0].score == 0.0


def test_sparse_lane_does_not_promote_secret_buried_in_non_indexed_metadata():
    """End-to-end: a sparse literal hit whose nested metadata carries a secret
    must NOT be promoted into results. Indexed text remains clean.
    """
    target_id = "550e8400-e29b-41d4-a716-446655440000"
    scope = {"profile_id": "p1"}
    dense_results: list[dict[str, Any]] = []
    bearer_frag = " ".join(["Authorization:", "Bearer", "".join(["y"] * 20)])
    scroll_points = [
        _payload_with_scope(
            target_id,
            scope,
            text=f"incident {target_id}",
            extra_metadata_block={"auth": bearer_frag},
        ),
        _payload_with_scope("clean-id", scope, text=f"clean fix {target_id}"),
    ]
    qdrant = _SparseAwareQdrant(dense_results=dense_results, scroll_points=scroll_points)
    retriever = MemoryRetriever(
        qdrant=qdrant,
        embeddings=_Embedding(),
        collection_name="memory",
        scope=scope,
        search_candidates=5,
    )
    results = retriever.search(f"recall {target_id}", top_k=5)
    ids = {chunk.id for chunk in results}
    # Secret-bearing target must NOT appear; only the clean-id sparse hit does.
    assert "clean-id" in ids
    assert target_id not in ids
    # And access metadata must not have been written for the blocked target.
    updated_ids = {update[1] for update in qdrant.payload_updates}
    assert target_id not in updated_ids
