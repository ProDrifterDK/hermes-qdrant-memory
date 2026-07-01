from __future__ import annotations

from qdrant_memory.retriever import MemoryRetriever
from qdrant_memory.tools import SEARCH_SCHEMA


class FakeEmbedding:
    def __init__(self):
        self.queries = []

    def embed_query(self, text):
        self.queries.append(text)
        return [0.1, 0.2]


class FakeQdrant:
    def __init__(self):
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
        return [
            {
                "id": "memory-1",
                "score": 0.9,
                "payload": {
                    "text": "Production API endpoint is /api/v2.",
                    "source_type": "project_doc",
                    "importance": 8,
                    "created_at": "2026-01-15T00:00:00+00:00",
                },
            }
        ]

    def update_payload(self, name, point_id, payload):
        self.payload_updates.append((name, point_id, payload))
        return {"status": "ok"}


def test_memory_search_adds_richer_filters_as_must_conditions():
    qdrant = FakeQdrant()
    embeddings = FakeEmbedding()
    retriever = MemoryRetriever(
        qdrant=qdrant,
        embeddings=embeddings,
        collection_name="memory",
        scope={"profile_id": "coder", "platform": "telegram"},
        search_candidates=3,
    )

    results = retriever.search(
        "api endpoint",
        top_k=2,
        source_type="project_doc",
        tags=["api", "v0.7"],
        source="api.md",
        file_path="/repo/docs/api.md",
        project_path="/repo",
        since="2026-01-01T00:00:00Z",
        until="2026-01-31T23:59:59Z",
    )

    assert len(results) == 1
    assert embeddings.queries == ["api endpoint"]
    call = qdrant.searches[0]
    assert call["name"] == "memory"
    assert call["limit"] == 3
    must = call["filter"]["must"]
    assert {"key": "profile_id", "match": {"value": "coder"}} in must
    assert {"key": "platform", "match": {"value": "telegram"}} in must
    assert {"key": "source_type", "match": {"value": "project_doc"}} in must
    assert {"key": "tags", "match": {"value": "api"}} in must
    assert {"key": "tags", "match": {"value": "v0.7"}} in must
    assert {"key": "source", "match": {"value": "api.md"}} in must
    assert {"key": "file_path", "match": {"value": "/repo/docs/api.md"}} in must
    assert {"key": "project_path", "match": {"value": "/repo"}} in must
    assert {"key": "created_at", "range": {"gte": "2026-01-01T00:00:00Z", "lte": "2026-01-31T23:59:59Z"}} in must


def test_memory_search_schema_exposes_richer_filters_with_strict_args():
    params = SEARCH_SCHEMA["parameters"]
    props = params["properties"]

    for key in ("tags", "source", "file_path", "project_path", "since", "until", "collection", "include_fact_history"):
        assert key in props
    assert props["include_fact_history"]["type"] == "boolean"
    assert props["collection"]["enum"] == ["memory", "learning"]
    assert params["additionalProperties"] is False


class FakeFactStatusQdrant(FakeQdrant):
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
        return [
            {
                "id": "active-1",
                "score": 0.91,
                "payload": {
                    "text": "Production endpoint is /api/v2.",
                    "memory_kind": "assertion",
                    "fact_status": "active",
                    "importance": 8,
                    "created_at": "2026-01-15T00:00:00+00:00",
                },
            },
            {
                "id": "deprecated-1",
                "score": 0.9,
                "payload": {
                    "text": "Production endpoint is /api/v1.",
                    "memory_kind": "assertion",
                    "fact_status": "deprecated",
                    "importance": 8,
                    "created_at": "2026-01-15T00:00:00+00:00",
                },
            },
            {
                "id": "superseded-1",
                "score": 0.89,
                "payload": {
                    "text": "Production endpoint is /api/beta.",
                    "memory_kind": "assertion",
                    "fact_status": "superseded",
                    "importance": 8,
                    "created_at": "2026-01-15T00:00:00+00:00",
                },
            },
        ]


def test_memory_search_hides_deprecated_and_superseded_fact_status_by_default():
    qdrant = FakeFactStatusQdrant()
    retriever = MemoryRetriever(qdrant=qdrant, embeddings=FakeEmbedding(), collection_name="memory", search_candidates=3)

    default_results = retriever.search("production endpoint", top_k=5)
    history_results = retriever.search("production endpoint", top_k=5, include_fact_history=True)

    assert [result.id for result in default_results] == ["active-1"]
    assert [result.id for result in history_results] == ["active-1", "deprecated-1", "superseded-1"]
    default_filter = qdrant.searches[0]["filter"]
    assert {"key": "fact_status", "match": {"value": "deprecated"}} in default_filter["must_not"]
    assert {"key": "fact_status", "match": {"value": "superseded"}} in default_filter["must_not"]
    assert "must_not" not in qdrant.searches[1]["filter"]


class FakeSparseAwareQdrant(FakeFactStatusQdrant):
    """Qdrant fake that exposes scroll_by_filter alongside the search() result list."""

    def __init__(self, search_results=None, scroll_results=None):
        super().__init__()
        self.search_results = list(search_results or [])
        self.scroll_results = list(scroll_results or [])
        self.scroll_calls = []

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
        return list(self.search_results)

    def scroll_by_filter(self, name, flt, *, limit=256, with_payload=True, with_vector=False, max_total=None):
        self.scroll_calls.append({"name": name, "filter": flt, "limit": limit, "max_total": max_total})
        return list(self.scroll_results)


def test_memory_search_does_not_invoke_sparse_lane_for_broad_queries():
    """The sparse lane is gated on exact-signal patterns; broad NL queries stay dense-only."""
    qdrant = FakeSparseAwareQdrant(search_results=[
        {
            "id": "memory-1",
            "score": 0.9,
            "payload": {
                "text": "Production API endpoint is /api/v2.",
                "source_type": "project_doc",
                "importance": 8,
                "created_at": "2026-01-15T00:00:00+00:00",
            },
        }
    ])
    retriever = MemoryRetriever(qdrant=qdrant, embeddings=FakeEmbedding(), collection_name="memory", search_candidates=3)

    results = retriever.search("how does the production endpoint behave?", top_k=3)

    assert [result.id for result in results] == ["memory-1"]
    # Sparse scroll must not have been called for a broad NL query.
    assert qdrant.scroll_calls == []


def test_memory_search_sparse_lane_reuses_scope_filter():
    """The scroll filter must include the same profile / user / chat scope as dense search."""
    scroll_results = [{
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "payload": {
            "text": "incident notes",
            "source_type": "project_doc",
            "profile_id": "coder",
            "user_id_hash": "u1",
            "chat_id_hash": "c1",
            "importance": 8,
            "created_at": "2026-01-15T00:00:00+00:00",
            "fact_status": "active",
        },
    }]
    qdrant = FakeSparseAwareQdrant(search_results=[], scroll_results=scroll_results)
    retriever = MemoryRetriever(
        qdrant=qdrant,
        embeddings=FakeEmbedding(),
        collection_name="memory",
        scope={"profile_id": "coder", "user_id_hash": "u1", "chat_id_hash": "c1"},
        search_candidates=3,
    )

    results = retriever.search("recall 550e8400-e29b-41d4-a716-446655440000", top_k=3)
    assert [result.id for result in results] == ["550e8400-e29b-41d4-a716-446655440000"]
    assert len(qdrant.scroll_calls) == 1
    flt = qdrant.scroll_calls[0]["filter"]
    must = flt.get("must", [])
    for key, value in {"profile_id": "coder", "user_id_hash": "u1", "chat_id_hash": "c1"}.items():
        assert {"key": key, "match": {"value": value}} in must


def test_memory_search_sparse_lane_degrades_when_scroll_missing():
    """If scroll_by_filter is absent, sparse lane is skipped and dense behavior is unchanged."""

    class NoScrollQdrant(FakeFactStatusQdrant):
        def __init__(self, search_results):
            super().__init__()
            self._search_results = list(search_results)

        def search(self, name, vector, limit, filter=None, with_payload=True, with_vector=False):
            self.searches.append({
                "name": name, "vector": vector, "limit": limit, "filter": filter,
                "with_payload": with_payload, "with_vector": with_vector,
            })
            return list(self._search_results)

        def scroll_by_filter(self, *args, **kwargs):  # pragma: no cover - defined to fail loudly
            raise AssertionError("scroll_by_filter should not be called when sparse is disabled")

    qdrant = NoScrollQdrant(search_results=[
        {
            "id": "memory-1",
            "score": 0.9,
            "payload": {
                "text": "Production API endpoint is /api/v2.",
                "source_type": "project_doc",
                "importance": 8,
                "created_at": "2026-01-15T00:00:00+00:00",
            },
        }
    ])
    retriever = MemoryRetriever(qdrant=qdrant, embeddings=FakeEmbedding(), collection_name="memory", search_candidates=3)

    # Even with a strong-signal query, no scroll happens because the qdrant
    # fake does not expose scroll_by_filter.
    results = retriever.search("550e8400-e29b-41d4-a716-446655440000", top_k=3)
    assert [result.id for result in results] == ["memory-1"]


def test_memory_search_allow_sparse_scroll_false_skips_scroll():
    """Phase 5 fix4: ``allow_sparse_scroll=False`` MUST suppress the
    sparse scroll lane even for strong-signal queries, so callers like
    ``HybridRouter.retrieve`` never invoke ``scroll_by_filter`` through
    the Phase 5 retrieve path.
    """

    class StrictQdrant(FakeSparseAwareQdrant):
        def scroll_by_filter(self, *args, **kwargs):  # pragma: no cover - failure sentinel
            raise AssertionError(
                "scroll_by_filter must NOT be invoked when allow_sparse_scroll=False"
            )

    qdrant = StrictQdrant(search_results=[
        {
            "id": "memory-1",
            "score": 0.9,
            "payload": {
                "text": "incident notes",
                "source_type": "project_doc",
                "importance": 8,
                "created_at": "2026-01-15T00:00:00+00:00",
            },
        }
    ])
    retriever = MemoryRetriever(qdrant=qdrant, embeddings=FakeEmbedding(), collection_name="memory", search_candidates=3)

    # Strong-signal query (UUID) — would normally trigger the sparse lane.
    results = retriever.search(
        "550e8400-e29b-41d4-a716-446655440000",
        top_k=3,
        allow_sparse_scroll=False,
    )
    assert [result.id for result in results] == ["memory-1"]
    assert qdrant.scroll_calls == []


def test_memory_search_default_still_runs_sparse_when_signal_present():
    """Phase 5 fix4: omitting ``allow_sparse_scroll`` (or passing ``True``)
    keeps the existing ``qdrant_memory_search`` backward-compatible —
    strong-signal queries still hit ``scroll_by_filter``.
    """

    qdrant = FakeSparseAwareQdrant(search_results=[], scroll_results=[
        {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "payload": {
                "text": "incident notes",
                "source_type": "project_doc",
                "profile_id": "coder",
                "importance": 8,
                "created_at": "2026-01-15T00:00:00+00:00",
                "fact_status": "active",
            },
        }
    ])
    retriever = MemoryRetriever(
        qdrant=qdrant,
        embeddings=FakeEmbedding(),
        collection_name="memory",
        scope={"profile_id": "coder"},
        search_candidates=3,
    )

    # Default: scroll IS called for a strong-signal query.
    retriever.search("550e8400-e29b-41d4-a716-446655440000", top_k=3)
    assert len(qdrant.scroll_calls) == 1

    qdrant.scroll_calls.clear()
    # Explicit True: scroll IS called for a strong-signal query.
    retriever.search(
        "550e8400-e29b-41d4-a716-446655440000",
        top_k=3,
        allow_sparse_scroll=True,
    )
    assert len(qdrant.scroll_calls) == 1
