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

    for key in ("tags", "source", "file_path", "project_path", "since", "until", "collection"):
        assert key in props
    assert props["collection"]["enum"] == ["memory", "learning"]
    assert params["additionalProperties"] is False
