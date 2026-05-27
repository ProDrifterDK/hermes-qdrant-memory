from __future__ import annotations

import json

from qdrant_memory.learning import LearningStore, build_learning_payload, classify_learning_type, make_learning_id
from qdrant_memory.tools import LEARNING_SEARCH_SCHEMA, LEARNING_STORE_SCHEMA, TOOL_SCHEMAS


class FakeEmbedding:
    def __init__(self):
        self.documents = []
        self.queries = []

    def embed_document(self, text):
        self.documents.append(text)
        return [0.7, 0.8]

    def embed_query(self, text):
        self.queries.append(text)
        return [0.1, 0.2]

    def health(self):
        return True


class FakeQdrant:
    def __init__(self):
        self.upserts = []
        self.searches = []
        self.payload_updates = []
        self.counts = {"memory": 3, "learnings": 2}
        self.collections = ["memory", "learnings"]
        self.health_ok = True

    def upsert(self, name, points):
        self.upserts.append((name, points))
        return {"status": "ok"}

    def search(self, name, vector, limit, filter=None, with_payload=True, with_vector=False):
        self.searches.append((name, vector, limit, filter, with_payload, with_vector))
        return [
            {
                "id": "lesson-1",
                "score": 0.91,
                "payload": {
                    "text": "When llama.cpp embedding batch fails, lower max_chunk_tokens.",
                    "source_type": "learning",
                    "chunk_type": "tool_failure_lesson",
                    "importance": 8,
                    "created_at": "2026-01-01T00:00:00+00:00",
                },
            }
        ]

    def update_payload(self, name, point_id, payload):
        self.payload_updates.append((name, point_id, payload))
        return {"status": "ok"}

    def health(self):
        return self.health_ok

    def get_collections(self):
        return self.collections

    def count(self, name):
        return self.counts[name]


def test_learning_id_is_deterministic_and_distinct_from_memory_ids():
    one = make_learning_id("tool_failure_lesson", "pytest", "missing pytest", "use venv python -m pytest")
    two = make_learning_id("tool_failure_lesson", "pytest", "missing pytest", "use venv python -m pytest")
    assert one == two
    assert len(one) == 36
    assert one != make_learning_id("user_correction", "pytest", "missing pytest", "use venv python -m pytest")


def test_classify_learning_type_uses_signal_words():
    assert classify_learning_type("tool failed with HTTP 500", "lower max_chunk_tokens") == "tool_failure_lesson"
    assert classify_learning_type("Alan corrected surname spelling", "Use Gárate") == "user_correction"
    assert classify_learning_type("workflow discovered", "run dry-run before live index") == "workflow_lesson"
    assert classify_learning_type("local quirk", "pytest is in Hermes venv") == "environment_quirk"


def test_build_learning_payload_has_procedural_fields_and_source_type():
    payload = build_learning_payload(
        lesson="Use the Hermes venv for pytest in this plugin.",
        learning_type="environment_quirk",
        trigger="pytest not found",
        mistake="ran pytest from system shell",
        correction="run <hermes-venv>/bin/python -m pytest",
        evidence="system pytest missing, venv pytest passes",
        tool_name="terminal",
        command="pytest tests -q",
        project_path="/repo",
        profile_id="default",
    )
    assert payload["source_type"] == "learning"
    assert payload["chunk_type"] == "environment_quirk"
    assert payload["learning_type"] == "environment_quirk"
    assert payload["trigger"] == "pytest not found"
    assert payload["mistake"] == "ran pytest from system shell"
    assert payload["correction"].startswith("run <hermes-venv>")
    assert payload["tool_name"] == "terminal"
    assert payload["command"] == "pytest tests -q"
    assert payload["project_path"] == "/repo"
    assert payload["promote_to_skill_candidate"] is False
    assert payload["topic"] == "environment_quirk"
    assert payload["entity"] == "terminal"
    assert payload["fact_key"] == "learning.environment.quirk.terminal"
    assert payload["reconsolidation_key"] == "learning.environment.quirk.terminal"


def test_learning_payload_does_not_fact_key_secret_command():
    secret = " ".join(["Authorization:", "Bearer", "".join(["abc", "def", "ghi", "jkl", "mno", "pqr", "stu", "vwx", "yz"])])
    payload = build_learning_payload(
        lesson="Do not leak API auth headers.",
        learning_type="tool_failure_lesson",
        trigger="curl failed",
        correction="remove the header from logs",
        tool_name="terminal",
        command=f"curl -H '{secret}' https://example.com",
    )

    assert "fact_key" not in payload
    assert "reconsolidation_key" not in payload


def test_learning_store_persists_fact_metadata_in_upsert_payload():
    qdrant = FakeQdrant()
    embeddings = FakeEmbedding()
    store = LearningStore(qdrant=qdrant, embeddings=embeddings, collection_name="learnings", profile_id="coder", platform="cli")

    store.store(
        lesson="Use the Hermes venv for pytest in this plugin.",
        learning_type="environment_quirk",
        trigger="pytest not found",
        tool_name="terminal",
    )

    payload = qdrant.upserts[0][1][0]["payload"]
    assert payload["fact_key"] == "learning.environment.quirk.terminal"
    assert payload["reconsolidation_key"] == "learning.environment.quirk.terminal"


def test_learning_store_writes_to_learning_collection_only():
    qdrant = FakeQdrant()
    embeddings = FakeEmbedding()
    store = LearningStore(qdrant=qdrant, embeddings=embeddings, collection_name="learnings", profile_id="coder", platform="cli")

    point_id = store.store(
        lesson="When a broad index changes, dry-run before live sync.",
        learning_type="workflow_lesson",
        trigger="directory indexing",
        correction="inspect deleted_file_ids first",
        importance=8,
    )

    assert point_id
    assert qdrant.upserts[0][0] == "learnings"
    point = qdrant.upserts[0][1][0]
    assert point["id"] == point_id
    assert point["vector"] == [0.7, 0.8]
    assert point["payload"]["source_type"] == "learning"
    assert point["payload"]["chunk_type"] == "workflow_lesson"
    assert embeddings.documents == [point["payload"]["text"]]


def test_learning_search_uses_learning_collection():
    qdrant = FakeQdrant()
    embeddings = FakeEmbedding()
    store = LearningStore(qdrant=qdrant, embeddings=embeddings, collection_name="learnings", scope={"profile_id": "coder"})

    results = store.search("embedding batch failure", top_k=3, learning_type="tool_failure_lesson")

    assert len(results) == 1
    assert qdrant.searches[0][0] == "learnings"
    filter_payload = qdrant.searches[0][3]
    assert {"key": "profile_id", "match": {"value": "coder"}} in filter_payload["must"]
    assert {"key": "source_type", "match": {"value": "learning"}} in filter_payload["must"]
    assert {"key": "learning_type", "match": {"value": "tool_failure_lesson"}} in filter_payload["must"]


def test_learning_search_adds_richer_filters_as_must_conditions():
    qdrant = FakeQdrant()
    embeddings = FakeEmbedding()
    store = LearningStore(qdrant=qdrant, embeddings=embeddings, collection_name="learnings", scope={"profile_id": "coder"})

    results = store.search(
        "embedding batch failure",
        top_k=3,
        learning_type="tool_failure_lesson",
        tags=["workflow", "cli"],
        source="hermes_learning",
        file_path="/repo/lessons.md",
        project_path="/repo",
        since="2026-02-01T00:00:00Z",
        until="2026-02-28T23:59:59Z",
    )

    assert len(results) == 1
    filter_payload = qdrant.searches[0][3]
    must = filter_payload["must"]
    assert {"key": "profile_id", "match": {"value": "coder"}} in must
    assert {"key": "source_type", "match": {"value": "learning"}} in must
    assert {"key": "learning_type", "match": {"value": "tool_failure_lesson"}} in must
    assert {"key": "tags", "match": {"value": "workflow"}} in must
    assert {"key": "tags", "match": {"value": "cli"}} in must
    assert {"key": "source", "match": {"value": "hermes_learning"}} in must
    assert {"key": "file_path", "match": {"value": "/repo/lessons.md"}} in must
    assert {"key": "project_path", "match": {"value": "/repo"}} in must
    assert {"key": "created_at", "range": {"gte": "2026-02-01T00:00:00Z", "lte": "2026-02-28T23:59:59Z"}} in must


def test_learning_semantic_duplicate_search_uses_learning_collection_without_access_update():
    qdrant = FakeQdrant()
    embeddings = FakeEmbedding()
    store = LearningStore(qdrant=qdrant, embeddings=embeddings, collection_name="learnings", scope={"profile_id": "coder"})

    duplicate = store.find_semantic_duplicate(
        "Use venv python when pytest command is missing",
        learning_type="tool_failure_lesson",
        threshold=0.9,
        top_k=3,
    )

    assert duplicate is not None
    assert duplicate["id"] == "lesson-1"
    assert qdrant.searches[0][0] == "learnings"
    assert qdrant.searches[0][2] == 3
    filter_payload = qdrant.searches[0][3]
    assert {"key": "profile_id", "match": {"value": "coder"}} in filter_payload["must"]
    assert {"key": "source_type", "match": {"value": "learning"}} in filter_payload["must"]
    assert {"key": "learning_type", "match": {"value": "tool_failure_lesson"}} in filter_payload["must"]
    assert embeddings.queries == ["Use venv python when pytest command is missing"]
    assert qdrant.payload_updates == []


def test_learning_semantic_duplicate_below_threshold_returns_none():
    qdrant = FakeQdrant()
    embeddings = FakeEmbedding()
    store = LearningStore(qdrant=qdrant, embeddings=embeddings, collection_name="learnings", scope={"profile_id": "coder"})

    duplicate = store.find_semantic_duplicate("weakly related lesson", learning_type="tool_failure_lesson", threshold=0.99, top_k=3)

    assert duplicate is None
    assert qdrant.payload_updates == []


def test_learning_tool_schemas_are_exposed():
    names = [schema["name"] for schema in TOOL_SCHEMAS]
    assert LEARNING_STORE_SCHEMA["name"] == "qdrant_learning_store"
    assert LEARNING_SEARCH_SCHEMA["name"] == "qdrant_learning_search"
    assert "qdrant_learning_store" in names
    assert "qdrant_learning_search" in names
    params = LEARNING_SEARCH_SCHEMA["parameters"]
    props = params["properties"]
    for key in ("tags", "source", "file_path", "project_path", "since", "until"):
        assert key in props
    assert params["additionalProperties"] is False


def test_provider_learning_tools_and_status_use_learning_collection():
    from __init__ import QdrantMemoryProvider

    provider = QdrantMemoryProvider()
    provider._qdrant = FakeQdrant()
    provider._embeddings = FakeEmbedding()
    provider._active = True
    provider._config.update(
        {
            "collection_name": "memory",
            "learning_collection_name": "learnings",
            "qdrant_url": "http://qdrant",
            "embedding_url": "http://embed/v1",
            "embedding_model": "bge-m3",
            "vector_size": 1024,
            "auto_recall": True,
            "sync_turns": True,
            "search_candidates": 20,
            "decay_rate": 0.001,
            "min_raw_score": 0.0,
            "min_final_score": 0.0,
        }
    )

    saved = json.loads(
        provider.handle_tool_call(
            "qdrant_learning_store",
            {"lesson": "Use dry-run before live directory sync", "learning_type": "workflow_lesson", "trigger": "indexing"},
        )
    )
    searched = json.loads(provider.handle_tool_call("qdrant_learning_search", {"query": "dry-run directory sync", "learning_type": "workflow_lesson"}))
    status = json.loads(provider.handle_tool_call("qdrant_memory_status", {}))

    assert saved["saved"] is True
    assert saved["write_decision"]["decision"] == "learning_candidate"
    assert provider._qdrant.upserts[0][0] == "learnings"
    assert searched["count"] == 1
    assert provider._qdrant.searches[0][0] == "learnings"
    assert status["learning_collection_exists"] is True
    assert status["learning_point_count"] == 2
    assert status["learning_enabled"] is True


def test_memory_search_collection_learning_routes_to_learning_collection_with_filters():
    from __init__ import QdrantMemoryProvider

    provider = QdrantMemoryProvider()
    provider._qdrant = FakeQdrant()
    provider._embeddings = FakeEmbedding()
    provider._active = True
    provider._config.update(
        {
            "collection_name": "memory",
            "learning_collection_name": "learnings",
            "qdrant_url": "http://qdrant",
            "embedding_url": "http://embed/v1",
            "embedding_model": "bge-m3",
            "vector_size": 1024,
            "auto_recall": True,
            "sync_turns": True,
            "search_candidates": 20,
            "decay_rate": 0.001,
            "min_raw_score": 0.0,
            "min_final_score": 0.0,
        }
    )

    searched = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_search",
            {
                "query": "dry-run directory sync",
                "collection": "learning",
                "tags": ["workflow", "cli"],
                "source": "hermes_learning",
                "file_path": "/repo/lessons.md",
                "project_path": "/repo",
                "since": "2026-02-01T00:00:00Z",
                "until": "2026-02-28T23:59:59Z",
            },
        )
    )

    assert searched["count"] == 1
    assert searched["collection_name"] == "learnings"
    assert provider._qdrant.searches[0][0] == "learnings"
    must = provider._qdrant.searches[0][3]["must"]
    assert {"key": "source_type", "match": {"value": "learning"}} in must
    assert {"key": "tags", "match": {"value": "workflow"}} in must
    assert {"key": "tags", "match": {"value": "cli"}} in must
    assert {"key": "source", "match": {"value": "hermes_learning"}} in must
    assert {"key": "file_path", "match": {"value": "/repo/lessons.md"}} in must
    assert {"key": "project_path", "match": {"value": "/repo"}} in must
    assert {"key": "created_at", "range": {"gte": "2026-02-01T00:00:00Z", "lte": "2026-02-28T23:59:59Z"}} in must


def test_learning_store_write_gate_skips_low_information_lessons():
    from __init__ import QdrantMemoryProvider

    provider = QdrantMemoryProvider()
    provider._qdrant = FakeQdrant()
    provider._embeddings = FakeEmbedding()
    provider._config.update({"learning_collection_name": "learnings", "learning_enabled": True})

    stored = json.loads(provider.handle_tool_call("qdrant_learning_store", {"lesson": "ok"}))

    assert stored["saved"] is False
    assert stored["write_decision"]["decision"] == "skip"
    assert provider._qdrant.upserts == []


def test_learning_store_write_gate_rejects_secrets_in_persisted_fields():
    from __init__ import QdrantMemoryProvider

    provider = QdrantMemoryProvider()
    provider._qdrant = FakeQdrant()
    provider._embeddings = FakeEmbedding()
    provider._config.update({"learning_collection_name": "learnings", "learning_enabled": True})
    secret_value = "".join(["abc", "def", "ghi", "jkl", "mnop"])

    stored = json.loads(
        provider.handle_tool_call(
            "qdrant_learning_store",
            {"lesson": "Use dry-run before live writes", "evidence": "Authorization: " + "Bearer " + secret_value},
        )
    )

    assert "error" in stored
    assert "rejected" in stored["error"]
    assert provider._qdrant.upserts == []


def test_provider_learning_tools_respect_learning_enabled_flag():
    from __init__ import QdrantMemoryProvider

    provider = QdrantMemoryProvider()
    provider._qdrant = FakeQdrant()
    provider._embeddings = FakeEmbedding()
    provider._config.update({"learning_collection_name": "learnings", "learning_enabled": False})

    stored = json.loads(provider.handle_tool_call("qdrant_learning_store", {"lesson": "do not save"}))
    searched = json.loads(provider.handle_tool_call("qdrant_learning_search", {"query": "anything"}))

    assert "disabled" in stored["error"]
    assert "disabled" in searched["error"]
    assert provider._qdrant.upserts == []
