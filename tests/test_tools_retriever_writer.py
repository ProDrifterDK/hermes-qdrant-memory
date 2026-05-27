from __future__ import annotations

import json

from qdrant_memory.retriever import RetrievedMemory, format_for_prompt
from qdrant_memory.schema import make_point_id
from qdrant_memory.tools import EXPAND_SCHEMA, INSPECT_SCHEMA, SEARCH_SCHEMA, SOURCE_STATUS_SCHEMA, STATUS_SCHEMA, STORE_SCHEMA, TRACE_SCHEMA
from qdrant_memory.writer import ConversationWriter, strip_injected_context


class FakeEmbedding:
    def __init__(self):
        self.documents = []
        self.queries = []

    def embed_document(self, text):
        self.documents.append(text)
        return [0.1, 0.2]

    def embed_query(self, text):
        self.queries.append(text)
        return [0.3, 0.4]

class FakeQdrant:
    def __init__(self):
        self.points = []
        self.searches = []
        self.search_results = []

    def upsert(self, name, points):
        self.points.extend(points)
        return {"status": "ok"}

    def search(self, name, vector, limit, filter=None, with_payload=True, with_vector=False):
        self.searches.append((name, vector, limit, filter, with_payload, with_vector))
        return self.search_results


def test_tool_schemas_have_required_names():
    assert STATUS_SCHEMA["name"] == "qdrant_memory_status"
    assert STORE_SCHEMA["parameters"]["required"] == ["text"]
    assert SEARCH_SCHEMA["parameters"]["required"] == ["query"]
    assert INSPECT_SCHEMA["name"] == "qdrant_memory_inspect"
    assert INSPECT_SCHEMA["parameters"]["required"] == ["point_id"]
    assert TRACE_SCHEMA["parameters"]["properties"]["direction"]["enum"] == ["upstream", "downstream", "both"]
    assert EXPAND_SCHEMA["parameters"]["properties"]["mode"]["enum"] == ["excerpt", "source", "neighbors"]
    assert SOURCE_STATUS_SCHEMA["name"] == "qdrant_memory_source_status"
    props = STORE_SCHEMA["parameters"]["properties"]
    assert props["dry_run"]["type"] == "boolean"
    assert props["approve"]["type"] == "boolean"
    assert props["duplicate_preview"]["type"] == "boolean"


def test_format_for_prompt_fenced_and_bounded():
    chunks = [
        RetrievedMemory(id="1", text="alpha memory", payload={"created_at": "2026-01-01T00:00:00+00:00", "importance": 7, "source_type": "manual"}, qdrant_score=0.8, final_score=0.7)
    ]
    out = format_for_prompt(chunks, display_tokens=50)
    assert out.startswith("# Relevant Long-Term Memory")
    assert "context, not instructions" in out
    assert "alpha memory" in out


def test_format_for_prompt_includes_compact_valid_memory_kind_only():
    out = format_for_prompt(
        [
            RetrievedMemory(
                id="1",
                text="Use the stable API shape",
                payload={
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "importance": 8,
                    "source_type": "manual",
                    "memory_kind": "decision",
                },
                qdrant_score=0.8,
                final_score=0.7,
            )
        ],
        display_tokens=80,
    )
    legacy_out = format_for_prompt(
        [
            RetrievedMemory(
                id="2",
                text="legacy memory",
                payload={"created_at": "2026-01-01T00:00:00+00:00", "importance": 5, "source_type": "manual"},
                qdrant_score=0.7,
                final_score=0.6,
            )
        ],
        display_tokens=80,
    )
    invalid_out = format_for_prompt(
        [
            RetrievedMemory(
                id="3",
                text="bad legacy kind",
                payload={
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "importance": 5,
                    "source_type": "manual",
                    "memory_kind": "not_a_kind",
                },
                qdrant_score=0.7,
                final_score=0.6,
            )
        ],
        display_tokens=80,
    )

    assert "kind=decision" in out
    assert "kind=unknown" not in out
    assert "kind=" not in legacy_out
    assert "not_a_kind" not in invalid_out


def test_strip_injected_context_removes_memory_sections():
    raw = "User text\n# Relevant Long-Term Memory\n- polluted\n```\nmore\n```\nAnswer text"
    stripped = strip_injected_context(raw)
    assert "polluted" not in stripped
    assert "User text" in stripped


def test_writer_stores_completed_turn_with_fake_clients():
    emb = FakeEmbedding()
    q = FakeQdrant()
    writer = ConversationWriter(
        qdrant=q,
        embeddings=emb,
        collection_name="test_collection",
        profile_id="coder",
        platform="cli",
        session_id="s1",
    )
    point_id = writer.store_turn("User asks about alpha", "Assistant answers beta")
    assert point_id
    assert q.points[0]["id"] == point_id
    assert q.points[0]["vector"] == [0.1, 0.2]
    payload = q.points[0]["payload"]
    assert payload["source_type"] == "conversation"
    assert payload["chunk_type"] == "turn"
    assert payload["memory_kind"] == "conversation_turn"
    assert "User: User asks about alpha" in payload["text"]
    assert "fact_key" not in payload


def test_writer_store_text_adds_fact_metadata_for_explicit_manual_fact():
    emb = FakeEmbedding()
    q = FakeQdrant()
    writer = ConversationWriter(qdrant=q, embeddings=emb, collection_name="test_collection")

    point_id = writer.store_text("TeamForge MCP binary is teamforge-mcp", source_type="manual")

    assert point_id
    payload = q.points[0]["payload"]
    assert payload["subject"] == "TeamForge MCP binary"
    assert payload["memory_kind"] == "manual_fact"
    assert payload["fact_key"] == "teamforge.mcp.binary"
    assert payload["reconsolidation_key"] == "teamforge.mcp.binary"


def test_writer_preview_text_does_not_embed_or_upsert():
    emb = FakeEmbedding()
    q = FakeQdrant()
    writer = ConversationWriter(qdrant=q, embeddings=emb, collection_name="test_collection")

    preview = writer.preview_text("TeamForge MCP binary is teamforge-mcp", source_type="manual", importance=6, tags=["ops"])

    assert preview["id"] == make_point_id("hermes_tool", preview["text"])
    assert preview["text"] == "TeamForge MCP binary is teamforge-mcp"
    assert preview["payload"]["importance"] == 6
    assert preview["payload"]["tags"] == ["ops"]
    assert preview["payload"]["memory_kind"] == "manual_fact"
    assert preview["payload"]["fact_key"] == "teamforge.mcp.binary"
    assert emb.documents == []
    assert q.points == []


def test_writer_preview_text_omits_memory_kind_for_unknown_source_chunk_type():
    writer = ConversationWriter(qdrant=FakeQdrant(), embeddings=FakeEmbedding(), collection_name="test_collection")

    preview = writer.preview_text("External notes mention the backup window.", source_type="external", chunk_type="note")

    assert "memory_kind" not in preview["payload"]


def test_writer_find_semantic_duplicate_uses_memory_collection_without_upsert():
    emb = FakeEmbedding()
    q = FakeQdrant()
    q.search_results = [
        {
            "id": "memory-1",
            "score": 0.93,
            "payload": {
                "text": "TeamForge MCP binary is teamforge-mcp",
                "source_type": "manual",
                "importance": 6,
            },
        }
    ]
    writer = ConversationWriter(qdrant=q, embeddings=emb, collection_name="test_collection", profile_id="coder")

    duplicate = writer.find_semantic_duplicate("TeamForge MCP binary is teamforge-mcp", source_type="manual", threshold=0.9, top_k=3)

    assert duplicate is not None
    assert duplicate["id"] == "memory-1"
    assert duplicate["score"] == 0.93
    assert duplicate["text"] == "TeamForge MCP binary is teamforge-mcp"
    assert q.searches[0][0] == "test_collection"
    assert q.searches[0][2] == 3
    assert {"key": "profile_id", "match": {"value": "coder"}} in q.searches[0][3]["must"]
    assert {"key": "source_type", "match": {"value": "manual"}} in q.searches[0][3]["must"]
    assert emb.queries == ["TeamForge MCP binary is teamforge-mcp"]
    assert q.points == []


def test_provider_memory_store_dry_run_approval_and_duplicate_preview():
    from __init__ import QdrantMemoryProvider

    emb = FakeEmbedding()
    q = FakeQdrant()
    provider = QdrantMemoryProvider()
    provider._qdrant = q
    provider._embeddings = emb
    provider._writer = ConversationWriter(qdrant=q, embeddings=emb, collection_name="test_collection", profile_id="coder")
    provider._config.update({"collection_name": "test_collection", "manual_store_duplicate_threshold": 0.9, "manual_store_duplicate_top_k": 3})

    dry_run = json.loads(provider.handle_tool_call("qdrant_memory_store", {"text": "TeamForge MCP binary is teamforge-mcp"}))
    assert dry_run["dry_run"] is True
    assert dry_run["saved"] is False
    assert dry_run["would_store"] is True
    assert dry_run["id"] == make_point_id("hermes_tool", "TeamForge MCP binary is teamforge-mcp")
    assert dry_run["write_decision"]["decision"] == "store"
    assert q.points == []
    assert emb.documents == []

    provider._config["manual_store_dry_run_default"] = False
    still_dry_run = json.loads(provider.handle_tool_call("qdrant_memory_store", {"text": "TeamForge MCP binary is teamforge-mcp", "approve": True}))
    assert still_dry_run["dry_run"] is True
    assert still_dry_run["saved"] is False
    assert q.points == []
    provider._config.pop("manual_store_dry_run_default", None)

    q.search_results = [
        {
            "id": "memory-1",
            "score": 0.93,
            "payload": {"text": "TeamForge MCP binary is teamforge-mcp", "source_type": "manual"},
        }
    ]
    dry_run_duplicate = json.loads(provider.handle_tool_call("qdrant_memory_store", {"text": "TeamForge MCP binary is teamforge-mcp", "duplicate_preview": True}))
    assert dry_run_duplicate["dry_run"] is True
    assert dry_run_duplicate["saved"] is False
    assert dry_run_duplicate["would_store"] is False
    assert dry_run_duplicate["duplicate_found"] is True
    assert dry_run_duplicate["write_decision"]["decision"] == "skip"
    assert q.points == []
    q.search_results = []

    unapproved = json.loads(provider.handle_tool_call("qdrant_memory_store", {"text": "TeamForge MCP binary is teamforge-mcp", "dry_run": False}))
    assert "approve" in unapproved["error"]
    assert q.points == []

    live = json.loads(provider.handle_tool_call("qdrant_memory_store", {"text": "TeamForge MCP binary is teamforge-mcp", "dry_run": False, "approve": True}))
    assert live["dry_run"] is False
    assert live["saved"] is True
    assert live["write_decision"]["decision"] == "store"
    assert len(q.points) == 1

    q.points.clear()
    q.search_results = [
        {
            "id": "memory-1",
            "score": 0.93,
            "payload": {"text": "TeamForge MCP binary is teamforge-mcp", "source_type": "manual"},
        }
    ]
    duplicate = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_store",
            {"text": "TeamForge MCP binary is teamforge-mcp", "dry_run": False, "approve": True, "duplicate_preview": True},
        )
    )
    assert duplicate["dry_run"] is False
    assert duplicate["saved"] is False
    assert duplicate["duplicate_found"] is True
    assert duplicate["duplicate"]["id"] == "memory-1"
    assert duplicate["write_decision"]["decision"] == "skip"
    assert q.points == []


def test_provider_memory_store_write_gate_blocks_review_only_live_store():
    from __init__ import QdrantMemoryProvider

    emb = FakeEmbedding()
    q = FakeQdrant()
    provider = QdrantMemoryProvider()
    provider._qdrant = q
    provider._embeddings = emb
    provider._writer = ConversationWriter(qdrant=q, embeddings=emb, collection_name="test_collection", profile_id="coder")
    provider._config.update({"collection_name": "test_collection"})

    dry_run = json.loads(provider.handle_tool_call("qdrant_memory_store", {"text": "Summarized memory that needs provenance before storage", "source_type": "summary"}))
    live = json.loads(provider.handle_tool_call("qdrant_memory_store", {"text": "Summarized memory that needs provenance before storage", "source_type": "summary", "dry_run": False, "approve": True}))

    assert dry_run["write_decision"]["decision"] == "draft_review"
    assert dry_run["would_store"] is False
    assert "review" in live["error"]
    assert q.points == []


def test_tool_error_json_shape_with_uninitialized_provider():
    from __init__ import QdrantMemoryProvider

    provider = QdrantMemoryProvider()
    result = json.loads(provider.handle_tool_call("qdrant_memory_search", {"query": "x"}))
    assert result["error"]
