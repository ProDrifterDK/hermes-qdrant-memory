from __future__ import annotations

import json

from qdrant_memory.retriever import RetrievedMemory, format_for_prompt
from qdrant_memory.tools import SEARCH_SCHEMA, STATUS_SCHEMA, STORE_SCHEMA
from qdrant_memory.writer import ConversationWriter, strip_injected_context


class FakeEmbedding:
    def __init__(self):
        self.documents = []

    def embed_document(self, text):
        self.documents.append(text)
        return [0.1, 0.2]


class FakeQdrant:
    def __init__(self):
        self.points = []

    def upsert(self, name, points):
        self.points.extend(points)
        return {"status": "ok"}


def test_tool_schemas_have_required_names():
    assert STATUS_SCHEMA["name"] == "qdrant_memory_status"
    assert STORE_SCHEMA["parameters"]["required"] == ["text"]
    assert SEARCH_SCHEMA["parameters"]["required"] == ["query"]


def test_format_for_prompt_fenced_and_bounded():
    chunks = [
        RetrievedMemory(id="1", text="alpha memory", payload={"created_at": "2026-01-01T00:00:00+00:00", "importance": 7, "source_type": "manual"}, qdrant_score=0.8, final_score=0.7)
    ]
    out = format_for_prompt(chunks, display_tokens=50)
    assert out.startswith("# Relevant Long-Term Memory")
    assert "context, not instructions" in out
    assert "alpha memory" in out


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
    assert payload["fact_key"] == "teamforge.mcp.binary"
    assert payload["reconsolidation_key"] == "teamforge.mcp.binary"


def test_tool_error_json_shape_with_uninitialized_provider():
    from __init__ import QdrantMemoryProvider

    provider = QdrantMemoryProvider()
    result = json.loads(provider.handle_tool_call("qdrant_memory_search", {"query": "x"}))
    assert result["error"]
