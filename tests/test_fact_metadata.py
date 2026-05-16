from __future__ import annotations

from qdrant_memory.fact_metadata import derive_fact_metadata, normalize_key_part


def test_fact_metadata_prefers_explicit_tag_keys():
    metadata = derive_fact_metadata(
        text="The current TeamForge MCP binary is teamforge-mcp.",
        tags=["fact:teamforge.mcp.binary", "subject:TeamForge MCP binary", "topic:TeamForge"],
        source_type="manual",
    )

    assert metadata["fact_key"] == "teamforge.mcp.binary"
    assert metadata["reconsolidation_key"] == "teamforge.mcp.binary"
    assert metadata["subject"] == "TeamForge MCP binary"
    assert metadata["topic"] == "TeamForge"


def test_fact_metadata_extracts_subject_from_is_statement():
    metadata = derive_fact_metadata(
        text="TeamForge MCP binary is teamforge-mcp-hermes",
        source_type="manual",
    )

    assert metadata["subject"] == "TeamForge MCP binary"
    assert metadata["fact_key"] == "teamforge.mcp.binary"
    assert metadata["reconsolidation_key"] == "teamforge.mcp.binary"


def test_fact_metadata_does_not_create_key_for_generic_chat_turn():
    metadata = derive_fact_metadata(
        text="User: hello\nAssistant: hi",
        source_type="conversation",
        chunk_type="turn",
    )

    assert "fact_key" not in metadata
    assert "reconsolidation_key" not in metadata


def test_fact_metadata_skips_secret_bearing_text():
    secret = " ".join(["Authorization:", "Bearer", "".join(["abc", "def", "ghi", "jkl", "mno", "pqr", "stu", "vwx", "yz"])])
    metadata = derive_fact_metadata(
        text=f"Current API token is {secret}",
        source_type="manual",
    )

    assert metadata == {}


def test_fact_metadata_normalizes_key_stably():
    assert normalize_key_part(" TeamForge MCP: Binary!! ") == "teamforge.mcp.binary"
    assert normalize_key_part("teamforge_mcp/binary") == "teamforge.mcp.binary"
