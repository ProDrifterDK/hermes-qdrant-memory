from __future__ import annotations

import json
import math

import pytest

from qdrant_memory.schema import DerivationEdge, MEMORY_KINDS, RELATION_TYPES, SourceLocator, build_payload


def _contains_non_finite_float(value):
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_contains_non_finite_float(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_non_finite_float(item) for item in value)
    return False


def test_build_payload_accepts_source_derivation_metadata_and_typed_helpers():
    payload = build_payload(
        text="Architecture notes",
        source="manual",
        source_type="manual",
        source_uri="file:///tmp/project.md",
        locator=SourceLocator(line_start=10, line_end=14, heading="Architecture"),
        content_hash="sha256:abc123",
        source_modified_at="2026-05-26T00:00:00Z",
        derivation_type="summary",
        derived_from=[
            DerivationEdge(
                source_uri="session://2026-05-25/example",
                locator={"message_id": 12345},
                derivation_type="completed_turn",
            )
        ],
        canonical=False,
        stale=True,
        requires_review=True,
    )

    assert payload["source_uri"] == "file:///tmp/project.md"
    assert payload["source_type"] == "manual"
    assert payload["locator"] == {"line_start": 10, "line_end": 14, "heading": "Architecture"}
    assert payload["content_hash"] == "sha256:abc123"
    assert payload["source_modified_at"] == "2026-05-26T00:00:00Z"
    assert payload["derivation_type"] == "summary"
    assert payload["derived_from"] == [
        {
            "source_uri": "session://2026-05-25/example",
            "locator": {"message_id": 12345},
            "derivation_type": "completed_turn",
        }
    ]
    assert payload["canonical"] is False
    assert payload["stale"] is True
    assert payload["requires_review"] is True
    json.dumps(payload)


def test_build_payload_source_ref_dict_cannot_override_core_payload_fields():
    payload = build_payload(
        text="original text",
        source="original-source",
        source_type="manual",
        chunk_type="fact",
        importance=3,
        confidence=0.75,
        tags=["original-tag"],
        source_ref={
            "text": "overridden text",
            "source": "other-source",
            "source_type": "other-type",
            "chunk_type": "other-chunk",
            "importance": 10,
            "confidence": 0.01,
            "tags": ["bad-tag"],
            "provider": "other-provider",
            "source_uri": "file:///safe/source.txt",
            "locator": {"line_start": 7, "text": "nested locator text is okay"},
        },
    )

    assert payload["text"] == "original text"
    assert payload["source"] == "original-source"
    assert payload["source_type"] == "manual"
    assert payload["chunk_type"] == "fact"
    assert payload["importance"] == 3
    assert payload["confidence"] == 0.75
    assert payload["tags"] == ["original-tag"]
    assert payload["provider"] == "qdrant"
    assert payload["source_uri"] == "file:///safe/source.txt"
    assert payload["locator"] == {"line_start": 7, "text": "nested locator text is okay"}


def test_build_payload_source_metadata_omits_non_finite_floats_for_strict_json():
    payload = build_payload(
        text="strict json",
        source="unit",
        source_type="manual",
        locator={
            "score": float("nan"),
            "positive_inf": float("inf"),
            "negative_inf": float("-inf"),
            "finite": 0.5,
        },
    )

    json.dumps(payload, allow_nan=False)
    assert payload["locator"] == {"finite": 0.5}
    assert not _contains_non_finite_float(payload)


def test_build_payload_omits_empty_source_derivation_metadata_for_legacy_callers():
    payload = build_payload(
        text="legacy memory",
        source="unit",
        source_type="manual",
        locator={},
        derived_from=[],
        source_uri="",
        content_hash=None,
        source_modified_at="",
        derivation_type=None,
        canonical=None,
        stale=None,
        requires_review=None,
    )

    for key in (
        "source_uri",
        "locator",
        "content_hash",
        "source_modified_at",
        "derivation_type",
        "derived_from",
        "canonical",
        "stale",
        "requires_review",
    ):
        assert key not in payload


def test_build_payload_secret_scans_user_controlled_source_metadata_strings():
    secret_heading = "".join(["api", "_key=", "super", "-secret", "-value"])
    bearer_value = "".join(["abcdef", "ghijkl", "mnop"])
    bearer_heading = " ".join(["Authorization:", "Bearer", bearer_value])
    payload = build_payload(
        text="safe text",
        source="unit",
        source_type="manual",
        source_uri="https://user:***@example.test/private.md",
        locator=SourceLocator(line_start=3, heading=secret_heading),
        derived_from=[
            DerivationEdge(
                source_uri="file:///tmp/source.md",
                locator={"heading": bearer_heading, "line_start": 1},
                derivation_type="summary",
            )
        ],
    )

    dumped = json.dumps(payload)
    assert "password@example" not in dumped
    assert secret_heading not in dumped
    assert bearer_heading not in dumped
    assert "source_uri" not in payload
    assert payload["locator"] == {"line_start": 3}
    assert payload["derived_from"] == [
        {"source_uri": "file:///tmp/source.md", "locator": {"line_start": 1}, "derivation_type": "summary"}
    ]


def test_memory_grammar_values_include_phase_6_terms():
    assert set(MEMORY_KINDS) >= {
        "conversation_turn",
        "manual_fact",
        "source_chunk",
        "learning",
        "assertion",
        "decision",
        "user_preference",
        "project_invariant",
        "tool_quirk",
        "workflow_lesson",
        "risk",
        "proposal",
        "summary",
    }
    assert set(RELATION_TYPES) >= {
        "DERIVED_FROM",
        "EXTRACTED_FROM",
        "SUMMARIZES",
        "SUPPORTS",
        "CONTRADICTS",
        "SUPERSEDES",
        "REFERENCES",
        "APPLIES_TO",
        "USES_TOOL",
        "PREFERS",
        "BLOCKS",
    }


def test_build_payload_accepts_valid_memory_kind():
    payload = build_payload(text="keep the API stable", source="unit", memory_kind="decision")

    assert payload["memory_kind"] == "decision"


def test_build_payload_omits_empty_memory_kind_for_legacy_callers():
    payload = build_payload(text="legacy memory", source="unit", memory_kind="")

    assert "memory_kind" not in payload


def test_build_payload_rejects_unknown_memory_kind():
    with pytest.raises(ValueError, match="memory_kind"):
        build_payload(text="bad kind", source="unit", memory_kind="unknown_kind")


def test_derivation_edge_accepts_valid_relation_type_and_serializes_it():
    edge = DerivationEdge(source_uri="memory://point/root", relation_type="SUPPORTS")

    assert edge.to_payload()["relation_type"] == "SUPPORTS"


def test_build_payload_rejects_unknown_relation_type_in_edges():
    with pytest.raises(ValueError, match="relation_type"):
        build_payload(
            text="bad edge",
            source="unit",
            derived_from=[{"source_uri": "memory://point/root", "relation_type": "NOT_A_RELATION"}],
        )

    with pytest.raises(ValueError, match="relation_type"):
        build_payload(
            text="bad source ref edge",
            source="unit",
            source_ref={"source_uri": "memory://point/root", "relation_type": "NOT_A_RELATION"},
        )
