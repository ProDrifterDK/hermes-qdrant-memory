"""Tests for graph entity/edge schema primitives.

Covers:
- Deterministic ID generation and validation.
- Payload serialization with safe redaction.
- No canonical auto-promotion.
- Provenance requirements for edges.
- Backward compatibility with existing schema grammar.
- Secret rejection in all user-controlled fields.
"""

from __future__ import annotations

import json
import math

import pytest

from qdrant_memory.graph_schema import (
    KNOWN_ENTITY_TYPES,
    GraphEdge,
    GraphEntity,
    build_edge_payload,
    build_entity_payload,
    make_edge_id,
    make_entity_id,
    sanitize_aliases,
    sanitize_source_point_ids,
    valid_edge_id,
    valid_entity_id,
    validate_edge_id,
    validate_entity_id,
)
from qdrant_memory.schema import (
    FACT_STATUSES,
    MEMORY_KINDS,
    RELATION_TYPES,
    MemoryKind,
    RelationType,
)


def _contains_non_finite(value):
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_contains_non_finite(v) for v in value.values())
    if isinstance(value, list):
        return any(_contains_non_finite(v) for v in value)
    return False


# ---------------------------------------------------------------------------
# ID generation tests
# ---------------------------------------------------------------------------

class TestEntityIDGeneration:
    def test_make_entity_id_is_deterministic(self):
        id1 = make_entity_id("concept", "Neural Plasticity")
        id2 = make_entity_id("concept", "Neural Plasticity")
        assert id1 == id2
        assert id1.startswith("entity-")

    def test_make_entity_id_changes_with_different_inputs(self):
        id1 = make_entity_id("concept", "Neural Plasticity")
        id2 = make_entity_id("concept", "Hebbian Learning")
        id3 = make_entity_id("person", "Neural Plasticity")
        id4 = make_entity_id("concept", "Neural Plasticity", profile_id="other")
        assert id1 != id2
        assert id1 != id3
        assert id1 != id4

    def test_make_entity_id_format(self):
        eid = make_entity_id("tool", "pytest")
        assert eid.startswith("entity-")
        assert len(eid) == len("entity-") + 16
        # Must be hex
        hex_part = eid[len("entity-"):]
        int(hex_part, 16)  # raises if not hex

    def test_make_entity_id_case_insensitive_label(self):
        id1 = make_entity_id("concept", "Neural Plasticity")
        id2 = make_entity_id("concept", "neural plasticity")
        id3 = make_entity_id("concept", "NEURAL PLASTICITY")
        assert id1 == id2 == id3

    def test_make_entity_id_rejects_empty_entity_type(self):
        with pytest.raises(ValueError, match="entity_type"):
            make_entity_id("", "label")

    def test_make_entity_id_rejects_empty_label(self):
        with pytest.raises(ValueError, match="label"):
            make_entity_id("concept", "")

    def test_make_entity_id_normalizes_whitespace(self):
        id1 = make_entity_id("concept", "Neural  Plasticity")
        id2 = make_entity_id("concept", "Neural Plasticity")
        assert id1 == id2


class TestEdgeIDGeneration:
    def test_make_edge_id_is_deterministic(self):
        src = make_entity_id("concept", "A")
        tgt = make_entity_id("concept", "B")
        id1 = make_edge_id(src, tgt, "SUPPORTS")
        id2 = make_edge_id(src, tgt, "SUPPORTS")
        assert id1 == id2
        assert id1.startswith("edge-")

    def test_make_edge_id_changes_with_different_inputs(self):
        src = make_entity_id("concept", "A")
        tgt = make_entity_id("concept", "B")
        other = make_entity_id("concept", "C")
        id1 = make_edge_id(src, tgt, "SUPPORTS")
        id2 = make_edge_id(src, tgt, "CONTRADICTS")
        id3 = make_edge_id(src, other, "SUPPORTS")
        id4 = make_edge_id(tgt, src, "SUPPORTS")
        id5 = make_edge_id(src, tgt, "SUPPORTS", profile_id="other")
        assert len({id1, id2, id3, id4, id5}) == 5

    def test_make_edge_id_format(self):
        src = make_entity_id("concept", "A")
        tgt = make_entity_id("concept", "B")
        eid = make_edge_id(src, tgt, "RELATED_TO")
        assert eid.startswith("edge-")
        hex_part = eid[len("edge-"):]
        assert len(hex_part) == 16
        int(hex_part, 16)

    def test_make_edge_id_rejects_empty_args(self):
        src = make_entity_id("concept", "A")
        tgt = make_entity_id("concept", "B")
        with pytest.raises(ValueError, match="source_entity_id"):
            make_edge_id("", tgt, "SUPPORTS")
        with pytest.raises(ValueError, match="target_entity_id"):
            make_edge_id(src, "", "SUPPORTS")
        with pytest.raises(ValueError, match="relation_type"):
            make_edge_id(src, tgt, "")


# ---------------------------------------------------------------------------
# ID validation tests
# ---------------------------------------------------------------------------

class TestIDValidation:
    def test_validate_entity_id_accepts_well_formed(self):
        eid = make_entity_id("concept", "Test")
        assert validate_entity_id(eid) == eid

    def test_validate_entity_id_rejects_non_string(self):
        with pytest.raises(ValueError, match="string"):
            validate_entity_id(123)

    def test_validate_entity_id_rejects_empty(self):
        with pytest.raises(ValueError, match="empty"):
            validate_entity_id("")

    def test_validate_entity_id_rejects_wrong_prefix(self):
        with pytest.raises(ValueError, match="entity-"):
            validate_entity_id("edge-abcdef0123456789")

    def test_validate_entity_id_rejects_wrong_length(self):
        with pytest.raises(ValueError):
            validate_entity_id("entity-short")

    def test_validate_entity_id_rejects_non_hex(self):
        with pytest.raises(ValueError):
            validate_entity_id("entity-xxxxxxxxxxxxxz12")

    def test_valid_entity_id_returns_none_on_invalid(self):
        assert valid_entity_id("bad") is None
        assert valid_entity_id(123) is None
        assert valid_entity_id(None) is None
        eid = make_entity_id("concept", "Test")
        assert valid_entity_id(eid) == eid

    def test_validate_edge_id_accepts_well_formed(self):
        src = make_entity_id("concept", "A")
        tgt = make_entity_id("concept", "B")
        e = make_edge_id(src, tgt, "SUPPORTS")
        assert validate_edge_id(e) == e

    def test_validate_edge_id_rejects_wrong_prefix(self):
        with pytest.raises(ValueError, match="edge-"):
            validate_edge_id("entity-abcdef0123456789")

    def test_valid_edge_id_returns_none_on_invalid(self):
        assert valid_edge_id("bad") is None
        src = make_entity_id("concept", "A")
        tgt = make_entity_id("concept", "B")
        e = make_edge_id(src, tgt, "SUPPORTS")
        assert valid_edge_id(e) == e


# ---------------------------------------------------------------------------
# Alias sanitization tests
# ---------------------------------------------------------------------------

class TestAliasSanitization:
    def test_strips_whitespace(self):
        assert sanitize_aliases(["  a  ", " b "]) == ["a", "b"]

    def test_deduplicates_case_insensitive(self):
        result = sanitize_aliases(["Python", "python", "PYTHON"])
        assert result == ["Python"]

    def test_rejects_empty(self):
        assert sanitize_aliases(["", "   ", "valid"]) == ["valid"]

    def test_rejects_non_strings(self):
        assert sanitize_aliases([1, None, "ok"]) == ["ok"]

    def test_rejects_secret_bearing(self):
        secret = "".join(["api", "_key=", "secret123"])
        result = sanitize_aliases([secret, "safe"])
        assert result == ["safe"]

    def test_caps_at_max(self):
        aliases = [f"alias-{i}" for i in range(100)]
        result = sanitize_aliases(aliases)
        assert len(result) == 32

    def test_rejects_non_list(self):
        assert sanitize_aliases("not a list") == []
        assert sanitize_aliases(None) == []
        assert sanitize_aliases(123) == []


# ---------------------------------------------------------------------------
# Source point ID sanitization tests
# ---------------------------------------------------------------------------

class TestSourcePointIDSanitization:
    def test_accepts_valid_point_ids(self):
        ids = ["abc123", "point-001", "node_xyz"]
        result = sanitize_source_point_ids(ids)
        assert result == ids

    def test_rejects_urls(self):
        result = sanitize_source_point_ids(["http://evil.com/x"])
        assert result == []

    def test_rejects_secrets(self):
        result = sanitize_source_point_ids(["api_key=secret"])
        assert result == []

    def test_deduplicates(self):
        result = sanitize_source_point_ids(["a", "a", "b"])
        assert result == ["a", "b"]

    def test_rejects_non_list(self):
        assert sanitize_source_point_ids(None) == []
        assert sanitize_source_point_ids("abc") == []


# ---------------------------------------------------------------------------
# Entity payload tests
# ---------------------------------------------------------------------------

class TestEntityPayload:
    def test_build_entity_payload_produces_safe_payload(self):
        payload = build_entity_payload(
            entity_type="concept",
            label="Hermes Qdrant Memory",
            description="A plugin for associative memory.",
            aliases=["qdrant-memory", "hermes-memory"],
            confidence=0.8,
            source_uri="file:///docs/architecture.md",
            content_hash="sha256:abc123",
            tags=["architecture", "memory"],
        )
        eid = make_entity_id("concept", "Hermes Qdrant Memory")
        assert payload["entity_id"] == eid
        assert payload["entity_type"] == "concept"
        assert payload["label"] == "Hermes Qdrant Memory"
        assert payload["memory_kind"] == "graph_entity"
        assert payload["source"] == "graph_entity"
        assert payload["source_type"] == "graph"
        assert payload["chunk_type"] == "entity"
        assert payload["confidence"] == 0.8
        assert payload["canonical"] is False
        assert payload["requires_review"] is True
        assert payload["usefulness_weight"] == 0.0
        assert payload["truth_confidence"] == 0.0
        assert payload["fact_status"] == "active"
        assert payload["source_uri"] == "file:///docs/architecture.md"
        assert payload["content_hash"] == "sha256:abc123"
        assert payload["aliases"] == ["qdrant-memory", "hermes-memory"]
        assert payload["description"] == "A plugin for associative memory."
        # Legacy search compatibility
        assert payload["text"] == "Hermes Qdrant Memory"
        # JSON serializable
        json.dumps(payload, allow_nan=False)

    def test_entity_payload_never_auto_promotes_canonical(self):
        # Even if we explicitly try to set canonical=True on the dataclass,
        # to_payload() must force it to False.
        entity = GraphEntity(
            entity_type="concept",
            label="Test",
            canonical=True,  # should be ignored
        )
        payload = entity.to_payload()
        assert payload["canonical"] is False
        assert payload["requires_review"] is True

    def test_entity_payload_rejects_empty_entity_type(self):
        with pytest.raises(ValueError, match="entity_type"):
            build_entity_payload(entity_type="", label="Test")

    def test_entity_payload_rejects_empty_label(self):
        with pytest.raises(ValueError, match="label"):
            build_entity_payload(entity_type="concept", label="")

    def test_entity_payload_rejects_secret_in_label(self):
        secret_label = "".join(["api", "_key=", "secret123"])
        with pytest.raises(ValueError, match="secrets"):
            build_entity_payload(entity_type="concept", label=secret_label)

    def test_entity_payload_rejects_secret_in_source_uri(self):
        secret_uri = "https://" + "user" + ":" + "pass" + "@evil.test/x"
        payload = build_entity_payload(
            entity_type="concept",
            label="Test",
            source_uri=secret_uri,
        )
        assert "source_uri" not in payload

    def test_entity_payload_rejects_secret_in_description(self):
        secret_desc = "".join(["Bearer ", "abcdefghijklmnopqrstuvwxyz0123456789"])
        payload = build_entity_payload(
            entity_type="concept",
            label="Test",
            description=secret_desc,
        )
        assert "description" not in payload

    def test_entity_payload_rejects_secret_in_aliases(self):
        secret_alias = "".join(["password=", "supersecret"])
        payload = build_entity_payload(
            entity_type="concept",
            label="Test",
            aliases=[secret_alias, "safe-alias"],
        )
        assert payload["aliases"] == ["safe-alias"]

    def test_entity_payload_clamps_confidence(self):
        payload = build_entity_payload(
            entity_type="concept",
            label="Test",
            confidence=5.0,
        )
        assert payload["confidence"] == 1.0

        payload = build_entity_payload(
            entity_type="concept",
            label="Test",
            confidence=-1.0,
        )
        assert payload["confidence"] == 0.0

    def test_entity_payload_clamps_weights(self):
        payload = build_entity_payload(
            entity_type="concept",
            label="Test",
            usefulness_weight=99.0,
            truth_confidence=-5.0,
        )
        assert payload["usefulness_weight"] == 1.0
        assert payload["truth_confidence"] == 0.0

    def test_entity_payload_usefulness_and_truth_are_separate(self):
        payload = build_entity_payload(
            entity_type="concept",
            label="Test",
            usefulness_weight=0.9,
            truth_confidence=0.1,
        )
        assert payload["usefulness_weight"] == 0.9
        assert payload["truth_confidence"] == 0.1
        assert payload["usefulness_weight"] != payload["truth_confidence"]

    def test_entity_payload_rejects_invalid_fact_status(self):
        payload = build_entity_payload(
            entity_type="concept",
            label="Test",
            fact_status="bogus",
        )
        # Falls back to "active" for invalid status
        assert payload["fact_status"] == "active"

    def test_entity_payload_valid_fact_status(self):
        for status in ("active", "stale", "deprecated", "disputed", "superseded", "review_required"):
            payload = build_entity_payload(
                entity_type="concept",
                label="Test",
                fact_status=status,
            )
            assert payload["fact_status"] == status

    def test_entity_payload_sanitizes_extra_metadata(self):
        secret_uri = "https://" + "user" + ":" + "pass" + "@evil.test/x"
        payload = build_entity_payload(
            entity_type="concept",
            label="Test",
            extra={
                "safe_field": "ok",
                "secret_field": secret_uri,
                "nested": {"deep": "value", "bad": "".join(["api", "_key=", "secret"])},
            },
        )
        dumped = json.dumps(payload)
        assert secret_uri not in dumped
        assert "api_key" not in dumped
        assert payload.get("safe_field") == "ok"

    def test_entity_payload_deterministic_id_across_calls(self):
        p1 = build_entity_payload(entity_type="concept", label="Same Entity")
        p2 = build_entity_payload(entity_type="concept", label="Same Entity")
        assert p1["entity_id"] == p2["entity_id"]

    def test_entity_payload_different_profile_different_id(self):
        p1 = build_entity_payload(entity_type="concept", label="Entity", profile_id="default")
        p2 = build_entity_payload(entity_type="concept", label="Entity", profile_id="work")
        assert p1["entity_id"] != p2["entity_id"]


# ---------------------------------------------------------------------------
# Edge payload tests
# ---------------------------------------------------------------------------

class TestEdgePayload:
    def _make_entities(self):
        src = make_entity_id("concept", "Source Concept")
        tgt = make_entity_id("concept", "Target Concept")
        return src, tgt

    def test_build_edge_payload_produces_safe_payload(self):
        src, tgt = self._make_entities()
        payload = build_edge_payload(
            source_entity_id=src,
            target_entity_id=tgt,
            relation_type="SUPPORTS",
            source_point_ids=["point-001", "point-002"],
            confidence=0.7,
            source_uri="file:///docs/evidence.md",
            observed_at="2026-06-26T10:00:00Z",
            valid_from="2026-06-26T00:00:00Z",
            tags=["evidence"],
        )
        expected_eid = make_edge_id(src, tgt, "SUPPORTS")
        assert payload["edge_id"] == expected_eid
        assert payload["source_entity_id"] == src
        assert payload["target_entity_id"] == tgt
        assert payload["relation_type"] == "SUPPORTS"
        assert payload["memory_kind"] == "graph_edge"
        assert payload["source"] == "graph_edge"
        assert payload["source_type"] == "graph"
        assert payload["chunk_type"] == "edge"
        assert payload["confidence"] == 0.7
        assert payload["canonical"] is False
        assert payload["requires_review"] is True
        assert payload["usefulness_weight"] == 0.0
        assert payload["truth_confidence"] == 0.0
        assert payload["fact_status"] == "active"
        assert payload["source_point_ids"] == ["point-001", "point-002"]
        assert payload["source_uri"] == "file:///docs/evidence.md"
        assert payload["observed_at"] == "2026-06-26T10:00:00Z"
        assert payload["valid_from"] == "2026-06-26T00:00:00Z"
        # Legacy search compatibility
        assert payload["text"] == f"{src} SUPPORTS {tgt}"
        json.dumps(payload, allow_nan=False)

    def test_edge_payload_never_auto_promotes_canonical(self):
        src, tgt = self._make_entities()
        edge = GraphEdge(
            source_entity_id=src,
            target_entity_id=tgt,
            relation_type="SUPPORTS",
            source_uri="file:///docs/x.md",
            canonical=True,
            requires_review=False,
        )
        payload = edge.to_payload()
        assert payload["canonical"] is False
        assert payload["requires_review"] is True

    def test_edge_payload_requires_provenance(self):
        src, tgt = self._make_entities()
        with pytest.raises(ValueError, match="provenance"):
            build_edge_payload(
                source_entity_id=src,
                target_entity_id=tgt,
                relation_type="SUPPORTS",
                # no source_point_ids or source_uri
            )

    def test_edge_payload_accepts_provenance_via_source_uri_only(self):
        src, tgt = self._make_entities()
        payload = build_edge_payload(
            source_entity_id=src,
            target_entity_id=tgt,
            relation_type="SUPPORTS",
            source_uri="file:///docs/evidence.md",
        )
        assert payload["source_uri"] == "file:///docs/evidence.md"
        assert "source_point_ids" not in payload

    def test_edge_payload_accepts_provenance_via_point_ids_only(self):
        src, tgt = self._make_entities()
        payload = build_edge_payload(
            source_entity_id=src,
            target_entity_id=tgt,
            relation_type="SUPPORTS",
            source_point_ids=["abc123"],
        )
        assert payload["source_point_ids"] == ["abc123"]
        assert "source_uri" not in payload

    def test_edge_payload_rejects_invalid_entity_ids(self):
        with pytest.raises(ValueError, match="entity_id"):
            build_edge_payload(
                source_entity_id="not-an-entity-id",
                target_entity_id="entity-abcdef0123456789",
                relation_type="SUPPORTS",
                source_uri="file:///x.md",
            )

    def test_edge_payload_rejects_invalid_relation_type(self):
        src, tgt = self._make_entities()
        with pytest.raises(ValueError, match="relation_type"):
            build_edge_payload(
                source_entity_id=src,
                target_entity_id=tgt,
                relation_type="NOT_A_RELATION",
                source_uri="file:///x.md",
            )

    def test_edge_payload_rejects_secret_in_source_uri(self):
        src, tgt = self._make_entities()
        secret_uri = "https://" + "user" + ":" + "pass" + "@evil.test/x"
        with pytest.raises(ValueError, match="provenance"):
            build_edge_payload(
                source_entity_id=src,
                target_entity_id=tgt,
                relation_type="SUPPORTS",
                source_uri=secret_uri,
            )

    def test_edge_payload_rejects_secret_source_point_ids(self):
        src, tgt = self._make_entities()
        payload = build_edge_payload(
            source_entity_id=src,
            target_entity_id=tgt,
            relation_type="SUPPORTS",
            source_point_ids=["api_key=secret", "valid-point"],
        )
        assert payload["source_point_ids"] == ["valid-point"]

    def test_edge_payload_deterministic_id(self):
        src, tgt = self._make_entities()
        p1 = build_edge_payload(
            source_entity_id=src,
            target_entity_id=tgt,
            relation_type="SUPPORTS",
            source_uri="file:///a.md",
        )
        p2 = build_edge_payload(
            source_entity_id=src,
            target_entity_id=tgt,
            relation_type="SUPPORTS",
            source_uri="file:///b.md",
        )
        assert p1["edge_id"] == p2["edge_id"]

    def test_edge_payload_clamps_weights(self):
        src, tgt = self._make_entities()
        payload = build_edge_payload(
            source_entity_id=src,
            target_entity_id=tgt,
            relation_type="SUPPORTS",
            source_uri="file:///x.md",
            confidence=10.0,
            usefulness_weight=-1.0,
            truth_confidence=99.0,
        )
        assert payload["confidence"] == 1.0
        assert payload["usefulness_weight"] == 0.0
        assert payload["truth_confidence"] == 1.0

    def test_edge_payload_all_new_relation_types(self):
        """Verify all graph-specific relation types from schema.py work."""
        src, tgt = self._make_entities()
        for rel in ("IS_A", "PART_OF", "RELATED_TO", "CREATED_BY", "DEPENDS_ON", "LOCATED_IN"):
            payload = build_edge_payload(
                source_entity_id=src,
                target_entity_id=tgt,
                relation_type=rel,
                source_uri="file:///x.md",
            )
            assert payload["relation_type"] == rel


# ---------------------------------------------------------------------------
# Backward compatibility tests
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_graph_memory_kinds_in_grammar(self):
        """GRAPH_ENTITY and GRAPH_EDGE must be in the MEMORY_KINDS tuple."""
        assert "graph_entity" in MEMORY_KINDS
        assert "graph_edge" in MEMORY_KINDS
        assert MemoryKind.GRAPH_ENTITY.value == "graph_entity"
        assert MemoryKind.GRAPH_EDGE.value == "graph_edge"

    def test_graph_relation_types_in_grammar(self):
        """New relation types must be in RELATION_TYPES tuple."""
        for rel in ("IS_A", "PART_OF", "RELATED_TO", "CREATED_BY", "DEPENDS_ON", "LOCATED_IN"):
            assert rel in RELATION_TYPES

    def test_existing_relation_types_preserved(self):
        """All original relation types must still be present."""
        for rel in (
            "DERIVED_FROM", "EXTRACTED_FROM", "SUMMARIZES", "SUPPORTS",
            "CONTRADICTS", "SUPERSEDES", "REFERENCES", "APPLIES_TO",
            "USES_TOOL", "PREFERS", "BLOCKS",
        ):
            assert rel in RELATION_TYPES

    def test_existing_memory_kinds_preserved(self):
        """All original memory kinds must still be present."""
        for kind in (
            "conversation_turn", "manual_fact", "source_chunk", "learning",
            "assertion", "decision", "user_preference", "project_invariant",
            "tool_quirk", "workflow_lesson", "risk", "proposal", "summary",
        ):
            assert kind in MEMORY_KINDS

    def test_fact_statuses_unchanged(self):
        assert set(FACT_STATUSES) == {
            "active", "stale", "deprecated", "disputed", "superseded", "review_required",
        }

    def test_entity_payload_has_text_field_for_legacy_search(self):
        payload = build_entity_payload(entity_type="concept", label="Searchable Label")
        assert payload["text"] == "Searchable Label"

    def test_edge_payload_has_text_field_for_legacy_search(self):
        src = make_entity_id("concept", "A")
        tgt = make_entity_id("concept", "B")
        payload = build_edge_payload(
            source_entity_id=src,
            target_entity_id=tgt,
            relation_type="SUPPORTS",
            source_uri="file:///x.md",
        )
        assert "text" in payload
        assert isinstance(payload["text"], str)
        assert len(payload["text"]) > 0


# ---------------------------------------------------------------------------
# Known entity types coverage
# ---------------------------------------------------------------------------

class TestKnownEntityTypes:
    def test_known_entity_types_include_core_types(self):
        for etype in ("concept", "person", "project", "tool", "technology", "organization"):
            assert etype in KNOWN_ENTITY_TYPES

    def test_known_entity_types_include_teamforge_domain(self):
        for etype in ("task", "agent", "worktree", "review", "blocker", "dependency"):
            assert etype in KNOWN_ENTITY_TYPES

    def test_known_entity_types_include_nucleogenesis_domain(self):
        for etype in ("hypothesis", "mechanism", "experiment", "metric", "seed", "failure_mode"):
            assert etype in KNOWN_ENTITY_TYPES


# ---------------------------------------------------------------------------
# JSON serialization safety
# ---------------------------------------------------------------------------

class TestJSONSerialization:
    def test_entity_payload_is_strict_json_serializable(self):
        payload = build_entity_payload(
            entity_type="concept",
            label="Test",
            aliases=["a", "b"],
            description="desc",
        )
        # allow_nan=False ensures no inf/nan
        json.dumps(payload, allow_nan=False)

    def test_edge_payload_is_strict_json_serializable(self):
        src = make_entity_id("concept", "A")
        tgt = make_entity_id("concept", "B")
        payload = build_edge_payload(
            source_entity_id=src,
            target_entity_id=tgt,
            relation_type="SUPPORTS",
            source_uri="file:///x.md",
        )
        json.dumps(payload, allow_nan=False)

    def test_entity_payload_no_non_finite_floats(self):
        payload = build_entity_payload(
            entity_type="concept",
            label="Test",
            aliases=["a", "b"],
            confidence=0.5,
        )
        json.dumps(payload, allow_nan=False)
        assert not _contains_non_finite(payload)

    def test_entity_payload_rejects_nan_confidence(self):
        with pytest.raises(ValueError, match="confidence"):
            build_entity_payload(
                entity_type="concept",
                label="Test",
                confidence=float("nan"),
            )
