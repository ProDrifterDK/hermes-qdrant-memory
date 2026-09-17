from __future__ import annotations

import json

import pytest

from qdrant_memory.write_gate import WriteDecision, decision_to_json, evaluate_write_candidate

from qdrant_memory import graph_schema, lineage


def test_write_gate_rejects_secret_text():
    secret = "".join(["api", "_key=", "secret", "-value"])

    decision = evaluate_write_candidate(text=f"store this {secret}", source_type="manual")

    assert decision.decision == "reject"
    assert decision.requires_review is True
    assert "possible_secret" in decision.reasons


def test_write_gate_skips_empty_trivial_and_duplicates():
    assert evaluate_write_candidate(text="   ").decision == "skip"
    assert evaluate_write_candidate(text="ok").decision == "skip"

    duplicate = evaluate_write_candidate(text="Remember this meaningful operational fact", duplicate={"id": "p1", "score": 0.97})
    assert duplicate.decision == "skip"
    assert duplicate.metadata["duplicate"]["id"] == "p1"


def test_write_gate_routes_derived_writes_without_provenance_to_draft_review():
    decision = evaluate_write_candidate(
        text="Summarized durable memory with enough information to matter.",
        derivation_type="summary",
        confidence=0.9,
    )

    assert decision.decision == "draft_review"
    assert decision.requires_review is True
    assert "missing_provenance" in decision.reasons


def test_write_gate_allows_provenance_rich_derived_write():
    decision = evaluate_write_candidate(
        text="Summarized durable memory with enough information to matter.",
        derivation_type="summary",
        derived_from=[{"source_uri": "session://abc", "derivation_type": "completed_turn"}],
        confidence=0.9,
    )

    assert decision.decision == "store"
    assert decision.requires_review is False


def test_write_gate_low_confidence_derived_write_requires_review():
    decision = evaluate_write_candidate(
        text="Summarized durable memory with enough information to matter.",
        derivation_type="summary",
        source_uri="session://abc",
        confidence=0.4,
    )

    assert decision.decision == "draft_review"
    assert "low_confidence_derived_write" in decision.reasons


def test_write_gate_learning_and_skill_candidates_are_reviewed():
    learning = evaluate_write_candidate(
        text="When this workflow appears, use the exact-ID apply path and verify dry-run output.",
        target="learning",
        confidence=0.8,
    )
    skill = evaluate_write_candidate(
        text="Remember this important correction: always preserve source provenance before durable writes.",
        source_type="manual",
        promote_to_skill_candidate=True,
        confidence=0.95,
    )

    assert learning.decision == "learning_candidate"
    assert learning.requires_review is True
    assert skill.decision == "skill_candidate"
    assert skill.requires_review is True


def test_write_decision_is_json_serializable():
    decision = WriteDecision(decision="store", reasons=["storeable"], confidence=0.75, requires_review=False, metadata={"importance": 5})

    payload = json.loads(decision_to_json(decision))

    assert payload == {
        "decision": "store",
        "reasons": ["storeable"],
        "confidence": 0.75,
        "requires_review": False,
        "metadata": {"importance": 5},
    }


# ===========================================================================
# Phase 1 regression tests: RAPTOR summary write-gate + recursive contamination
# ===========================================================================

from qdrant_memory.write_gate import evaluate_raptor_summary_write
from qdrant_memory.schema import clean_text_for_memory


class TestRaptorSummaryWriteGate:
    """Model-authored RAPTOR summaries must route to review/reject unless
    they carry full provenance (child hashes, citations) and never claim
    canonical=true or requires_review=false.

    Phase 4 fix4 tightens the post-enrichment gate to mirror the
    pre-enrichment gate: ``canonical`` must be exactly the boolean
    ``False``, ``requires_review`` must be exactly the boolean ``True``,
    ``source_hashes`` must be a non-empty list of 64-char lowercase hex
    SHA-256 strings, and ``derived_from`` must be a non-empty list of
    RAPTOR provenance edges with non-empty ``source_uri``/``child_node_id``,
    ``derivation_type == "raptor_summary"``, and ``relation_type ==
    "SUMMARIZES"``.
    """

    # 64-char lowercase hex SHA-256 strings (matches the builder's
    # ``hashlib.sha256(...).hexdigest()`` output).
    _GOOD_HASH_A = "a" * 64
    _GOOD_HASH_B = "b" * 64
    _GOOD_METADATA = {
        "raptor_node_id": "raptor-node-001",
        "raptor_child_ids": ["child-1", "child-2"],
        "source_hashes": [_GOOD_HASH_A, _GOOD_HASH_B],
        "derived_from": [
            {
                "source_uri": "raptor://node/raptor-tree-x/child-1",
                "derivation_type": "raptor_summary",
                "relation_type": "SUMMARIZES",
                "child_node_id": "child-1",
            }
        ],
        # Pre-enrichment gate mirrors the post-enrichment gate in fix4.
        "canonical": False,
        "requires_review": True,
    }

    def test_raptor_rejects_canonical_true(self):
        decision = evaluate_raptor_summary_write(
            text="Summary of cluster alpha with enough text to matter.",
            metadata={**self._GOOD_METADATA, "canonical": True},
        )
        assert decision.decision == "reject"
        assert "raptor_summary_must_not_be_canonical" in decision.reasons

    def test_raptor_rejects_requires_review_false(self):
        decision = evaluate_raptor_summary_write(
            text="Summary of cluster alpha with enough text to matter.",
            metadata={**self._GOOD_METADATA, "requires_review": False},
        )
        assert decision.decision == "reject"
        assert "raptor_summary_must_not_skip_review" in decision.reasons

    def test_raptor_draft_review_missing_provenance(self):
        # With fix4 strict checks, a metadata dict that omits the
        # required trust flags is rejected on the canonical check first
        # (canonical is missing, not exactly False). The decision is
        # still non-store and requires_review=True.
        decision = evaluate_raptor_summary_write(
            text="Summary of cluster alpha with enough text to matter.",
            metadata={
                "raptor_node_id": "raptor-node-001",
                # Missing raptor_child_ids, source_hashes, and the trust
                # flags (canonical, requires_review) — strict gate rejects.
            },
        )
        assert decision.requires_review is True
        assert decision.decision in {"draft_review", "reject"}

    def test_raptor_draft_review_missing_citations(self):
        # With strict source-hash + derived_from checks, a metadata dict
        # with a valid raptor_node_id + raptor_child_ids but no
        # source_hashes/derived_from fails on the strict hash/edge gate
        # first. We still expect ``requires_review`` is True.
        decision = evaluate_raptor_summary_write(
            text="Summary of cluster alpha with enough text to matter.",
            metadata={
                "raptor_node_id": "raptor-node-001",
                "raptor_child_ids": ["child-1"],
                # No source_hashes or derived_from — strict gate rejects.
            },
        )
        assert decision.requires_review is True
        assert decision.decision in {"draft_review", "reject"}

    def test_raptor_routes_to_review_with_full_provenance(self):
        """Even with full provenance, RAPTOR summaries go to draft_review —
        they are never auto-stored as canonical facts."""
        decision = evaluate_raptor_summary_write(
            text="Summary of cluster alpha with enough text to matter.",
            metadata=self._GOOD_METADATA,
        )
        assert decision.decision == "draft_review"
        assert "raptor_summary_review_required" in decision.reasons
        assert decision.requires_review is True

    def test_raptor_rejects_secret_text(self):
        secret = "".join(["api", "_key=", "secret", "-raptor-value"])
        decision = evaluate_raptor_summary_write(
            text=f"Summary containing {secret} accidentally",
            metadata=self._GOOD_METADATA,
        )
        assert decision.decision == "reject"
        assert "possible_secret" in decision.reasons

    def test_raptor_skips_empty_text(self):
        decision = evaluate_raptor_summary_write(
            text="   ",
            metadata=self._GOOD_METADATA,
        )
        assert decision.decision == "skip"

    def test_raptor_metadata_flags_raptor(self):
        """All RAPTOR decisions must include raptor=True in metadata."""
        decision = evaluate_raptor_summary_write(
            text="Summary of cluster alpha with enough text to matter.",
            metadata=self._GOOD_METADATA,
        )
        assert decision.metadata.get("raptor") is True


# ===========================================================================
# Phase 4 fix4 regression tests: post-enrichment gate strict trust/provenance
# ===========================================================================


class TestRaptorStrictTrustFlags:
    """fix4: ``canonical`` must be exactly ``False`` and ``requires_review``
    must be exactly ``True``. Every other value must produce a
    ``decision="reject"`` so the post-enrichment gate cannot diverge from
    the pre-enrichment gate in :mod:`qdrant_memory.raptor.apply`.
    """

    _BASE = {
        "raptor_node_id": "raptor-node-strict",
        "raptor_child_ids": ["child-x"],
        "source_hashes": ["a" * 64],
        "derived_from": [
            {
                "source_uri": "raptor://node/raptor-tree-x/child-x",
                "derivation_type": "raptor_summary",
                "relation_type": "SUMMARIZES",
                "child_node_id": "child-x",
            }
        ],
    }

    @pytest.mark.parametrize("bad_value", ["true", "false", "1", "0", 1, 0, None, "yes", "no", 0.0])
    def test_raptor_rejects_non_boolean_canonical(self, bad_value):
        decision = evaluate_raptor_summary_write(
            text="Summary of cluster alpha with enough text to matter.",
            metadata={**self._BASE, "canonical": bad_value, "requires_review": True},
        )
        assert decision.decision == "reject"
        assert "raptor_summary_must_not_be_canonical" in decision.reasons
        assert decision.requires_review is True

    @pytest.mark.parametrize("bad_value", ["false", "0", 0, None, "yes", 0.0, "no"])
    def test_raptor_rejects_non_true_requires_review(self, bad_value):
        decision = evaluate_raptor_summary_write(
            text="Summary of cluster alpha with enough text to matter.",
            metadata={**self._BASE, "canonical": False, "requires_review": bad_value},
        )
        assert decision.decision == "reject"
        assert "raptor_summary_must_not_skip_review" in decision.reasons
        assert decision.requires_review is True

    def test_raptor_rejects_missing_canonical_key(self):
        # No canonical key at all: ``metadata.get("canonical")`` is None,
        # which is not exactly False, so the strict gate rejects.
        decision = evaluate_raptor_summary_write(
            text="Summary of cluster alpha with enough text to matter.",
            metadata={**self._BASE, "requires_review": True},
        )
        assert decision.decision == "reject"
        assert "raptor_summary_must_not_be_canonical" in decision.reasons

    def test_raptor_rejects_missing_requires_review_key(self):
        decision = evaluate_raptor_summary_write(
            text="Summary of cluster alpha with enough text to matter.",
            metadata={**self._BASE, "canonical": False},
        )
        assert decision.decision == "reject"
        assert "raptor_summary_must_not_skip_review" in decision.reasons


class TestRaptorStrictSourceHashes:
    """fix4: ``source_hashes`` must be a non-empty list of 64-char
    lowercase hex SHA-256 strings."""

    _BASE = {
        "raptor_node_id": "raptor-node-hash",
        "raptor_child_ids": ["child-x"],
        "canonical": False,
        "requires_review": True,
        "derived_from": [
            {
                "source_uri": "raptor://node/raptor-tree-x/child-x",
                "derivation_type": "raptor_summary",
                "relation_type": "SUMMARIZES",
                "child_node_id": "child-x",
            }
        ],
    }

    def test_raptor_rejects_source_hashes_with_none_entry(self):
        decision = evaluate_raptor_summary_write(
            text="Summary of cluster alpha with enough text to matter.",
            metadata={**self._BASE, "source_hashes": [None]},
        )
        assert decision.decision == "reject"
        assert "raptor_source_hashes_malformed" in decision.reasons

    def test_raptor_rejects_source_hashes_with_dict_entry(self):
        decision = evaluate_raptor_summary_write(
            text="Summary of cluster alpha with enough text to matter.",
            metadata={**self._BASE, "source_hashes": [{}]},
        )
        assert decision.decision == "reject"
        assert "raptor_source_hashes_malformed" in decision.reasons

    def test_raptor_rejects_source_hashes_with_empty_string(self):
        decision = evaluate_raptor_summary_write(
            text="Summary of cluster alpha with enough text to matter.",
            metadata={**self._BASE, "source_hashes": [""]},
        )
        assert decision.decision == "reject"
        assert "raptor_source_hashes_malformed" in decision.reasons

    def test_raptor_rejects_source_hashes_short(self):
        decision = evaluate_raptor_summary_write(
            text="Summary of cluster alpha with enough text to matter.",
            metadata={**self._BASE, "source_hashes": ["abcdef"]},
        )
        assert decision.decision == "reject"
        assert "raptor_source_hashes_malformed" in decision.reasons

    def test_raptor_rejects_source_hashes_non_hex(self):
        # 64 chars but contains a non-hex character.
        bad = "z" * 64
        decision = evaluate_raptor_summary_write(
            text="Summary of cluster alpha with enough text to matter.",
            metadata={**self._BASE, "source_hashes": [bad]},
        )
        assert decision.decision == "reject"
        assert "raptor_source_hashes_malformed" in decision.reasons

    def test_raptor_rejects_source_hashes_uppercase(self):
        # 64 hex chars but uppercase — strict pattern is lowercase.
        bad = "A" * 64
        decision = evaluate_raptor_summary_write(
            text="Summary of cluster alpha with enough text to matter.",
            metadata={**self._BASE, "source_hashes": [bad]},
        )
        assert decision.decision == "reject"
        assert "raptor_source_hashes_malformed" in decision.reasons

    def test_raptor_rejects_source_hashes_with_prefixed_string(self):
        # Old shape: "sha256:..." prefix is no longer accepted.
        decision = evaluate_raptor_summary_write(
            text="Summary of cluster alpha with enough text to matter.",
            metadata={**self._BASE, "source_hashes": ["sha256:" + "a" * 60]},
        )
        assert decision.decision == "reject"
        assert "raptor_source_hashes_malformed" in decision.reasons


class TestRaptorStrictDerivedFrom:
    """fix4: ``derived_from`` must be a non-empty list of structurally
    valid RAPTOR provenance edges."""

    _BASE = {
        "raptor_node_id": "raptor-node-edge",
        "raptor_child_ids": ["child-x"],
        "source_hashes": ["a" * 64],
        "canonical": False,
        "requires_review": True,
    }

    def test_raptor_rejects_derived_from_with_dict(self):
        decision = evaluate_raptor_summary_write(
            text="Summary of cluster alpha with enough text to matter.",
            metadata={**self._BASE, "derived_from": [{}]},
        )
        assert decision.decision == "reject"
        assert "raptor_derived_from_malformed" in decision.reasons

    def test_raptor_rejects_derived_from_with_none(self):
        decision = evaluate_raptor_summary_write(
            text="Summary of cluster alpha with enough text to matter.",
            metadata={**self._BASE, "derived_from": [None]},
        )
        assert decision.decision == "reject"
        assert "raptor_derived_from_malformed" in decision.reasons

    def test_raptor_rejects_derived_from_with_string(self):
        decision = evaluate_raptor_summary_write(
            text="Summary of cluster alpha with enough text to matter.",
            metadata={**self._BASE, "derived_from": ["not-a-provenance-edge"]},
        )
        assert decision.decision == "reject"
        assert "raptor_derived_from_malformed" in decision.reasons

    def test_raptor_rejects_derived_from_wrong_derivation_type(self):
        decision = evaluate_raptor_summary_write(
            text="Summary of cluster alpha with enough text to matter.",
            metadata={
                **self._BASE,
                "derived_from": [
                    {
                        "source_uri": "raptor://node/raptor-tree-x/child-x",
                        "derivation_type": "summary",  # wrong type
                        "relation_type": "SUMMARIZES",
                        "child_node_id": "child-x",
                    }
                ],
            },
        )
        assert decision.decision == "reject"
        assert "raptor_derived_from_malformed" in decision.reasons

    def test_raptor_rejects_derived_from_wrong_relation_type(self):
        decision = evaluate_raptor_summary_write(
            text="Summary of cluster alpha with enough text to matter.",
            metadata={
                **self._BASE,
                "derived_from": [
                    {
                        "source_uri": "raptor://node/raptor-tree-x/child-x",
                        "derivation_type": "raptor_summary",
                        "relation_type": "REFERENCES",  # wrong relation
                        "child_node_id": "child-x",
                    }
                ],
            },
        )
        assert decision.decision == "reject"
        assert "raptor_derived_from_malformed" in decision.reasons

    def test_raptor_rejects_derived_from_empty_source_uri(self):
        decision = evaluate_raptor_summary_write(
            text="Summary of cluster alpha with enough text to matter.",
            metadata={
                **self._BASE,
                "derived_from": [
                    {
                        "source_uri": "",
                        "derivation_type": "raptor_summary",
                        "relation_type": "SUMMARIZES",
                        "child_node_id": "child-x",
                    }
                ],
            },
        )
        assert decision.decision == "reject"
        assert "raptor_derived_from_malformed" in decision.reasons

    def test_raptor_rejects_derived_from_empty_child_node_id(self):
        decision = evaluate_raptor_summary_write(
            text="Summary of cluster alpha with enough text to matter.",
            metadata={
                **self._BASE,
                "derived_from": [
                    {
                        "source_uri": "raptor://node/raptor-tree-x/child-x",
                        "derivation_type": "raptor_summary",
                        "relation_type": "SUMMARIZES",
                        "child_node_id": "",
                    }
                ],
            },
        )
        assert decision.decision == "reject"
        assert "raptor_derived_from_malformed" in decision.reasons

    def test_raptor_accepts_valid_derived_from_edge(self):
        decision = evaluate_raptor_summary_write(
            text="Summary of cluster alpha with enough text to matter.",
            metadata={
                **self._BASE,
                "derived_from": [
                    {
                        "source_uri": "raptor://node/raptor-tree-x/child-x",
                        "derivation_type": "raptor_summary",
                        "relation_type": "SUMMARIZES",
                        "child_node_id": "child-x",
                    }
                ],
            },
        )
        # Even with full provenance, RAPTOR summaries go to review
        # (never auto-store as canonical). The new strict gate routes
        # this to ``draft_review`` with requires_review=True.
        assert decision.decision == "draft_review"
        assert decision.requires_review is True
        assert "raptor_summary_review_required" in decision.reasons


class TestRecursiveContamination:
    """Recursive contamination markers must be stripped/rejected so memory
    ingestion does not embed prior retrieval output."""

    def test_clean_strips_relevant_long_term_memory_section(self):
        text = """Some important fact about the system.

# Relevant Long-Term Memory

1. [2026-06-20 | score=0.900] This should be stripped
2. [2026-06-19 | score=0.850] This too should be stripped

# Next Section

Some other content that should remain."""

        cleaned = clean_text_for_memory(text)
        assert "# Relevant Long-Term Memory" not in cleaned
        assert "This should be stripped" not in cleaned
        assert "This too should be stripped" not in cleaned
        assert "Some other content" in cleaned

    def test_clean_strips_past_learnings_section(self):
        text = """A learning about the deploy process.

# Past Learnings

- Lesson: always check logs before restarting
- Trigger: deploy fails

# Next

More content after."""

        cleaned = clean_text_for_memory(text)
        assert "# Past Learnings" not in cleaned
        assert "always check logs" not in cleaned
        assert "More content after" in cleaned

    def test_clean_strips_qdrant_memory_fenced_block(self):
        text = """A normal memory fact.

```qdrant-memory
{"point_id": "fake-1", "text": "injected memory should not be stored"}
```

Remaining content stays."""

        cleaned = clean_text_for_memory(text)
        assert "qdrant-memory" not in cleaned.lower()
        assert "fake-1" not in cleaned
        assert "injected memory" not in cleaned
        assert "Remaining content stays" in cleaned

    def test_clean_strips_multiple_contamination_markers(self):
        text = """# Relevant Long-Term Memory

Old memory that should not be re-ingested.

# Past Learnings

Old learning that should not be re-ingested.

```qdrant-memory
{"text": "injected"}
```

# Actual Content

Actual new fact: the API uses v2 endpoints."""

        cleaned = clean_text_for_memory(text)
        assert "# Relevant Long-Term Memory" not in cleaned
        assert "# Past Learnings" not in cleaned
        assert "qdrant-memory" not in cleaned.lower()
        assert "Actual new fact" in cleaned

    def test_clean_strips_contamination_at_start(self):
        """When the entire text is a contamination section, result is empty."""
        text = """# Relevant Long-Term Memory

Everything here is old retrieval output."""

        cleaned = clean_text_for_memory(text)
        assert cleaned == ""

# ===========================================================================
# W0: mechanical lineage write gate
# ===========================================================================

import hashlib as _hashlib


def _mechanical_fixtures(relation="PART_OF", operation="index_capture"):
    from qdrant_memory import lineage

    profile = "gate-profile"
    collection = "memory"
    scope_key = lineage.make_scope_key(collection_name=collection, profile_id=profile)
    source_key = lineage.make_source_key(scope_key=scope_key, resolved_file_path="/repo/Gate File.md")
    file_sha = _hashlib.sha256(b"gate bytes").hexdigest()

    src = lineage.build_source_node_payload(
        source_key=source_key,
        scope_key=scope_key,
        profile_id=profile,
        file_path="/repo/Gate File.md",
        source_uri="file:///repo/Gate%20File.md",
        lineage_operation=operation,
        observation="read_bytes",
    )
    ver = lineage.build_version_node_payload(
        source_key=source_key,
        scope_key=scope_key,
        profile_id=profile,
        file_path="/repo/Gate File.md",
        file_sha256=file_sha,
        source_uri="file:///repo/Gate%20File.md",
        lineage_operation=operation,
        observation="read_bytes",
    )
    version_logical = ver["entity_id"]
    source_storage = lineage.storage_point_id(src["entity_id"])
    version_storage = lineage.storage_point_id(version_logical)
    edge = lineage.build_mechanical_edge_payload(
        relation_type=relation,
        source_entity_id=version_logical,
        target_entity_id=src["entity_id"],
        profile_id=profile,
        source_point_id=version_storage,
        target_point_id=source_storage,
        source_entity_type="source",
        target_entity_type="source",
        lineage_operation=operation,
        lineage_source_key=source_key,
        lineage_scope_key=scope_key,
        observation="read_bytes",
        provenance_content_hash=f"sha256:{file_sha}",
        source_content_hash=f"sha256:{file_sha}",
        file_version_id=version_storage,
        file_path="/repo/Gate File.md",
        file_sha256=file_sha,
    )
    evidence = lineage.LineageEvidence(
        operation=operation,
        collection_name=collection,
        profile_id=profile,
        scope_key=scope_key,
        source_key=source_key,
        source_uri="file:///repo/Gate%20File.md",
        file_path="/repo/Gate File.md",
        file_sha256=file_sha,
        content_hash=f"sha256:{file_sha}",
        observation="read_bytes",
        relation_type=relation,
        source_point_id=version_storage,
        target_point_id=source_storage,
        source_entity_type="source",
        target_entity_type="source",
        source_content_hash=f"sha256:{file_sha}",
        file_version_id=version_storage,
    )
    return lineage, src, ver, edge, evidence


def _derived_from_chunk_fixtures():
    """A chunk -> file-version DERIVED_FROM edge with complete independent
    provenance: the chunk's own content hash, a bounded locator, and the
    file-version URI — everything the direction table demands."""
    from qdrant_memory import lineage

    profile = "gate-profile"
    collection = "memory"
    scope_key = lineage.make_scope_key(collection_name=collection, profile_id=profile)
    source_key = lineage.make_source_key(scope_key=scope_key, resolved_file_path="/repo/Gate File.md")
    file_sha = _hashlib.sha256(b"gate bytes").hexdigest()
    chunk_point_id = "11111111-2222-3333-4444-555555555555"
    chunk_sha = _hashlib.sha256(b"chunk bytes").hexdigest()
    ver = lineage.build_version_node_payload(
        source_key=source_key,
        scope_key=scope_key,
        profile_id=profile,
        file_path="/repo/Gate File.md",
        file_sha256=file_sha,
        source_uri="file:///repo/Gate%20File.md",
    )
    # The chunk endpoint handle is derived from the exact scoped point ID
    # (section 4), never from a free-text label.
    chunk_endpoint = lineage.memory_point_endpoint_logical_id(
        scope_key=scope_key, point_id=chunk_point_id, profile_id=profile
    )
    version_storage = lineage.storage_point_id(ver["entity_id"])
    edge = lineage.build_mechanical_edge_payload(
        relation_type="DERIVED_FROM",
        source_entity_id=chunk_endpoint,
        target_entity_id=ver["entity_id"],
        profile_id=profile,
        source_point_id=chunk_point_id,
        target_point_id=version_storage,
        source_entity_type="memory_point",
        target_entity_type="source",
        lineage_operation="index_capture",
        lineage_source_key=source_key,
        lineage_scope_key=scope_key,
        provenance_content_hash=f"sha256:{file_sha}",
        source_content_hash=f"sha256:{chunk_sha}",
        target_content_hash=f"sha256:{file_sha}",
        file_version_id=version_storage,
        file_path="/repo/Gate File.md",
        file_sha256=file_sha,
        locator={"line_start": 1},
    )
    edge["user_id_hash"] = ""
    edge["chat_id_hash"] = ""
    evidence = lineage.LineageEvidence(
        operation="index_capture",
        collection_name=collection,
        profile_id=profile,
        relation_type="DERIVED_FROM",
        source_point_id=chunk_point_id,
        target_point_id=version_storage,
        source_entity_type="memory_point",
        target_entity_type="source",
        file_path="/repo/Gate File.md",
        file_sha256=file_sha,
        source_content_hash=f"sha256:{chunk_sha}",
        target_content_hash=f"sha256:{file_sha}",
        observation="indexed_payload",
        locator={"line_start": 1},
        file_version_id=version_storage,
        source_uri="file:///repo/Gate%20File.md",
        scope_key=scope_key,
        source_key=source_key,
        content_hash=f"sha256:{file_sha}",
    )
    return lineage, edge, evidence


class TestMechanicalLineageGateMatrix:
    """store only for the approved operation/type combinations."""

    def test_source_and_version_records_store_for_capture(self):
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        lineage, src, ver, _, evidence = _mechanical_fixtures()
        for payload in (src, ver):
            decision = evaluate_mechanical_lineage_write(payload, collection_name="memory", operation="index_capture", evidence=evidence)
            assert decision.decision == "store", decision.reasons
            assert decision.metadata["candidate_type"] == "mechanical_lineage_candidate"
            # Structural persistence is not assertion use.
            assert decision.requires_review is True
            assert payload["canonical"] is False and payload["requires_review"] is True

    def test_derived_from_and_part_of_store_for_capture_and_reconcile(self):
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        lineage, derived, evidence = _derived_from_chunk_fixtures()
        assert lineage.validate_lineage_payload(derived) == [], lineage.validate_lineage_payload(derived)
        for operation in ("index_capture", "index_reconcile"):
            derived["lineage_operation"] = operation
            op_evidence = lineage.LineageEvidence(**{**evidence.to_dict(), "operation": operation})
            decision = evaluate_mechanical_lineage_write(derived, collection_name="memory", operation=operation, evidence=op_evidence)
            assert decision.decision == "store", decision.reasons

        _, _, _, part_of_edge, part_of_evidence = _mechanical_fixtures()
        for operation in ("index_capture", "index_reconcile"):
            part_of_edge["lineage_operation"] = operation
            op_evidence = lineage.LineageEvidence(**{**part_of_evidence.to_dict(), "operation": operation})
            decision = evaluate_mechanical_lineage_write(part_of_edge, collection_name="memory", operation=operation, evidence=op_evidence)
            assert decision.decision == "store", decision.reasons

    def test_supersedes_requires_distinct_version_hashes_predecessor_and_event(self):
        """SUPERSEDES runs only between two distinct content-bearing file
        versions with distinct, evidence-bound event UUID reference tokens."""
        import hashlib as _hashlib

        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        lineage, _, _, _, _ = _mechanical_fixtures()
        profile = "gate-profile"
        collection = "memory"
        scope_key = lineage.make_scope_key(collection_name=collection, profile_id=profile)
        source_key = lineage.make_source_key(scope_key=scope_key, resolved_file_path="/repo/Gate File.md")
        new_sha = _hashlib.sha256(b"new bytes").hexdigest()
        old_sha = _hashlib.sha256(b"old bytes").hexdigest()
        new_ver = lineage.build_version_node_payload(
            source_key=source_key,
            scope_key=scope_key,
            profile_id=profile,
            file_path="/repo/Gate File.md",
            file_sha256=new_sha,
            source_uri="file:///repo/Gate%20File.md",
            lineage_operation="index_reconcile",
            observation="read_bytes",
        )
        old_ver = lineage.build_version_node_payload(
            source_key=source_key,
            scope_key=scope_key,
            profile_id=profile,
            file_path="/repo/Gate File.md",
            file_sha256=old_sha,
            source_uri="file:///repo/Gate%20File.md",
            lineage_operation="index_reconcile",
            observation="indexed_payload",
        )
        event_uuid = "99999999-9999-4999-8999-999999999999"

        def _supersedes_edge():
            return lineage.build_mechanical_edge_payload(
                relation_type="SUPERSEDES",
                source_entity_id=new_ver["entity_id"],
                target_entity_id=old_ver["entity_id"],
                profile_id=profile,
                source_point_id=lineage.storage_point_id(new_ver["entity_id"]),
                target_point_id=lineage.storage_point_id(old_ver["entity_id"]),
                source_entity_type="source",
                target_entity_type="source",
                lineage_operation="index_reconcile",
                lineage_source_key=source_key,
                lineage_scope_key=scope_key,
                observation="indexed_payload",
                source_content_hash=f"sha256:{new_sha}",
                target_content_hash=f"sha256:{old_sha}",
                provenance_content_hash=f"sha256:{new_sha}",
                file_version_id=lineage.storage_point_id(new_ver["entity_id"]),
                file_path="/repo/Gate File.md",
                file_sha256=new_sha,
                lineage_event_id=event_uuid,
            )

        evidence = lineage.LineageEvidence(
            operation="index_reconcile",
            collection_name=collection,
            profile_id=profile,
            scope_key=scope_key,
            source_key=source_key,
            source_uri="file:///repo/Gate%20File.md",
            file_path="/repo/Gate File.md",
            file_sha256=new_sha,
            content_hash=f"sha256:{new_sha}",
            observation="indexed_payload",
            relation_type="SUPERSEDES",
            source_point_id=lineage.storage_point_id(new_ver["entity_id"]),
            target_point_id=lineage.storage_point_id(old_ver["entity_id"]),
            source_entity_type="source",
            target_entity_type="source",
            source_content_hash=f"sha256:{new_sha}",
            target_content_hash=f"sha256:{old_sha}",
            file_version_id=lineage.storage_point_id(new_ver["entity_id"]),
            event_id=event_uuid,
        )

        # Capture never supersedes; here the payload is reconcile-labeled, so
        # the operation binding is the first refusal to fire.
        sup = _supersedes_edge()
        capture_evidence = lineage.LineageEvidence(**{**evidence.to_dict(), "operation": "index_capture"})
        decision = evaluate_mechanical_lineage_write(sup, collection_name=collection, operation="index_capture", evidence=capture_evidence)
        assert decision.decision == "reject"
        assert "lineage_operation" in decision.reasons

        # A capture-labeled SUPERSEDES payload is refused by the matrix.
        sup = _supersedes_edge()
        sup["lineage_operation"] = "index_capture"
        capture_payload_evidence = lineage.LineageEvidence(**{**evidence.to_dict(), "operation": "index_capture"})
        decision = evaluate_mechanical_lineage_write(sup, collection_name=collection, operation="index_capture", evidence=capture_payload_evidence)
        assert decision.decision == "reject"
        assert "lineage_operation_relation_mismatch" in decision.reasons

        # Reconcile without an explicit predecessor fails closed.
        sup = _supersedes_edge()
        no_predecessor = lineage.LineageEvidence(
            **{key: value for key, value in evidence.to_dict().items() if key != "predecessor_event_id"}
        )
        decision = evaluate_mechanical_lineage_write(sup, collection_name=collection, operation="index_reconcile", evidence=no_predecessor)
        assert decision.decision == "reject"
        assert "supersedes_predecessor_required" in decision.reasons

        # Without a current event reference token the transition is unbound.
        # An event id the payload picked while the evidence lacks one is caught
        # even earlier by the lineage_event_id evidence binding.
        sup = _supersedes_edge()
        sup["lineage_event_id"] = ""
        no_event = lineage.LineageEvidence(
            **{key: value for key, value in evidence.to_dict().items() if key != "event_id"}
        )
        decision = evaluate_mechanical_lineage_write(sup, collection_name=collection, operation="index_reconcile", evidence=no_event)
        assert decision.decision == "reject"
        # Caught at payload validation here (no event on the payload either);
        # a payload event with no evidence event is caught by the binding.
        assert decision.reasons == [
            "lineage_payload_invalid",
            "lineage_event_id must be an exact canonical UUID",
            "SUPERSEDES requires a lineage_event_id UUID reference token",
        ], decision.reasons

        # Equal version hashes are not a supersession (A->A is forbidden).
        sup = _supersedes_edge()
        sup["target_content_hash"] = f"sha256:{new_sha}"
        same_hash_evidence = lineage.LineageEvidence(**{**evidence.to_dict(), "target_content_hash": f"sha256:{new_sha}"})
        decision = evaluate_mechanical_lineage_write(sup, collection_name=collection, operation="index_reconcile", evidence=same_hash_evidence)
        assert decision.decision == "reject"
        assert any("distinct version hashes" in reason for reason in decision.reasons)

        # Valid version -> version supersession stores.
        sup = _supersedes_edge()
        with_predecessor = lineage.LineageEvidence(**{**evidence.to_dict(), "predecessor_event_id": "99999999-9999-4999-8999-999999999998"})
        decision = evaluate_mechanical_lineage_write(sup, collection_name=collection, operation="index_reconcile", evidence=with_predecessor)
        assert decision.decision == "store", decision.reasons

        # The payload's event UUID must equal the evidence UUID, not one the
        # payload picked for itself after enrichment.
        sup = _supersedes_edge()
        sup["lineage_event_id"] = "88888888-8888-4888-8888-888888888888"
        decision = evaluate_mechanical_lineage_write(sup, collection_name=collection, operation="index_reconcile", evidence=with_predecessor)
        assert decision.decision == "reject"
        # The general lineage_event_id binding fires first; the gate's named
        # supersedes_event_mismatch check is the belt-and-braces duplicate.
        assert "lineage_event_id" in decision.reasons or "supersedes_event_mismatch" in decision.reasons

        # The current event and its predecessor must be distinct events.
        sup = _supersedes_edge()
        same_event = lineage.LineageEvidence(
            **{**with_predecessor.to_dict(), "predecessor_event_id": with_predecessor.event_id}
        )
        decision = evaluate_mechanical_lineage_write(sup, collection_name=collection, operation="index_reconcile", evidence=same_event)
        assert decision.decision == "reject"
        assert "supersedes_event_not_distinct" in decision.reasons

    def test_approved_citation_relations_require_approval_reference(self):
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        lineage, src, _, _, _ = _mechanical_fixtures(relation="SUMMARIZES", operation="approved_citation")
        profile = "gate-profile"
        scope_key = lineage.make_scope_key(collection_name="memory", profile_id=profile)
        summary_point_id = "99999999-8888-7777-6666-555555555555"
        cited_point_id = "99999999-8888-7777-6666-555555555556"
        summary_handle = lineage.memory_point_endpoint_logical_id(
            scope_key=scope_key, point_id=summary_point_id, profile_id=profile
        )
        cited_handle = lineage.memory_point_endpoint_logical_id(
            scope_key=scope_key, point_id=cited_point_id, profile_id=profile
        )
        edge = lineage.build_mechanical_edge_payload(
            relation_type="SUMMARIZES",
            source_entity_id=summary_handle,
            target_entity_id=cited_handle,
            profile_id=profile,
            source_point_id=summary_point_id,
            target_point_id=cited_point_id,
            source_entity_type="memory_point",
            target_entity_type="memory_point",
            lineage_operation="approved_citation",
            lineage_scope_key=scope_key,
            observation="approved_manifest",
            source_content_hash="sha256:" + "c" * 64,
            target_content_hash="sha256:" + "c" * 64,
            provenance_content_hash="sha256:" + "c" * 64,
        )
        edge["user_id_hash"] = ""
        edge["chat_id_hash"] = ""
        evidence = lineage.LineageEvidence(
            operation="approved_citation",
            collection_name="memory",
            profile_id=profile,
            relation_type="SUMMARIZES",
            source_point_id=summary_point_id,
            target_point_id=cited_point_id,
            source_entity_type="memory_point",
            target_entity_type="memory_point",
            source_content_hash="sha256:" + "c" * 64,
            target_content_hash="sha256:" + "c" * 64,
            observation="approved_manifest",
        )
        decision = evaluate_mechanical_lineage_write(edge, collection_name="memory", operation="approved_citation", evidence=evidence)
        assert decision.decision == "reject"
        assert "approved_citation_required" in decision.reasons

        with_ref = lineage.LineageEvidence(**{**evidence.to_dict(), "approval_ref": "proposal-abc123"})
        decision = evaluate_mechanical_lineage_write(edge, collection_name="memory", operation="approved_citation", evidence=with_ref)
        assert decision.decision == "store", decision.reasons


class TestMechanicalLineageGateFailClosed:
    """Missing/altered provenance, claims, forged roles, mismatched ownership."""

    def test_claim_edges_rejected_regardless_of_edge_class(self):
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        for relation in ("SUPPORTS", "CONTRADICTS"):
            _, _, _, claim, evidence = _mechanical_fixtures(relation=relation)
            assert claim["edge_class"] == "mechanical"
            decision = evaluate_mechanical_lineage_write(claim, collection_name="memory", operation="index_capture", evidence=evidence)
            assert decision.decision == "reject"
            assert any("claim relation" in reason for reason in decision.reasons)

    def test_claim_level_supersedes_rejected(self):
        """SUPERSEDES between non-version endpoints is claim-level and rejected."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        lineage, _, _, _, _ = _mechanical_fixtures()
        claim_sup = lineage.build_mechanical_edge_payload(
            relation_type="SUPERSEDES",
            source_entity_id=lineage.make_entity_id("memory_point", "claim-a", profile_id="gate-profile"),
            target_entity_id=lineage.make_entity_id("memory_point", "claim-b", profile_id="gate-profile"),
            profile_id="gate-profile",
            source_point_id="11111111-2222-3333-4444-555555555555",
            target_point_id="66666666-7777-8888-9999-aaaaaaaaaaaa",
            source_entity_type="memory_point",
            target_entity_type="memory_point",
            lineage_operation="index_reconcile",
            provenance_content_hash="sha256:" + "b" * 64,
        )
        claim_sup["user_id_hash"] = ""
        claim_sup["chat_id_hash"] = ""
        evidence = lineage.LineageEvidence(
            operation="index_reconcile",
            collection_name="memory",
            profile_id="gate-profile",
            relation_type="SUPERSEDES",
            source_point_id="11111111-2222-3333-4444-555555555555",
            target_point_id="66666666-7777-8888-9999-aaaaaaaaaaaa",
            source_entity_type="memory_point",
            target_entity_type="memory_point",
            predecessor_event_id="entity-bbbbbbbbbbbbbbbb",
            content_hash="sha256:" + "b" * 64,
            observation="indexed_payload",
        )
        decision = evaluate_mechanical_lineage_write(claim_sup, collection_name="memory", operation="index_reconcile", evidence=evidence)
        assert decision.decision == "reject"

    def test_missing_or_altered_provenance_fails_closed(self):
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        lineage, _, _, edge, evidence = _mechanical_fixtures()
        # Altered after enrichment: swapped file hash.
        tampered = dict(edge)
        tampered["file_sha256"] = "e" * 64
        decision = evaluate_mechanical_lineage_write(tampered, collection_name="memory", operation="index_capture", evidence=evidence)
        assert decision.decision == "reject"
        assert decision.reasons[0] == "lineage_evidence_mismatch"

        # Sanitized-away locator that the evidence still carries.
        stripped = {key: value for key, value in edge.items() if key != "file_path"}
        decision = evaluate_mechanical_lineage_write(stripped, collection_name="memory", operation="index_capture", evidence=evidence)
        assert decision.decision == "reject"
        assert decision.reasons[0] == "lineage_evidence_mismatch"

        # Missing structural marker entirely (provenance absent); scope keys
        # match so the refusal is attributable to payload validation.
        decision = evaluate_mechanical_lineage_write(
            {
                "memory_kind": "graph_edge",
                "relation_type": "PART_OF",
                "profile_id": "gate-profile",
                "user_id_hash": "",
                "chat_id_hash": "",
            },
            collection_name="memory", operation="index_capture",
            evidence=evidence,
        )
        assert decision.decision == "reject"
        assert decision.reasons[0] == "lineage_payload_invalid"

        # Ownership is checked before everything else and fails closed.
        decision = evaluate_mechanical_lineage_write(
            {"memory_kind": "graph_edge", "relation_type": "PART_OF"},
            collection_name="memory", operation="index_capture",
            evidence=evidence,
        )
        assert decision.decision == "reject"
        assert decision.reasons == ["lineage_ownership_mismatch"]

    def test_ownership_is_exact_tuple_and_missing_means_empty(self):
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        lineage, _, ver, _, evidence = _mechanical_fixtures()
        assert ver["user_id_hash"] == "" and ver["chat_id_hash"] == ""
        decision = evaluate_mechanical_lineage_write(ver, collection_name="memory", operation="index_capture", evidence=evidence)
        assert decision.decision == "store"

        for override in ({"profile_id": "other"}, {"user_id_hash": "u1"}, {"chat_id_hash": "c1"}):
            mismatched = lineage.LineageEvidence(**{**evidence.to_dict(), **override})
            decision = evaluate_mechanical_lineage_write(ver, collection_name="memory", operation="index_capture", evidence=mismatched)
            assert decision.decision == "reject"
            assert "lineage_ownership_mismatch" in decision.reasons

    def test_forged_endpoint_types_rejected(self):
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        lineage, _, _, edge, evidence = _mechanical_fixtures()
        forged = dict(edge)
        forged["source_entity_type"] = "intern"  # not a known entity type
        decision = evaluate_mechanical_lineage_write(forged, collection_name="memory", operation="index_capture", evidence=evidence)
        assert decision.decision == "reject"

        # Endpoints claiming the wrong role for a PART_OF edge — now caught
        # at payload validation, before any evidence binding is even needed.
        wrong_role = dict(edge)
        wrong_role["target_entity_type"] = "memory_point"
        decision = evaluate_mechanical_lineage_write(wrong_role, collection_name="memory", operation="index_capture", evidence=evidence)
        assert decision.decision == "reject"
        assert decision.reasons[0] == "lineage_payload_invalid"
        assert any("PART_OF" in reason for reason in decision.reasons)

    def test_operation_and_evidence_mismatch_rejected(self):
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        lineage, _, _, edge, evidence = _mechanical_fixtures()
        decision = evaluate_mechanical_lineage_write(edge, collection_name="memory", operation="restore", evidence=evidence)
        assert decision.decision == "reject"
        assert decision.reasons == ["operation_mismatch"]

        decision = evaluate_mechanical_lineage_write(edge, collection_name="memory", operation="totally_new", evidence=evidence)
        assert decision.decision == "reject"
        assert decision.reasons == ["unsupported_lineage_operation"]

        bad_evidence = lineage.LineageEvidence(**{**evidence.to_dict(), "collection_name": ""})
        decision = evaluate_mechanical_lineage_write(edge, collection_name="memory", operation="index_capture", evidence=bad_evidence)
        assert decision.decision == "reject"
        assert decision.reasons == ["lineage_evidence_invalid"]

        decision = evaluate_mechanical_lineage_write(edge, collection_name="memory", operation="index_capture", evidence={"operation": "index_capture"})
        assert decision.decision == "reject"
        assert decision.reasons == ["lineage_evidence_invalid"]

    def test_secrets_fail_closed(self):
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        _, _, ver, _, evidence = _mechanical_fixtures()
        seeded = dict(ver)
        seeded["description"] = "credential: " + "api_" + "key=supersecretvalue99"  # scanner-safe runtime construction
        decision = evaluate_mechanical_lineage_write(seeded, collection_name="memory", operation="index_capture", evidence=evidence)
        assert decision.decision == "reject"
        assert decision.reasons == ["possible_secret"]

    def test_observation_mismatch_after_enrichment_rejected(self):
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        lineage, _, _, edge, evidence = _mechanical_fixtures()
        assert evaluate_mechanical_lineage_write(edge, collection_name="memory", operation="index_capture", evidence=evidence).decision == "store"
        changed = lineage.LineageEvidence(**{**evidence.to_dict(), "observation": "legacy_unpinned"})
        decision = evaluate_mechanical_lineage_write(edge, collection_name="memory", operation="index_capture", evidence=changed)
        assert decision.decision == "reject"
        assert decision.reasons[0] == "lineage_evidence_mismatch"

    def test_post_enrichment_trust_flag_change_rejected(self):
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        _, _, ver, _, evidence = _mechanical_fixtures()
        assert evaluate_mechanical_lineage_write(ver, collection_name="memory", operation="index_capture", evidence=evidence).decision == "store"
        demoted = dict(ver)
        demoted["requires_review"] = False
        decision = evaluate_mechanical_lineage_write(demoted, collection_name="memory", operation="index_capture", evidence=evidence)
        assert decision.decision == "reject"
        promoted = dict(ver)
        promoted["canonical"] = True
        decision = evaluate_mechanical_lineage_write(promoted, collection_name="memory", operation="index_capture", evidence=evidence)
        assert decision.decision == "reject"

    def test_lineage_operation_bound_to_final_payload(self):
        """The final payload's lineage_operation must equal the authorized
        operation; a relabeled record is altered provenance."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        lineage, _, _, edge, evidence = _mechanical_fixtures()
        decision = evaluate_mechanical_lineage_write(edge, collection_name="memory", operation="index_capture", evidence=evidence)
        assert decision.decision == "store"
        tampered = dict(edge)
        tampered["lineage_operation"] = "restore"
        decision = evaluate_mechanical_lineage_write(tampered, collection_name="memory", operation="index_capture", evidence=evidence)
        assert decision.decision == "reject"
        assert decision.reasons[0] == "lineage_evidence_mismatch"
        assert "lineage_operation" in decision.reasons

    def test_scope_digest_recomputed_not_trusted(self):
        """The payload scope digest must equal the scope recomputed from the
        evidence's own collection plus ownership tuple; claimed evidence
        digests that contradict their own material fail closed too."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        lineage, _, ver, _, evidence = _mechanical_fixtures()
        decision = evaluate_mechanical_lineage_write(ver, collection_name="memory", operation="index_capture", evidence=evidence)
        assert decision.decision == "store"

        # Payload claims another collection's scope.
        other_scope = lineage.make_scope_key(collection_name="learnings", profile_id="gate-profile")
        tampered = dict(ver)
        tampered["lineage_scope_key"] = other_scope
        decision = evaluate_mechanical_lineage_write(tampered, collection_name="memory", operation="index_capture", evidence=evidence)
        assert decision.decision == "reject"
        assert decision.reasons[0] == "lineage_evidence_mismatch"
        assert "lineage_scope_key" in decision.reasons

        # Evidence whose claimed scope digest contradicts its own tuple.
        bad_evidence = lineage.LineageEvidence(**{**evidence.to_dict(), "scope_key": other_scope})
        decision = evaluate_mechanical_lineage_write(ver, collection_name="memory", operation="index_capture", evidence=bad_evidence)
        assert decision.decision == "reject"
        assert "evidence_scope_key" in decision.reasons

        # A version node cannot drop its source URI and keep its hash.
        stripped = {key: value for key, value in ver.items() if key != "source_uri"}
        decision = evaluate_mechanical_lineage_write(stripped, collection_name="memory", operation="index_capture", evidence=evidence)
        assert decision.decision == "reject"
        assert "source_uri" in decision.reasons

    def test_same_prefix_different_suffix_identity_collision_rejected(self):
        """A full identity sharing the handle's 16-hex truncation but differing
        in the suffix is a collision and must never store."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        lineage, src, _, _, evidence = _mechanical_fixtures()
        full_digest = lineage.entity_identity_digest("source", lineage.make_source_key(scope_key=lineage.make_scope_key(collection_name="memory", profile_id="gate-profile"), resolved_file_path="/repo/Gate File.md"), profile_id="gate-profile")
        # Forge a digest with the SAME 16-hex prefix and a different suffix.
        flipped = "0" if full_digest[16] != "0" else "1"
        forged_digest = full_digest[:16] + flipped + full_digest[17:]
        assert lineage.logical_id_matches_digest(src["entity_id"], forged_digest)
        tampered = dict(src)
        tampered["lineage_identity_digest"] = forged_digest
        decision = evaluate_mechanical_lineage_write(tampered, collection_name="memory", operation="index_capture", evidence=evidence)
        assert decision.decision == "reject"
        assert decision.reasons[0] == "lineage_identity_mismatch"
        assert "lineage_identity_digest" in decision.reasons

    def test_approved_citation_operation_requires_approval_for_every_relation(self):
        """DERIVED_FROM under approved_citation is still a citation
        materialization: without a recorded approval reference it fails.
        Approved derived point -> cited source point runs between two
        ordinary memory points, never into structural file nodes."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        lineage, _, ver, _, _ = _mechanical_fixtures()
        profile = "gate-profile"
        scope_key = lineage.make_scope_key(collection_name="memory", profile_id=profile)
        derived_point_id = "12121212-3434-4567-8901-234567890123"
        cited_point_id = "12121212-3434-4567-8901-234567890124"
        derived_handle = lineage.memory_point_endpoint_logical_id(
            scope_key=scope_key, point_id=derived_point_id, profile_id=profile
        )
        cited_handle = lineage.memory_point_endpoint_logical_id(
            scope_key=scope_key, point_id=cited_point_id, profile_id=profile
        )
        edge = lineage.build_mechanical_edge_payload(
            relation_type="DERIVED_FROM",
            source_entity_id=derived_handle,
            target_entity_id=cited_handle,
            profile_id=profile,
            source_point_id=derived_point_id,
            target_point_id=cited_point_id,
            source_entity_type="memory_point",
            target_entity_type="memory_point",
            lineage_operation="approved_citation",
            lineage_scope_key=scope_key,
            observation="approved_manifest",
            source_content_hash=f"sha256:{'a' * 64}",
            target_content_hash=f"sha256:{'a' * 64}",
            provenance_content_hash="sha256:" + "a" * 64,
        )
        edge["user_id_hash"] = ""
        edge["chat_id_hash"] = ""
        evidence = lineage.LineageEvidence(
            operation="approved_citation",
            collection_name="memory",
            profile_id=profile,
            relation_type="DERIVED_FROM",
            source_point_id=derived_point_id,
            target_point_id=cited_point_id,
            source_entity_type="memory_point",
            target_entity_type="memory_point",
            source_content_hash=f"sha256:{'a' * 64}",
            target_content_hash=f"sha256:{'a' * 64}",
            observation="approved_manifest",
        )
        decision = evaluate_mechanical_lineage_write(edge, collection_name="memory", operation="approved_citation", evidence=evidence)
        assert decision.decision == "reject"
        assert "approved_citation_required" in decision.reasons

        with_ref = lineage.LineageEvidence(**{**evidence.to_dict(), "approval_ref": "proposal-abc123"})
        decision = evaluate_mechanical_lineage_write(edge, collection_name="memory", operation="approved_citation", evidence=with_ref)
        assert decision.decision == "store", decision.reasons

    def test_citation_relation_into_structural_file_node_rejected(self):
        """SUMMARIZES/EXTRACTED_FROM run between ordinary memory points; a
        citation edge aimed at a structural file node is a forged direction."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        lineage, src, _, _, _ = _mechanical_fixtures(relation="SUMMARIZES", operation="approved_citation")
        edge = lineage.build_mechanical_edge_payload(
            relation_type="SUMMARIZES",
            source_entity_id=lineage.make_entity_id("memory_point", "summary", profile_id="gate-profile"),
            target_entity_id=src["entity_id"],
            profile_id="gate-profile",
            source_point_id="99999999-8888-7777-6666-555555555555",
            target_point_id=lineage.storage_point_id(src["entity_id"]),
            source_entity_type="memory_point",
            target_entity_type="source",
            lineage_operation="approved_citation",
            lineage_scope_key=lineage.make_scope_key(collection_name="memory", profile_id="gate-profile"),
            observation="approved_manifest",
            source_content_hash="sha256:" + "c" * 64,
            target_content_hash="sha256:" + "c" * 64,
            provenance_content_hash="sha256:" + "c" * 64,
        )
        edge["user_id_hash"] = ""
        edge["chat_id_hash"] = ""
        evidence = lineage.LineageEvidence(
            operation="approved_citation",
            collection_name="memory",
            profile_id="gate-profile",
            relation_type="SUMMARIZES",
            source_point_id="99999999-8888-7777-6666-555555555555",
            target_point_id=lineage.storage_point_id(src["entity_id"]),
            source_entity_type="memory_point",
            target_entity_type="source",
            source_content_hash="sha256:" + "c" * 64,
            target_content_hash="sha256:" + "c" * 64,
            observation="approved_manifest",
            approval_ref="proposal-abc123",
        )
        decision = evaluate_mechanical_lineage_write(edge, collection_name="memory", operation="approved_citation", evidence=evidence)
        assert decision.decision == "reject"
        assert any("ordinary memory points" in reason for reason in decision.reasons)

    def test_gate_binds_the_write_target_collection(self):
        """One evidence object can never authorize a write into a different
        collection than the one named at the gate."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        _, _, ver, _, evidence = _mechanical_fixtures()
        decision = evaluate_mechanical_lineage_write(ver, collection_name="learnings", operation="index_capture", evidence=evidence)
        assert decision.decision == "reject"
        assert decision.reasons == ["lineage_collection_mismatch"]

    def test_validation_errors_propagate_not_swallowed(self, monkeypatch):
        """Unlike the semantic path's permissive ``except ImportError: pass``,
        errors escaping lineage validation must surface, never be swallowed."""
        import pytest

        from qdrant_memory import lineage
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        _, _, _, edge, evidence = _mechanical_fixtures()

        def _boom(_payload):
            raise RuntimeError("simulated lineage validation crash")

        monkeypatch.setattr(lineage, "validate_lineage_payload", _boom)
        with pytest.raises(RuntimeError):
            evaluate_mechanical_lineage_write(edge, collection_name="memory", operation="index_capture", evidence=evidence)

    def test_malformed_payload_types_rejected_not_crashed(self):
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        _, _, _, _, evidence = _mechanical_fixtures()
        for bad_payload in (None, "entity-0123456789abcdef", ["lineage"]):
            decision = evaluate_mechanical_lineage_write(bad_payload, collection_name="memory", operation="index_capture", evidence=evidence)
            assert decision.decision == "reject"


class TestMechanicalLineageEndpointAndTokenBinding:
    """Escalation fixes: endpoint derivation, independent evidence, exact tokens."""

    def test_edge_endpoint_handles_and_storage_ids_derived_from_evidence(self):
        """Edge endpoint handles, storage UUIDs, and the edge digest are
        derived from the evidence's own material, never trusted from the
        payload's declarations."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        lineage, src, ver, edge, evidence = _mechanical_fixtures()
        assert evaluate_mechanical_lineage_write(edge, collection_name="memory", operation="index_capture", evidence=evidence).decision == "store"

        # A different well-formed handle for the version endpoint is an
        # unbound endpoint, whatever its shape.
        other_version = lineage.build_version_node_payload(
            source_key=edge["lineage_source_key"],
            scope_key=edge["lineage_scope_key"],
            profile_id="gate-profile",
            file_path="/repo/Gate File.md",
            file_sha256=_hashlib.sha256(b"other bytes").hexdigest(),
        )
        tampered = dict(edge)
        tampered["source_entity_id"] = other_version["entity_id"]
        decision = evaluate_mechanical_lineage_write(tampered, collection_name="memory", operation="index_capture", evidence=evidence)
        assert decision.decision == "reject"
        assert decision.reasons[0] == "lineage_identity_mismatch"
        assert "source_entity_id" in decision.reasons

        # A valid UUID that is not the derived storage ID of the declared
        # endpoint is an unbound storage binding.
        tampered = dict(edge)
        tampered["source_point_id"] = lineage.storage_point_id(other_version["entity_id"])
        decision = evaluate_mechanical_lineage_write(tampered, collection_name="memory", operation="index_capture", evidence=evidence)
        assert decision.decision == "reject"
        assert decision.reasons[0] == "lineage_evidence_mismatch"
        assert "source_point_id" in decision.reasons

        # An internally consistent but forged edge — handle, digest, and edge
        # id all agree with each other and every evidence-supplied binding
        # (point IDs, content hashes, file identity) is kept, yet the declared
        # source handle is not the handle the evidence derives: identity
        # derivation refuses it.
        forged = lineage.build_mechanical_edge_payload(
            relation_type="PART_OF",
            source_entity_id=other_version["entity_id"],
            target_entity_id=edge["target_entity_id"],
            profile_id="gate-profile",
            source_point_id=edge["source_point_id"],
            target_point_id=edge["target_point_id"],
            source_entity_type="source",
            target_entity_type="source",
            lineage_operation="index_capture",
            lineage_source_key=edge["lineage_source_key"],
            lineage_scope_key=edge["lineage_scope_key"],
            observation="read_bytes",
            provenance_content_hash=edge["content_hash"],
            source_content_hash=evidence.source_content_hash,
            file_version_id=edge["file_version_id"],
            file_path="/repo/Gate File.md",
            file_sha256=evidence.file_sha256,
        )
        forged["user_id_hash"] = ""
        forged["chat_id_hash"] = ""
        decision = evaluate_mechanical_lineage_write(forged, collection_name="memory", operation="index_capture", evidence=evidence)
        assert decision.decision == "reject"
        assert decision.reasons[0] == "lineage_identity_mismatch"
        assert "source_entity_id" in decision.reasons
        assert "lineage_identity_digest" in decision.reasons
        assert "edge_id" in decision.reasons

    def test_part_of_requires_file_node_endpoints(self):
        """PART_OF only ever runs file version -> file source; a concept
        endpoint is a forged direction and fails payload validation."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        lineage, src, ver, edge, evidence = _mechanical_fixtures()
        forged = dict(edge)
        forged["source_entity_type"] = "concept"
        forged_evidence = lineage.LineageEvidence(**{**evidence.to_dict(), "source_entity_type": "concept"})
        decision = evaluate_mechanical_lineage_write(forged, collection_name="memory", operation="index_capture", evidence=forged_evidence)
        assert decision.decision == "reject"
        assert decision.reasons[0] == "lineage_payload_invalid"
        assert any("PART_OF" in reason for reason in decision.reasons)

    def test_part_of_cannot_target_another_version(self):
        """PART_OF is version -> file source. The old generic
        entity_type/content-hash inference let evidence carrying a target
        content hash turn the target endpoint into a second version node;
        relation-specific role derivation must refuse that here, at identity
        time, even when every evidence-supplied binding is internally
        consistent."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        lineage, src, ver, edge, evidence = _mechanical_fixtures()
        other_sha = _hashlib.sha256(b"other bytes").hexdigest()
        other = lineage.build_version_node_payload(
            source_key=edge["lineage_source_key"],
            scope_key=edge["lineage_scope_key"],
            profile_id="gate-profile",
            file_path="/repo/Gate File.md",
            file_sha256=other_sha,
            source_uri="file:///repo/Gate%20File.md",
        )
        other_storage = lineage.storage_point_id(other["entity_id"])
        # Internally consistent version -> version PART_OF: handle, digest,
        # and edge id all agree, and every evidence binding matches the
        # payload exactly. Only endpoint-role derivation can refuse it.
        forged = lineage.build_mechanical_edge_payload(
            relation_type="PART_OF",
            source_entity_id=ver["entity_id"],
            target_entity_id=other["entity_id"],
            profile_id="gate-profile",
            source_point_id=edge["source_point_id"],
            target_point_id=other_storage,
            source_entity_type="source",
            target_entity_type="source",
            lineage_operation="index_capture",
            lineage_source_key=edge["lineage_source_key"],
            lineage_scope_key=edge["lineage_scope_key"],
            observation="read_bytes",
            provenance_content_hash=edge["content_hash"],
            source_content_hash=evidence.source_content_hash,
            target_content_hash=f"sha256:{other_sha}",
            file_version_id=edge["file_version_id"],
            file_path="/repo/Gate File.md",
            file_sha256=evidence.file_sha256,
        )
        forged["user_id_hash"] = ""
        forged["chat_id_hash"] = ""
        forged_evidence = lineage.LineageEvidence(
            **{
                **evidence.to_dict(),
                "target_point_id": other_storage,
                "target_content_hash": f"sha256:{other_sha}",
            }
        )
        decision = evaluate_mechanical_lineage_write(forged, collection_name="memory", operation="index_capture", evidence=forged_evidence)
        assert decision.decision == "reject"
        # R3 role validity: PART_OF's target endpoint is the file-source node,
        # so the forged target snapshot hash now refuses at payload validation
        # (earlier and equally fail-closed), before identity derivation runs.
        assert decision.reasons[0] == "lineage_payload_invalid", decision.reasons
        assert any("target_content_hash" in reason for reason in decision.reasons), decision.reasons

    def test_part_of_file_version_id_never_accepts_source_node_uuid(self):
        """``file_version_id`` designates a file-version endpoint exactly; the
        file-source node's storage UUID is a different record and must never
        satisfy a version binding."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        lineage, src, ver, edge, evidence = _mechanical_fixtures()
        source_storage = lineage.storage_point_id(src["entity_id"])
        tampered = dict(edge)
        tampered["file_version_id"] = source_storage
        decision = evaluate_mechanical_lineage_write(tampered, collection_name="memory", operation="index_capture", evidence=evidence)
        assert decision.decision == "reject"
        # The evidence twin is now recorded (independent-binding class), so
        # the altered binding refuses at the mismatch layer, before identity
        # derivation; the refusal is equally fail-closed and field-specific.
        assert decision.reasons[0] == "lineage_evidence_mismatch", decision.reasons
        assert "file_version_id" in decision.reasons

    def test_derived_from_chunk_requires_independent_chunk_provenance(self):
        """chunk -> version DERIVED_FROM needs the chunk's own content hash,
        a bounded locator, and the file-version URI. Simultaneous absence
        from BOTH payload and evidence still fails closed — the payload
        cannot certify its own provenance by omission."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        lineage, edge, evidence = _derived_from_chunk_fixtures()
        assert evaluate_mechanical_lineage_write(edge, collection_name="memory", operation="index_capture", evidence=evidence).decision == "store"

        def _without(payload: dict, evidence_fields: set[str], payload_fields: set[str]) -> tuple[dict, lineage.LineageEvidence]:
            stripped_payload = {key: value for key, value in payload.items() if key not in payload_fields}
            stripped_evidence = lineage.LineageEvidence(
                **{key: value for key, value in evidence.to_dict().items() if key not in evidence_fields}
            )
            return stripped_payload, stripped_evidence

        # Chunk hash absent from both sides.
        payload, ev = _without(edge, {"source_content_hash"}, {"source_content_hash"})
        decision = evaluate_mechanical_lineage_write(payload, collection_name="memory", operation="index_capture", evidence=ev)
        assert decision.decision == "reject"
        assert decision.reasons[0] == "lineage_evidence_missing"
        assert "source_content_hash" in decision.reasons

        # Locator absent from both sides.
        payload, ev = _without(edge, {"locator"}, {"locator"})
        decision = evaluate_mechanical_lineage_write(payload, collection_name="memory", operation="index_capture", evidence=ev)
        assert decision.decision == "reject"
        assert decision.reasons[0] == "lineage_evidence_missing"
        assert "locator" in decision.reasons

        # File-version URI absent from both sides.
        payload, ev = _without(edge, {"source_uri"}, {"source_uri"})
        decision = evaluate_mechanical_lineage_write(payload, collection_name="memory", operation="index_capture", evidence=ev)
        assert decision.decision == "reject"
        assert decision.reasons[0] == "lineage_evidence_missing"
        assert "source_uri" in decision.reasons

    def test_file_chunk_locator_must_match_real_indexer_shape(self):
        """The locator is provenance only in the exact indexer shape: a
        non-empty mapping with positive non-bool line_start, optional ordered
        line_end, optional bounded string heading, no unknown keys, bounded
        serialized size. Presence alone proves nothing."""
        from qdrant_memory import lineage

        # Valid shapes, including the boundaries.
        assert lineage.file_chunk_locator_problems({"line_start": 1}) == []
        assert lineage.file_chunk_locator_problems({"line_start": 1, "line_end": 1}) == []
        assert lineage.file_chunk_locator_problems({"line_start": 3, "line_end": 9, "heading": "H" * 200}) == []

        # Missing / non-mapping / empty.
        assert lineage.file_chunk_locator_problems(None)
        assert lineage.file_chunk_locator_problems("1-2")
        assert lineage.file_chunk_locator_problems({})
        # Malformed line numbers: zero, negative, bool, string, reversed.
        assert lineage.file_chunk_locator_problems({"line_start": 0})
        assert lineage.file_chunk_locator_problems({"line_start": -3})
        assert lineage.file_chunk_locator_problems({"line_start": True})
        assert lineage.file_chunk_locator_problems({"line_start": "1"})
        assert lineage.file_chunk_locator_problems({"line_start": 1, "line_end": True})
        assert lineage.file_chunk_locator_problems({"line_start": 5, "line_end": 2})
        # Unknown keys are not indexer locator fields.
        assert lineage.file_chunk_locator_problems({"line_start": 1, "byte_offset": 512})
        # Oversized heading / oversized serialized locator.
        assert lineage.file_chunk_locator_problems({"line_start": 1, "heading": "H" * 201})
        assert lineage.file_chunk_locator_problems({"line_start": 1, "heading": "H" * 600})
        # Non-string heading.
        assert lineage.file_chunk_locator_problems({"line_start": 1, "heading": 7})

    def test_derived_from_chunk_rejects_malformed_evidence_locator(self):
        """A malformed non-empty locator is not provenance: the independent
        evidence is validated in the real indexer shape BEFORE equality with
        the payload, so garbage metadata never reaches store even when
        payload and evidence agree perfectly."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        lineage, edge, evidence = _derived_from_chunk_fixtures()

        def _gate(locator):
            payload = dict(edge)
            payload["locator"] = locator
            ev = lineage.LineageEvidence(**{**evidence.to_dict(), "locator": locator})
            return evaluate_mechanical_lineage_write(payload, collection_name="memory", operation="index_capture", evidence=ev)

        for malformed in (
            {"line_start": 0},
            {"line_start": 5, "line_end": 2},
            {"line_start": 1, "byte_offset": 512},
            {"line_start": 1, "heading": "H" * 600},
        ):
            decision = _gate(malformed)
            assert decision.decision == "reject", malformed
            # R3: a PRESENT locator on the final payload is shape-validated
            # against the shared domain at payload validation itself, so a
            # malformed locator carried by both sides refuses at the payload
            # layer (earlier, equally fail-closed) instead of the evidence
            # requirement layer.
            assert decision.reasons[0] == "lineage_payload_invalid", (malformed, decision.reasons)
            assert any("locator" in reason for reason in decision.reasons), malformed

        # Evidence-only malformation (payload keeps a valid locator) is still
        # caught by the evidence requirement layer BEFORE equality: garbage
        # metadata never becomes provenance even when the payload is clean.
        for malformed in ({"line_start": 0}, {"line_start": 1, "byte_offset": 512}):
            ev = lineage.LineageEvidence(**{**evidence.to_dict(), "locator": malformed})
            decision = evaluate_mechanical_lineage_write(edge, collection_name="memory", operation="index_capture", evidence=ev)
            assert decision.decision == "reject", malformed
            assert decision.reasons[0] == "lineage_evidence_missing", (malformed, decision.reasons)
            assert "locator" in decision.reasons, malformed

        # The bounded boundary still stores.
        decision = _gate({"line_start": 1, "line_end": 1, "heading": "H" * 200})
        assert decision.decision == "store", decision.reasons

    def test_file_uri_evidence_is_exact_derivation_of_evidenced_path(self):
        """An evidenced file URI must be the exact stdlib file URI of the
        evidenced resolved path (case/spelling preserved, spaces encoded):
        whitespace-bearing tokens, other schemes, and URIs for a different
        path are different tokens and fail closed — for entity evidence and
        edge evidence alike."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        lineage, src, ver, _, evidence = _mechanical_fixtures()
        assert lineage.expected_file_uri("/repo/Gate File.md") == "file:///repo/Gate%20File.md"
        assert lineage.expected_file_uri("relative.md") is None

        def _verdict(ev):
            return evaluate_mechanical_lineage_write(ver, collection_name="memory", operation="index_capture", evidence=ev)

        # Whitespace-bearing URI token.
        padded = lineage.LineageEvidence(**{**evidence.to_dict(), "source_uri": "file:///repo/Gate%20File.md "})
        decision = _verdict(padded)
        assert decision.decision == "reject"
        assert decision.reasons[0] == "lineage_evidence_invalid"
        assert "evidence_source_uri" in decision.reasons

        # URI for a different path (also covers non-file schemes).
        wrong_path = lineage.LineageEvidence(**{**evidence.to_dict(), "source_uri": "file:///repo/Other.md"})
        decision = _verdict(wrong_path)
        assert decision.decision == "reject"
        assert "evidence_source_uri" in decision.reasons

        # file_version evidence missing its URI entirely.
        no_uri = lineage.LineageEvidence(**{key: value for key, value in evidence.to_dict().items() if key != "source_uri"})
        decision = _verdict(no_uri)
        assert decision.decision == "reject"
        assert decision.reasons[0] == "lineage_evidence_missing"
        assert "source_uri" in decision.reasons

        # Edge evidence URIs are bound to the path exactly the same way.
        _, edge, chunk_evidence = _derived_from_chunk_fixtures()
        mismatched_uri = lineage.LineageEvidence(**{**chunk_evidence.to_dict(), "source_uri": "file:///repo/other%20file.md"})
        decision = evaluate_mechanical_lineage_write(edge, collection_name="memory", operation="index_capture", evidence=mismatched_uri)
        assert decision.decision == "reject"
        assert "evidence_source_uri" in decision.reasons

        # The encoded-space fixture URI itself is the exact derivation and stores.
        decision = evaluate_mechanical_lineage_write(src, collection_name="memory", operation="index_capture", evidence=evidence)
        assert decision.decision == "store", decision.reasons

    def test_derived_from_chunk_rejects_payload_locator_type_drift(self):
        """Payload-side locator drift alone must reject: Python dict equality
        says {'line_start': True} == {'line_start': 1} and 1.0 == 1, so the
        final payload locator is schema-validated in its own right before
        equality can authorize it. Evidence stays valid in every case."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        lineage, edge, evidence = _derived_from_chunk_fixtures()
        assert evidence.locator == {"line_start": 1}

        for drifted in (
            {"line_start": True},   # == 1 under dict equality
            {"line_start": 1.0},    # == 1 under dict equality
            {"line_start": 1, "line_end": True},
        ):
            payload = dict(edge)
            payload["locator"] = drifted
            decision = evaluate_mechanical_lineage_write(payload, collection_name="memory", operation="index_capture", evidence=evidence)
            assert decision.decision == "reject", drifted
            # R3: the drifted locator is refused at payload shape validation
            # (shared domain) with a locator-specific problem.
            assert any("locator" in reason for reason in decision.reasons), drifted

        # The untouched valid fixture still stores.
        decision = evaluate_mechanical_lineage_write(edge, collection_name="memory", operation="index_capture", evidence=evidence)
        assert decision.decision == "store", decision.reasons

    def test_missing_independent_evidence_fails_closed(self):
        """A payload cannot certify its own provenance: the fields its
        identity is derived from must exist on the evidence record first."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        lineage, _, ver, edge, evidence = _mechanical_fixtures()
        # A version node whose hash the evidence never observed.
        no_hash = lineage.LineageEvidence(**{key: value for key, value in evidence.to_dict().items() if key != "file_sha256"})
        decision = evaluate_mechanical_lineage_write(ver, collection_name="memory", operation="index_capture", evidence=no_hash)
        assert decision.decision == "reject"
        assert decision.reasons[0] == "lineage_evidence_missing"
        assert "file_sha256" in decision.reasons

        # No recorded observation means nothing to bind the payload's
        # observation claim against.
        no_observation = lineage.LineageEvidence(**{key: value for key, value in evidence.to_dict().items() if key != "observation"})
        decision = evaluate_mechanical_lineage_write(ver, collection_name="memory", operation="index_capture", evidence=no_observation)
        assert decision.decision == "reject"
        assert decision.reasons[0] == "lineage_evidence_missing"
        assert "observation" in decision.reasons

        # A PART_OF edge whose evidence lacks the version endpoint's content
        # hash cannot derive its own source endpoint.
        no_hash = lineage.LineageEvidence(
            **{
                **evidence.to_dict(),
                "file_sha256": "",
                "content_hash": "",
                "source_content_hash": "",
            }
        )
        decision = evaluate_mechanical_lineage_write(edge, collection_name="memory", operation="index_capture", evidence=no_hash)
        assert decision.decision == "reject"
        assert decision.reasons[0] == "lineage_evidence_missing"

    def test_change_event_structural_records_fail_closed(self):
        """W0 has no typed change-event identity, so a change_event structural
        record is refused outright — including the exact former bypass shape:
        a fully relabeled event node (entity_type='event' AND
        lineage_role='change_event') that still carries every consistent
        source-node field, so only the explicit change_event refusal (and no
        generic entity/label/URI check) rejects it."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        lineage, src, _, _, evidence = _mechanical_fixtures()
        # Former bypass shape: ONLY the two role fields change. source_uri,
        # file_path, label, text, and identity all stay the valid source-node
        # values, so no other validation rule fires.
        relabeled = dict(src)
        relabeled["entity_type"] = "event"
        relabeled["lineage_role"] = "change_event"
        # Proof that the explicit refusal is the ONLY problem: without that
        # one rule this otherwise-consistent record validated cleanly — the
        # former gate accepted exactly this shape.
        problems = lineage.validate_lineage_payload(relabeled)
        assert len(problems) == 1
        assert "change_event" in problems[0]
        decision = evaluate_mechanical_lineage_write(relabeled, collection_name="memory", operation="index_capture", evidence=evidence)
        assert decision.decision == "reject"
        assert decision.reasons[0] == "lineage_payload_invalid"
        assert any("change_event" in reason for reason in decision.reasons)

        # The source-typed relabel is rejected too (regression guard).
        relabeled = dict(src)
        relabeled["lineage_role"] = "change_event"
        decision = evaluate_mechanical_lineage_write(relabeled, collection_name="memory", operation="index_capture", evidence=evidence)
        assert decision.decision == "reject"
        assert any("change_event" in reason for reason in decision.reasons)

    def test_collection_and_operation_tokens_are_exact_not_normalized(self):
        """Original tokens are validated before normalization: whitespace-
        bearing or padded tokens are different tokens and fail closed."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        _, _, ver, _, evidence = _mechanical_fixtures()
        # Whitespace-bearing write target.
        decision = evaluate_mechanical_lineage_write(ver, collection_name=" memory", operation="index_capture", evidence=evidence)
        assert decision.decision == "reject"
        assert decision.reasons == ["lineage_collection_invalid"]

        # Whitespace-bearing evidence collection name.
        padded = lineage_evidence = evidence.__class__(**{**evidence.to_dict(), "collection_name": "memory "})
        decision = evaluate_mechanical_lineage_write(ver, collection_name="memory", operation="index_capture", evidence=padded)
        assert decision.decision == "reject"
        assert decision.reasons == ["lineage_collection_invalid"]

        # A padded operation token is unsupported, not normalized to the enum.
        decision = evaluate_mechanical_lineage_write(ver, collection_name="memory", operation=" index_capture", evidence=evidence)
        assert decision.decision == "reject"
        assert decision.reasons == ["unsupported_lineage_operation"]

    def test_payload_event_id_requires_evidence_event(self):
        """An event id on a non-SUPERSEDES record must be the event the
        evidence recorded; a payload-chosen event id is unbound."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        lineage, _, _, edge, evidence = _mechanical_fixtures()
        tampered = dict(edge)
        tampered["lineage_event_id"] = "77777777-7777-4777-8777-777777777777"
        decision = evaluate_mechanical_lineage_write(tampered, collection_name="memory", operation="index_capture", evidence=evidence)
        assert decision.decision == "reject"
        # R3 independent-binding class: the evidence records NO event, so the
        # payload-chosen event id now fails closed at the missing-evidence
        # layer (before comparison) instead of the mismatch layer.
        assert decision.reasons[0] == "lineage_evidence_missing", decision.reasons
        assert any("lineage_event_id" in reason for reason in decision.reasons), decision.reasons

        with_event = lineage.LineageEvidence(
            **{**evidence.to_dict(), "event_id": "77777777-7777-4777-8777-777777777777"}
        )
        decision = evaluate_mechanical_lineage_write(tampered, collection_name="memory", operation="index_capture", evidence=with_event)
        assert decision.decision == "store", decision.reasons


class TestMechanicalCandidatePathIsolation:
    """The internal candidate class is not submittable through extraction."""

    def test_mechanical_candidate_type_rejected_by_extraction_gate(self):
        from qdrant_memory.extraction_candidates import ExtractionCandidate
        from qdrant_memory.write_gate import evaluate_extraction_candidate_write

        _, _, _, edge, _ = _mechanical_fixtures()
        candidate = ExtractionCandidate(
            candidate_id="mech-1",
            candidate_type="mechanical_lineage_candidate",
            source_uri="file:///repo/Gate%20File.md",
            derived_from=[{"source_uri": "file:///repo/Gate%20File.md", "relation_type": "PART_OF"}],
            proposed_payload=edge,
        )
        decision = evaluate_extraction_candidate_write(candidate)
        assert decision.decision == "reject"
        # The refusal names the detected structural marker (the payload is a
        # mechanical-class edge), then the shared refusal reason.
        assert decision.reasons[0] == "mechanical_lineage_candidate_forbidden"
        assert "edge_class=mechanical" in decision.reasons

    def test_structural_payload_rejected_through_extraction_gate(self):
        from qdrant_memory.extraction_candidates import ExtractionCandidate
        from qdrant_memory.write_gate import evaluate_extraction_candidate_write

        _, _, ver, _, _ = _mechanical_fixtures()
        candidate = ExtractionCandidate(
            candidate_id="mech-2",
            candidate_type="graph_entity_candidate",
            source_uri="file:///repo/Gate%20File.md",
            derived_from=[{"source_uri": "file:///repo/Gate%20File.md", "relation_type": "EXTRACTED_FROM"}],
            proposed_payload=ver,
        )
        decision = evaluate_extraction_candidate_write(candidate)
        assert decision.decision == "reject"
        assert decision.reasons[0] == "mechanical_lineage_candidate_forbidden"
        assert "lineage_record" in decision.reasons

    def test_extraction_gate_rejects_bare_lineage_pending(self):
        """A semantic candidate carrying only lineage_pending=True would create
        a point permanently excluded from every retrieval pool: reject it."""
        from qdrant_memory.extraction_candidates import ExtractionCandidate
        from qdrant_memory.write_gate import evaluate_extraction_candidate_write

        candidate = ExtractionCandidate(
            candidate_id="mech-3",
            candidate_type="fact_candidate",
            source_uri="file:///repo/Gate%20File.md",
            derived_from=[{"source_uri": "file:///repo/Gate%20File.md", "relation_type": "EXTRACTED_FROM"}],
            proposed_payload={"text": "harmless fact", "lineage_pending": True},
        )
        decision = evaluate_extraction_candidate_write(candidate)
        assert decision.decision == "reject"
        assert decision.reasons[0] == "mechanical_lineage_candidate_forbidden"
        assert "lineage_pending" in decision.reasons

    def test_no_public_tool_surface_for_mechanical_gate(self):
        """The mechanical gate takes a LineageEvidence object, not a dict from
        tool args: a user-controlled boolean or payload cannot request it."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        _, _, ver, _, _ = _mechanical_fixtures()
        decision = evaluate_mechanical_lineage_write(ver, collection_name="memory", operation="index_capture", evidence=True)
        assert decision.decision == "reject"
        assert decision.reasons == ["lineage_evidence_invalid"]


# ===========================================================================
# W0 correction regressions (post adversarial review blockers B1-B4)
# ===========================================================================


def _w0_semantic_edge_candidate():
    """A normal reviewed semantic SUPPORTS edge candidate (store path)."""
    from qdrant_memory.improve import extract_improve_candidates_from_text

    return extract_improve_candidates_from_text(
        "Graph edge: project:DeploymentSafety -[SUPPORTS]-> concept:Validation",
        source_uri="session://review/edge",
    )[0]


class TestW0B1StructuralMarkerValueSemantics:
    """B1: the semantic gate must refuse structural write-class metadata by
    VALUE semantics — any truthy lineage_record/lineage_pending of any type,
    and the mechanical edge class on the semantic path. Absent and falsy
    markers stay allowed shared metadata; ordinary semantic metadata (endpoint
    types, source paths) is untouched."""

    @pytest.mark.parametrize("delta", [
        {"lineage_record": 1},
        {"lineage_record": "true"},
        {"lineage_pending": 1},
        {"lineage_pending": "true"},
        {"edge_class": "mechanical"},
        {"edge_class": "Mechanical"},
    ])
    def test_extraction_gate_refuses_structural_markers_by_value(self, delta):
        from qdrant_memory.source_extraction import evaluate_source_extraction_candidate

        candidate = _w0_semantic_edge_candidate()
        candidate.proposed_payload.update(delta)
        decision = evaluate_source_extraction_candidate(candidate)
        assert decision.decision == "reject", decision.to_dict()
        assert decision.reasons[0] == "mechanical_lineage_candidate_forbidden"

    def test_normal_semantic_edge_still_stores(self):
        """Positive control: a normal reviewed semantic SUPPORTS edge (with
        absent markers) still stores through the same gate."""
        from qdrant_memory.source_extraction import evaluate_source_extraction_candidate

        decision = evaluate_source_extraction_candidate(_w0_semantic_edge_candidate())
        assert decision.decision == "store", decision.to_dict()

    def test_falsy_markers_and_semantic_edge_class_are_allowed(self):
        """Absent and falsy markers (False, '') and edge_class='semantic' are
        legitimate shared metadata, never a structural classification."""
        from qdrant_memory.source_extraction import evaluate_source_extraction_candidate

        candidate = _w0_semantic_edge_candidate()
        candidate.proposed_payload.update({
            "lineage_record": False,
            "lineage_pending": False,
            "edge_class": "semantic",
        })
        decision = evaluate_source_extraction_candidate(candidate)
        assert decision.decision == "store", decision.to_dict()

    @pytest.mark.parametrize("delta", [
        {"lineage_record": 1},
        {"lineage_pending": "true"},
        {"edge_class": "mechanical"},
    ])
    def test_persisted_payload_drift_refused_post_enrichment(self, delta):
        """The post-enrichment seam (persisted_payload) runs the same
        value-semantic refusal: review-time approval cannot certify a
        structural payload class added before the final write."""
        from qdrant_memory.source_extraction import evaluate_source_extraction_candidate

        candidate = _w0_semantic_edge_candidate()
        persisted = {
            **candidate.proposed_payload,
            "profile_id": "reviewer",
            "user_id_hash": "",
            "chat_id_hash": "",
            **delta,
        }
        decision = evaluate_source_extraction_candidate(candidate, persisted_payload=persisted)
        assert decision.decision == "reject", decision.to_dict()
        assert decision.reasons[0] == "mechanical_lineage_candidate_forbidden"


class TestW0B2RaptorGateStructuralMarkers:
    """B2: the shared RAPTOR gate refuses structural/pending markers (any
    truthy type) and mechanical-class annotations — including post-enrichment
    drift. The pre-enrichment manifest refusal is covered at the provider seam
    in test_raptor_apply.py."""

    _GOOD = {
        "raptor_node_id": "raptor-node-w0b2",
        "raptor_child_ids": ["child-w0b2"],
        "source_hashes": ["a" * 64],
        "derived_from": [
            {
                "source_uri": "raptor://node/raptor-tree-w0b2/child-w0b2",
                "derivation_type": "raptor_summary",
                "relation_type": "SUMMARIZES",
                "child_node_id": "child-w0b2",
            }
        ],
        "canonical": False,
        "requires_review": True,
    }

    @pytest.mark.parametrize("delta", [
        {"lineage_record": True},
        {"lineage_pending": True},
        {"lineage_record": 1},
        {"lineage_pending": "true"},
        {"edge_class": "mechanical"},
    ])
    def test_raptor_gate_rejects_structural_markers(self, delta):
        decision = evaluate_raptor_summary_write(
            text="Summary of cluster alpha with enough text to matter.",
            metadata={**self._GOOD, **delta},
        )
        assert decision.decision == "reject", decision.to_dict()
        assert "structural_lineage_marker_forbidden" in decision.reasons

    def test_raptor_gate_allows_falsy_markers(self):
        """Falsy markers are not a structural classification: the ordinary
        full-provenance summary still routes to review, not reject."""
        decision = evaluate_raptor_summary_write(
            text="Summary of cluster alpha with enough text to matter.",
            metadata={**self._GOOD, "lineage_record": False, "lineage_pending": False, "edge_class": "semantic"},
        )
        assert decision.decision == "draft_review", decision.to_dict()


def _w0_citation_fixtures(relation="SUMMARIZES"):
    """A complete approved-citation edge between two UUID-bound memory-point
    endpoints, with independent evidence and a non-empty approval reference."""
    from qdrant_memory import lineage

    profile = "gate-profile"
    scope = lineage.make_scope_key(collection_name="memory", profile_id=profile)
    a = "99999999-8888-7777-6666-555555555555"
    b = "99999999-8888-7777-6666-555555555556"
    payload = lineage.build_mechanical_edge_payload(
        relation_type=relation,
        source_entity_id=lineage.memory_point_endpoint_logical_id(scope_key=scope, point_id=a, profile_id=profile),
        target_entity_id=lineage.memory_point_endpoint_logical_id(scope_key=scope, point_id=b, profile_id=profile),
        profile_id=profile, source_point_id=a, target_point_id=b,
        source_entity_type="memory_point", target_entity_type="memory_point",
        lineage_operation="approved_citation", lineage_scope_key=scope,
        observation="approved_manifest", source_content_hash="sha256:" + "c" * 64,
        target_content_hash="sha256:" + "d" * 64,
        provenance_content_hash="sha256:" + "c" * 64,
    )
    evidence = lineage.LineageEvidence(
        operation="approved_citation", collection_name="memory", profile_id=profile,
        relation_type=relation, source_point_id=a, target_point_id=b,
        source_entity_type="memory_point", target_entity_type="memory_point",
        source_content_hash="sha256:" + "c" * 64, target_content_hash="sha256:" + "d" * 64,
        observation="approved_manifest", approval_ref="proposal-abc123",
        scope_key=scope,
    )
    return lineage, payload, evidence


def _w0_supersedes_fixtures():
    """A valid version -> version SUPERSEDES edge (index_reconcile) with
    distinct event and predecessor UUID reference tokens, so tests can mutate
    event/predecessor/file_version_id tokens end to end."""
    import hashlib as _hashlib

    lineage, _, _, _, _ = _mechanical_fixtures()
    profile = "gate-profile"
    collection = "memory"
    scope_key = lineage.make_scope_key(collection_name=collection, profile_id=profile)
    source_key = lineage.make_source_key(scope_key=scope_key, resolved_file_path="/repo/Gate File.md")
    new_sha = _hashlib.sha256(b"new bytes").hexdigest()
    old_sha = _hashlib.sha256(b"old bytes").hexdigest()
    new_ver = lineage.build_version_node_payload(
        source_key=source_key, scope_key=scope_key, profile_id=profile,
        file_path="/repo/Gate File.md", file_sha256=new_sha,
        source_uri="file:///repo/Gate%20File.md",
        lineage_operation="index_reconcile", observation="read_bytes",
    )
    old_ver = lineage.build_version_node_payload(
        source_key=source_key, scope_key=scope_key, profile_id=profile,
        file_path="/repo/Gate File.md", file_sha256=old_sha,
        source_uri="file:///repo/Gate%20File.md",
        lineage_operation="index_reconcile", observation="read_bytes",
    )
    event_uuid = "99999999-9999-4999-8999-999999999999"
    payload = lineage.build_mechanical_edge_payload(
        relation_type="SUPERSEDES",
        source_entity_id=new_ver["entity_id"],
        target_entity_id=old_ver["entity_id"],
        profile_id=profile,
        source_point_id=lineage.storage_point_id(new_ver["entity_id"]),
        target_point_id=lineage.storage_point_id(old_ver["entity_id"]),
        source_entity_type="source",
        target_entity_type="source",
        lineage_operation="index_reconcile",
        lineage_source_key=source_key,
        lineage_scope_key=scope_key,
        observation="indexed_payload",
        source_content_hash=f"sha256:{new_sha}",
        target_content_hash=f"sha256:{old_sha}",
        provenance_content_hash=f"sha256:{new_sha}",
        file_version_id=lineage.storage_point_id(new_ver["entity_id"]),
        file_path="/repo/Gate File.md",
        file_sha256=new_sha,
        lineage_event_id=event_uuid,
    )
    evidence = lineage.LineageEvidence(
        operation="index_reconcile",
        collection_name=collection,
        profile_id=profile,
        scope_key=scope_key,
        source_key=source_key,
        source_uri="file:///repo/Gate%20File.md",
        file_path="/repo/Gate File.md",
        file_sha256=new_sha,
        content_hash=f"sha256:{new_sha}",
        observation="indexed_payload",
        relation_type="SUPERSEDES",
        source_point_id=lineage.storage_point_id(new_ver["entity_id"]),
        target_point_id=lineage.storage_point_id(old_ver["entity_id"]),
        source_entity_type="source",
        target_entity_type="source",
        source_content_hash=f"sha256:{new_sha}",
        target_content_hash=f"sha256:{old_sha}",
        file_version_id=lineage.storage_point_id(new_ver["entity_id"]),
        event_id=event_uuid,
        predecessor_event_id="99999999-9999-4999-8999-999999999998",
    )
    return lineage, payload, evidence


class TestW0B3CitationHashDomain:
    """B3: approved-citation endpoint content hashes must be validated
    against the documented sha256:<64 lowercase hex> domain in BOTH the
    evidence requirements and the final payload validation, for both endpoint
    positions and all three approved-citation relations, before any
    comparison."""

    @pytest.mark.parametrize("relation", ["SUMMARIZES", "EXTRACTED_FROM", "DERIVED_FROM"])
    def test_valid_citation_stores(self, relation):
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        _, payload, evidence = _w0_citation_fixtures(relation)
        decision = evaluate_mechanical_lineage_write(payload, operation="approved_citation", evidence=evidence, collection_name="memory")
        assert decision.decision == "store", decision.reasons

    @pytest.mark.parametrize("bad_hash", ["not-a-content-hash", "sha256:xyz"])
    @pytest.mark.parametrize("position", ["source", "target"])
    @pytest.mark.parametrize("relation", ["SUMMARIZES", "EXTRACTED_FROM", "DERIVED_FROM"])
    def test_malformed_hash_rejected_for_both_positions_and_all_relations(self, relation, position, bad_hash):
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        lineage, payload, evidence = _w0_citation_fixtures(relation)
        key = f"{position}_content_hash"
        payload[key] = bad_hash
        evidence = lineage.LineageEvidence(**{**evidence.to_dict(), key: bad_hash})
        decision = evaluate_mechanical_lineage_write(payload, operation="approved_citation", evidence=evidence, collection_name="memory")
        assert decision.decision == "reject", decision.to_dict()
        # Hash-specific reason, not a bare field name.
        assert any("sha256" in reason for reason in decision.reasons), decision.reasons

    def test_payload_hash_shape_validated_before_comparison(self):
        """The final payload validation itself flags the malformed hash with a
        hash-specific problem — shape validation is not deferred to the
        equality comparison."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        lineage, payload, evidence = _w0_citation_fixtures("SUMMARIZES")
        payload["source_content_hash"] = "sha256:xyz"
        problems = lineage.validate_lineage_payload(payload)
        assert any("source_content_hash" in p and "sha256" in p for p in problems), problems
        decision = evaluate_mechanical_lineage_write(payload, operation="approved_citation", evidence=evidence, collection_name="memory")
        assert decision.decision == "reject"
        assert decision.reasons[0] == "lineage_payload_invalid"
        assert any("sha256" in reason for reason in decision.reasons)

    def test_evidence_hash_shape_validated_before_comparison(self):
        """A malformed evidence hash fails the evidence requirements with a
        hash-specific label even when the payload carries the same
        malformation (equal malformation is not validation)."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        lineage, payload, evidence = _w0_citation_fixtures("SUMMARIZES")
        bad_evidence = lineage.LineageEvidence(**{**evidence.to_dict(), "target_content_hash": "not-a-content-hash"})
        decision = evaluate_mechanical_lineage_write(payload, operation="approved_citation", evidence=bad_evidence, collection_name="memory")
        assert decision.decision == "reject"
        assert decision.reasons[0] == "lineage_evidence_missing"
        assert any("sha256" in reason for reason in decision.reasons), decision.reasons


class TestW0B4BidirectionalProvenanceBinding:
    """B4: a mechanical edge may carry an optional provenance field only when
    the evidence observed exactly that value. All four states per field:
    absent on both sides and equal on both are accepted where the field is
    optional; payload-only and unequal are refused."""

    def test_mechanical_edge_rejects_unevidenced_uri(self):
        """The reported counterexample: payload-only source_uri with an empty
        evidence URI must not store."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        _, _, _, payload, evidence = _mechanical_fixtures()
        no_uri_evidence = evidence.__class__(**{**evidence.to_dict(), "source_uri": ""})
        tampered = dict(payload)
        tampered["source_uri"] = "https://unrelated.example/unsupported-provenance"
        decision = evaluate_mechanical_lineage_write(tampered, operation="index_capture", evidence=no_uri_evidence, collection_name="memory")
        assert decision.decision == "reject", decision.to_dict()
        # R3: refused at the missing-evidence binding layer with a
        # source_uri-specific problem.
        assert any("source_uri" in reason for reason in decision.reasons), decision.reasons

    @pytest.mark.parametrize("state", ["absent_both", "equal", "payload_only", "unequal"])
    def test_part_of_source_uri_four_states(self, state):
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        _, _, _, payload, evidence = _mechanical_fixtures()
        evidenced_uri = str(evidence.source_uri)
        payload = dict(payload)
        if state == "absent_both":
            evidence = evidence.__class__(**{**evidence.to_dict(), "source_uri": ""})
            payload.pop("source_uri", None)
            expected = "store"
        elif state == "equal":
            payload["source_uri"] = evidenced_uri
            expected = "store"
        elif state == "payload_only":
            evidence = evidence.__class__(**{**evidence.to_dict(), "source_uri": ""})
            payload["source_uri"] = evidenced_uri
            expected = "reject"
        else:
            payload["source_uri"] = "https://unrelated.example/other-provenance"
            expected = "reject"
        decision = evaluate_mechanical_lineage_write(payload, operation="index_capture", evidence=evidence, collection_name="memory")
        assert decision.decision == expected, (state, decision.to_dict())
        if expected == "reject":
            assert any("source_uri" in reason for reason in decision.reasons)

    @pytest.mark.parametrize("state", ["absent_both", "equal", "payload_only", "unequal"])
    def test_citation_source_uri_four_states(self, state):
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        lineage, payload, evidence = _w0_citation_fixtures("SUMMARIZES")
        payload = dict(payload)
        if state == "absent_both":
            expected = "store"
        elif state == "equal":
            payload["source_uri"] = "session://approved/summary"
            evidence = lineage.LineageEvidence(**{**evidence.to_dict(), "source_uri": "session://approved/summary"})
            expected = "store"
        elif state == "payload_only":
            payload["source_uri"] = "https://unrelated.example/unsupported-provenance"
            expected = "reject"
        else:
            payload["source_uri"] = "session://approved/summary"
            evidence = lineage.LineageEvidence(**{**evidence.to_dict(), "source_uri": "session://other/summary"})
            expected = "reject"
        decision = evaluate_mechanical_lineage_write(payload, operation="approved_citation", evidence=evidence, collection_name="memory")
        assert decision.decision == expected, (state, decision.to_dict())
        if expected == "reject":
            assert any("source_uri" in reason for reason in decision.reasons)

    @pytest.mark.parametrize("state", ["absent_both", "equal", "payload_only", "unequal"])
    def test_citation_locator_four_states(self, state):
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        lineage, payload, evidence = _w0_citation_fixtures("SUMMARIZES")
        payload = dict(payload)
        if state == "absent_both":
            expected = "store"
        elif state == "equal":
            payload["locator"] = {"line_start": 3}
            evidence = lineage.LineageEvidence(**{**evidence.to_dict(), "locator": {"line_start": 3}})
            expected = "store"
        elif state == "payload_only":
            payload["locator"] = {"line_start": 3}
            expected = "reject"
        else:
            payload["locator"] = {"line_start": 3}
            evidence = lineage.LineageEvidence(**{**evidence.to_dict(), "locator": {"line_start": 9}})
            expected = "reject"
        decision = evaluate_mechanical_lineage_write(payload, operation="approved_citation", evidence=evidence, collection_name="memory")
        assert decision.decision == expected, (state, decision.to_dict())
        if expected == "reject":
            assert any("locator" in reason for reason in decision.reasons)

    @pytest.mark.parametrize("state", ["absent_both", "equal_both", "payload_only", "unequal"])
    @pytest.mark.parametrize("field", ["file_path", "file_sha256"])
    def test_citation_file_field_four_states(self, field, state):
        """B4 four-state matrix for the citation edge's optional file-observed
        fields: absent on both sides and equal on both store; payload-only and
        unequal reject with the field-specific mismatch. The payload carries
        the exact recomputed source key of the EVIDENCED path so the
        file-binding shape requirement is satisfied and the parametrized
        field's own mismatch is the only problem."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        lineage, payload, evidence = _w0_citation_fixtures("SUMMARIZES")
        payload = dict(payload)
        if state == "absent_both":
            decision = evaluate_mechanical_lineage_write(payload, operation="approved_citation", evidence=evidence, collection_name="memory")
            assert decision.decision == "store", (field, state, decision.to_dict())
            return

        cited_path = "/repo/cited-source.md"
        cited_sha = _hashlib.sha256(b"cited source bytes").hexdigest()
        other_sha = _hashlib.sha256(b"different cited bytes").hexdigest()
        scope = lineage.make_scope_key(collection_name="memory", profile_id="gate-profile")
        evidence_path = cited_path
        payload_path = cited_path
        evidence_sha = cited_sha
        payload_sha = cited_sha
        if state == "unequal" and field == "file_path":
            payload_path = "/repo/other-cited-source.md"
        if state == "unequal" and field == "file_sha256":
            payload_sha = other_sha

        # Complete evidence for the field under test; the evidence source key
        # is the exact recomputed key of the evidenced path (internal-binding
        # consistent), and the payload carries the same key so the file-binding
        # source-key requirement never masks the field-specific refusal.
        evidence = lineage.LineageEvidence(
            **{
                **evidence.to_dict(),
                "source_key": lineage.make_source_key(scope_key=scope, resolved_file_path=evidence_path),
            }
        )
        payload["lineage_source_key"] = lineage.make_source_key(scope_key=scope, resolved_file_path=evidence_path)
        if field == "file_path":
            evidence = lineage.LineageEvidence(**{**evidence.to_dict(), "file_path": evidence_path})
            payload["file_path"] = payload_path
        else:
            evidence = lineage.LineageEvidence(**{**evidence.to_dict(), "file_sha256": evidence_sha})
            payload["file_sha256"] = payload_sha

        if state == "payload_only":
            # The evidence never observed the field the payload now carries.
            drop = {"file_path", "file_sha256", "source_key"} if field == "file_path" else {"file_sha256", "source_key"}
            evidence = lineage.LineageEvidence(**{key: value for key, value in evidence.to_dict().items() if key not in drop})
            expected, label = "reject", field
        elif state == "unequal":
            expected, label = "reject", field
        else:
            expected, label = "store", None

        decision = evaluate_mechanical_lineage_write(payload, operation="approved_citation", evidence=evidence, collection_name="memory")
        assert decision.decision == expected, (field, state, decision.to_dict())
        if label:
            assert any(label in reason for reason in decision.reasons), (field, state, decision.reasons)


class TestW0B5ExactTokenShapeDomain:
    """B5 (r2): every documented hash/ID shape the mechanical gate enforces is
    an exact-token check. Python ``re`` '$' also matches immediately before a
    trailing newline, so a '\\n'-suffixed digest or UUID passed the
    '$'-anchored patterns; the '\\Z'-anchored patterns refuse any
    non-canonical spelling."""

    @pytest.mark.parametrize("relation", ["SUMMARIZES", "EXTRACTED_FROM", "DERIVED_FROM"])
    @pytest.mark.parametrize("position", ["source", "target"])
    def test_citation_endpoint_hash_trailing_newline_rejected(self, relation, position):
        """The reported counterexample: a '\\n'-suffixed endpoint hash is not
        a canonical sha256 token and must be refused even when payload and
        evidence carry the identical malformation (equal malformation is not
        validation)."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write
        _, payload, evidence = _w0_citation_fixtures(relation)
        key = f"{position}_content_hash"
        malformed = payload[key] + "\n"
        payload[key] = malformed
        newline_evidence = evidence.__class__(**{**evidence.to_dict(), key: malformed})
        decision = evaluate_mechanical_lineage_write(
            payload, operation="approved_citation", evidence=newline_evidence, collection_name="memory"
        )
        assert decision.decision == "reject", decision.to_dict()
        # Refused at payload shape validation, before any comparison.
        assert decision.reasons[0] == "lineage_payload_invalid", decision.reasons
        assert any("sha256" in reason for reason in decision.reasons), decision.reasons

    @pytest.mark.parametrize("bad_hash", ["  sha256:" + "c" * 64, "sha256:" + "c" * 64 + "  ", "SHA256:" + "c" * 64, "sha256:" + "C" * 64])
    def test_citation_hash_other_noncanonical_spellings_refused(self, bad_hash):
        """Leading/trailing whitespace and uppercase spellings are different
        tokens, not the documented digest."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write
        _, payload, evidence = _w0_citation_fixtures()
        payload["source_content_hash"] = bad_hash
        bad_evidence = evidence.__class__(**{**evidence.to_dict(), "source_content_hash": bad_hash})
        decision = evaluate_mechanical_lineage_write(
            payload, operation="approved_citation", evidence=bad_evidence, collection_name="memory"
        )
        assert decision.decision == "reject", decision.to_dict()

    def test_supersedes_newline_event_identity_tokens_rejected(self):
        """Canonical UUID identity tokens on a SUPERSEDES transition (payload
        lineage_event_id, evidence event_id/predecessor_event_id) are exact
        tokens: a newline-suffixed UUID is refused at payload validation,
        never silently accepted by a '$'-anchored match."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        lineage, payload, evidence = _w0_supersedes_fixtures()
        baseline = evaluate_mechanical_lineage_write(payload, operation="index_reconcile", evidence=evidence, collection_name="memory")
        assert baseline.decision == "store", baseline.reasons

        newline_event = str(payload["lineage_event_id"]) + chr(10)
        tampered_payload = dict(payload)
        tampered_payload["lineage_event_id"] = newline_event
        tampered_evidence = evidence.__class__(
            **{
                **evidence.to_dict(),
                "event_id": newline_event,
                "predecessor_event_id": str(evidence.predecessor_event_id) + chr(10),
            }
        )
        decision = evaluate_mechanical_lineage_write(
            tampered_payload, operation="index_reconcile", evidence=tampered_evidence, collection_name="memory"
        )
        assert decision.decision == "reject", decision.to_dict()
        assert decision.reasons[0] == "lineage_payload_invalid", decision.reasons
        assert any("lineage_event_id" in reason for reason in decision.reasons), decision.reasons

    def test_supersedes_newline_file_version_id_rejected_at_shape_layer(self):
        """file_version_id is shape-validated as a canonical UUID whenever it
        is present: a newline-suffixed storage UUID is refused by payload
        validation itself, not merely by the downstream identity recompute."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        lineage, payload, evidence = _w0_supersedes_fixtures()
        tampered = dict(payload)
        tampered["file_version_id"] = str(payload["file_version_id"]) + chr(10)
        decision = evaluate_mechanical_lineage_write(tampered, operation="index_reconcile", evidence=evidence, collection_name="memory")
        assert decision.decision == "reject", decision.to_dict()
        assert decision.reasons[0] == "lineage_payload_invalid", decision.reasons
        assert any("file_version_id" in reason for reason in decision.reasons), decision.reasons

    def test_scope_and_source_digest_newlines_rejected(self):
        """The scope/source-key digests are 64-hex tokens: the same
        exact-token class applies at every validation layer that documents
        the digest shape."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write
        lineage, _, _, payload, evidence = _mechanical_fixtures()
        baseline = evaluate_mechanical_lineage_write(payload, operation="index_capture", evidence=evidence, collection_name="memory")
        assert baseline.decision == "store", baseline.reasons

        tampered = dict(payload)
        tampered["lineage_scope_key"] = str(payload["lineage_scope_key"]) + "\n"
        decision = evaluate_mechanical_lineage_write(tampered, operation="index_capture", evidence=evidence, collection_name="memory")
        assert decision.decision == "reject", decision.to_dict()
        assert decision.reasons[0] == "lineage_payload_invalid", decision.reasons
        assert any("lineage_scope_key" in reason for reason in decision.reasons), decision.reasons


class TestW0B6ProvenanceDigestBound:
    """B6 (r2): the edge's own provenance digest (payload ``content_hash``)
    must satisfy the documented hash domain whenever it is present, whatever
    the relation, and must be bound to content the evidence actually
    observed — never a compare-when-both courtesy and never a payload-only
    free pass."""

    @pytest.mark.parametrize("kind", ["citation", "indexed_chunk", "part_of"])
    def test_malformed_provenance_digest_rejected_for_every_edge_kind(self, kind):
        """A present ``content_hash`` outside the documented sha256 domain is
        refused at payload validation, whatever the relation."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write
        if kind == "citation":
            _, payload, evidence = _w0_citation_fixtures()
        elif kind == "indexed_chunk":
            _, payload, evidence = _derived_from_chunk_fixtures()
        else:
            _, _, _, payload, evidence = _mechanical_fixtures()
            evidence = evidence.__class__(**{**evidence.to_dict(), "content_hash": ""})
        assert evaluate_mechanical_lineage_write(payload, operation=evidence.operation, evidence=evidence, collection_name="memory").decision == "store"
        payload["content_hash"] = "not-a-content-hash"
        decision = evaluate_mechanical_lineage_write(payload, operation=evidence.operation, evidence=evidence, collection_name="memory")
        assert decision.decision == "reject", decision.to_dict()
        assert decision.reasons[0] == "lineage_payload_invalid", decision.reasons
        assert any("sha256" in reason for reason in decision.reasons), decision.reasons

    @pytest.mark.parametrize("kind", ["citation", "indexed_chunk", "part_of"])
    def test_unbound_payload_only_provenance_digest_rejected(self, kind):
        """A well-formed digest the evidence never observed is unbound
        provenance: payload-only values must designate evidenced content,
        so an unrelated sha256 digest is refused for every edge kind."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write
        if kind == "citation":
            _, payload, evidence = _w0_citation_fixtures()
        elif kind == "indexed_chunk":
            _, payload, evidence = _derived_from_chunk_fixtures()
        else:
            _, _, _, payload, evidence = _mechanical_fixtures()
            evidence = evidence.__class__(**{**evidence.to_dict(), "content_hash": ""})
        assert evaluate_mechanical_lineage_write(payload, operation=evidence.operation, evidence=evidence, collection_name="memory").decision == "store"
        payload["content_hash"] = "sha256:" + "e" * 64
        decision = evaluate_mechanical_lineage_write(payload, operation=evidence.operation, evidence=evidence, collection_name="memory")
        assert decision.decision == "reject", decision.to_dict()
        assert decision.reasons[0] == "lineage_evidence_mismatch", decision.reasons
        assert "content_hash" in decision.reasons, decision.reasons

    def test_documented_payload_only_digests_remain_storable(self):
        """The positive controls that legitimately carry a payload-only
        provenance digest stay storable: the citation fixture's digest is the
        cited source snapshot; the file-bound fixture's digest is the
        evidenced file digest."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write
        _, citation_payload, citation_evidence = _w0_citation_fixtures()
        assert citation_payload["content_hash"] == citation_payload["source_content_hash"]
        citation_decision = evaluate_mechanical_lineage_write(
            citation_payload, operation="approved_citation", evidence=citation_evidence, collection_name="memory"
        )
        assert citation_decision.decision == "store", citation_decision.reasons

        _, _, _, edge_payload, edge_evidence = _mechanical_fixtures()
        no_digest_evidence = edge_evidence.__class__(**{**edge_evidence.to_dict(), "content_hash": ""})
        assert edge_payload["content_hash"] == f"sha256:{edge_payload['file_sha256']}"
        edge_decision = evaluate_mechanical_lineage_write(
            edge_payload, operation="index_capture", evidence=no_digest_evidence, collection_name="memory"
        )
        assert edge_decision.decision == "store", edge_decision.reasons

    def test_evidence_bound_provenance_digest_is_strictly_bound(self):
        """When the evidence records the edge's own digest, the payload must
        carry exactly that value: emptying the payload digest fails payload
        validation (presence is presence), and altering it is a binding
        mismatch."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        _, _, _, payload, evidence = _mechanical_fixtures()
        assert str(evidence.content_hash or "")

        emptied = dict(payload)
        emptied["content_hash"] = ""
        decision = evaluate_mechanical_lineage_write(emptied, operation="index_capture", evidence=evidence, collection_name="memory")
        assert decision.decision == "reject", decision.to_dict()
        # Presence-based shape validation fires first: an explicit empty
        # digest is a failed validation, not an omission.
        assert decision.reasons[0] == "lineage_payload_invalid", decision.reasons
        assert any("sha256" in reason for reason in decision.reasons), decision.reasons

        altered = dict(payload)
        altered["content_hash"] = "sha256:" + "e" * 64
        decision = evaluate_mechanical_lineage_write(altered, operation="index_capture", evidence=evidence, collection_name="memory")
        assert decision.decision == "reject", decision.to_dict()
        assert "content_hash" in decision.reasons, decision.reasons

    def test_evidence_bound_provenance_digest_cannot_be_omitted(self):
        """Deleting the payload key must still fail strict evidence binding.

        This case reaches the comparison layer; unlike an explicit empty
        value, shape validation cannot reject it first.
        """
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        _, _, _, payload, evidence = _mechanical_fixtures()
        missing = dict(payload)
        del missing["content_hash"]
        decision = evaluate_mechanical_lineage_write(
            missing,
            operation="index_capture",
            evidence=evidence,
            collection_name="memory",
        )
        assert decision.decision == "reject", decision.to_dict()
        assert decision.reasons == ["lineage_evidence_mismatch", "content_hash"], decision.reasons

    def test_supersedes_evidence_target_hash_requires_documented_domain(self):
        """The SUPERSEDES predecessor hash is a documented sha256 digest on
        the evidence side too: truthiness alone accepts any non-empty
        spelling, and the payload-side shape check must not be the only
        exact-token enforcement in the chain."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write
        lineage, payload, evidence = _w0_supersedes_fixtures()
        malformed_evidence = evidence.__class__(**{**evidence.to_dict(), "target_content_hash": "not-a-content-hash\n"})
        payload = dict(payload)
        payload["target_content_hash"] = "not-a-content-hash\n"
        decision = evaluate_mechanical_lineage_write(payload, operation="index_reconcile", evidence=malformed_evidence, collection_name="memory")
        assert decision.decision == "reject", decision.to_dict()
        # Payload shape validation fires first and refuses the equally
        # malformed payload value.
        assert decision.reasons[0] == "lineage_payload_invalid", decision.reasons


class TestW0R2FollowupPresenceAndBuilderExactness:
    """Parent-review follow-ups: (1) a payload content_hash that is PRESENT
    must satisfy the documented domain - an explicit empty string, None, or
    non-string is a failed validation, not an omission; (2) file_version_id
    is a canonical UUID whenever the key is present; (3) the builder seam
    refuses noncanonical hash/UUID input exactly - it never strips or
    normalizes whitespace/newline input into storable tokens."""

    @pytest.mark.parametrize("bad_value", ["", None, 123])
    def test_edge_content_hash_present_means_valid_domain(self, bad_value):
        """Key presence is presence: content_hash="" or None on a direct
        payload cannot slip past the shape guard when the evidence is also
        empty."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        _, payload, evidence = _w0_citation_fixtures()
        assert evaluate_mechanical_lineage_write(payload, operation="approved_citation", evidence=evidence, collection_name="memory").decision == "store"
        payload["content_hash"] = bad_value
        decision = evaluate_mechanical_lineage_write(payload, operation="approved_citation", evidence=evidence, collection_name="memory")
        assert decision.decision == "reject", decision.to_dict()
        assert decision.reasons[0] == "lineage_payload_invalid", decision.reasons
        assert any("sha256" in reason for reason in decision.reasons), decision.reasons

    @pytest.mark.parametrize("bad_value", ["", None])
    def test_entity_content_hash_present_means_valid_domain(self, bad_value):
        """Structural entity records (which legitimately omit the key) may
        not carry it empty: a file_source payload with content_hash="" or
        None is refused."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        _, src, _, _, evidence = _mechanical_fixtures()
        assert evaluate_mechanical_lineage_write(src, operation="index_capture", evidence=evidence, collection_name="memory").decision == "store"
        assert "content_hash" not in src
        src["content_hash"] = bad_value
        decision = evaluate_mechanical_lineage_write(src, operation="index_capture", evidence=evidence, collection_name="memory")
        assert decision.decision == "reject", decision.to_dict()
        assert decision.reasons[0] == "lineage_payload_invalid", decision.reasons

    @pytest.mark.parametrize("bad_value", ["", None])
    def test_file_version_id_present_means_canonical_uuid(self, bad_value):
        """file_version_id: when the key exists, an explicit empty or None
        value is a failed validation, not an omission."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        _, _, _, payload, evidence = _mechanical_fixtures()
        assert evaluate_mechanical_lineage_write(payload, operation="index_capture", evidence=evidence, collection_name="memory").decision == "store"
        payload["file_version_id"] = bad_value
        decision = evaluate_mechanical_lineage_write(payload, operation="index_capture", evidence=evidence, collection_name="memory")
        assert decision.decision == "reject", decision.to_dict()
        assert decision.reasons[0] == "lineage_payload_invalid", decision.reasons
        assert any("file_version_id" in reason for reason in decision.reasons), decision.reasons

    @pytest.mark.parametrize("bad_hash", ["sha256:" + "c" * 64 + "\n", "  " + "c" * 64, "sha256:" + "c" * 64 + "  "])
    def test_builder_refuses_noncanonical_provenance_hash(self, bad_hash):
        """The mechanical builder refuses a provenance digest outside the
        documented domain BEFORE the legacy generic sanitizer can strip it:
        no whitespace/newline normalization into a storable token."""
        import pytest as _pytest

        from qdrant_memory.lineage import build_mechanical_edge_payload

        _, _, _, payload, _ = _mechanical_fixtures()
        with _pytest.raises(ValueError, match="provenance_content_hash"):
            build_mechanical_edge_payload(
                relation_type="PART_OF",
                source_entity_id=payload["source_entity_id"],
                target_entity_id=payload["target_entity_id"],
                profile_id=payload["profile_id"],
                source_point_id=payload["source_point_id"],
                target_point_id=payload["target_point_id"],
                source_entity_type="source",
                target_entity_type="source",
                lineage_operation="index_capture",
                lineage_source_key=payload["lineage_source_key"],
                lineage_scope_key=payload["lineage_scope_key"],
                observation="read_bytes",
                provenance_content_hash=bad_hash,
                source_content_hash=payload["source_content_hash"],
                file_version_id=payload["file_version_id"],
                file_path=payload["file_path"],
                file_sha256=payload["file_sha256"],
            )

    def test_builder_strips_nothing_w0_hash_and_uuid_fields(self):
        """Every newly introduced W0 builder normalization site now enforces
        exact matching: whitespace/newline input raises instead of being
        stripped and stored."""
        import hashlib as _hashlib
        import pytest as _pytest

        from qdrant_memory.lineage import build_mechanical_edge_payload

        _, _, _, payload, _ = _mechanical_fixtures()
        file_sha = payload["file_sha256"]
        canonical_uuid = payload["file_version_id"]
        newline_uuid = canonical_uuid + "\n"
        padded_hex = "  " + _hashlib.sha256(b"gate bytes").hexdigest() + " "
        # A canonical provenance digest satisfies the generic provenance
        # requirement so each case reaches the exact-token field validation.
        canonical_provenance = "sha256:" + "c" * 64
        cases = [
            ({"source_point_id": newline_uuid}, "source_point_id"),
            ({"target_point_id": " " + canonical_uuid}, "target_point_id"),
            ({"source_content_hash": f"sha256:{file_sha}\n"}, "source_content_hash"),
            ({"target_content_hash": f"sha256:{file_sha} "}, "target_content_hash"),
            ({"file_version_id": newline_uuid}, "file_version_id"),
            ({"file_version_id": f" {canonical_uuid}"}, "file_version_id"),
            ({"lineage_event_id": newline_uuid}, "lineage_event_id"),
            ({"file_sha256": padded_hex}, "file_sha256"),
            ({"lineage_source_key": payload["lineage_source_key"] + "\n"}, "lineage_source_key"),
        ]
        for overrides, expected_name in cases:
            kwargs = dict(
                relation_type="PART_OF",
                source_entity_id=payload["source_entity_id"],
                target_entity_id=payload["target_entity_id"],
                profile_id=payload["profile_id"],
                source_point_id=payload["source_point_id"],
                target_point_id=payload["target_point_id"],
                source_entity_type="source",
                target_entity_type="source",
                lineage_operation="index_capture",
                lineage_source_key=payload["lineage_source_key"],
                lineage_scope_key=payload["lineage_scope_key"],
                observation="read_bytes",
                provenance_content_hash=canonical_provenance,
                source_content_hash=f"sha256:{file_sha}",
                file_version_id=canonical_uuid,
                file_path=payload["file_path"],
                file_sha256=file_sha,
            )
            kwargs.update(overrides)
            with _pytest.raises(ValueError) as excinfo:
                build_mechanical_edge_payload(**kwargs)
            assert expected_name in str(excinfo.value), (expected_name, str(excinfo.value))

    def test_builder_empty_provenance_hash_still_means_omitted(self):
        """The empty default omits the edge's own optional digest. Structural
        endpoint provenance is serialized, and the final gate admits it."""
        from qdrant_memory.lineage import build_mechanical_edge_payload
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        _, _, _, payload, evidence = _mechanical_fixtures()
        built = build_mechanical_edge_payload(
            relation_type="PART_OF",
            source_entity_id=payload["source_entity_id"],
            target_entity_id=payload["target_entity_id"],
            profile_id=payload["profile_id"],
            source_point_id=payload["source_point_id"],
            target_point_id=payload["target_point_id"],
            source_entity_type="source",
            target_entity_type="source",
            lineage_operation="index_capture",
            lineage_source_key=payload["lineage_source_key"],
            lineage_scope_key=payload["lineage_scope_key"],
            observation="read_bytes",
            provenance_content_hash="",
            source_content_hash=payload["source_content_hash"],
            file_version_id=payload["file_version_id"],
            file_path=payload["file_path"],
            file_sha256=payload["file_sha256"],
        )
        assert "content_hash" not in built
        no_own_hash_evidence = evidence.__class__(**{**evidence.to_dict(), "content_hash": ""})
        decision = evaluate_mechanical_lineage_write(
            built,
            operation="index_capture",
            evidence=no_own_hash_evidence,
            collection_name="memory",
        )
        assert decision.reasons == ["mechanical_lineage_store"], decision.reasons


class TestW0R3SharedShapeDomains:
    """R3 correction pass: ONE shared validation surface (graph_schema
    ``w0_*`` domain functions) enforces every documented W0 shape domain at
    BOTH the mechanical gate and the W0-capable builder paths, by raw type
    and domain, whenever a field's key is present and BEFORE any
    relation-specific requirement or evidence comparison. Presence is
    presence: omission is key absence on payloads and the None/"" sentinel at
    builders — 0, False and {} are present, wrong-typed values, never
    "absent"."""

    @pytest.mark.parametrize("value", ["not-a-content-hash", "sha256:" + "d" * 64 + "\n", "", None, 0, {}])
    def test_part_of_present_target_hash_outside_domain_rejects(self, value):
        """A present endpoint hash must be in the documented domain whatever
        the relation: PART_OF used to store a malformed target hash because
        shape validation was conditional on the relation."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        _, _, _, payload, evidence = _mechanical_fixtures()
        assert evaluate_mechanical_lineage_write(payload, operation="index_capture", evidence=evidence, collection_name="memory").decision == "store"
        tampered = dict(payload)
        tampered["target_content_hash"] = value
        decision = evaluate_mechanical_lineage_write(tampered, operation="index_capture", evidence=evidence, collection_name="memory")
        assert decision.decision == "reject", (value, decision.to_dict())
        assert decision.reasons[0] == "lineage_payload_invalid", decision.reasons
        assert any("sha256" in reason for reason in decision.reasons), decision.reasons

    @pytest.mark.parametrize("value", ["99999999-9999-4999-8999-999999999999\n", "not-a-uuid", 0, False, {}])
    @pytest.mark.parametrize("kind", ["source", "version", "SUMMARIZES"])
    def test_present_event_reference_held_to_uuid_domain_everywhere(self, kind, value):
        """lineage_event_id is a UUID reference token whenever its key is
        present: on edges for every relation (not only when truthy) and on
        entity records not at all — the field must not escape its domain."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        if kind == "source":
            _, payload, _, _, evidence = _mechanical_fixtures()
            operation = "index_capture"
        elif kind == "version":
            _, _, payload, _, evidence = _mechanical_fixtures()
            operation = "index_capture"
        else:
            _, payload, evidence = _w0_citation_fixtures()
            operation = "approved_citation"
        assert evaluate_mechanical_lineage_write(payload, operation=operation, evidence=evidence, collection_name="memory").decision == "store"
        tampered = dict(payload)
        tampered["lineage_event_id"] = value
        decision = evaluate_mechanical_lineage_write(tampered, operation=operation, evidence=evidence, collection_name="memory")
        assert decision.decision == "reject", (kind, repr(value), decision.to_dict())
        assert decision.reasons[0] == "lineage_payload_invalid", decision.reasons

    @pytest.mark.parametrize(
        "field,values",
        [
            ("lineage_source_key", ["", None, 0, False, {}]),
            ("locator", [[], False, {}]),
        ],
    )
    def test_present_falsy_values_do_not_collapse_to_absence(self, field, values):
        """lineage_source_key and locator collapse 0/False/{}/"" to absence on
        the direct payload path; presence must be presence."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        _, payload, evidence = _w0_citation_fixtures()
        assert evaluate_mechanical_lineage_write(payload, operation="approved_citation", evidence=evidence, collection_name="memory").decision == "store"
        for bad in values:
            tampered = dict(payload)
            tampered[field] = bad
            decision = evaluate_mechanical_lineage_write(tampered, operation="approved_citation", evidence=evidence, collection_name="memory")
            assert decision.decision == "reject", (field, repr(bad), decision.to_dict())
            assert decision.reasons[0] == "lineage_payload_invalid", decision.reasons

    def test_present_file_sha256_outside_domain_rejects(self):
        """A present file_sha256 is a 64-hex token by RAW type: the old
        comparator coerced with str(), so a 64-digit Python integer stored."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        _, payload, evidence = _w0_citation_fixtures()
        path = "/repo/citation.md"
        key = lineage.make_source_key(scope_key=payload["lineage_scope_key"], resolved_file_path=path)
        bad = int("1" * 64)
        tampered = dict(payload, file_path=path, file_sha256=bad, lineage_source_key=key)
        bad_evidence = lineage.LineageEvidence(**{**evidence.to_dict(), "file_path": path, "file_sha256": bad, "source_key": key})
        decision = evaluate_mechanical_lineage_write(tampered, operation="approved_citation", evidence=bad_evidence, collection_name="memory")
        assert decision.decision == "reject", decision.to_dict()
        assert decision.reasons[0] == "lineage_payload_invalid", decision.reasons
        assert any("file_sha256" in reason for reason in decision.reasons), decision.reasons

    def test_file_source_content_hash_is_prohibited_but_omission_stores(self):
        """R3-B2 resolution: a file_source record must not carry content_hash
        at all. The source node designates the exact path, not a content
        snapshot, and no evidence observes content for it — any digest it
        carried (well-formed or not) is unbindable provenance. The omission
        control (no key) keeps storing, and the file_version binding is
        untouched."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        _, src, _, _, evidence = _mechanical_fixtures()
        assert "content_hash" not in src
        assert evaluate_mechanical_lineage_write(src, operation="index_capture", evidence=evidence, collection_name="memory").decision == "store"
        tampered = dict(src, content_hash="sha256:" + "e" * 64)
        decision = evaluate_mechanical_lineage_write(tampered, operation="index_capture", evidence=evidence, collection_name="memory")
        assert decision.decision == "reject", decision.to_dict()
        assert decision.reasons[0] == "lineage_payload_invalid", decision.reasons
        assert any("file_source" in reason for reason in decision.reasons), decision.reasons

    def test_evidence_fields_held_to_same_shared_domains(self):
        """The identical shape rule applies to the evidence record itself: a
        present evidence value outside its documented domain refuses before
        any comparison (the old comparator coerced evidence and payload
        through str() and compared)."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        _, payload, evidence = _w0_citation_fixtures()
        bad_evidence = lineage.LineageEvidence(**{**evidence.to_dict(), "file_sha256": int("1" * 64)})
        decision = evaluate_mechanical_lineage_write(payload, operation="approved_citation", evidence=bad_evidence, collection_name="memory")
        assert decision.decision == "reject", decision.to_dict()
        assert decision.reasons[0] == "lineage_evidence_invalid", decision.reasons
        assert any("file_sha256" in reason for reason in decision.reasons), decision.reasons

        newline_uri_evidence = lineage.LineageEvidence(**{**evidence.to_dict(), "source_uri": "session://approved/summary\n"})
        decision = evaluate_mechanical_lineage_write(payload, operation="approved_citation", evidence=newline_uri_evidence, collection_name="memory")
        assert decision.decision == "reject", decision.to_dict()
        assert decision.reasons[0] == "lineage_evidence_invalid", decision.reasons

    def test_generic_builder_refuses_unnormalized_provenance_hash(self):
        """The GENERIC structural serialization boundary (GraphEntity /
        GraphEdge ``to_payload``) refuses a content_hash outside the shared
        domain when lineage_record=True; the legacy lax sanitizer must not
        strip 'sha256:<64hex>\\n' into a storable token."""
        import pytest as _pytest

        _, payload, _ = _w0_citation_fixtures()
        for bad in ("sha256:" + "c" * 64 + "\n", "  sha256:" + "c" * 64 + "  "):
            with _pytest.raises(ValueError, match="content_hash"):
                graph_schema.build_edge_payload(
                    source_entity_id=payload["source_entity_id"],
                    target_entity_id=payload["target_entity_id"],
                    relation_type=payload["relation_type"],
                    profile_id=payload["profile_id"],
                    content_hash=bad,
                    lineage_record=True,
                    lineage_schema_version=1,
                    lineage_operation=payload["lineage_operation"],
                    lineage_identity_digest=payload["lineage_identity_digest"],
                    lineage_scope_key=payload["lineage_scope_key"],
                    lineage_observation=payload["lineage_observation"],
                    edge_class="mechanical",
                    source_point_id=payload["source_point_id"],
                    target_point_id=payload["target_point_id"],
                    source_entity_type=payload["source_entity_type"],
                    target_entity_type=payload["target_entity_type"],
                    source_content_hash=payload["source_content_hash"],
                    target_content_hash=payload["target_content_hash"],
                )
            with _pytest.raises(ValueError, match="content_hash"):
                graph_schema.build_entity_payload(
                    entity_type="source",
                    label="file-source-x",
                    profile_id="gate-profile",
                    content_hash=bad,
                    logical_entity_id=payload["source_entity_id"],
                    lineage_record=True,
                    lineage_schema_version=1,
                    lineage_operation="index_capture",
                    lineage_identity_digest="a" * 64,
                    lineage_source_key="b" * 64,
                    lineage_scope_key="c" * 64,
                    lineage_observation="read_bytes",
                    lineage_role="file_source",
                    file_path="/repo/x.md",
                )

    @pytest.mark.parametrize(
        "overrides",
        [
            {"lineage_operation": " approved_citation "},
            {"observation": "approved_manifest\n"},
            {"source_entity_type": " MEMORY_POINT "},
            {"relation_type": " summarizes "},
        ],
    )
    def test_mechanical_builder_refuses_unnormalized_vocabulary(self, overrides):
        """Controlled vocabulary is exact-token at the mechanical builder:
        padded, newline-suffixed, or case-shifted tokens refuse instead of
        being stripped/lowercased into the storable spelling."""
        import pytest as _pytest

        _, payload, _ = _w0_citation_fixtures()
        kwargs = dict(
            relation_type=payload["relation_type"],
            source_entity_id=payload["source_entity_id"],
            target_entity_id=payload["target_entity_id"],
            profile_id=payload["profile_id"],
            source_point_id=payload["source_point_id"],
            target_point_id=payload["target_point_id"],
            source_entity_type=payload["source_entity_type"],
            target_entity_type=payload["target_entity_type"],
            lineage_operation=payload["lineage_operation"],
            lineage_scope_key=payload["lineage_scope_key"],
            observation=payload["lineage_observation"],
            provenance_content_hash=payload["content_hash"],
            source_content_hash=payload["source_content_hash"],
            target_content_hash=payload["target_content_hash"],
        )
        kwargs.update(overrides)
        with _pytest.raises(ValueError):
            lineage.build_mechanical_edge_payload(**kwargs)

    def test_mechanical_builder_refuses_invalid_raw_locator(self):
        """The W0 builder path validates the RAW locator against the shared
        real-indexer shape; the generic sanitizer used to erase unknown keys
        and truncate an overlong heading, turning an invalid raw locator
        into the exact valid evidence locator."""
        import pytest as _pytest

        _, edge, _ = _derived_from_chunk_fixtures()
        for locator in ({"line_start": 1, "extra": None}, {"line_start": 1, "heading": "x" * 201}):
            with _pytest.raises(ValueError, match="locator"):
                lineage.build_mechanical_edge_payload(
                    relation_type="DERIVED_FROM",
                    source_entity_id=edge["source_entity_id"],
                    target_entity_id=edge["target_entity_id"],
                    profile_id=edge["profile_id"],
                    source_point_id=edge["source_point_id"],
                    target_point_id=edge["target_point_id"],
                    source_entity_type="memory_point",
                    target_entity_type="source",
                    lineage_operation="index_capture",
                    lineage_source_key=edge["lineage_source_key"],
                    lineage_scope_key=edge["lineage_scope_key"],
                    provenance_content_hash=edge["content_hash"],
                    source_content_hash=edge["source_content_hash"],
                    target_content_hash=edge["target_content_hash"],
                    file_version_id=edge["file_version_id"],
                    file_path=edge["file_path"],
                    file_sha256=edge["file_sha256"],
                    locator=locator,
                )

    def test_builders_emit_handles_and_uri_verbatim_and_gate_refuses_raw_spelling(self):
        """Logical handles and source URIs serialize VERBATIM for structural
        records (never stripped): a raw token the direct gate rejects builds
        into a payload the gate still rejects — normalization can no longer
        manufacture a storable record."""
        _, payload, evidence = _w0_citation_fixtures()
        bad_handle = payload["source_entity_id"] + "\n"
        built = graph_schema.build_edge_payload(
            source_entity_id=bad_handle,
            target_entity_id=payload["target_entity_id"],
            relation_type=payload["relation_type"],
            profile_id=payload["profile_id"],
            content_hash=payload["content_hash"],
            lineage_record=True,
            lineage_schema_version=1,
            lineage_operation=payload["lineage_operation"],
            lineage_identity_digest=payload["lineage_identity_digest"],
            lineage_scope_key=payload["lineage_scope_key"],
            lineage_observation=payload["lineage_observation"],
            edge_class="mechanical",
            source_point_id=payload["source_point_id"],
            target_point_id=payload["target_point_id"],
            source_entity_type=payload["source_entity_type"],
            target_entity_type=payload["target_entity_type"],
            source_content_hash=payload["source_content_hash"],
            target_content_hash=payload["target_content_hash"],
        )
        assert built["source_entity_id"] == bad_handle
        assert lineage.validate_lineage_payload(built), "raw handle must fail payload validation"
        assert any("source_entity_id" in reason for reason in lineage.validate_lineage_payload(built))

        _, src, _, _, evidence = _mechanical_fixtures()
        bad_uri = "file:///repo/Gate%20File.md\n"
        built_source = lineage.build_source_node_payload(
            source_key=src["lineage_source_key"],
            scope_key=src["lineage_scope_key"],
            profile_id=src["profile_id"],
            file_path=src["file_path"],
            source_uri=bad_uri,
        )
        assert built_source["source_uri"] == bad_uri
        problems = lineage.validate_lineage_payload(built_source)
        assert problems and any("source_uri" in reason for reason in problems), problems


class TestW0R3RepairRound:
    """Independent-review repair round (blockers 1-5): the version builder
    must refuse wrong-typed digests before coercion; the evidence locator's
    only omission sentinel is the empty dict and locator optional members are
    validated by key presence; kind guards cover edge_id / entity_id /
    entity_type; legacy non-lineage builder behavior is byte-for-byte
    preserved; and the structural serializers classify handles/URIs with the
    shared domains BEFORE any legacy normalizer — unsafe values raise,
    malformed-but-serializable strings serialize verbatim for the final gate
    (the enforcing write boundary) to refuse."""

    def test_version_builder_refuses_wrong_typed_file_sha256(self):
        """The version builder validates the RAW file_sha256 (shared domain)
        before identity/interpolation: a 64-digit int must never be coerced
        through str() into a look-alike token that identity, content_hash,
        and string evidence would then agree on."""
        import pytest as _pytest

        with _pytest.raises(ValueError, match="file_sha256"):
            lineage.build_version_node_payload(
                source_key="a" * 64,
                scope_key="b" * 64,
                profile_id="gate-profile",
                file_path="/repo/x.md",
                file_sha256=int("1" * 64),
            )
        with _pytest.raises(ValueError, match="file_sha256"):
            lineage.build_version_node_payload(
                source_key="a" * 64,
                scope_key="b" * 64,
                profile_id="gate-profile",
                file_path="/repo/x.md",
                file_sha256="1" * 64 + "\n",
            )

    @pytest.mark.parametrize("value", [[], False, 0, None])
    def test_evidence_locator_only_empty_dict_sentinel_omits(self, value):
        """The evidence locator's documented omission sentinel is the empty
        dict; every other present value — including falsy ones — is held to
        the shared locator domain."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        _, _, _, payload, evidence = _mechanical_fixtures()
        assert evidence.locator == {}
        assert evaluate_mechanical_lineage_write(payload, operation="index_capture", evidence=evidence, collection_name="memory").decision == "store"
        bad_evidence = lineage.LineageEvidence(**{**evidence.to_dict(), "locator": value})
        decision = evaluate_mechanical_lineage_write(payload, operation="index_capture", evidence=bad_evidence, collection_name="memory")
        assert decision.decision == "reject", (repr(value), decision.to_dict())
        assert decision.reasons[0] == "lineage_evidence_invalid", decision.reasons

    def test_locator_optional_members_validated_by_key_presence(self):
        """An explicit line_end=None or heading=None is a present, wrong-typed
        value and refuses; genuinely absent optional keys and the bounded
        positive control still pass."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        _, edge, evidence = _derived_from_chunk_fixtures()
        for locator in ({"line_start": 1, "line_end": None}, {"line_start": 1, "heading": None}):
            tampered = dict(edge, locator=locator)
            tampered_evidence = lineage.LineageEvidence(**{**evidence.to_dict(), "locator": locator})
            decision = evaluate_mechanical_lineage_write(tampered, operation="index_capture", evidence=tampered_evidence, collection_name="memory")
            assert decision.decision == "reject", (locator, decision.to_dict())
            assert decision.reasons[0] == "lineage_payload_invalid", decision.reasons
            assert any("locator" in reason for reason in decision.reasons), locator
        # Valid omissions and the bounded positive control still store.
        for locator in ({"line_start": 1}, {"line_start": 1, "line_end": 9, "heading": "H" * 200}):
            tampered = dict(edge, locator=locator)
            tampered_evidence = lineage.LineageEvidence(**{**evidence.to_dict(), "locator": locator})
            decision = evaluate_mechanical_lineage_write(tampered, operation="index_capture", evidence=tampered_evidence, collection_name="memory")
            assert decision.decision == "store", (locator, decision.to_dict())

    @pytest.mark.parametrize("kind,key,value", [
        ("entity", "edge_id", "edge-0123456789abcdef"),
        ("edge", "entity_id", "entity-0123456789abcdef"),
        ("edge", "entity_type", "source"),
    ])
    def test_kind_guards_cover_edge_id_entity_id_entity_type(self, kind, key, value):
        """A structural field only the other record kind interprets must not
        ride on a record: edge_id on entities, entity_id/entity_type on
        edges all refuse by key presence."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        if kind == "entity":
            _, payload, _, _, evidence = _mechanical_fixtures()
            operation = "index_capture"
        else:
            _, _, _, payload, evidence = _mechanical_fixtures()
            operation = "index_capture"
        tampered = dict(payload)
        tampered[key] = value
        decision = evaluate_mechanical_lineage_write(tampered, operation=operation, evidence=evidence, collection_name="memory")
        assert decision.decision == "reject", (kind, key, decision.to_dict())
        assert decision.reasons[0] == "lineage_payload_invalid", decision.reasons
        assert any(key in reason for reason in decision.reasons), decision.reasons

    def test_legacy_builder_behavior_preserved_for_non_lineage_callers(self):
        """lineage_record=False keeps the exact pre-pass legacy behavior:
        edge_class=False omits the field, and a truthy non-string file_path
        stringifies ('123') instead of raising. Strict raw typing applies
        only when lineage_record=True."""
        built = graph_schema.build_edge_payload(
            source_entity_id="entity-0123456789abcdef",
            target_entity_id="entity-0123456789abcdee",
            relation_type="SUPPORTS",
            source_uri="legacy",
            edge_class=False,
        )
        assert "edge_class" not in built
        built = graph_schema.build_edge_payload(
            source_entity_id="entity-0123456789abcdef",
            target_entity_id="entity-0123456789abcdee",
            relation_type="SUPPORTS",
            source_uri="legacy",
            file_path=123,
        )
        assert built["file_path"] == "123"
        # The strict structural path still refuses the same input.
        import pytest as _pytest

        with _pytest.raises(ValueError, match="file_path"):
            graph_schema.build_edge_payload(
                source_entity_id="entity-0123456789abcdef",
                target_entity_id="entity-0123456789abcdee",
                relation_type="SUPPORTS",
                source_uri="legacy",
                file_path=123,
                lineage_record=True,
                lineage_schema_version=1,
                lineage_operation="index_capture",
                lineage_identity_digest="a" * 64,
                lineage_observation="read_bytes",
            )

    def test_structural_serializer_classifies_before_normalizing(self):
        """At the structural serialization boundary the shared handle/URI
        domains are classified BEFORE any legacy normalizer: malformed-but-
        serializable raw strings serialize with their exact spelling (the
        final gate is the enforcing write boundary and refuses them), while
        non-string and secret-bearing unsafe values raise at the builder."""
        import pytest as _pytest

        _, src, _, _, _ = _mechanical_fixtures()
        bad_uri = "file:///repo/Gate%20File.md\n"
        built = lineage.build_source_node_payload(
            source_key=src["lineage_source_key"],
            scope_key=src["lineage_scope_key"],
            profile_id=src["profile_id"],
            file_path=src["file_path"],
            source_uri=bad_uri,
        )
        assert built["source_uri"] == bad_uri, "malformed serializable string must serialize verbatim"
        problems = lineage.validate_lineage_payload(built)
        assert problems and any("source_uri" in reason for reason in problems), problems

        _, payload, _ = _w0_citation_fixtures()
        bad_handle = payload["source_entity_id"] + "\n"
        built = graph_schema.build_edge_payload(
            source_entity_id=bad_handle,
            target_entity_id=payload["target_entity_id"],
            relation_type=payload["relation_type"],
            profile_id=payload["profile_id"],
            content_hash=payload["content_hash"],
            lineage_record=True,
            lineage_schema_version=1,
            lineage_operation=payload["lineage_operation"],
            lineage_identity_digest=payload["lineage_identity_digest"],
            lineage_scope_key=payload["lineage_scope_key"],
            lineage_observation=payload["lineage_observation"],
            edge_class="mechanical",
            source_point_id=payload["source_point_id"],
            target_point_id=payload["target_point_id"],
            source_entity_type=payload["source_entity_type"],
            target_entity_type=payload["target_entity_type"],
            source_content_hash=payload["source_content_hash"],
            target_content_hash=payload["target_content_hash"],
        )
        assert built["source_entity_id"] == bad_handle
        problems = lineage.validate_lineage_payload(built)
        assert any("source_entity_id" in reason for reason in problems), problems

        # Unsafe values raise at the builder boundary.
        with _pytest.raises(ValueError, match="source_uri"):
            lineage.build_source_node_payload(
                source_key=src["lineage_source_key"],
                scope_key=src["lineage_scope_key"],
                profile_id=src["profile_id"],
                file_path=src["file_path"],
                source_uri=123,
            )
        with _pytest.raises(ValueError, match="source_entity_id"):
            graph_schema.build_edge_payload(
                # Scanner-shaped secret constructed at runtime (repo scan rule).
            source_entity_id="".join(["api", "_key=", "secret", "-handle"]),
                target_entity_id="entity-0123456789abcdee",
                relation_type="SUPPORTS",
                source_uri="legacy",
                lineage_record=True,
                lineage_schema_version=1,
                lineage_operation="index_capture",
                lineage_identity_digest="a" * 64,
                lineage_observation="read_bytes",
            )


class TestW0R3IndependentBindingAndRoleValidity:
    """Third repair round: (1) record-role validity — snapshot digests exist
    only on file_version records and file-source endpoints never carry
    snapshot hashes, whatever the evidence supplies; (2) the independent-
    binding class — whenever the payload carries an interpreted evidence/
    provenance field, a sentinel-omitted evidence twin fails closed at the
    missing-evidence layer before any comparison, systematically over the
    payload->evidence field map for the fixture kinds this table exercises
    (source, version, part_of, indexed_chunk, supersedes); (3) legacy
    non-lineage file_path emits only when the coerced path is truthy."""

    @staticmethod
    def _fixture(kind):
        if kind in ("source", "version", "part_of"):
            _, src, ver, edge, evidence = _mechanical_fixtures()
            payload = {"source": src, "version": ver, "part_of": edge}[kind]
            operation = "index_capture"
        elif kind == "indexed_chunk":
            _, payload, evidence = _derived_from_chunk_fixtures()
            operation = "index_capture"
        elif kind == "supersedes":
            _, payload, evidence = _w0_supersedes_fixtures()[0], _w0_supersedes_fixtures()[1], _w0_supersedes_fixtures()[2]
            operation = "index_reconcile"
        else:
            _, payload, evidence = _w0_citation_fixtures()
            operation = "approved_citation"
        return payload, evidence, operation

    @pytest.mark.parametrize(
        "kind,payload_key,evidence_field,expected",
        [
            ("source", "lineage_scope_key", "scope_key", "store"),
            ("source", "lineage_source_key", "source_key", "reject"),
            ("version", "lineage_scope_key", "scope_key", "store"),
            ("version", "lineage_source_key", "source_key", "reject"),
            ("version", "content_hash", "content_hash", "reject"),
            ("version", "file_sha256", "file_sha256", "reject"),
            ("part_of", "lineage_scope_key", "scope_key", "store"),
            ("part_of", "lineage_source_key", "source_key", "reject"),
            ("part_of", "content_hash", "content_hash", "store"),
            ("part_of", "file_version_id", "file_version_id", "reject"),
            ("part_of", "file_sha256", "file_sha256", "reject"),
            ("part_of", "file_path", "file_path", "reject"),
            ("indexed_chunk", "file_version_id", "file_version_id", "reject"),
            ("indexed_chunk", "locator", "locator", "reject"),
            ("supersedes", "lineage_scope_key", "scope_key", "store"),
            ("supersedes", "lineage_source_key", "source_key", "reject"),
            ("supersedes", "content_hash", "content_hash", "store"),
            ("supersedes", "file_version_id", "file_version_id", "reject"),
            ("supersedes", "lineage_event_id", "event_id", "reject"),
            ("supersedes", "file_sha256", "file_sha256", "reject"),
        ],
    )
    def test_sentinel_evidence_twin_fails_closed_systematically(self, kind, payload_key, evidence_field, expected):
        """Systematic unbound-evidence sweep, kept durable: a valid payload
        whose evidence twin is set to its documented sentinel must reject at
        the missing-evidence layer for every bound field. The only stores are
        the two documented independence exceptions (scope via recomputation;
        edge provenance digest via the anchors rule), which the independence
        test below proves are independently bound."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        payload, evidence, operation = self._fixture(kind)
        assert payload_key in payload, (kind, payload_key)
        assert evaluate_mechanical_lineage_write(payload, operation=operation, evidence=evidence, collection_name="memory").decision == "store"
        sentinel = {} if evidence_field == "locator" else ""
        emptied = lineage.LineageEvidence(**{**evidence.to_dict(), evidence_field: sentinel})
        decision = evaluate_mechanical_lineage_write(payload, operation=operation, evidence=emptied, collection_name="memory")
        if expected == "reject":
            assert decision.decision == "reject", (kind, payload_key, decision.to_dict())
            assert decision.reasons[0] == "lineage_evidence_missing", (kind, payload_key, decision.reasons)
            assert payload_key in decision.reasons[1], (kind, payload_key, decision.reasons)
        else:
            assert decision.decision == "store", (kind, payload_key, decision.to_dict())
            # Independence proof for the exception: a TAMPERED payload value
            # with the same omitted twin still fails closed.
            tampered = dict(payload)
            tampered[payload_key] = "sha256:" + "e" * 64 if payload_key == "content_hash" else "f" * 64
            decision = evaluate_mechanical_lineage_write(tampered, operation=operation, evidence=emptied, collection_name="memory")
            assert decision.decision == "reject", (kind, payload_key, decision.to_dict())

    def test_file_source_record_may_not_carry_file_sha256(self):
        """Record-role validity: the file_source node designates the exact
        path, never a content snapshot — file_sha256 refuses with both
        absent and equal evidence (mirrors the content_hash prohibition)."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        _, src, ver, _, evidence = _mechanical_fixtures()
        assert "file_sha256" not in src
        digest = ver["file_sha256"]
        for evidence_value in ("", digest):
            tampered = dict(src, file_sha256=digest)
            tampered_evidence = lineage.LineageEvidence(**{**evidence.to_dict(), "file_sha256": evidence_value})
            decision = evaluate_mechanical_lineage_write(tampered, operation="index_capture", evidence=tampered_evidence, collection_name="memory")
            assert decision.decision == "reject", (evidence_value, decision.to_dict())
            assert decision.reasons[0] == "lineage_payload_invalid", decision.reasons
            assert any("file_sha256" in reason for reason in decision.reasons), decision.reasons
        # The file_version record keeps its required snapshot digest.
        assert evaluate_mechanical_lineage_write(ver, operation="index_capture", evidence=evidence, collection_name="memory").decision == "store"

    @pytest.mark.parametrize("evidence_value", ["", "sha256:" + "d" * 64])
    def test_part_of_target_snapshot_hash_refused_even_with_equal_evidence(self, evidence_value):
        """PART_OF's target endpoint is the file-source node: a target
        snapshot hash is illegal by ROLE, so it refuses with no evidence AND
        with an equally supplied evidence value (validation precedes every
        comparison)."""
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        _, _, _, payload, evidence = _mechanical_fixtures()
        target_hash = "sha256:" + "d" * 64
        tampered = dict(payload, target_content_hash=target_hash)
        tampered_evidence = lineage.LineageEvidence(**{**evidence.to_dict(), "target_content_hash": evidence_value})
        decision = evaluate_mechanical_lineage_write(tampered, operation="index_capture", evidence=tampered_evidence, collection_name="memory")
        assert decision.decision == "reject", (evidence_value, decision.to_dict())
        assert decision.reasons[0] == "lineage_payload_invalid", decision.reasons
        assert any("target_content_hash" in reason for reason in decision.reasons), decision.reasons

    def test_legacy_falsy_file_path_omits_key(self):
        """Legacy non-lineage file_path: after str(value or '') coercion the
        key is emitted only when truthy — 123 emits '123'; 0/False/[]/{}
        omit the key entirely (exact pre-pass behavior). Structural records
        keep strict raw typing."""
        for falsy in (0, False, [], {}):
            built = graph_schema.build_edge_payload(
                source_entity_id="entity-0123456789abcdef",
                target_entity_id="entity-0123456789abcdee",
                relation_type="SUPPORTS",
                source_uri="legacy",
                file_path=falsy,
            )
            assert "file_path" not in built, repr(falsy)
        built = graph_schema.build_edge_payload(
            source_entity_id="entity-0123456789abcdef",
            target_entity_id="entity-0123456789abcdee",
            relation_type="SUPPORTS",
            source_uri="legacy",
            file_path=123,
        )
        assert built["file_path"] == "123"


class TestW0FileVersionEvidenceContentBinding:
    """Final narrow correction: the file_version record's snapshot digest must
    equal the evidence's observed content digest. The binding layer already
    fails an OMITTED twin; an UNRELATED well-formed evidence digest now fails
    at the mismatch layer instead of storing against a valid payload."""

    @staticmethod
    def _gate(payload, evidence):
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        return evaluate_mechanical_lineage_write(payload, operation="index_capture", evidence=evidence, collection_name="memory")

    def test_file_version_content_hash_evidence_twins(self):
        _, _, ver, _, evidence = _mechanical_fixtures()
        assert evidence.content_hash == ver["content_hash"]
        # Matching twin: stores.
        assert self._gate(ver, evidence).decision == "store", evidence.to_dict()
        # Omitted twin: fails closed at the missing-evidence binding layer.
        omitted = lineage.LineageEvidence(**{**evidence.to_dict(), "content_hash": ""})
        decision = self._gate(ver, omitted)
        assert decision.decision == "reject", decision.to_dict()
        assert decision.reasons[0] == "lineage_evidence_missing", decision.reasons
        assert any("content_hash" in reason for reason in decision.reasons), decision.reasons
        # Unrelated WELL-FORMED twin: fails closed at the mismatch layer.
        unrelated = lineage.LineageEvidence(**{**evidence.to_dict(), "content_hash": "sha256:" + "e" * 64})
        decision = self._gate(ver, unrelated)
        assert decision.decision == "reject", decision.to_dict()
        assert decision.reasons[0] == "lineage_evidence_mismatch", decision.reasons
        assert "content_hash" in decision.reasons, decision.reasons
        # The file_source prohibition is untouched: no content_hash at all.
        _, src, _, _, _ = _mechanical_fixtures()
        assert "content_hash" not in src
        assert self._gate(src, evidence).decision == "store"


class TestW0FileVersionEndpointIdentityBinding:
    """A paired payload/evidence UUID cannot replace the deterministic
    file-version endpoint binding."""

    _ARBITRARY_VERSION_UUID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"

    @classmethod
    def _assert_rejected(cls, payload, evidence):
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        baseline = evaluate_mechanical_lineage_write(
            payload,
            operation=evidence.operation,
            evidence=evidence,
            collection_name="memory",
        )
        assert baseline.decision == "store", baseline.to_dict()
        assert cls._ARBITRARY_VERSION_UUID not in {
            payload["source_point_id"],
            payload["target_point_id"],
        }

        tampered_payload = dict(payload, file_version_id=cls._ARBITRARY_VERSION_UUID)
        tampered_evidence = lineage.LineageEvidence(
            **{**evidence.to_dict(), "file_version_id": cls._ARBITRARY_VERSION_UUID}
        )
        decision = evaluate_mechanical_lineage_write(
            tampered_payload,
            operation=evidence.operation,
            evidence=tampered_evidence,
            collection_name="memory",
        )
        assert decision.decision == "reject", decision.to_dict()
        assert decision.reasons == [
            "lineage_identity_mismatch",
            "file_version_id",
        ], decision.to_dict()

    def test_part_of_rejects_paired_arbitrary_file_version_id(self):
        _, _, _, payload, evidence = _mechanical_fixtures()
        self._assert_rejected(payload, evidence)

    def test_indexed_derived_from_rejects_paired_arbitrary_file_version_id(self):
        _, payload, evidence = _derived_from_chunk_fixtures()
        self._assert_rejected(payload, evidence)

    def test_supersedes_rejects_paired_arbitrary_file_version_id(self):
        _, payload, evidence = _w0_supersedes_fixtures()
        self._assert_rejected(payload, evidence)

    @staticmethod
    def _assert_near_match_rejected(payload, evidence, version_id):
        from uuid import UUID

        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        baseline = evaluate_mechanical_lineage_write(
            payload,
            operation=evidence.operation,
            evidence=evidence,
            collection_name="memory",
        )
        assert baseline.decision == "store", baseline.to_dict()
        endpoints = {payload["source_point_id"], payload["target_point_id"]}
        assert version_id in endpoints

        replacement = "0" if version_id[-1] != "0" else "1"
        near_match = version_id[:-1] + replacement
        assert str(UUID(near_match)) == near_match
        assert near_match[:-1] == version_id[:-1]
        assert near_match[-1] != version_id[-1]
        assert near_match not in endpoints

        tampered_payload = dict(payload, file_version_id=near_match)
        tampered_evidence = lineage.LineageEvidence(
            **{**evidence.to_dict(), "file_version_id": near_match}
        )
        decision = evaluate_mechanical_lineage_write(
            tampered_payload,
            operation=evidence.operation,
            evidence=tampered_evidence,
            collection_name="memory",
        )
        assert decision.decision == "reject", decision.to_dict()
        assert decision.reasons == [
            "lineage_identity_mismatch",
            "file_version_id",
        ], decision.to_dict()

    def test_part_of_rejects_near_match_source_version_id(self):
        _, _, _, payload, evidence = _mechanical_fixtures()
        self._assert_near_match_rejected(payload, evidence, payload["source_point_id"])

    def test_indexed_derived_from_rejects_near_match_target_version_id(self):
        _, payload, evidence = _derived_from_chunk_fixtures()
        self._assert_near_match_rejected(payload, evidence, payload["target_point_id"])

    def test_supersedes_rejects_near_match_current_version_id(self):
        _, payload, evidence = _w0_supersedes_fixtures()
        self._assert_near_match_rejected(payload, evidence, payload["source_point_id"])

    def test_supersedes_rejects_near_match_predecessor_version_id(self):
        _, payload, evidence = _w0_supersedes_fixtures()
        self._assert_near_match_rejected(payload, evidence, payload["target_point_id"])

    def test_part_of_rejects_paired_file_source_endpoint_as_version(self):
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        _, _, _, payload, evidence = _mechanical_fixtures()
        baseline = evaluate_mechanical_lineage_write(
            payload,
            operation=evidence.operation,
            evidence=evidence,
            collection_name="memory",
        )
        assert baseline.decision == "store", baseline.to_dict()
        file_source_id = payload["target_point_id"]
        tampered_payload = dict(payload, file_version_id=file_source_id)
        tampered_evidence = lineage.LineageEvidence(
            **{**evidence.to_dict(), "file_version_id": file_source_id}
        )
        decision = evaluate_mechanical_lineage_write(
            tampered_payload,
            operation=evidence.operation,
            evidence=tampered_evidence,
            collection_name="memory",
        )
        assert decision.decision == "reject", decision.to_dict()
        assert decision.reasons == [
            "lineage_identity_mismatch",
            "file_version_id",
        ], decision.to_dict()

    def test_approved_citation_rejects_reference_without_version_endpoint(self):
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        _, payload, evidence = _w0_citation_fixtures("DERIVED_FROM")
        baseline = evaluate_mechanical_lineage_write(
            payload,
            operation=evidence.operation,
            evidence=evidence,
            collection_name="memory",
        )
        assert baseline.decision == "store", baseline.to_dict()
        file_path = "/repo/Cited File.md"
        source_key = lineage.make_source_key(
            scope_key=payload["lineage_scope_key"],
            resolved_file_path=file_path,
        )
        tampered_payload = dict(
            payload,
            file_path=file_path,
            lineage_source_key=source_key,
            file_version_id=self._ARBITRARY_VERSION_UUID,
        )
        tampered_evidence = lineage.LineageEvidence(
            **{
                **evidence.to_dict(),
                "file_path": file_path,
                "source_key": source_key,
                "source_uri": "file:///repo/Cited%20File.md",
                "file_version_id": self._ARBITRARY_VERSION_UUID,
            }
        )
        decision = evaluate_mechanical_lineage_write(
            tampered_payload,
            operation=evidence.operation,
            evidence=tampered_evidence,
            collection_name="memory",
        )
        assert decision.decision == "reject", decision.to_dict()
        assert decision.reasons == [
            "lineage_identity_mismatch",
            "file_version_id",
        ], decision.to_dict()


class TestW0R4BuilderGateAgreement:
    """Structural serializers neither normalize rejected identity labels into
    accepted records nor reject gate-valid edges that omit their own optional
    provenance digest."""

    @staticmethod
    def _fixture(kind):
        if kind in ("source", "version", "part_of"):
            _, source, version, edge, evidence = _mechanical_fixtures()
            return {"source": source, "version": version, "part_of": edge}[kind], evidence
        if kind == "indexed_chunk":
            return _derived_from_chunk_fixtures()[1:]
        if kind == "supersedes":
            return _w0_supersedes_fixtures()[1:]
        return _w0_citation_fixtures(kind)[1:]

    @pytest.mark.parametrize("kind", ["source", "version"])
    @pytest.mark.parametrize("builder", ["helper", "dataclass"])
    def test_structural_entity_label_serializes_verbatim_for_gate_rejection(self, kind, builder):
        """Both public entity serialization paths preserve a malformed safe
        label/text pair, so the final gate sees and rejects the raw spelling."""
        import inspect

        from qdrant_memory import graph_schema
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        payload, evidence = self._fixture(kind)
        raw = payload["label"] + "\n"
        direct = dict(payload, label=raw, text=raw)
        direct_decision = evaluate_mechanical_lineage_write(
            direct, operation=evidence.operation, evidence=evidence, collection_name="memory"
        )
        assert direct_decision.reasons == ["lineage_identity_mismatch", "label"], direct_decision.reasons

        factory = graph_schema.build_entity_payload if builder == "helper" else graph_schema.GraphEntity
        accepted = inspect.signature(factory).parameters
        args = {key: value for key, value in payload.items() if key in accepted}
        args.update(label=raw, logical_entity_id=payload["entity_id"])
        built = factory(**args)
        if builder == "dataclass":
            built = built.to_payload()
        built.update(user_id_hash="", chat_id_hash="")
        if kind == "version":
            built["file_sha256"] = payload["file_sha256"]
        assert built["label"] == raw
        assert built["text"] == raw
        decision = evaluate_mechanical_lineage_write(
            built, operation=evidence.operation, evidence=evidence, collection_name="memory"
        )
        assert decision.reasons == ["lineage_identity_mismatch", "label"], decision.reasons

    @pytest.mark.parametrize(
        "kind,builder",
        [
            ("source", "generic"),
            ("source", "dataclass"),
            ("SUMMARIZES", "generic"),
            ("SUMMARIZES", "dataclass"),
            ("SUMMARIZES", "mechanical"),
        ],
    )
    def test_structural_profile_id_is_not_normalized_into_another_owner(self, kind, builder):
        """Structural serializers preserve a safe raw ownership token, so a
        gate-rejected owner cannot be stripped into the evidenced owner."""
        import inspect

        from qdrant_memory import graph_schema, lineage
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        payload, evidence = self._fixture(kind)
        raw = payload["profile_id"] + "\n"
        direct = evaluate_mechanical_lineage_write(
            dict(payload, profile_id=raw),
            operation=evidence.operation,
            evidence=evidence,
            collection_name="memory",
        )
        assert direct.decision == "reject", direct.to_dict()

        if kind == "source":
            factory = graph_schema.build_entity_payload if builder == "generic" else graph_schema.GraphEntity
            accepted = inspect.signature(factory).parameters
            args = {key: value for key, value in payload.items() if key in accepted}
            args.update(profile_id=raw, logical_entity_id=payload["entity_id"])
        elif builder == "mechanical":
            factory = lineage.build_mechanical_edge_payload
            accepted = inspect.signature(factory).parameters
            args = {key: value for key, value in payload.items() if key in accepted}
            args.update(profile_id=raw, observation=payload["lineage_observation"])
        else:
            factory = graph_schema.build_edge_payload if builder == "generic" else graph_schema.GraphEdge
            accepted = inspect.signature(factory).parameters
            args = {key: value for key, value in payload.items() if key in accepted}
            args["profile_id"] = raw
        built = factory(**args)
        if builder == "dataclass":
            built = built.to_payload()
        built.update(user_id_hash="", chat_id_hash="")
        assert built["profile_id"] == raw
        decision = evaluate_mechanical_lineage_write(
            built, operation=evidence.operation, evidence=evidence, collection_name="memory"
        )
        assert decision.decision == "reject", decision.to_dict()

    @pytest.mark.parametrize(
        "kind",
        ["part_of", "indexed_chunk", "supersedes", "SUMMARIZES", "EXTRACTED_FROM", "DERIVED_FROM"],
    )
    @pytest.mark.parametrize("builder", ["mechanical", "generic", "dataclass"])
    def test_structural_edge_builders_accept_gate_valid_omitted_own_hash(self, kind, builder):
        """Endpoint provenance is sufficient for structural serialization;
        the final mechanical gate remains the admission decision."""
        import dataclasses
        import inspect

        from qdrant_memory import graph_schema, lineage
        from qdrant_memory.write_gate import evaluate_mechanical_lineage_write

        payload, evidence = self._fixture(kind)
        payload = dict(payload)
        payload.pop("content_hash", None)
        evidence = dataclasses.replace(evidence, content_hash="")
        assert evaluate_mechanical_lineage_write(
            payload, operation=evidence.operation, evidence=evidence, collection_name="memory"
        ).reasons == ["mechanical_lineage_store"]

        if builder == "mechanical":
            factory = lineage.build_mechanical_edge_payload
            accepted = inspect.signature(factory).parameters
            args = {key: value for key, value in payload.items() if key in accepted}
            args["observation"] = payload["lineage_observation"]
            built = factory(**args)
        else:
            factory = graph_schema.build_edge_payload if builder == "generic" else graph_schema.GraphEdge
            accepted = inspect.signature(factory).parameters
            args = {key: value for key, value in payload.items() if key in accepted}
            built = factory(**args)
            if builder == "dataclass":
                built = built.to_payload()
            built.update(user_id_hash="", chat_id_hash="")
        assert "content_hash" not in built
        decision = evaluate_mechanical_lineage_write(
            built, operation=evidence.operation, evidence=evidence, collection_name="memory"
        )
        assert decision.reasons == ["mechanical_lineage_store"], decision.reasons
