from __future__ import annotations

import json

import pytest

from qdrant_memory.write_gate import WriteDecision, decision_to_json, evaluate_write_candidate


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
