from __future__ import annotations

import json

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
