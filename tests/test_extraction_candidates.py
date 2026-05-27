from __future__ import annotations

import json

import pytest

from qdrant_memory.extraction_candidates import (
    EXTRACTION_CANDIDATE_TYPES,
    build_extraction_candidate,
    preview_extraction_candidates,
)
from qdrant_memory.schema import SourceLocator, build_assertion_payload, build_payload


class FakeEmbeddings:
    def __init__(self):
        self.documents = []

    def embed_document(self, text):  # pragma: no cover - must not be called
        self.documents.append(text)
        raise AssertionError("candidate preview must not construct embeddings")


class FakeQdrant:
    def __init__(self):
        self.upserts = []
        self.deletes = []

    def upsert(self, collection_name, points):  # pragma: no cover - must not be called
        self.upserts.append((collection_name, points))
        raise AssertionError("candidate preview must not mutate Qdrant")

    def delete_ids(self, collection_name, ids):  # pragma: no cover - must not be called
        self.deletes.append((collection_name, ids))
        raise AssertionError("candidate preview must not mutate Qdrant")


def test_building_and_previewing_candidate_is_serializable_without_embedding_or_qdrant_mutation():
    embeddings = FakeEmbeddings()
    qdrant = FakeQdrant()
    payload = build_assertion_payload(
        claim_text="Hermes extraction candidates remain review gated.",
        subject="hermes.extraction_candidates",
        predicate="remain",
        object="review_gated",
        source_uri="session://session-1/message-2",
        locator=SourceLocator(line_start=2, line_end=2),
        evidence=[{"source_uri": "session://session-1/message-2", "relation_type": "SUPPORTS"}],
        created_at="2026-05-27T00:00:00+00:00",
    )

    candidate = build_extraction_candidate(
        candidate_type="assertion_candidate",
        source_uri="session://session-1/message-2",
        locator={"message_index": 2},
        derived_from=[{"source_uri": "session://session-1", "relation_type": "EXTRACTED_FROM"}],
        proposed_payload=payload,
        reason="source-backed assertion candidate",
        confidence=0.82,
        risk="medium",
        created_at="2026-05-27T00:00:01+00:00",
    )
    preview = preview_extraction_candidates([candidate])

    assert set(EXTRACTION_CANDIDATE_TYPES) >= {
        "memory_candidate",
        "assertion_candidate",
        "preference_candidate",
        "invariant_candidate",
        "risk_candidate",
        "status_update_candidate",
        "ontology_suggestion",
    }
    assert preview["dry_run"] is True
    assert preview["count"] == 1
    item = preview["candidates"][0]
    assert {
        "candidate_id",
        "candidate_type",
        "source_uri",
        "locator",
        "derived_from",
        "proposed_payload",
        "reason",
        "confidence",
        "risk",
        "requires_review",
        "created_at",
    } <= item.keys()
    assert item["candidate_type"] == "assertion_candidate"
    assert item["requires_review"] is True
    json.dumps(preview, sort_keys=True, allow_nan=False)
    assert embeddings.documents == []
    assert qdrant.upserts == []
    assert qdrant.deletes == []


def test_candidate_ids_are_deterministic_for_identical_pending_lifecycle_inputs():
    payload = build_payload(
        text="Prefer exact IDs for source-backed memory approval.",
        source="candidate-preview",
        memory_kind="user_preference",
        source_uri="session://stable/input",
        derived_from=[{"source_uri": "session://stable/input", "relation_type": "EXTRACTED_FROM"}],
        created_at="2026-05-27T00:00:00+00:00",
    )

    first = build_extraction_candidate(
        candidate_type="preference_candidate",
        source_uri="session://stable/input",
        locator={"message_index": 4},
        derived_from=[{"source_uri": "session://stable/input", "relation_type": "EXTRACTED_FROM"}],
        proposed_payload=payload,
        reason="explicit user preference",
        confidence=0.91,
        risk="low",
        lifecycle_id="pending-buffer-1",
        created_at="2026-05-27T00:00:01+00:00",
    )
    second = build_extraction_candidate(
        candidate_type="preference_candidate",
        source_uri="session://stable/input",
        locator={"message_index": 4},
        derived_from=[{"source_uri": "session://stable/input", "relation_type": "EXTRACTED_FROM"}],
        proposed_payload=payload,
        reason="explicit user preference",
        confidence=0.91,
        risk="low",
        lifecycle_id="pending-buffer-1",
        created_at="2026-05-27T00:00:09+00:00",
    )

    assert first.candidate_id == second.candidate_id


def test_invalid_candidate_type_is_rejected():
    with pytest.raises(ValueError, match="candidate_type"):
        build_extraction_candidate(
            candidate_type="learning_candidate",
            source_uri="session://invalid/type",
            proposed_payload={"text": "bad candidate type"},
            reason="unsupported type",
        )


def test_invalid_proposed_payload_memory_grammar_is_rejected():
    with pytest.raises(ValueError, match="memory_kind"):
        build_extraction_candidate(
            candidate_type="memory_candidate",
            source_uri="session://bad/memory-kind",
            proposed_payload={"text": "bad kind", "memory_kind": "unknown_kind"},
            reason="bad grammar",
        )

    with pytest.raises(ValueError, match="relation_type"):
        build_extraction_candidate(
            candidate_type="assertion_candidate",
            source_uri="session://bad/relation",
            proposed_payload={
                "text": "bad relation",
                "memory_kind": "assertion",
                "derived_from": [{"source_uri": "session://root", "relation_type": "NOT_A_RELATION"}],
            },
            reason="bad grammar",
        )

    with pytest.raises(ValueError, match="fact_status"):
        build_extraction_candidate(
            candidate_type="status_update_candidate",
            source_uri="session://bad/fact-status",
            proposed_payload={"text": "bad status", "memory_kind": "assertion", "fact_status": "not_a_status"},
            reason="bad grammar",
        )
