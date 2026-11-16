from __future__ import annotations

import json
from pathlib import Path

from __init__ import QdrantMemoryProvider
from qdrant_memory.extraction_candidates import ExtractionCandidate, build_extraction_candidate
from qdrant_memory.schema import build_payload
from qdrant_memory.source_extraction import (
    evaluate_source_extraction_candidate,
    extract_source_candidates_from_text,
)


class FakeEmbedding:
    def __init__(self):
        self.documents = []

    def embed_document(self, text):
        self.documents.append(text)
        return [0.3, 0.4]


class FakeQdrant:
    def __init__(self):
        self.upserts = []

    def upsert(self, name, points):
        self.upserts.append((name, points))
        return {"status": "ok"}

    def retrieve(self, name, ids, *, with_payload=True, with_vector=False):
        # This double keeps no point state, so it has none to return. The
        # extraction-approval path reads the target id before writing to refuse a
        # lineage-managed overwrite, and needs the interface to exist.
        return []


def _provider_for_source_extraction(tmp_path: Path) -> QdrantMemoryProvider:
    provider = QdrantMemoryProvider()
    provider._qdrant = FakeQdrant()
    provider._embeddings = FakeEmbedding()
    provider._active = True
    provider._hermes_home = str(tmp_path / "hermes")
    provider._session_id = "session-11-2"
    provider._config.update(
        {
            "collection_name": "memory",
            "learning_collection_name": "learnings",
        }
    )
    return provider


def test_source_text_extractor_emits_only_source_backed_priority_signals():
    source = "\n".join(
        [
            "Today we discussed generic progress and several minor updates.",
            "Decision: keep qdrant_memory_index dry-run by default.",
            "Tool quirk: terminal commands must use venv/bin/python -m pytest because pytest is not on PATH.",
            "Project invariant: qdrant_memory_consolidate is report-only and must not mutate Qdrant.",
            "Resolved conflict: source-backed assertions supersede unsupported summaries after review.",
            "Actually, prefer exact point IDs for approvals, not query-based deletion.",
        ]
    )

    candidates = extract_source_candidates_from_text(
        source,
        source_uri="session://source-first/turn-1",
        source_type="completed_turn",
        lifecycle_id="buffer-11-2",
    )

    payload_texts = [candidate.proposed_payload["text"] for candidate in candidates]
    assert len(candidates) == 5
    assert not any("generic progress" in text for text in payload_texts)
    assert {candidate.candidate_type for candidate in candidates} >= {
        "memory_candidate",
        "preference_candidate",
        "invariant_candidate",
        "status_update_candidate",
    }
    assert {candidate.proposed_payload["memory_kind"] for candidate in candidates} >= {
        "decision",
        "tool_quirk",
        "project_invariant",
        "user_preference",
        "assertion",
    }
    for candidate in candidates:
        assert candidate.source_uri == "session://source-first/turn-1"
        assert candidate.derived_from
        assert candidate.proposed_payload["source_uri"] == "session://source-first/turn-1"
        assert candidate.proposed_payload["derived_from"][0]["relation_type"] == "EXTRACTED_FROM"
        assert candidate.proposed_payload["derivation_type"] == "source_extraction"
        assert evaluate_source_extraction_candidate(candidate).decision == "store"


def test_secret_like_source_candidate_is_rejected_without_persisting_secret_text():
    token_value = "".join(["ghp", "_", "1234567890abcdef", "1234567890abcdef", "123456"])
    secret_line = "Decision: use " + "Authorization: " + "Bearer " + token_value + " for the next request."

    extracted = extract_source_candidates_from_text(secret_line, source_uri="session://unsafe/turn")

    assert extracted == []

    unsafe_payload = build_payload(
        text=secret_line,
        source="source_extraction",
        source_type="source_extraction",
        memory_kind="decision",
        source_uri="session://unsafe/turn",
        derived_from=[{"source_uri": "session://unsafe/turn", "relation_type": "EXTRACTED_FROM"}],
        derivation_type="source_extraction",
    )
    unsafe_candidate = build_extraction_candidate(
        candidate_type="memory_candidate",
        source_uri="session://unsafe/turn",
        proposed_payload=unsafe_payload,
        derived_from=[{"source_uri": "session://unsafe/turn", "relation_type": "EXTRACTED_FROM"}],
        reason="unsafe regression candidate",
        confidence=0.9,
        risk="high",
    )

    decision = evaluate_source_extraction_candidate(unsafe_candidate)

    assert decision.decision == "reject"
    assert "possible_secret" in decision.reasons


def test_low_confidence_candidate_live_approval_creates_draft_proposal_not_memory(tmp_path):
    provider = _provider_for_source_extraction(tmp_path)
    candidates = extract_source_candidates_from_text(
        "Possible decision: keep source-first extraction dry-run until approval.",
        source_uri="session://source-first/low-confidence",
        min_confidence=0.0,
    )
    candidate = candidates[0]

    dry_run = json.loads(provider.approve_extraction_candidate(candidate))
    live = json.loads(provider.approve_extraction_candidate(candidate, dry_run=False, approve=True))

    assert dry_run["dry_run"] is True
    assert dry_run["would_create_proposal"] is True
    assert dry_run["would_store"] is False
    assert dry_run["proposal_id"] == candidate.candidate_id
    assert live["dry_run"] is False
    assert live["proposal_created"] is True
    assert live["proposal_id"] == candidate.candidate_id
    assert live["saved"] is False
    assert provider._qdrant.upserts == []
    assert Path(live["proposal_draft_path"]).exists()


def test_source_extraction_live_approval_requires_prior_exact_dry_run(tmp_path):
    provider = _provider_for_source_extraction(tmp_path)
    candidate = extract_source_candidates_from_text(
        "Decision: keep source extraction approvals tied to exact pending candidate IDs.",
        source_uri="session://source-first/exact-dry-run",
    )[0]

    live_without_review = json.loads(provider.approve_extraction_candidate(candidate, dry_run=False, approve=True))
    without_approve = json.loads(provider.approve_extraction_candidate(candidate, dry_run=False))
    reviewed = json.loads(provider.approve_extraction_candidate(candidate))
    live_after_review = json.loads(provider.approve_extraction_candidate(candidate, dry_run=False, approve=True))

    assert "error" in live_without_review
    assert "dry-run" in live_without_review["error"]
    assert "approve=true" in without_approve["error"]
    assert reviewed["dry_run"] is True
    assert reviewed["candidate"]["candidate_id"] == candidate.candidate_id
    assert live_after_review["dry_run"] is False
    assert live_after_review["saved"] is True
    assert provider._qdrant.upserts[0][0] == "memory"


def test_source_extraction_approval_refuses_missing_source_provenance(tmp_path):
    provider = _provider_for_source_extraction(tmp_path)
    payload = build_payload(
        text="Decision: source extraction candidates require source provenance before approval.",
        source="source_extraction",
        source_type="source_extraction",
        memory_kind="decision",
        derivation_type="source_extraction",
    )
    candidate = ExtractionCandidate(
        candidate_id="missing-source-provenance",
        candidate_type="memory_candidate",
        source_uri="",
        locator={},
        derived_from=[],
        proposed_payload=payload,
        reason="regression candidate with no source provenance",
        confidence=0.92,
        risk="low",
    )

    preview = json.loads(provider.approve_extraction_candidate(candidate))
    live = json.loads(provider.approve_extraction_candidate(candidate, dry_run=False, approve=True))

    assert preview["write_decision"]["decision"] == "reject"
    assert "missing_source_provenance" in preview["write_decision"]["reasons"]
    assert "error" in live
    assert "provenance" in live["error"]
    assert provider._qdrant.upserts == []


def test_source_extraction_approval_revalidates_full_persisted_payload_before_upsert(tmp_path):
    provider = _provider_for_source_extraction(tmp_path)
    candidate = extract_source_candidates_from_text(
        "Decision: validate every persisted field before live source extraction approval.",
        source_uri="session://source-first/full-payload-secret",
    )[0]
    reviewed = json.loads(provider.approve_extraction_candidate(candidate))
    secret_value = "".join(["abcdefghijkl", "mnopqrstu"])
    provider._config["embedding_model"] = "Authorization: " + "Bearer " + secret_value

    live = json.loads(provider.approve_extraction_candidate(candidate, dry_run=False, approve=True))

    assert reviewed["write_decision"]["decision"] == "store"
    assert "error" in live
    assert "rejected" in live["error"]
    assert provider._qdrant.upserts == []
