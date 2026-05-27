from __future__ import annotations

import json
from pathlib import Path

from __init__ import QdrantMemoryProvider
from qdrant_memory.extraction_candidates import ExtractionCandidate, build_extraction_candidate
from qdrant_memory.schema import build_payload
from qdrant_memory.source_extraction import (
    evaluate_source_extraction_candidate,
    extract_source_candidates_from_messages,
    extract_source_candidates_from_point,
    extract_source_candidates_from_text,
    preview_source_extraction_candidates,
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
            "source_extraction_enabled": True,
            "source_extraction_mode": "preview",
            "source_extraction_min_confidence": 0.65,
            "source_extraction_max_candidates_per_session": 8,
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
    preview = preview_source_extraction_candidates(candidates)

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
    assert preview["dry_run"] is True
    assert preview["count"] == 5
    assert all(item["write_decision"]["decision"] == "store" for item in preview["candidates"])


def test_completed_turn_and_recalled_point_extraction_preserve_source_provenance():
    turn_candidates = extract_source_candidates_from_messages(
        [
            {"role": "user", "content": "Actually, use OpenAI Codex provider, not OpenRouter."},
            {"role": "assistant", "content": "Acknowledged."},
        ],
        source_uri="session://source-first/completed-turn",
    )
    point_candidates = extract_source_candidates_from_point(
        {
            "id": "point-123",
            "payload": {
                "text": "Decision: project recipes are retrieval plans, not new memory authority.",
                "source_uri": "file:///tmp/roadmap.md",
                "locator": {"line_start": 894, "line_end": 895},
            },
        }
    )

    assert turn_candidates[0].candidate_type == "preference_candidate"
    assert turn_candidates[0].locator["message_index"] == 0
    assert turn_candidates[0].derived_from[0]["derivation_type"] == "completed_turn"
    assert point_candidates[0].source_uri == "file:///tmp/roadmap.md"
    assert point_candidates[0].derived_from[0]["point_id"] == "point-123"
    assert point_candidates[0].locator == {"line_start": 894, "line_end": 895}


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
    provider._pending_extraction_candidates[candidate.candidate_id] = candidate

    dry_run = json.loads(provider.handle_tool_call("qdrant_memory_extraction_approve", {"candidate_id": candidate.candidate_id}))
    live = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_extraction_approve",
            {"candidate_id": candidate.candidate_id, "dry_run": False, "approve": True},
        )
    )

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
    provider._pending_extraction_candidates[candidate.candidate_id] = candidate

    live_without_review = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_extraction_approve",
            {"candidate_id": candidate.candidate_id, "dry_run": False, "approve": True},
        )
    )
    partial_id_review = json.loads(
        provider.handle_tool_call("qdrant_memory_extraction_approve", {"candidate_id": candidate.candidate_id[:8]})
    )
    reviewed = json.loads(provider.handle_tool_call("qdrant_memory_extraction_approve", {"candidate_id": candidate.candidate_id}))
    live_after_review = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_extraction_approve",
            {"candidate_id": candidate.candidate_id, "dry_run": False, "approve": True},
        )
    )

    assert "error" in live_without_review
    assert "dry-run" in live_without_review["error"]
    assert "error" in partial_id_review
    assert "Unknown extraction candidate" in partial_id_review["error"]
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
    provider._pending_extraction_candidates[candidate.candidate_id] = candidate

    preview = json.loads(provider.handle_tool_call("qdrant_memory_extraction_approve", {"candidate_id": candidate.candidate_id}))
    live = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_extraction_approve",
            {"candidate_id": candidate.candidate_id, "dry_run": False, "approve": True},
        )
    )

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
    provider._pending_extraction_candidates[candidate.candidate_id] = candidate
    reviewed = json.loads(provider.handle_tool_call("qdrant_memory_extraction_approve", {"candidate_id": candidate.candidate_id}))
    secret_value = "".join(["abcdefghijkl", "mnopqrstu"])
    provider._config["embedding_model"] = "Authorization: " + "Bearer " + secret_value

    live = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_extraction_approve",
            {"candidate_id": candidate.candidate_id, "dry_run": False, "approve": True},
        )
    )

    assert reviewed["write_decision"]["decision"] == "store"
    assert "error" in live
    assert "rejected" in live["error"]
    assert provider._qdrant.upserts == []


def test_source_extraction_default_disabled_and_approval_dry_run_do_not_mutate(tmp_path):
    provider = QdrantMemoryProvider()
    provider._qdrant = FakeQdrant()
    provider._embeddings = FakeEmbedding()
    provider._active = True
    provider._hermes_home = str(tmp_path / "hermes")
    provider._config.update({"collection_name": "memory", "learning_collection_name": "learnings"})

    provider.on_session_end([
        {"role": "user", "content": "Decision: keep default source extraction disabled."},
        {"role": "assistant", "content": "Acknowledged."},
    ])

    disabled_preview = json.loads(provider.handle_tool_call("qdrant_memory_extraction_preview", {}))
    assert disabled_preview["count"] == 0

    provider._config["source_extraction_enabled"] = True
    provider.on_session_end([
        {"role": "user", "content": "Decision: source extraction approvals default to dry-run."},
        {"role": "assistant", "content": "Acknowledged."},
    ])
    enabled_preview = json.loads(provider.handle_tool_call("qdrant_memory_extraction_preview", {}))
    candidate_id = enabled_preview["candidates"][0]["candidate_id"]
    approved = json.loads(provider.handle_tool_call("qdrant_memory_extraction_approve", {"candidate_id": candidate_id}))

    assert enabled_preview["dry_run"] is True
    assert approved["dry_run"] is True
    assert approved["saved"] is False
    assert provider._qdrant.upserts == []
