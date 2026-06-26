from __future__ import annotations

import json
from pathlib import Path

import pytest

from __init__ import QdrantMemoryProvider
from qdrant_memory.improve import (
    IMPROVE_EXTRACTOR_VERSION,
    _check_no_external_graph_deps,
    build_improve_report,
    extract_improve_candidates_from_text,
    make_candidate_digest,
)
from qdrant_memory.schema import RELATION_TYPES


class FakeEmbedding:
    def __init__(self):
        self.documents = []

    def embed_document(self, text):
        self.documents.append(text)
        return [0.3, 0.4]


class FakeQdrant:
    def __init__(self):
        self.upserts = []
        self.scroll_results: dict[str, list[dict]] = {}

    def upsert(self, name, points):
        self.upserts.append((name, points))
        return {"status": "ok"}

    def scroll(self, collection_name, limit=10, with_payload=True, with_vectors=False, _filter=None, **kw):
        # Return format expected by provider: (points_list, next_page_offset)
        if _filter and "must" in _filter:
            for clause in _filter["must"]:
                if clause.get("key") == "id":
                    pid = clause.get("match", {}).get("value", "")
                    if pid in self.scroll_results:
                        return (self.scroll_results[pid], None)
        return ([], None)


def _provider_for_improve(tmp_path: Path) -> QdrantMemoryProvider:
    provider = QdrantMemoryProvider()
    provider._qdrant = FakeQdrant()
    provider._embeddings = FakeEmbedding()
    provider._active = True
    provider._hermes_home = str(tmp_path / "hermes")
    provider._session_id = "improve-session-1"
    provider._config.update(
        {
            "collection_name": "memory",
            "learning_collection_name": "learnings",
        }
    )
    return provider


# ---------------------------------------------------------------------------
# Dependency guard
# ---------------------------------------------------------------------------

def test_improve_module_does_not_import_forbidden_dependencies():
    _check_no_external_graph_deps()


# ---------------------------------------------------------------------------
# Preview from explicit source text
# ---------------------------------------------------------------------------

def test_preview_from_source_text_creates_report_without_mutating_qdrant(tmp_path):
    provider = _provider_for_improve(tmp_path)
    source = "\n".join([
        "Graph entity: project: Nucleogenesis",
        "Graph entity: tool: PicoClaw",
        "Graph edge: project:Nucleogenesis -[DEPENDS_ON]-> tool:PicoClaw",
        "Decision: keep improve preview dry-run by default.",
    ])

    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_improve_preview",
            {"source_text": source, "source_uri": "session://test/improve"},
        )
    )

    assert result["dry_run"] is True
    assert result["report_type"] == "graph_improve_preview"
    assert result["extractor_version"] == IMPROVE_EXTRACTOR_VERSION
    assert result["source_scope"] == "source_text"
    assert result["counts"]["total"] >= 3  # 2 entities + 1 edge + maybe decision
    assert result["counts"]["store_eligible"] >= 1
    assert "report_id" in result
    assert result["report_id"].startswith("improve-")
    assert "report_digest" in result
    assert "candidates" in result
    assert len(result["candidates"]) == result["counts"]["total"]

    # Each candidate has required fields
    for item in result["candidates"]:
        assert "candidate_id" in item
        assert "candidate_digest" in item
        assert "target_point_id" in item
        assert "write_decision" in item
        assert "apply_eligible" in item
        assert "would_store" in item
        assert item["candidate_digest"].startswith("sha256:")

    # No Qdrant mutation
    assert provider._qdrant.upserts == []
    assert provider._embeddings.documents == []


def test_preview_graph_entity_candidates_have_deterministic_ids(tmp_path):
    provider = _provider_for_improve(tmp_path)
    source = "Graph entity: project: TestProject"

    result1 = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_improve_preview",
            {"source_text": source, "source_uri": "session://test/deterministic"},
        )
    )
    result2 = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_improve_preview",
            {"source_text": source, "source_uri": "session://test/deterministic"},
        )
    )

    # Same inputs should produce the same report_id and candidate_ids
    assert result1["report_id"] == result2["report_id"]
    assert result1["candidates"][0]["candidate_id"] == result2["candidates"][0]["candidate_id"]
    assert result1["candidates"][0]["candidate_digest"] == result2["candidates"][0]["candidate_digest"]


# ---------------------------------------------------------------------------
# Preview from point_ids scope
# ---------------------------------------------------------------------------

def test_preview_from_point_ids_uses_bounded_qdrant_scroll(tmp_path):
    provider = _provider_for_improve(tmp_path)
    # Set up fake scroll results
    provider._qdrant.scroll_results = {
        "point-aaa": [
            {"id": "point-aaa", "payload": {"text": "Graph entity: concept: Emergence", "source_uri": "session://test/point"}},
        ],
    }

    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_improve_preview",
            {"point_ids": ["point-aaa"], "source_scope": "point_ids"},
        )
    )

    assert result["dry_run"] is True
    assert result["source_scope"] == "point_ids"
    assert result["counts"]["total"] >= 1
    assert "qdrant://memory/point-aaa" in result["source_handles"]
    assert provider._qdrant.upserts == []


# ---------------------------------------------------------------------------
# Live apply refuses without prior dry-run and without approve
# ---------------------------------------------------------------------------

def test_live_apply_refuses_without_prior_dry_run(tmp_path):
    provider = _provider_for_improve(tmp_path)
    source = "Graph entity: project: ApplyTest"
    preview = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_improve_preview",
            {"source_text": source, "source_uri": "session://test/apply"},
        )
    )
    report_id = preview["report_id"]
    candidate_id = preview["candidates"][0]["candidate_id"]

    # Live apply without dry-run first
    live = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_improve_apply",
            {"report_id": report_id, "candidate_id": candidate_id, "dry_run": False, "approve": True},
        )
    )
    assert "error" in live
    assert "dry-run" in live["error"]


def test_live_apply_refuses_without_approve_true(tmp_path):
    provider = _provider_for_improve(tmp_path)
    source = "Graph entity: project: ApproveTest"
    preview = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_improve_preview",
            {"source_text": source, "source_uri": "session://test/approve"},
        )
    )
    report_id = preview["report_id"]
    candidate_id = preview["candidates"][0]["candidate_id"]

    # Dry-run first
    provider.handle_tool_call(
        "qdrant_memory_improve_apply",
        {"report_id": report_id, "candidate_id": candidate_id},
    )

    # Live apply without approve
    live = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_improve_apply",
            {"report_id": report_id, "candidate_id": candidate_id, "dry_run": False, "approve": False},
        )
    )
    assert "error" in live
    assert "approve" in live["error"]


def test_apply_rejects_unknown_report_and_candidate(tmp_path):
    provider = _provider_for_improve(tmp_path)

    unknown_report = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_improve_apply",
            {"report_id": "improve-nonexistent", "candidate_id": "fake"},
        )
    )
    assert "error" in unknown_report
    assert "Unknown" in unknown_report["error"]


# ---------------------------------------------------------------------------
# Live apply stores only exact approved storeable candidate
# ---------------------------------------------------------------------------

def test_live_apply_stores_graph_entity_candidate(tmp_path):
    provider = _provider_for_improve(tmp_path)
    source = "Graph entity: project: StoreTest"
    preview = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_improve_preview",
            {"source_text": source, "source_uri": "session://test/store"},
        )
    )
    report_id = preview["report_id"]
    # Find the store-eligible candidate
    store_candidate = None
    for c in preview["candidates"]:
        if c["would_store"]:
            store_candidate = c
            break
    assert store_candidate is not None, "Expected at least one store-eligible candidate"
    candidate_id = store_candidate["candidate_id"]
    target_pid = store_candidate["target_point_id"]

    # Dry-run first
    provider.handle_tool_call(
        "qdrant_memory_improve_apply",
        {"report_id": report_id, "candidate_id": candidate_id},
    )

    # Live apply
    live = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_improve_apply",
            {"report_id": report_id, "candidate_id": candidate_id, "dry_run": False, "approve": True},
        )
    )

    assert live["dry_run"] is False
    assert live["saved"] is True
    assert live["applied"] is True
    assert live["id"] == target_pid
    # Verify exactly one Qdrant upsert with target_pid
    assert len(provider._qdrant.upserts) == 1
    assert provider._qdrant.upserts[0][0] == "memory"
    upserted_point = provider._qdrant.upserts[0][1][0]
    assert upserted_point["id"] == target_pid


def test_live_apply_removes_candidate_from_pending_after_success(tmp_path):
    provider = _provider_for_improve(tmp_path)
    source = "Graph entity: project: RemoveTest"
    preview = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_improve_preview",
            {"source_text": source, "source_uri": "session://test/remove"},
        )
    )
    report_id = preview["report_id"]
    store_candidate = next(c for c in preview["candidates"] if c["would_store"])
    candidate_id = store_candidate["candidate_id"]

    # Dry-run + live apply
    provider.handle_tool_call(
        "qdrant_memory_improve_apply",
        {"report_id": report_id, "candidate_id": candidate_id},
    )
    provider.handle_tool_call(
        "qdrant_memory_improve_apply",
        {"report_id": report_id, "candidate_id": candidate_id, "dry_run": False, "approve": True},
    )

    # Report should be gone from pending state (no more candidates)
    assert report_id not in provider._pending_improve_reports


# ---------------------------------------------------------------------------
# Low-confidence/unsafe candidate routes to draft or reject
# ---------------------------------------------------------------------------

def test_low_confidence_candidate_routes_to_draft_review_not_store(tmp_path):
    provider = _provider_for_improve(tmp_path)
    # Use "Possible decision:" which creates a 0.5 confidence candidate
    source = "Possible decision: this should be draft review."
    preview = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_improve_preview",
            {"source_text": source, "source_uri": "session://test/low-conf"},
        )
    )

    if preview["counts"]["total"] > 0:
        draft_candidate = next(c for c in preview["candidates"] if c["would_create_proposal"])
        assert draft_candidate is not None
        report_id = preview["report_id"]
        candidate_id = draft_candidate["candidate_id"]

        # Dry-run
        provider.handle_tool_call(
            "qdrant_memory_improve_apply",
            {"report_id": report_id, "candidate_id": candidate_id},
        )

        # Live apply should create draft, not store
        live = json.loads(
            provider.handle_tool_call(
                "qdrant_memory_improve_apply",
                {"report_id": report_id, "candidate_id": candidate_id, "dry_run": False, "approve": True},
            )
        )
        assert live["saved"] is False
        assert live.get("proposal_created") is True
        assert "proposal_draft_path" in live
        assert Path(live["proposal_draft_path"]).exists()
        assert provider._qdrant.upserts == []


# ---------------------------------------------------------------------------
# Secret-bearing source is redacted/rejected
# ---------------------------------------------------------------------------

def test_secret_bearing_source_text_is_not_leaked_in_report(tmp_path):
    provider = _provider_for_improve(tmp_path)
    token_value = "".join(["ghp", "_", "1234567890abcdef", "1234567890abcdef", "123456"])
    secret_line = f"Graph entity: tool: api_key_{token_value}"

    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_improve_preview",
            {"source_text": secret_line, "source_uri": "session://test/secret"},
        )
    )

    # Should have 0 candidates (secret line rejected)
    assert result["counts"]["total"] == 0
    # Token must not appear anywhere in the report
    assert token_value not in json.dumps(result)


def test_secret_bearing_candidate_is_rejected_by_write_gate(tmp_path):
    """A candidate with a secret in payload should be rejected, not stored."""
    from qdrant_memory.extraction_candidates import ExtractionCandidate
    from qdrant_memory.schema import build_payload

    secret_parts = [chr(c) for c in range(97, 97 + 19)]
    token_value = "".join(secret_parts)
    unsafe_payload = build_payload(
        text="Decision: use " + "".join(["Bea", "rer ", token_value]) + " for auth.",
        source="source_extraction",
        source_type="source_extraction",
        memory_kind="decision",
        source_uri="session://test/secret-candidate",
        derived_from=[{"source_uri": "session://test/secret-candidate", "relation_type": "EXTRACTED_FROM"}],
        derivation_type="source_extraction",
    )
    candidate = ExtractionCandidate(
        candidate_id="unsafe-candidate",
        candidate_type="memory_candidate",
        source_uri="session://test/secret-candidate",
        locator={},
        derived_from=[{"source_uri": "session://test/secret-candidate", "relation_type": "EXTRACTED_FROM"}],
        proposed_payload=unsafe_payload,
        reason="unsafe candidate",
        confidence=0.9,
        risk="high",
    )

    report = build_improve_report(
        [candidate],
        profile_id="default",
        session_id="test",
        source_scope="source_text",
        source_handles=["session://test/secret-candidate"],
    )

    # Write gate should reject
    assert report["counts"]["rejected"] == 1
    assert report["counts"]["store_eligible"] == 0
    # Token must not appear in report
    assert token_value not in json.dumps(report)


# ---------------------------------------------------------------------------
# Report/candidate IDs are deterministic/idempotent
# ---------------------------------------------------------------------------

def test_report_id_is_deterministic_for_identical_inputs():
    source = "Graph entity: project: Idempotency"
    uri = "session://test/idempotent"

    candidates1 = extract_improve_candidates_from_text(
        source, source_uri=uri, profile_id="default", lifecycle_id="test",
    )
    candidates2 = extract_improve_candidates_from_text(
        source, source_uri=uri, profile_id="default", lifecycle_id="test",
    )

    report1 = build_improve_report(
        candidates1, profile_id="default", session_id="test",
        source_scope="source_text", source_handles=[uri],
    )
    report2 = build_improve_report(
        candidates2, profile_id="default", session_id="test",
        source_scope="source_text", source_handles=[uri],
    )

    assert report1["report_id"] == report2["report_id"]
    assert report1["report_digest"] == report2["report_digest"]
    for c1, c2 in zip(report1["candidates"], report2["candidates"]):
        assert c1["candidate_id"] == c2["candidate_id"]
        assert c1["candidate_digest"] == c2["candidate_digest"]


def test_report_id_changes_when_candidates_change():
    source1 = "Graph entity: project: ProjectA"
    source2 = "Graph entity: project: ProjectB"
    uri = "session://test/change"

    candidates1 = extract_improve_candidates_from_text(
        source1, source_uri=uri, profile_id="default", lifecycle_id="test",
    )
    candidates2 = extract_improve_candidates_from_text(
        source2, source_uri=uri, profile_id="default", lifecycle_id="test",
    )

    report1 = build_improve_report(
        candidates1, profile_id="default", session_id="test",
        source_scope="source_text", source_handles=[uri],
    )
    report2 = build_improve_report(
        candidates2, profile_id="default", session_id="test",
        source_scope="source_text", source_handles=[uri],
    )

    assert report1["report_id"] != report2["report_id"]


# ---------------------------------------------------------------------------
# Graph edge candidate extraction
# ---------------------------------------------------------------------------

def test_graph_edge_candidate_is_extracted_and_storeable():
    source = "Graph edge: project:TestProject -[DEPENDS_ON]-> tool:TestTool"
    candidates = extract_improve_candidates_from_text(
        source, source_uri="session://test/edge", profile_id="default",
    )

    edge_candidates = [c for c in candidates if c.candidate_type == "graph_edge_candidate"]
    assert len(edge_candidates) == 1
    edge = edge_candidates[0]
    assert edge.proposed_payload["memory_kind"] == "graph_edge"
    assert edge.proposed_payload["relation_type"] == "DEPENDS_ON"
    assert "edge_id" in edge.proposed_payload
    assert edge.proposed_payload["edge_id"].startswith("edge-")


def test_graph_entity_candidate_payload_has_safe_fields():
    source = "Graph entity: concept: Emergence"
    candidates = extract_improve_candidates_from_text(
        source, source_uri="session://test/entity", profile_id="default",
    )

    entity_candidates = [c for c in candidates if c.candidate_type == "graph_entity_candidate"]
    assert len(entity_candidates) == 1
    entity = entity_candidates[0]
    assert entity.proposed_payload["memory_kind"] == "graph_entity"
    assert entity.proposed_payload["canonical"] is False
    assert entity.proposed_payload["requires_review"] is True
    assert "entity_id" in entity.proposed_payload
    assert entity.proposed_payload["entity_id"].startswith("entity-")


# ---------------------------------------------------------------------------
# Session reset clears improve state
# ---------------------------------------------------------------------------

def test_session_reset_clears_improve_state(tmp_path):
    provider = _provider_for_improve(tmp_path)
    source = "Graph entity: project: ResetTest"
    preview = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_improve_preview",
            {"source_text": source, "source_uri": "session://test/reset"},
        )
    )
    report_id = preview["report_id"]
    assert report_id in provider._pending_improve_reports

    # Simulate session reset
    provider.on_session_switch("new-session", reset=True)

    assert report_id not in provider._pending_improve_reports
    assert len(provider._reviewed_improve_candidate_keys) == 0


# ---------------------------------------------------------------------------
# Candidate digest detects stale reports
# ---------------------------------------------------------------------------

def test_candidate_digest_detects_tampering():
    item = {
        "candidate_id": "test-id",
        "candidate_type": "graph_entity_candidate",
        "target_point_id": "entity-abc123",
        "proposed_payload": {"text": "test"},
        "derived_from": [],
        "source_uri": "session://test",
        "locator": {},
        "write_decision": {"decision": "store", "reasons": ["storeable"]},
    }
    digest1 = make_candidate_digest(item)

    tampered = dict(item)
    tampered["proposed_payload"] = {"text": "tampered"}
    digest2 = make_candidate_digest(tampered)

    assert digest1 != digest2
