from __future__ import annotations

import json
from pathlib import Path

import pytest

from __init__ import QdrantMemoryProvider
from qdrant_memory.improve import (
    IMPROVE_EXTRACTOR_VERSION,
    REPORT_ID_RE,
    _check_no_external_graph_deps,
    build_improve_report,
    extract_improve_candidates_from_text,
    is_candidate_applied,
    is_identity_bearing_entity_type,
    is_identity_bearing_graph_candidate,
    is_identity_bearing_value,
    load_improve_report,
    make_candidate_digest,
    persist_improve_report,
    record_candidate_applied,
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
        self.retrieve_results: dict[str, dict] = {}

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

    def retrieve(self, name, ids, *, with_payload=True, with_vector=False):
        results = []
        for pid in ids:
            if pid in self.retrieve_results:
                results.append(self.retrieve_results[pid])
        return results


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

    # Use canonical-format ID that does not exist
    unknown_report = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_improve_apply",
            {"report_id": "improve-aaaaaaaaaaaa", "candidate_id": "fake"},
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


# ===========================================================================
# Blocker 1: Persisted report exact-ID gate
# ===========================================================================

def test_report_id_regex_accepts_canonical():
    assert REPORT_ID_RE.match("improve-82cf21a4bc3f")
    assert REPORT_ID_RE.match("improve-9e065979c8b7")


def test_report_id_regex_rejects_non_canonical():
    # Punctuation
    assert not REPORT_ID_RE.match("improve-82cf21a4bc3f!!!")
    # Path separators
    assert not REPORT_ID_RE.match("improve-82cf/../bc3f")
    # Whitespace
    assert not REPORT_ID_RE.match(" improve-82cf21a4bc3f")
    assert not REPORT_ID_RE.match("improve-82cf21a4bc3f ")
    # Partial ID
    assert not REPORT_ID_RE.match("improve-82cf21a4")
    # Non-hex chars
    assert not REPORT_ID_RE.match("improve-82cf21a4bc3g")
    # Missing prefix
    assert not REPORT_ID_RE.match("82cf21a4bc3f")
    # Empty
    assert not REPORT_ID_RE.match("")


def test_load_improve_report_rejects_non_exact_ids(tmp_path):
    """Non-canonical IDs must return None even if a report artifact exists."""
    report = {
        "report_id": "improve-abcdef123456",
        "report_type": "graph_improve_preview",
        "candidates": [],
        "profile_id": "default",
    }
    hermes_home = str(tmp_path)
    persist_improve_report(report, hermes_home=hermes_home)

    # Canonical ID loads
    loaded = load_improve_report("improve-abcdef123456", hermes_home=hermes_home)
    assert loaded is not None
    assert loaded["report_id"] == "improve-abcdef123456"

    # Non-exact IDs fail to load even though the file exists
    assert load_improve_report("improve-abcdef123456!!!", hermes_home=hermes_home) is None
    assert load_improve_report("improve-abcdef/../123456", hermes_home=hermes_home) is None
    assert load_improve_report("improve-abcdef12345", hermes_home=hermes_home) is None  # partial


def test_apply_rejects_non_canonical_report_id(tmp_path):
    """Live apply must reject non-canonical report IDs at the gate."""
    provider = _provider_for_improve(tmp_path)
    source = "Graph entity: project: NonCanonical"
    preview = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_improve_preview",
            {"source_text": source, "source_uri": "session://test/non-canonical"},
        )
    )
    report_id = preview["report_id"]

    # Aliased/decorated IDs that fail canonical regex
    for alias in [report_id + "!!!", report_id + "/../", report_id + "zzz"]:
        result = json.loads(
            provider.handle_tool_call(
                "qdrant_memory_improve_apply",
                {"report_id": alias, "candidate_id": "fake"},
            )
        )
        assert "error" in result
        assert "canonical" in result["error"]


def test_load_rejects_mismatched_report_id_in_file(tmp_path):
    """Loaded report must have matching report_id or be rejected."""
    import json as _json
    from pathlib import Path as _Path

    # Manually write a file whose report_id doesn't match its filename
    artifact_dir = _Path(str(tmp_path)) / "qdrant_memory" / "improve_reports"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    mismatched = {"report_id": "improve-deadbeef0000", "candidates": []}
    (artifact_dir / "improve-abcdef123456.json").write_text(
        _json.dumps(mismatched), encoding="utf-8"
    )
    loaded = load_improve_report("improve-abcdef123456", hermes_home=str(tmp_path))
    assert loaded is None  # report_id in file doesn't match


# ===========================================================================
# Blocker 2: Repeat apply / idempotency
# ===========================================================================

def test_repeat_live_apply_is_idempotent_after_persisted_reload(tmp_path):
    """After a successful live apply, reloading from disk and re-applying
    must return already_applied without additional upserts."""
    provider = _provider_for_improve(tmp_path)
    source = "Graph entity: project: IdempotencyTest"
    preview = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_improve_preview",
            {"source_text": source, "source_uri": "session://test/idem"},
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
    live1 = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_improve_apply",
            {"report_id": report_id, "candidate_id": candidate_id, "dry_run": False, "approve": True},
        )
    )
    assert live1["saved"] is True
    assert live1["already_applied"] is False
    upserts_after_live1 = len(provider._qdrant.upserts)
    assert upserts_after_live1 == 1

    # Simulate process restart: clear in-memory state, reload from disk
    provider._pending_improve_reports.clear()
    provider._reviewed_improve_candidate_keys.clear()

    # Dry-run again from persisted report, then try to live apply again
    provider.handle_tool_call(
        "qdrant_memory_improve_apply",
        {"report_id": report_id, "candidate_id": candidate_id},
    )
    live2 = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_improve_apply",
            {"report_id": report_id, "candidate_id": candidate_id, "dry_run": False, "approve": True},
        )
    )
    # Must be already_applied, no new upsert
    assert live2.get("already_applied") is True
    assert live2.get("saved") is False
    assert len(provider._qdrant.upserts) == upserts_after_live1  # no new upsert
    assert len(provider._embeddings.documents) == 1  # no new embedding


def test_application_record_persistence_and_check(tmp_path):
    """Application record is persisted on disk and retrievable."""
    hermes_home = str(tmp_path)
    record = record_candidate_applied(
        "improve-abcdef123456", "cand-1",
        hermes_home=hermes_home,
        target_point_id="entity-1",
        candidate_digest="sha256:abc",
    )
    assert record["target_point_id"] == "entity-1"
    loaded = is_candidate_applied("improve-abcdef123456", "cand-1", hermes_home=hermes_home)
    assert loaded is not None
    assert loaded["target_point_id"] == "entity-1"
    # Different candidate has no record
    assert is_candidate_applied("improve-abcdef123456", "cand-2", hermes_home=hermes_home) is None


# ===========================================================================
# Blocker 3: Existing point conflict safety
# ===========================================================================

def test_live_apply_exact_replay_existing_point_is_noop(tmp_path):
    """If target_point_id exists with same candidate/provenance marker,
    apply returns already_applied without embedding/upserting."""
    provider = _provider_for_improve(tmp_path)
    source = "Graph entity: project: ReplayTest"
    preview = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_improve_preview",
            {"source_text": source, "source_uri": "session://test/replay"},
        )
    )
    report_id = preview["report_id"]
    store_candidate = next(c for c in preview["candidates"] if c["would_store"])
    candidate_id = store_candidate["candidate_id"]
    target_pid = store_candidate["target_point_id"]

    # Pre-populate Qdrant with an existing point that has the same markers
    provider._qdrant.retrieve_results[target_pid] = {
        "id": target_pid,
        "payload": {
            "improve_candidate_id": candidate_id,
            "improve_report_id": report_id,
            "text": "project: ReplayTest",
        },
    }

    # Dry-run + live apply
    provider.handle_tool_call(
        "qdrant_memory_improve_apply",
        {"report_id": report_id, "candidate_id": candidate_id},
    )
    live = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_improve_apply",
            {"report_id": report_id, "candidate_id": candidate_id, "dry_run": False, "approve": True},
        )
    )
    assert live.get("already_applied") is True
    assert live.get("saved") is False
    assert len(provider._qdrant.upserts) == 0
    assert len(provider._embeddings.documents) == 0


def test_live_apply_conflicting_existing_point_fails_closed(tmp_path):
    """If target_point_id exists with DIFFERENT payload/provenance,
    apply must fail closed and NOT overwrite."""
    provider = _provider_for_improve(tmp_path)
    source = "Graph entity: project: ConflictTest"
    preview = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_improve_preview",
            {"source_text": source, "source_uri": "session://test/conflict"},
        )
    )
    report_id = preview["report_id"]
    store_candidate = next(c for c in preview["candidates"] if c["would_store"])
    candidate_id = store_candidate["candidate_id"]
    target_pid = store_candidate["target_point_id"]

    # Pre-populate Qdrant with a DIFFERENT existing point (different markers)
    provider._qdrant.retrieve_results[target_pid] = {
        "id": target_pid,
        "payload": {
            "improve_candidate_id": "different-candidate-id",
            "improve_report_id": "improve-000000000000",
            "text": "old data",
        },
    }

    # Dry-run + live apply
    provider.handle_tool_call(
        "qdrant_memory_improve_apply",
        {"report_id": report_id, "candidate_id": candidate_id},
    )
    live = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_improve_apply",
            {"report_id": report_id, "candidate_id": candidate_id, "dry_run": False, "approve": True},
        )
    )
    # Must fail closed
    assert "error" in live
    assert "refusing to overwrite" in live["error"]
    assert len(provider._qdrant.upserts) == 0
    assert len(provider._embeddings.documents) == 0


def test_live_apply_no_conflict_stores_normally(tmp_path):
    """If target_point_id does not exist, apply stores normally."""
    provider = _provider_for_improve(tmp_path)
    source = "Graph entity: project: NoConflict"
    preview = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_improve_preview",
            {"source_text": source, "source_uri": "session://test/no-conflict"},
        )
    )
    report_id = preview["report_id"]
    store_candidate = next(c for c in preview["candidates"] if c["would_store"])
    candidate_id = store_candidate["candidate_id"]

    # No retrieve_results set -> retrieve returns empty -> no conflict

    # Dry-run + live apply
    provider.handle_tool_call(
        "qdrant_memory_improve_apply",
        {"report_id": report_id, "candidate_id": candidate_id},
    )
    live = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_improve_apply",
            {"report_id": report_id, "candidate_id": candidate_id, "dry_run": False, "approve": True},
        )
    )
    assert live["saved"] is True
    assert live["already_applied"] is False
    assert len(provider._qdrant.upserts) == 1


# ===========================================================================
# Blocker 4: Identity-bearing graph inputs
# ===========================================================================

def test_identity_detection_for_emails():
    assert is_identity_bearing_value("alan@example.com")
    assert is_identity_bearing_value("contact: user@domain.org")


def test_identity_detection_for_phone_like():
    assert is_identity_bearing_value("+12345678901")
    assert is_identity_bearing_value("phone: +56 9 8765 4321")


def test_identity_detection_for_identity_keywords():
    assert is_identity_bearing_value("my_email_field")
    assert is_identity_bearing_value("user_phone_number")


def test_identity_detection_for_safe_values():
    assert not is_identity_bearing_value("Nucleogenesis")
    assert not is_identity_bearing_value("project")
    assert not is_identity_bearing_value("DEPENDS_ON")


def test_identity_entity_types():
    for etype in ["person", "user", "customer", "account", "contact", "profile"]:
        assert is_identity_bearing_entity_type(etype), f"{etype} should be identity-bearing"
    assert not is_identity_bearing_entity_type("project")
    assert not is_identity_bearing_entity_type("concept")
    assert not is_identity_bearing_entity_type("tool")


def test_identity_bearing_graph_candidate_detection():
    # Entity with identity type
    assert is_identity_bearing_graph_candidate(entity_type="person", label="John")
    # Entity with email label
    assert is_identity_bearing_graph_candidate(entity_type="account", label="user@domain.com")
    # Safe entity
    assert not is_identity_bearing_graph_candidate(entity_type="project", label="Nucleogenesis")


def test_identity_entity_rejected_in_preview(tmp_path):
    """Graph entity with person/user/etc. type must not appear in preview."""
    provider = _provider_for_improve(tmp_path)
    source = "\n".join([
        "Graph entity: person: Alan Gárate",
        "Graph entity: account: alan@example.com",
        "Graph entity: project: SafeProject",
    ])
    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_improve_preview",
            {"source_text": source, "source_uri": "session://test/identity"},
        )
    )
    report_json = json.dumps(result)

    # Identity strings must NOT appear anywhere in the report
    assert "alan@example.com" not in report_json
    assert "Alan Gárate" not in report_json

    # The safe entity should still be present
    labels = []
    for c in result["candidates"]:
        payload = c.get("proposed_payload") or {}
        labels.append(payload.get("label") or payload.get("text") or "")
    assert any("SafeProject" in l for l in labels), "Safe entity should survive"
    assert not any("Alan" in l or "alan@" in l for l in labels)


def test_identity_email_in_label_rejected_in_preview(tmp_path):
    """Graph entity label containing email must not appear in preview."""
    provider = _provider_for_improve(tmp_path)
    source = "Graph entity: contact: support@company.com"
    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_improve_preview",
            {"source_text": source, "source_uri": "session://test/email"},
        )
    )
    assert result["counts"]["total"] == 0
    assert "support@company.com" not in json.dumps(result)


def test_identity_phone_in_label_rejected_in_preview(tmp_path):
    """Graph entity label containing phone-like value must not appear in preview."""
    provider = _provider_for_improve(tmp_path)
    source = "Graph entity: account: +15551234567"
    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_improve_preview",
            {"source_text": source, "source_uri": "session://test/phone"},
        )
    )
    assert result["counts"]["total"] == 0
    assert "+15551234567" not in json.dumps(result)


def test_identity_bearing_entity_not_in_persisted_report(tmp_path):
    """Identity strings must not appear in the persisted report JSON file."""
    provider = _provider_for_improve(tmp_path)
    source = "\n".join([
        "Graph entity: user: john.doe@email.com",
        "Graph entity: project: SafeEntity",
    ])
    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_improve_preview",
            {"source_text": source, "source_uri": "session://test/persist-id"},
        )
    )
    report_id = result["report_id"]
    # Load persisted report from disk
    loaded = load_improve_report(report_id, hermes_home=provider._hermes_home)
    assert loaded is not None
    loaded_json = json.dumps(loaded)
    assert "john.doe@email.com" not in loaded_json
    # Safe entity should be present
    assert "SafeEntity" in loaded_json


def test_identity_entity_in_edge_rejected_in_preview(tmp_path):
    """Graph edge with identity-bearing entity type must not appear in preview."""
    provider = _provider_for_improve(tmp_path)
    source = "Graph edge: person:John Doe -[WORKS_AT]-> company:AcmeCorp"
    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_improve_preview",
            {"source_text": source, "source_uri": "session://test/edge-identity"},
        )
    )
    # Should have 0 candidates since the edge has person type
    edge_candidates = [
        c for c in result["candidates"]
        if c.get("candidate_type") == "graph_edge_candidate"
    ]
    assert len(edge_candidates) == 0
    assert "John Doe" not in json.dumps(result)


def test_identity_rejected_at_write_gate():
    """Write gate must reject identity-bearing graph candidates even if
    they somehow reach it via persisted payload."""
    from qdrant_memory.extraction_candidates import ExtractionCandidate
    from qdrant_memory.write_gate import evaluate_extraction_candidate_write

    candidate = ExtractionCandidate(
        candidate_id="test-id",
        candidate_type="graph_entity_candidate",
        source_uri="session://test",
        locator={},
        derived_from=[{"source_uri": "session://test", "relation_type": "EXTRACTED_FROM"}],
        proposed_payload={
            "text": "user@example.com",
            "label": "user@example.com",
            "entity_type": "account",
            "memory_kind": "graph_entity",
            "source_type": "graph",
            "source_uri": "session://test",
            "derivation_type": "source_extraction",
        },
        reason="test",
        confidence=0.9,
        risk="low",
    )
    decision = evaluate_extraction_candidate_write(candidate)
    assert decision.decision == "reject"
    assert "identity_bearing_graph_candidate" in decision.reasons
