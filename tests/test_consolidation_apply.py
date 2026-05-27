from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from qdrant_memory.config import load_config
from qdrant_memory.tools import TOOL_SCHEMAS


def _provider(tmp_path):
    from __init__ import QdrantMemoryProvider

    provider = QdrantMemoryProvider()
    provider._active = True
    provider._hermes_home = str(tmp_path)
    provider._profile_id = "architect"
    provider._platform = "cli"
    provider._session_id = "s1"
    provider._config = load_config(hermes_home=str(tmp_path), hermes_config={})
    provider._config["collection_name"] = "memory"
    provider._config["learning_collection_name"] = "learnings"
    return provider


class FakeQdrant:
    def __init__(self, by_collection=None):
        self.by_collection = by_collection or {}
        self.scrolls = []
        self.upserts = []
        self.payload_updates = []
        self.deleted_ids = []
        self.deleted_filters = []
        self.searches = []
        self.ensure_calls = []

    def scroll_by_filter(self, name, filter, *, limit=256, with_payload=True, with_vector=False, max_total=None):
        self.scrolls.append(
            {
                "name": name,
                "filter": filter,
                "limit": limit,
                "with_payload": with_payload,
                "with_vector": with_vector,
                "max_total": max_total,
            }
        )
        points = self.by_collection.get(name, [])
        return points[:max_total] if max_total is not None else points

    def retrieve(self, name, ids, *, with_payload=True, with_vector=False):
        wanted = {str(item) for item in ids}
        return [p for p in self.by_collection.get(name, []) if str(p.get("id")) in wanted]

    def upsert(self, name, points):
        self.upserts.append((name, points))

    def update_payload(self, name, point_id, payload):
        self.payload_updates.append((name, point_id, payload))

    def delete_ids(self, name, ids):
        self.deleted_ids.append((name, ids))

    def delete_filter(self, name, filter):
        self.deleted_filters.append((name, filter))

    def ensure_collection(self, name, vector_size, distance):
        self.ensure_calls.append((name, vector_size, distance))

    def collection_vector_size(self, name):
        return 2

    def collection_info(self, name):
        return {"config": {"params": {"vectors": {"size": 2, "distance": "Cosine"}}}}

    def search(self, *args, **kwargs):
        self.searches.append((args, kwargs))
        return []


def _point(point_id, text, vector=None, **payload):
    return {"id": point_id, "vector": vector or [0.1, 0.2], "payload": {"text": text, **payload}}


def _persist_duplicate_report(provider):
    result = json.loads(provider.handle_tool_call("qdrant_memory_consolidate", {"scope": "memory", "persist": True}))
    proposal = next(p for p in result["proposals"] if p["proposal_type"] == "duplicate_cluster")
    return result, proposal


def _persist_stale_report(provider):
    result = json.loads(provider.handle_tool_call("qdrant_memory_consolidate", {"scope": "memory", "persist": True}))
    proposal = next(p for p in result["proposals"] if p["proposal_type"] == "stale_low_value")
    return result, proposal


def _persist_promotion_report(provider):
    result = json.loads(provider.handle_tool_call("qdrant_memory_consolidate", {"scope": "learning", "persist": True}))
    proposal = next(p for p in result["proposals"] if p["proposal_type"] == "learning_promotion_candidate")
    return result, proposal


def test_consolidation_apply_schema_is_exposed():
    schemas = {schema["name"]: schema for schema in TOOL_SCHEMAS}

    assert "qdrant_memory_consolidation_apply" in schemas
    apply_schema = schemas["qdrant_memory_consolidation_apply"]
    assert apply_schema["parameters"]["additionalProperties"] is False
    assert "report_id" in apply_schema["parameters"]["required"]
    assert "proposal_id" in apply_schema["parameters"]["required"]
    assert apply_schema["parameters"]["properties"]["action"]["enum"] == ["merge", "delete", "quarantine", "promote_to_skill", "draft_review"]


def test_consolidate_persists_report_artifact_with_report_id(tmp_path):
    provider = _provider(tmp_path)
    provider._qdrant = FakeQdrant(
        {
            "memory": [
                _point("m1", "Always dry-run before live vault indexing", source_type="manual"),
                _point("m2", "Always dry-run before live vault indexing", source_type="conversation"),
            ],
            "learnings": [],
        }
    )

    result = json.loads(provider.handle_tool_call("qdrant_memory_consolidate", {"scope": "memory", "persist": True}))

    assert result["persisted"] is True
    assert result["report_id"]
    artifact_path = tmp_path / "qdrant_memory" / "consolidation" / f"report-{result['report_id']}.json"
    assert artifact_path.exists()
    persisted = json.loads(artifact_path.read_text())
    assert persisted["report_id"] == result["report_id"]
    assert persisted["proposals"][0]["proposal_id"]
    assert persisted["profile_id"] == "architect"
    assert provider._qdrant.upserts == []
    assert provider._qdrant.deleted_ids == []
    assert provider._qdrant.deleted_filters == []


def test_persisted_report_redacts_secret_examples(tmp_path):
    provider = _provider(tmp_path)
    fake_secret = "raw-" + "secret-" + "token"
    provider._qdrant = FakeQdrant({"memory": [_point("m1", "Authorization: " + "Bearer " + fake_secret, source_type="manual")], "learnings": []})

    result_text = provider.handle_tool_call("qdrant_memory_consolidate", {"scope": "memory", "include_examples": True, "persist": True})
    result = json.loads(result_text)
    artifact_path = tmp_path / "qdrant_memory" / "consolidation" / f"report-{result['report_id']}.json"
    artifact_text = artifact_path.read_text()

    assert fake_secret not in result_text
    assert "Bearer" not in result_text
    assert fake_secret not in artifact_text
    assert "Bearer" not in artifact_text


def test_apply_requires_proposal_id(tmp_path):
    provider = _provider(tmp_path)
    provider._qdrant = FakeQdrant()

    result = json.loads(provider.handle_tool_call("qdrant_memory_consolidation_apply", {}))

    assert "error" in result
    assert "proposal_id" in result["error"]
    assert provider._qdrant.deleted_ids == []


def test_apply_dry_run_returns_plan_without_mutation(tmp_path):
    provider = _provider(tmp_path)
    old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    provider._qdrant = FakeQdrant({"memory": [_point("m1", "old weak memory", source_type="conversation", importance=1, confidence=0.3, access_count=0, created_at=old)], "learnings": []})
    report, proposal = _persist_stale_report(provider)

    result = json.loads(provider.handle_tool_call("qdrant_memory_consolidation_apply", {"report_id": report["report_id"], "proposal_id": proposal["proposal_id"], "action": "delete"}))

    assert result["dry_run"] is True
    assert result["would_apply"] is True
    assert result["action"] == "delete"
    assert result["affected_ids"] == ["m1"]
    assert provider._qdrant.deleted_ids == []
    assert provider._qdrant.upserts == []
    assert provider._qdrant.payload_updates == []


def test_apply_live_requires_approve_true(tmp_path):
    provider = _provider(tmp_path)
    old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    provider._qdrant = FakeQdrant({"memory": [_point("m1", "old weak memory", source_type="conversation", importance=1, confidence=0.3, access_count=0, created_at=old)], "learnings": []})
    report, proposal = _persist_stale_report(provider)

    result = json.loads(provider.handle_tool_call("qdrant_memory_consolidation_apply", {"report_id": report["report_id"], "proposal_id": proposal["proposal_id"], "action": "delete", "dry_run": "false"}))

    assert "error" in result
    assert "approve" in result["error"]
    assert provider._qdrant.deleted_ids == []


def test_apply_rejects_action_mismatch(tmp_path):
    provider = _provider(tmp_path)
    old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    provider._qdrant = FakeQdrant({"memory": [_point("m1", "old weak memory", source_type="conversation", importance=1, confidence=0.3, access_count=0, created_at=old)], "learnings": []})
    report, proposal = _persist_stale_report(provider)

    result = json.loads(provider.handle_tool_call("qdrant_memory_consolidation_apply", {"report_id": report["report_id"], "proposal_id": proposal["proposal_id"], "action": "merge", "dry_run": False, "approve": True}))

    assert "error" in result
    assert "mismatch" in result["error"]
    assert provider._qdrant.deleted_ids == []


def test_apply_quarantine_marks_stale_low_value_without_deleting(tmp_path):
    provider = _provider(tmp_path)
    old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    provider._qdrant = FakeQdrant({"memory": [_point("m1", "old weak memory", source_type="conversation", importance=1, confidence=0.3, access_count=0, created_at=old)], "learnings": []})
    report, proposal = _persist_stale_report(provider)

    result = json.loads(provider.handle_tool_call("qdrant_memory_consolidation_apply", {"report_id": report["report_id"], "proposal_id": proposal["proposal_id"], "action": "quarantine", "dry_run": False, "approve": True, "quarantine_days": 14}))

    assert result["applied"] is True
    assert result["action"] == "quarantine"
    assert result["quarantined_ids"] == ["m1"]
    assert provider._qdrant.deleted_ids == []
    assert provider._qdrant.deleted_filters == []
    assert provider._qdrant.payload_updates[0][0:2] == ("memory", "m1")
    payload = provider._qdrant.payload_updates[0][2]
    assert payload["consolidation_quarantined"] is True
    assert payload["consolidation_quarantine_reason"] == "guarded-auto stale_low_value"
    assert payload["consolidation_proposal_id"] == proposal["proposal_id"]
    assert result["application_artifact"]


def test_apply_delete_live_deletes_only_explicit_ids(tmp_path):
    provider = _provider(tmp_path)
    old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    provider._qdrant = FakeQdrant({"memory": [_point("m1", "old weak memory", source_type="conversation", importance=1, confidence=0.3, access_count=0, created_at=old)], "learnings": []})
    report, proposal = _persist_stale_report(provider)

    result = json.loads(provider.handle_tool_call("qdrant_memory_consolidation_apply", {"report_id": report["report_id"], "proposal_id": proposal["proposal_id"], "action": "delete", "dry_run": "false", "approve": True}))

    assert result["applied"] is True
    assert result["action"] == "delete"
    assert result["deleted_ids"] == ["m1"]
    assert provider._qdrant.deleted_ids == [("memory", ["m1"])]
    assert provider._qdrant.deleted_filters == []
    applications = list((tmp_path / "qdrant_memory" / "consolidation" / "applications").glob("*.json"))
    assert applications


def test_apply_merge_live_updates_canonical_then_deletes_duplicates(tmp_path):
    provider = _provider(tmp_path)
    provider._qdrant = FakeQdrant(
        {
            "memory": [
                _point("m1", "Always dry-run before live vault indexing", source_type="manual", importance=5, confidence=0.8),
                _point("m2", "Always dry-run before live vault indexing", source_type="conversation", importance=9, confidence=0.7),
            ],
            "learnings": [],
        }
    )
    report, proposal = _persist_duplicate_report(provider)

    result = json.loads(provider.handle_tool_call("qdrant_memory_consolidation_apply", {"report_id": report["report_id"], "proposal_id": proposal["proposal_id"], "action": "merge", "dry_run": False, "approve": True}))

    assert result["applied"] is True
    assert result["action"] == "merge"
    assert result["canonical_id"] == "m2"
    assert result["deleted_ids"] == ["m1"]
    assert provider._qdrant.payload_updates[0][0:2] == ("memory", "m2")
    assert provider._qdrant.payload_updates[0][2]["consolidated_from"] == ["m1"]
    assert provider._qdrant.deleted_ids == [("memory", ["m1"])]
    assert provider._qdrant.deleted_filters == []


def test_apply_refuses_quality_warning_manual_only(tmp_path):
    provider = _provider(tmp_path)
    secret_like = " ".join(["Authorization:", "Bearer", "".join(["abc", "def", "ghi", "jkl", "mnop"])])
    provider._qdrant = FakeQdrant({"memory": [_point("m1", f'{secret_like} source_type="manual')], "learnings": []})
    result = json.loads(provider.handle_tool_call("qdrant_memory_consolidate", {"scope": "memory", "persist": True}))
    proposal = next(p for p in result["proposals"] if p["proposal_type"] == "quality_warning")

    applied = json.loads(provider.handle_tool_call("qdrant_memory_consolidation_apply", {"report_id": result["report_id"], "proposal_id": proposal["proposal_id"], "action": "delete", "dry_run": False, "approve": True}))

    assert "error" in applied
    assert "manual" in applied["error"]
    assert provider._qdrant.deleted_ids == []


def test_apply_draft_review_creates_neutral_proposal_draft(tmp_path):
    from qdrant_memory.consolidation import persist_consolidation_report

    provider = _provider(tmp_path)
    provider._qdrant = FakeQdrant(
        {
            "memory": [
                _point("m1", "Alan prefers concise status summaries", source_type="manual", confidence=0.9, importance=8),
                _point("m2", "Alan prefers detailed status summaries", source_type="conversation", confidence=0.7, importance=5),
            ],
            "learnings": [],
        }
    )
    report = persist_consolidation_report(
        {
            "dry_run": True,
            "report_only": True,
            "scope": "memory",
            "profile_id": "architect",
            "proposals": [
                {
                    "proposal_id": "recon-review",
                    "proposal_type": "reconsolidation_candidate",
                    "collection_name": "memory",
                    "affected_ids": ["m1", "m2"],
                    "suggested_action": "reconsolidate_review_only",
                    "risk": "high",
                    "confidence": 0.8,
                    "manual_review_required": True,
                }
            ],
        },
        hermes_home=str(tmp_path),
    )

    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_consolidation_apply",
            {"report_id": report["report_id"], "proposal_id": "recon-review", "action": "draft_review", "dry_run": False, "approve": True},
        )
    )

    assert result["applied"] is True
    assert result["proposal_draft_path"]
    assert result["reconsolidation_draft_path"] == result["proposal_draft_path"]
    assert result["write_decision"]["decision"] == "draft_review"
    draft_path = tmp_path / "qdrant_memory" / "proposals" / result["proposal_draft_path"].split("/")[-1]
    assert draft_path.exists()
    text = draft_path.read_text()
    assert "recon-review" in text
    assert "m1" in text and "m2" in text
    assert provider._qdrant.payload_updates == []
    assert provider._qdrant.deleted_ids == []


def test_apply_promote_live_creates_skill_draft_and_marks_learning(tmp_path):
    provider = _provider(tmp_path)
    provider._qdrant = FakeQdrant(
        {
            "memory": [],
            "learnings": [
                _point(
                    "l1",
                    "Always run pytest after changing consolidation actions",
                    source_type="learning",
                    learning_type="workflow_lesson",
                    trigger="consolidation change",
                    correction="run pytest tests -q",
                    evidence="75 passed",
                    confidence=0.95,
                    importance=9,
                    promote_to_skill_candidate=True,
                )
            ],
        }
    )
    report, proposal = _persist_promotion_report(provider)

    result = json.loads(provider.handle_tool_call("qdrant_memory_consolidation_apply", {"report_id": report["report_id"], "proposal_id": proposal["proposal_id"], "action": "promote_to_skill", "dry_run": False, "approve": True}))

    assert result["applied"] is True
    assert result["action"] == "promote_to_skill"
    assert result["proposal_draft_path"]
    assert result["skill_draft_path"] == result["proposal_draft_path"]
    draft_path = tmp_path / "qdrant_memory" / "proposals" / result["proposal_draft_path"].split("/")[-1]
    assert draft_path.exists()
    assert "Always run pytest" in draft_path.read_text()
    assert result["write_decision"]["decision"] == "skill_candidate"
    assert provider._qdrant.payload_updates[0][0:2] == ("learnings", "l1")
    update = provider._qdrant.payload_updates[0][2]
    assert update["promoted_to_skill_draft"] is True
    assert update["proposal_draft_path"] == result["proposal_draft_path"]
    assert update["skill_draft_path"] == result["skill_draft_path"]
    assert provider._qdrant.deleted_ids == []


def test_apply_promote_refuses_secret_bearing_learning_even_with_persisted_proposal(tmp_path):
    from qdrant_memory.consolidation import persist_consolidation_report

    provider = _provider(tmp_path)
    secret_like = "".join(["s", "k", "-", "abcdefghijklmnopqrstuvwxyz"])
    provider._qdrant = FakeQdrant(
        {
            "memory": [],
            "learnings": [
                _point(
                    "l-secret",
                    f"Never persist this credential-shaped value {secret_like}",
                    source_type="learning",
                    learning_type="workflow_lesson",
                    confidence=0.99,
                    importance=10,
                    promote_to_skill_candidate=True,
                )
            ],
        }
    )
    report = persist_consolidation_report(
        {
            "dry_run": True,
            "report_only": True,
            "scope": "learning",
            "profile_id": "architect",
            "proposals": [
                {
                    "proposal_id": "learning-secret",
                    "proposal_type": "learning_promotion_candidate",
                    "collection_name": "learnings",
                    "affected_ids": ["l-secret"],
                    "suggested_action": "promote_to_skill_review_only",
                    "risk": "low",
                    "confidence": 0.99,
                    "guarded_auto_eligible": True,
                    "preauthorized_policy": "guarded-auto:learning-skill-draft",
                }
            ],
        },
        hermes_home=str(tmp_path),
    )

    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_consolidation_apply",
            {
                "report_id": report["report_id"],
                "proposal_id": "learning-secret",
                "action": "promote_to_skill",
                "dry_run": False,
                "approve": True,
            },
        )
    )

    assert "error" in result
    assert "secret-bearing" in result["error"]
    assert provider._qdrant.payload_updates == []
    assert provider._qdrant.deleted_ids == []
    assert list((tmp_path / "qdrant_memory" / "consolidation" / "skill_drafts").glob("*.md")) == []


def test_apply_guarded_auto_preauthorized_maintenance_refuses_sensitive_points(tmp_path):
    from qdrant_memory.consolidation import persist_consolidation_report

    provider = _provider(tmp_path)
    provider._qdrant = FakeQdrant(
        {
            "memory": [
                _point("u1", "Alan prefers explicit reboot warnings", source_type="user_profile"),
                _point("u2", "Alan prefers explicit reboot warnings", source_type="user_profile"),
            ],
            "learnings": [],
        }
    )
    report = persist_consolidation_report(
        {
            "dry_run": True,
            "report_only": True,
            "scope": "memory",
            "profile_id": "architect",
            "proposals": [
                {
                    "proposal_id": "profile-duplicate",
                    "proposal_type": "duplicate_cluster",
                    "collection_name": "memory",
                    "affected_ids": ["u1", "u2"],
                    "suggested_action": "merge_review_only",
                    "risk": "low",
                    "confidence": 0.99,
                    "match_kind": "exact_normalized",
                    "guarded_auto_eligible": True,
                    "preauthorized_policy": "guarded-auto:exact-duplicate-merge",
                }
            ],
        },
        hermes_home=str(tmp_path),
    )

    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_consolidation_apply",
            {"report_id": report["report_id"], "proposal_id": "profile-duplicate", "action": "merge", "dry_run": False, "approve": True},
        )
    )

    assert "error" in result
    assert "profile or fact-like" in result["error"]
    assert provider._qdrant.payload_updates == []
    assert provider._qdrant.deleted_ids == []

    secret_like = "".join(["s", "k", "-", "abcdefghijklmnopqrstuvwxyz"])
    provider = _provider(tmp_path)
    provider._qdrant = FakeQdrant({"memory": [_point("s1", f"stale weak {secret_like}", source_type="conversation")], "learnings": []})
    report = persist_consolidation_report(
        {
            "dry_run": True,
            "report_only": True,
            "scope": "memory",
            "profile_id": "architect",
            "proposals": [
                {
                    "proposal_id": "secret-stale",
                    "proposal_type": "stale_low_value",
                    "collection_name": "memory",
                    "affected_ids": ["s1"],
                    "suggested_action": "quarantine_guarded_auto_eligible",
                    "risk": "low",
                    "confidence": 0.75,
                    "guarded_auto_eligible": True,
                    "preauthorized_policy": "guarded-auto:stale-low-value-quarantine",
                }
            ],
        },
        hermes_home=str(tmp_path),
    )

    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_consolidation_apply",
            {"report_id": report["report_id"], "proposal_id": "secret-stale", "action": "quarantine", "dry_run": False, "approve": True},
        )
    )

    assert "error" in result
    assert "secret-bearing" in result["error"]
    assert provider._qdrant.payload_updates == []
    assert provider._qdrant.deleted_ids == []


def test_apply_backup_first_creates_backup_before_live_mutation(tmp_path):
    provider = _provider(tmp_path)
    old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    provider._qdrant = FakeQdrant(
        {"memory": [_point("m1", "old weak memory for backup-first", source_type="conversation", importance=1, confidence=0.3, access_count=0, created_at=old)], "learnings": []}
    )
    report, proposal = _persist_stale_report(provider)

    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_consolidation_apply",
            {
                "report_id": report["report_id"],
                "proposal_id": proposal["proposal_id"],
                "action": "delete",
                "dry_run": False,
                "approve": True,
                "backup_first": True,
            },
        )
    )

    assert result["applied"] is True
    assert result["pre_apply_backup_id"]
    backup_dir = tmp_path / "qdrant_memory" / "backups" / result["pre_apply_backup_id"]
    assert (backup_dir / "manifest.json").exists()
    assert (backup_dir / "memory.jsonl").exists()
    assert provider._qdrant.deleted_ids == [("memory", ["m1"])]
    assert provider._qdrant.deleted_filters == []
    assert provider._qdrant.ensure_calls == []
