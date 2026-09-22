from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

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


def _payload_value(payload, key):
    value = payload
    for part in key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _lineage_filter_matches(payload, filter_value):
    for clause in filter_value.get("must", []):
        if "is_empty" in clause:
            if _payload_value(payload, clause["is_empty"]["key"]) not in (None, "", [], {}):
                return False
            continue
        if "nested" in clause:
            nested = clause["nested"]
            items = _payload_value(payload, nested["key"])
            if not isinstance(items, list) or not any(
                isinstance(item, dict) and _lineage_filter_matches(item, nested["filter"])
                for item in items
            ):
                return False
            continue
        actual = _payload_value(payload, clause["key"])
        match = clause.get("match", {})
        if "value" in match and actual != match["value"]:
            return False
        if "any" in match and actual not in match["any"]:
            return False
    for clause in filter_value.get("must_not", []):
        actual = _payload_value(payload, clause["key"])
        match = clause.get("match", {})
        if "value" in match and actual == match["value"]:
            return False
        if "any" in match and actual in match["any"]:
            return False
    return True


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
        must = filter.get("must", [])
        if any("nested" in clause or clause.get("key") == "target_point_id" for clause in must):
            points = [
                point for point in points
                if _lineage_filter_matches(point.get("payload") or {}, filter)
            ]
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


class FakeEmbedding:
    def embed_document(self, _text):
        return [0.3, 0.4]


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


def _report_artifact(tmp_path, report_id):
    return tmp_path / "qdrant_memory" / "consolidation" / f"report-{report_id}.json"


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
    assert result["lineage_fence"] == {
        "mode": "off", "status": "clear", "complete": True,
        "blocked": False, "dependent_count": 0,
    }
    assert proposal["lineage_fence"] == result["lineage_fence"]
    persisted = json.loads(_report_artifact(tmp_path, report["report_id"]).read_text())
    persisted_proposal = next(
        item for item in persisted["proposals"]
        if item["proposal_id"] == proposal["proposal_id"]
    )
    assert persisted_proposal["lineage_fence"] == result["lineage_fence"]
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
    assert proposal["guarded_auto_eligible"] is True
    assert proposal["preauthorized_policy"] == "guarded-auto:learning-skill-draft"
    assert proposal["guarded_auto_snapshot"]["point_digests"]
    assert proposal["guarded_auto_proposal_sha256"]

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


def test_apply_promote_fact_like_learning_is_manual_review_only_but_human_apply_still_drafts(tmp_path):
    provider = _provider(tmp_path)
    provider._qdrant = FakeQdrant(
        {
            "memory": [],
            "learnings": [
                _point(
                    "l-fact",
                    "Always run pytest after changing consolidation actions",
                    source_type="learning",
                    learning_type="workflow_lesson",
                    fact_key="workflow.consolidation",
                    confidence=0.95,
                    importance=9,
                    promote_to_skill_candidate=True,
                )
            ],
        }
    )
    report, proposal = _persist_promotion_report(provider)

    assert proposal["guarded_auto_eligible"] is False
    assert proposal["manual_review_required"] is True
    assert proposal["manual_review_reason"] == "profile or fact-like memory requires manual review"
    assert "preauthorized_policy" not in proposal
    assert "guarded_auto_snapshot" not in proposal
    assert "guarded_auto_proposal_sha256" not in proposal

    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_consolidation_apply",
            {
                "report_id": report["report_id"],
                "proposal_id": proposal["proposal_id"],
                "action": "promote_to_skill",
                "dry_run": False,
                "approve": True,
            },
        )
    )

    assert result["applied"] is True
    assert result["action"] == "promote_to_skill"
    assert result["skill_draft_path"]
    assert result["write_decision"]["decision"] == "skill_candidate"
    assert provider._qdrant.payload_updates[0][0:2] == ("learnings", "l-fact")


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


def test_guarded_auto_duplicate_rejects_point_changed_after_report(tmp_path):
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
    assert proposal["guarded_auto_snapshot"]["point_digests"]
    provider._qdrant.by_collection["memory"][1]["payload"]["text"] = "Changed after report generation"

    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_consolidation_apply",
            {"report_id": report["report_id"], "proposal_id": proposal["proposal_id"], "action": "merge", "dry_run": False, "approve": True},
        )
    )

    assert "error" in result
    assert "fresh report" in result["error"]
    assert provider._qdrant.payload_updates == []
    assert provider._qdrant.deleted_ids == []


def test_manual_lineage_impact_tampering_is_digest_bound(tmp_path):
    provider = _provider(tmp_path)
    provider._config["lineage_mode"] = "reconcile"
    provider._qdrant = FakeQdrant(
        {
            "memory": [
                _point("m1", "same", source_type="manual", fact_key="profile.one"),
                _point("m2", "same", source_type="conversation", fact_key="profile.two"),
            ],
            "learnings": [],
        }
    )
    report, proposal = _persist_duplicate_report(provider)
    assert proposal.get("guarded_auto_proposal_sha256") is None
    assert proposal["lineage_impact_proposal_sha256"]
    artifact = _report_artifact(tmp_path, report["report_id"])
    persisted = json.loads(artifact.read_text(encoding="utf-8"))
    persisted_proposal = next(item for item in persisted["proposals"] if item["proposal_id"] == proposal["proposal_id"])
    persisted_proposal["lineage_impact"]["complete"] = True
    persisted_proposal["lineage_impact"]["errors"] = []
    artifact.write_text(json.dumps(persisted), encoding="utf-8")

    result = json.loads(provider.handle_tool_call(
        "qdrant_memory_consolidation_apply",
        {
            "report_id": report["report_id"], "proposal_id": proposal["proposal_id"],
            "action": "merge", "dry_run": False, "approve": True,
        },
    ))

    assert "lineage impact proposal digest changed" in result["error"]
    assert provider._qdrant.payload_updates == []
    assert provider._qdrant.deleted_ids == []


@pytest.mark.parametrize("mode", ["off", "capture"])
@pytest.mark.parametrize("action", ["merge", "delete", "quarantine"])
def test_destructive_consolidation_refuses_lineage_dependents_outside_reconcile(mode, action, tmp_path):
    provider = _provider(tmp_path)
    provider._config["lineage_mode"] = mode
    provider._config["lineage_lock_dir"] = str(tmp_path / "locks")
    if action == "merge":
        roots = [
            _point(
                "m1", "same", source_type="manual", importance=5, confidence=0.8,
            ),
            _point("m2", "same", source_type="conversation", importance=9, confidence=0.7),
        ]
        provider._qdrant = FakeQdrant({"memory": roots, "learnings": []})
        report, proposal = _persist_duplicate_report(provider)
    else:
        old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
        roots = [
            _point(
                "m1", "old weak memory", source_type="conversation", importance=1,
                confidence=0.3, access_count=0, created_at=old,
            )
        ]
        provider._qdrant = FakeQdrant({"memory": roots, "learnings": []})
        report, proposal = _persist_stale_report(provider)
    dependent = _point(
        "dependent", "derived memory", source_type="conversation",
        profile_id=provider._profile_id, user_id_hash=provider._user_id_hash,
        chat_id_hash=provider._chat_id_hash,
    )
    edge = _point(
        "dependency-edge", "", profile_id=provider._profile_id,
        user_id_hash=provider._user_id_hash, chat_id_hash=provider._chat_id_hash,
        memory_kind="graph_edge", source_point_id="dependent", target_point_id="m1",
        relation_type="DERIVED_FROM",
    )
    assert not {
        "derived_from", "file_version_id", "lineage_entity_id", "lineage_participates",
        "lineage_review_event_ids", "lineage_schema_version", "lineage_scope_key",
        "lineage_source_key", "session_origin",
    } & set(roots[0]["payload"])
    provider._qdrant.by_collection["memory"].extend([dependent, edge])
    before = copy.deepcopy(provider._qdrant.by_collection)

    result = json.loads(provider.handle_tool_call(
        "qdrant_memory_consolidation_apply",
        {
            "report_id": report["report_id"], "proposal_id": proposal["proposal_id"],
            "action": action, "dry_run": False, "approve": True, "backup_first": True,
        },
    ))

    assert result["error"] == "lineage dependents block destructive consolidation"
    assert provider._qdrant.by_collection == before
    assert provider._qdrant.upserts == []
    assert provider._qdrant.payload_updates == []
    assert provider._qdrant.deleted_ids == []
    assert not any(
        (point.get("payload") or {}).get("lineage_role") == "change_event"
        or (point.get("payload") or {}).get("lineage_retired_by_event_id")
        for point in provider._qdrant.by_collection["memory"]
    )
    assert not (tmp_path / "qdrant_memory" / "consolidation" / "applications").exists()
    assert not (tmp_path / "qdrant_memory" / "backups").exists()


def test_destructive_consolidation_refuses_when_dependency_lookup_fails(tmp_path):
    provider = _provider(tmp_path)
    provider._config["lineage_mode"] = "capture"
    provider._config["lineage_lock_dir"] = str(tmp_path / "locks")
    old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()

    class FailingLineageLookup(FakeQdrant):
        def scroll_by_filter(self, name, filter, **kwargs):
            if any(clause.get("key") == "target_point_id" for clause in filter.get("must", [])):
                raise RuntimeError("lookup unavailable")
            return super().scroll_by_filter(name, filter, **kwargs)

    provider._qdrant = FailingLineageLookup(
        {
            "memory": [
                _point(
                    "m1", "old weak memory", source_type="conversation", importance=1,
                    confidence=0.3, access_count=0, created_at=old,
                )
            ],
            "learnings": [],
        }
    )
    report, proposal = _persist_stale_report(provider)
    before = copy.deepcopy(provider._qdrant.by_collection)

    result = json.loads(provider.handle_tool_call(
        "qdrant_memory_consolidation_apply",
        {
            "report_id": report["report_id"], "proposal_id": proposal["proposal_id"],
            "action": "delete", "dry_run": False, "approve": True,
        },
    ))

    assert result["error"] == "lineage dependency fence lookup failed"
    assert provider._qdrant.by_collection == before
    assert provider._qdrant.payload_updates == []
    assert provider._qdrant.deleted_ids == []
    assert provider._qdrant.upserts == []
    assert not any(
        (point.get("payload") or {}).get("lineage_role") == "change_event"
        or (point.get("payload") or {}).get("lineage_retired_by_event_id")
        for point in provider._qdrant.by_collection["memory"]
    )
    assert not (tmp_path / "qdrant_memory" / "consolidation" / "applications").exists()


def test_destructive_consolidation_refuses_when_dependency_lookup_is_incomplete(tmp_path):
    provider = _provider(tmp_path)
    provider._config["lineage_mode"] = "off"
    provider._config["lineage_lock_dir"] = str(tmp_path / "locks")
    old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()

    class IncompleteLineageLookup(FakeQdrant):
        def scroll_page(self, name, filter, **kwargs):
            if any(clause.get("key") == "target_point_id" for clause in filter.get("must", [])):
                return [], "more"
            return [], None

    provider._qdrant = IncompleteLineageLookup(
        {
            "memory": [
                _point(
                    "m1", "old weak memory", source_type="conversation", importance=1,
                    confidence=0.3, access_count=0, created_at=old,
                )
            ],
            "learnings": [],
        }
    )
    report, proposal = _persist_stale_report(provider)
    before = copy.deepcopy(provider._qdrant.by_collection)

    result = json.loads(provider.handle_tool_call(
        "qdrant_memory_consolidation_apply",
        {
            "report_id": report["report_id"], "proposal_id": proposal["proposal_id"],
            "action": "delete", "dry_run": False, "approve": True,
        },
    ))

    assert result["error"] == "lineage dependency fence is incomplete"
    assert result["lineage_fence"]["status"] == "incomplete"
    assert provider._qdrant.by_collection == before
    assert provider._qdrant.payload_updates == []
    assert provider._qdrant.deleted_ids == []
    assert provider._qdrant.upserts == []
    assert not (tmp_path / "qdrant_memory" / "consolidation" / "applications").exists()


@pytest.mark.parametrize("mode", ["off", "capture"])
def test_destructive_consolidation_without_dependents_completes_after_lineage_check(mode, tmp_path):
    provider = _provider(tmp_path)
    provider._config["lineage_mode"] = mode
    provider._config["lineage_lock_dir"] = str(tmp_path / "locks")
    provider._qdrant = FakeQdrant(
        {
            "memory": [
                _point("m1", "same", source_type="manual", importance=5, confidence=0.8),
                _point("m2", "same", source_type="conversation", importance=9, confidence=0.7),
            ],
            "learnings": [],
        }
    )
    report, proposal = _persist_duplicate_report(provider)
    before = copy.deepcopy(provider._qdrant.by_collection)
    reads_before_apply = len(provider._qdrant.scrolls)

    result = json.loads(provider.handle_tool_call(
        "qdrant_memory_consolidation_apply",
        {
            "report_id": report["report_id"], "proposal_id": proposal["proposal_id"],
            "action": "merge", "dry_run": False, "approve": True,
        },
    ))

    assert result["applied"] is True
    assert result["canonical_id"] == "m2"
    assert result["deleted_ids"] == ["m1"]
    target_clauses = [
        clause
        for scroll in provider._qdrant.scrolls[reads_before_apply:]
        for clause in scroll["filter"].get("must", [])
        if clause.get("key") == "target_point_id"
    ]
    assert any(set(clause["match"].get("any", [])) == {"m1", "m2"} for clause in target_clauses)
    assert provider._qdrant.by_collection == before
    assert provider._qdrant.payload_updates[0][0:2] == ("memory", "m2")
    assert provider._qdrant.deleted_ids == [("memory", ["m1"])]


def test_destructive_consolidation_report_and_dry_run_expose_blocked_fence(tmp_path):
    provider = _provider(tmp_path)
    provider._config["lineage_mode"] = "capture"
    provider._config["lineage_lock_dir"] = str(tmp_path / "locks")
    old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    provider._qdrant = FakeQdrant({
        "memory": [
            _point(
                "m1", "old weak memory", source_type="conversation", importance=1,
                confidence=0.3, access_count=0, created_at=old,
            ),
            _point(
                "dependent", "derived memory", source_type="conversation",
                profile_id=provider._profile_id, user_id_hash=provider._user_id_hash,
                chat_id_hash=provider._chat_id_hash,
            ),
            _point(
                "foreign-edge-id", "", profile_id=provider._profile_id,
                user_id_hash=provider._user_id_hash, chat_id_hash="foreign-chat",
                memory_kind="graph_edge", source_point_id="dependent",
                target_point_id="m1", relation_type="DERIVED_FROM",
            ),
        ],
        "learnings": [],
    })

    report, proposal = _persist_stale_report(provider)
    persisted = json.loads(_report_artifact(tmp_path, report["report_id"]).read_text())
    persisted_proposal = next(
        item for item in persisted["proposals"]
        if item["proposal_id"] == proposal["proposal_id"]
    )
    result_text = provider.handle_tool_call(
        "qdrant_memory_consolidation_apply",
        {"report_id": report["report_id"], "proposal_id": proposal["proposal_id"], "action": "delete"},
    )
    result = json.loads(result_text)

    assert proposal["lineage_fence"]["status"] == "incomplete"
    assert persisted_proposal["lineage_fence"] == proposal["lineage_fence"]
    assert result["error"] == "lineage dependency fence is incomplete"
    assert result["lineage_fence"]["status"] == "incomplete"
    assert "foreign-edge-id" not in result_text
    assert provider._qdrant.deleted_ids == []
    assert provider._qdrant.payload_updates == []
    assert provider._qdrant.upserts == []


def test_consolidation_lock_contention_is_distinguishable_and_respects_timeout(tmp_path):
    from qdrant_memory.lineage import collection_write_lock

    provider = _provider(tmp_path)
    provider._config["lineage_mode"] = "off"
    provider._config["lineage_lock_dir"] = str(tmp_path / "locks")
    provider._config["lineage_lock_timeout_seconds"] = 0.15
    old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    provider._qdrant = FakeQdrant({
        "memory": [_point(
            "m1", "old weak memory", source_type="conversation", importance=1,
            confidence=0.3, access_count=0, created_at=old,
        )],
        "learnings": [],
    })
    report, proposal = _persist_stale_report(provider)
    before = copy.deepcopy(provider._qdrant.by_collection)
    acquired = threading.Event()
    release = threading.Event()

    def hold_lock():
        with collection_write_lock(
            collection_name="memory", timeout=1.0,
            lock_dir=provider._config["lineage_lock_dir"],
        ):
            acquired.set()
            assert release.wait(2.0)

    thread = threading.Thread(target=hold_lock)
    thread.start()
    assert acquired.wait(1.0)
    started = time.monotonic()
    try:
        result = json.loads(provider.handle_tool_call(
            "qdrant_memory_consolidation_apply",
            {
                "report_id": report["report_id"], "proposal_id": proposal["proposal_id"],
                "action": "delete", "dry_run": False, "approve": True,
            },
        ))
    finally:
        elapsed = time.monotonic() - started
        release.set()
        thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert result == {"error": "lineage collection lock unavailable"}
    assert elapsed >= 0.10
    assert provider._qdrant.by_collection == before
    assert provider._qdrant.deleted_ids == []
    assert provider._qdrant.payload_updates == []
    assert provider._qdrant.upserts == []


def test_consolidation_rereads_roots_under_lock_before_mutation(tmp_path):
    class InjectDependentOnLockedReread(FakeQdrant):
        armed = False
        armed_retrieves = 0
        injected = False

        def retrieve(self, name, ids, *, with_payload=True, with_vector=False):
            if self.armed:
                self.armed_retrieves += 1
                if self.armed_retrieves == 2:
                    self.injected = True
                    self.by_collection["memory"].extend([
                        _point(
                            "late-dependent", "late derived memory", source_type="conversation",
                            profile_id=provider._profile_id, user_id_hash=provider._user_id_hash,
                            chat_id_hash=provider._chat_id_hash,
                        ),
                        _point(
                            "late-edge", "", profile_id=provider._profile_id,
                            user_id_hash=provider._user_id_hash, chat_id_hash=provider._chat_id_hash,
                            memory_kind="graph_edge", source_point_id="late-dependent",
                            target_point_id="m1", relation_type="DERIVED_FROM",
                        ),
                    ])
            return super().retrieve(name, ids, with_payload=with_payload, with_vector=with_vector)

    provider = _provider(tmp_path)
    provider._config["lineage_mode"] = "capture"
    provider._config["lineage_lock_dir"] = str(tmp_path / "locks")
    old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    provider._qdrant = InjectDependentOnLockedReread({
        "memory": [_point(
            "m1", "old weak memory", source_type="conversation", importance=1,
            confidence=0.3, access_count=0, created_at=old,
        )],
        "learnings": [],
    })
    report, proposal = _persist_stale_report(provider)
    provider._qdrant.armed = True

    result = json.loads(provider.handle_tool_call(
        "qdrant_memory_consolidation_apply",
        {
            "report_id": report["report_id"], "proposal_id": proposal["proposal_id"],
            "action": "delete", "dry_run": False, "approve": True,
        },
    ))

    assert result["error"] == "lineage dependents block destructive consolidation"
    assert provider._qdrant.injected is True
    assert provider._qdrant.deleted_ids == []
    assert provider._qdrant.payload_updates == []
    assert provider._qdrant.upserts == []
    assert not (tmp_path / "qdrant_memory" / "consolidation" / "applications").exists()


def test_consolidation_apply_holds_lock_while_fence_lookup_is_paused(tmp_path):
    """The apply path must acquire the collection write lock BEFORE it computes
    the lineage dependency fence. While the fence's direct-dependent graph scan
    is paused, the configured lock is already held, so no competing writer can
    land a DERIVED_FROM edge between the fence verdict and the destructive
    mutation. Computing the fence first leaves that window open (G1)."""
    from qdrant_memory.lineage import collection_write_lock

    lookup_started = threading.Event()
    continue_lookup = threading.Event()

    class PausingDependencyLookup(FakeQdrant):
        armed = False

        def scroll_by_filter(self, name, filter, **kwargs):
            if self.armed and any(
                clause.get("key") == "target_point_id" for clause in filter.get("must", [])
            ):
                lookup_started.set()
                assert continue_lookup.wait(2.0)
            return super().scroll_by_filter(name, filter, **kwargs)

    provider = _provider(tmp_path)
    provider._config["lineage_mode"] = "off"
    provider._config["lineage_lock_dir"] = str(tmp_path / "locks")
    old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    provider._qdrant = FakeQdrant({
        "memory": [_point(
            "m1", "old weak memory", source_type="conversation", importance=1,
            confidence=0.3, access_count=0, created_at=old,
        )],
        "learnings": [],
    })
    # Report generation runs the fence too; arm the pause only for the apply call.
    report, proposal = _persist_stale_report(provider)

    provider._qdrant = PausingDependencyLookup({
        "memory": [
            _point(
                "m1", "old weak memory", source_type="conversation", importance=1,
                confidence=0.3, access_count=0, created_at=old,
            ),
            _point(
                "dependent", "derived memory", source_type="conversation",
                profile_id=provider._profile_id, user_id_hash=provider._user_id_hash,
                chat_id_hash=provider._chat_id_hash,
            ),
            _point(
                "dependency-edge", "", profile_id=provider._profile_id,
                user_id_hash=provider._user_id_hash, chat_id_hash=provider._chat_id_hash,
                memory_kind="graph_edge", source_point_id="dependent",
                target_point_id="m1", relation_type="DERIVED_FROM",
            ),
        ],
        "learnings": [],
    })
    provider._qdrant.armed = True
    before = copy.deepcopy(provider._qdrant.by_collection)
    outcome = {}

    def apply():
        outcome["result"] = json.loads(provider.handle_tool_call(
            "qdrant_memory_consolidation_apply",
            {
                "report_id": report["report_id"], "proposal_id": proposal["proposal_id"],
                "action": "delete", "dry_run": False, "approve": True,
            },
        ))

    thread = threading.Thread(target=apply)
    thread.start()
    assert lookup_started.wait(1.0)
    try:
        with pytest.raises(TimeoutError, match="timed out"):
            with collection_write_lock(
                collection_name="memory", timeout=0,
                lock_dir=provider._config["lineage_lock_dir"],
            ):
                pass
    finally:
        continue_lookup.set()
        thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert outcome["result"]["error"] == "lineage dependents block destructive consolidation"
    assert outcome["result"]["lineage_fence"]["blocked"] is True
    assert provider._qdrant.by_collection == before
    assert provider._qdrant.deleted_ids == []
    assert provider._qdrant.deleted_filters == []
    assert provider._qdrant.payload_updates == []
    assert provider._qdrant.upserts == []


def test_forget_lock_contention_refuses_without_any_write_and_respects_timeout(tmp_path):
    from qdrant_memory.lineage import collection_write_lock

    provider = _provider(tmp_path)
    provider._config["lineage_mode"] = "off"
    provider._config["lineage_lock_dir"] = str(tmp_path / "locks")
    provider._config["lineage_lock_timeout_seconds"] = 0.15
    provider._qdrant = FakeQdrant({
        "memory": [_point("root", "must survive", source_type="manual")],
        "learnings": [],
    })
    before = json.dumps(
        provider._qdrant.by_collection, sort_keys=True, separators=(",", ":"),
    ).encode()
    acquired = threading.Event()
    release = threading.Event()

    def hold_lock():
        with collection_write_lock(
            collection_name="memory", timeout=1.0,
            lock_dir=provider._config["lineage_lock_dir"],
        ):
            acquired.set()
            assert release.wait(2.0)

    thread = threading.Thread(target=hold_lock)
    thread.start()
    assert acquired.wait(1.0)
    started = time.monotonic()
    try:
        result = json.loads(provider.handle_tool_call(
            "qdrant_memory_forget", {"ids": ["root"], "dry_run": False},
        ))
    finally:
        elapsed = time.monotonic() - started
        release.set()
        thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert result == {"error": "lineage collection lock unavailable"}
    assert elapsed >= 0.10
    assert json.dumps(
        provider._qdrant.by_collection, sort_keys=True, separators=(",", ":"),
    ).encode() == before
    assert provider._qdrant.deleted_ids == []
    assert provider._qdrant.deleted_filters == []
    assert provider._qdrant.payload_updates == []
    assert provider._qdrant.upserts == []


def test_forget_holds_configured_lock_and_releases_it_after_refusal(tmp_path):
    from qdrant_memory.lineage import collection_write_lock

    lookup_started = threading.Event()
    continue_lookup = threading.Event()

    class PausingDependencyLookup(FakeQdrant):
        def scroll_by_filter(self, name, filter, **kwargs):
            if any(clause.get("key") == "target_point_id" for clause in filter.get("must", [])):
                lookup_started.set()
                assert continue_lookup.wait(2.0)
            return super().scroll_by_filter(name, filter, **kwargs)

    provider = _provider(tmp_path)
    provider._config["lineage_mode"] = "off"
    provider._config["lineage_lock_dir"] = str(tmp_path / "locks")
    provider._qdrant = PausingDependencyLookup({
        "memory": [
            _point("root", "root", source_type="manual"),
            _point(
                "dependent", "derived", source_type="conversation",
                profile_id=provider._profile_id, user_id_hash=provider._user_id_hash,
                chat_id_hash=provider._chat_id_hash,
            ),
            _point(
                "dependency-edge", "", profile_id=provider._profile_id,
                user_id_hash=provider._user_id_hash, chat_id_hash=provider._chat_id_hash,
                memory_kind="graph_edge", source_point_id="dependent",
                target_point_id="root", relation_type="DERIVED_FROM",
            ),
        ],
        "learnings": [],
    })
    before = copy.deepcopy(provider._qdrant.by_collection)
    outcome = {}

    def forget():
        outcome["result"] = json.loads(provider.handle_tool_call(
            "qdrant_memory_forget", {"ids": ["root"], "dry_run": False},
        ))

    thread = threading.Thread(target=forget)
    thread.start()
    assert lookup_started.wait(1.0)
    try:
        with pytest.raises(TimeoutError, match="timed out"):
            with collection_write_lock(
                collection_name="memory", timeout=0,
                lock_dir=provider._config["lineage_lock_dir"],
            ):
                pass
    finally:
        continue_lookup.set()
        thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert outcome["result"] == {"error": "lineage dependents block forget"}
    assert provider._qdrant.by_collection == before
    assert provider._qdrant.deleted_ids == []
    assert provider._qdrant.deleted_filters == []
    assert provider._qdrant.payload_updates == []
    assert provider._qdrant.upserts == []

    reacquired = threading.Event()

    def acquire_after_refusal():
        with collection_write_lock(
            collection_name="memory", timeout=1.0,
            lock_dir=provider._config["lineage_lock_dir"],
        ):
            reacquired.set()

    later = threading.Thread(target=acquire_after_refusal)
    later.start()
    later.join(timeout=2.0)
    assert not later.is_alive()
    assert reacquired.is_set()


def _invoke_writer_while_configured_lock_is_held(provider, invoke):
    from qdrant_memory.lineage import collection_write_lock

    before = json.dumps(
        provider._qdrant.by_collection, sort_keys=True, separators=(",", ":"),
    ).encode()
    acquired = threading.Event()
    release = threading.Event()

    def hold_lock():
        with collection_write_lock(
            collection_name="memory", timeout=1.0,
            lock_dir=provider._config["lineage_lock_dir"],
        ):
            acquired.set()
            assert release.wait(2.0)

    thread = threading.Thread(target=hold_lock)
    thread.start()
    assert acquired.wait(1.0)
    started = time.monotonic()
    try:
        result = json.loads(invoke())
    finally:
        elapsed = time.monotonic() - started
        release.set()
        thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert json.dumps(
        provider._qdrant.by_collection, sort_keys=True, separators=(",", ":"),
    ).encode() == before
    assert provider._qdrant.upserts == []
    assert elapsed >= 0.10
    return result


def test_extraction_approval_uses_configured_lineage_upsert_lock(tmp_path):
    from qdrant_memory.source_extraction import extract_source_candidates_from_text

    provider = _provider(tmp_path)
    provider._qdrant = FakeQdrant({"memory": [], "learnings": []})
    provider._embeddings = FakeEmbedding()
    provider._config.update({
        "source_extraction_enabled": True,
        "source_extraction_mode": "preview",
        "source_extraction_min_confidence": 0.65,
        "source_extraction_max_candidates_per_session": 8,
        "lineage_lock_dir": str(tmp_path / "writer-locks"),
        "lineage_lock_timeout_seconds": 0.15,
    })
    candidate = extract_source_candidates_from_text(
        "Decision: keep extraction approvals behind the configured collection lock.",
        source_uri="session://lineage-lock/extraction",
    )[0]
    provider._pending_extraction_candidates[candidate.candidate_id] = candidate
    reviewed = json.loads(provider.handle_tool_call(
        "qdrant_memory_extraction_approve", {"candidate_id": candidate.candidate_id},
    ))
    assert reviewed["dry_run"] is True

    result = _invoke_writer_while_configured_lock_is_held(
        provider,
        lambda: provider.handle_tool_call(
            "qdrant_memory_extraction_approve",
            {"candidate_id": candidate.candidate_id, "dry_run": False, "approve": True},
        ),
    )

    assert result == {
        "error": "Extraction approval failed: lineage collection lock unavailable",
    }
    assert candidate.candidate_id in provider._pending_extraction_candidates


def test_improve_apply_uses_configured_lineage_upsert_lock(tmp_path):
    provider = _provider(tmp_path)
    provider._qdrant = FakeQdrant({"memory": [], "learnings": []})
    provider._embeddings = FakeEmbedding()
    provider._config.update({
        "lineage_lock_dir": str(tmp_path / "writer-locks"),
        "lineage_lock_timeout_seconds": 0.15,
    })
    preview = json.loads(provider.handle_tool_call(
        "qdrant_memory_improve_preview",
        {
            "source_text": "Graph entity: project: LockedImprove",
            "source_uri": "session://lineage-lock/improve",
        },
    ))
    candidate = next(item for item in preview["candidates"] if item["would_store"])
    reviewed = json.loads(provider.handle_tool_call(
        "qdrant_memory_improve_apply",
        {"report_id": preview["report_id"], "candidate_id": candidate["candidate_id"]},
    ))
    assert reviewed["dry_run"] is True

    result = _invoke_writer_while_configured_lock_is_held(
        provider,
        lambda: provider.handle_tool_call(
            "qdrant_memory_improve_apply",
            {
                "report_id": preview["report_id"],
                "candidate_id": candidate["candidate_id"],
                "dry_run": False,
                "approve": True,
            },
        ),
    )

    assert result == {
        "error": "Improve apply failed: lineage collection lock unavailable",
    }


def test_raptor_apply_uses_configured_lineage_upsert_lock(tmp_path):
    from qdrant_memory.raptor import RaptorBuilder
    from qdrant_memory.raptor.apply import persist_manifest_report, wrap_manifest

    provider = _provider(tmp_path)
    provider._profile_id = "default"
    provider._qdrant = FakeQdrant({"memory": [], "learnings": []})
    provider._embeddings = FakeEmbedding()
    provider._config.update({
        "embedding_model": "test-model",
        "lineage_lock_dir": str(tmp_path / "writer-locks"),
        "lineage_lock_timeout_seconds": 0.15,
    })
    manifest = RaptorBuilder().build([
        _point(
            "leaf-1", "alpha note about RAPTOR", source="test://leaf1",
            source_uri="test://leaf1", source_type="manual", profile_id="default",
        ),
        _point(
            "leaf-2", "beta note about RAPTOR", source="test://leaf2",
            source_uri="test://leaf2", source_type="manual", profile_id="default",
        ),
    ]).to_dict()
    report_id = f"raptor-{manifest['manifest_digest'][:12]}"
    wrapper = wrap_manifest(manifest, report_id=report_id)
    persist_manifest_report(wrapper, hermes_home=provider._hermes_home)
    provider._pending_raptor_manifests[report_id] = wrapper
    args = {
        "report_id": report_id,
        "build_id": manifest["build_id"],
        "manifest_digest": manifest["manifest_digest"],
    }
    reviewed = json.loads(provider.handle_tool_call("qdrant_memory_raptor_apply", args))
    assert reviewed["dry_run"] is True

    result = _invoke_writer_while_configured_lock_is_held(
        provider,
        lambda: provider.handle_tool_call(
            "qdrant_memory_raptor_apply", {**args, "dry_run": False, "approve": True},
        ),
    )

    assert result == {
        "error": "RAPTOR apply failed: lineage collection lock unavailable",
    }


@pytest.mark.parametrize("mode", ["off", "capture", "reconcile"])
def test_forget_refuses_any_dependent_and_deletes_all_clear_targets(mode, tmp_path):
    class MutableFakeQdrant(FakeQdrant):
        def delete_ids(self, name, ids):
            super().delete_ids(name, ids)
            wanted = {str(point_id) for point_id in ids}
            self.by_collection[name] = [
                point for point in self.by_collection.get(name, [])
                if str(point.get("id")) not in wanted
            ]

    def points(with_dependency):
        values = [
            _point("root", "root without lineage markers", source_type="manual"),
            _point("second-root", "second root", source_type="manual"),
            _point("unrelated", "must survive", source_type="manual"),
        ]
        if with_dependency:
            values.extend([
                _point(
                    "dependent", "derived memory", source_type="conversation",
                    profile_id=provider._profile_id, user_id_hash=provider._user_id_hash,
                    chat_id_hash=provider._chat_id_hash,
                ),
                _point(
                    "dependency-edge", "", profile_id=provider._profile_id,
                    user_id_hash=provider._user_id_hash, chat_id_hash=provider._chat_id_hash,
                    memory_kind="graph_edge", source_point_id="dependent",
                    # Deliberately NOT the first affected id: the fence must
                    # inspect every named target, so a lookup that reads only
                    # root ids[:1] has to miss this dependent.
                    target_point_id="second-root", relation_type="DERIVED_FROM",
                ),
            ])
        return {"memory": values, "learnings": []}

    provider = _provider(tmp_path)
    provider._config["lineage_mode"] = mode
    provider._config["lineage_lock_dir"] = str(tmp_path / "locks")
    provider._qdrant = MutableFakeQdrant(points(with_dependency=True))
    before = copy.deepcopy(provider._qdrant.by_collection)

    refused = json.loads(provider.handle_tool_call(
        "qdrant_memory_forget",
        {"ids": ["root", "second-root"], "dry_run": False},
    ))

    assert refused == {"error": "lineage dependents block forget"}
    assert provider._qdrant.by_collection == before
    assert provider._qdrant.deleted_ids == []
    assert provider._qdrant.payload_updates == []
    assert provider._qdrant.upserts == []

    provider._qdrant = MutableFakeQdrant(points(with_dependency=False))
    expected = copy.deepcopy(provider._qdrant.by_collection)
    expected["memory"] = [
        point for point in expected["memory"]
        if point["id"] not in {"root", "second-root"}
    ]
    deleted = json.loads(provider.handle_tool_call(
        "qdrant_memory_forget",
        {"ids": ["root", "second-root"], "dry_run": False},
    ))

    assert deleted == {
        "dry_run": False, "ids": ["root", "second-root"], "deleted": 2,
    }
    assert provider._qdrant.by_collection == expected
    assert provider._qdrant.deleted_ids == [("memory", ["root", "second-root"])]
    assert provider._qdrant.payload_updates == []
    assert provider._qdrant.upserts == []


@pytest.mark.parametrize(
    ("failure", "expected_error"),
    [
        ("raise", "lineage dependency fence lookup failed"),
        ("incomplete", "lineage dependency fence is incomplete"),
    ],
)
def test_forget_lookup_failure_refuses_without_any_write(failure, expected_error, tmp_path):
    class FailingForgetLookup(FakeQdrant):
        def scroll_by_filter(self, name, filter, **kwargs):
            if any(clause.get("key") == "target_point_id" for clause in filter.get("must", [])):
                if failure == "raise":
                    raise RuntimeError("private lookup detail")
            return super().scroll_by_filter(name, filter, **kwargs)

        def scroll_page(self, name, filter, **kwargs):
            is_dependency_lookup = any(
                clause.get("key") == "target_point_id" for clause in filter.get("must", [])
            )
            if failure == "raise" and is_dependency_lookup:
                raise RuntimeError("private lookup detail")
            if failure == "incomplete" and is_dependency_lookup:
                return [], "more"
            return [], None

    provider = _provider(tmp_path)
    provider._config["lineage_mode"] = "off"
    provider._config["lineage_lock_dir"] = str(tmp_path / "locks")
    provider._qdrant = FailingForgetLookup({
        "memory": [_point("root", "root", source_type="manual")],
        "learnings": [],
    })
    before = copy.deepcopy(provider._qdrant.by_collection)

    result_text = provider.handle_tool_call(
        "qdrant_memory_forget", {"ids": ["root"], "dry_run": False},
    )

    assert json.loads(result_text) == {"error": expected_error}
    assert "private lookup detail" not in result_text
    assert provider._qdrant.by_collection == before
    assert provider._qdrant.deleted_ids == []
    assert provider._qdrant.payload_updates == []
    assert provider._qdrant.upserts == []


def test_consolidation_apply_redacts_unexpected_exception_text(tmp_path):
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
    report, proposal = _persist_duplicate_report(provider)
    sentinel = "credential-" + "must-not-escape"

    class RaisingRetrieveQdrant(FakeQdrant):
        def retrieve(self, _name, _ids, *, with_payload=True, with_vector=False):
            raise RuntimeError(f"provider path /private/location leaked {sentinel}")

    provider._qdrant = RaisingRetrieveQdrant(provider._qdrant.by_collection)
    result_text = provider.handle_tool_call(
        "qdrant_memory_consolidation_apply",
        {"report_id": report["report_id"], "proposal_id": proposal["proposal_id"], "action": "merge", "dry_run": False, "approve": True},
    )

    assert sentinel not in result_text
    assert "/private/location" not in result_text
    assert json.loads(result_text) == {"error": "consolidation_apply_failed"}


def test_guarded_auto_rechecks_points_immediately_before_mutation(tmp_path):
    class RacingFakeQdrant(FakeQdrant):
        def __init__(self, by_collection=None):
            super().__init__(by_collection)
            self.armed = False
            self.retrieve_count = 0

        def retrieve(self, name, ids, *, with_payload=True, with_vector=False):
            if self.armed:
                self.retrieve_count += 1
                if self.retrieve_count == 2:
                    self.by_collection["memory"][1]["payload"]["text"] = "Changed inside the apply window"
            return super().retrieve(name, ids, with_payload=with_payload, with_vector=with_vector)

    provider = _provider(tmp_path)
    provider._qdrant = RacingFakeQdrant(
        {
            "memory": [
                _point("m1", "Always dry-run before live vault indexing", source_type="manual"),
                _point("m2", "Always dry-run before live vault indexing", source_type="conversation"),
            ],
            "learnings": [],
        }
    )
    report, proposal = _persist_duplicate_report(provider)
    provider._qdrant.armed = True

    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_consolidation_apply",
            {"report_id": report["report_id"], "proposal_id": proposal["proposal_id"], "action": "merge", "dry_run": False, "approve": True},
        )
    )

    assert "error" in result
    assert "fresh report" in result["error"]
    assert provider._qdrant.retrieve_count == 2
    assert provider._qdrant.payload_updates == []
    assert provider._qdrant.deleted_ids == []


def test_guarded_auto_heading_rejects_changed_fingerprint_after_report(tmp_path):
    provider = _provider(tmp_path)
    provider._qdrant = FakeQdrant({"memory": [_point("h1", "# Tareas", source_type="conversation")], "learnings": []})
    report = json.loads(provider.handle_tool_call("qdrant_memory_consolidate", {"scope": "memory", "persist": True}))
    proposal = next(p for p in report["proposals"] if p["proposal_type"] == "heading_noise")
    assert proposal["guarded_auto_eligible"] is True
    provider._qdrant.by_collection["memory"][0]["payload"]["text"] = "# Project Phoenix"

    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_consolidation_apply",
            {"report_id": report["report_id"], "proposal_id": proposal["proposal_id"], "action": "delete", "dry_run": False, "approve": True},
        )
    )

    assert "error" in result
    assert "fresh report" in result["error"]
    assert provider._qdrant.deleted_ids == []


def test_guarded_auto_stale_rejects_current_point_that_is_now_accessed(tmp_path):
    provider = _provider(tmp_path)
    old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    provider._qdrant = FakeQdrant(
        {"memory": [_point("m1", "old weak memory", source_type="conversation", importance=1, confidence=0.3, access_count=0, created_at=old)], "learnings": []}
    )
    report, proposal = _persist_stale_report(provider)
    provider._qdrant.by_collection["memory"][0]["payload"]["access_count"] = 1

    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_consolidation_apply",
            {"report_id": report["report_id"], "proposal_id": proposal["proposal_id"], "action": "quarantine", "dry_run": False, "approve": True},
        )
    )

    assert "error" in result
    assert "fresh report" in result["error"]
    assert provider._qdrant.payload_updates == []
    assert provider._qdrant.deleted_ids == []


def test_guarded_auto_rejects_tampered_proposal_metadata(tmp_path):
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
    report, proposal = _persist_duplicate_report(provider)
    artifact = _report_artifact(tmp_path, report["report_id"])
    persisted = json.loads(artifact.read_text())
    persisted_proposal = next(p for p in persisted["proposals"] if p["proposal_id"] == proposal["proposal_id"])
    persisted_proposal["risk"] = "medium"
    artifact.write_text(json.dumps(persisted))

    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_consolidation_apply",
            {"report_id": report["report_id"], "proposal_id": proposal["proposal_id"], "action": "merge", "dry_run": False, "approve": True},
        )
    )

    assert "error" in result
    assert "report metadata changed" in result["error"]
    assert provider._qdrant.payload_updates == []
    assert provider._qdrant.deleted_ids == []


def test_guarded_auto_caller_rejects_removed_preauthorization_metadata(tmp_path):
    from qdrant_memory.guarded_auto import GuardedAutoPolicy, apply_guarded_auto

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
    report, proposal = _persist_duplicate_report(provider)
    artifact = _report_artifact(tmp_path, report["report_id"])
    persisted = json.loads(artifact.read_text())
    persisted_proposal = next(p for p in persisted["proposals"] if p["proposal_id"] == proposal["proposal_id"])
    persisted_proposal.pop("preauthorized_policy")
    artifact.write_text(json.dumps(persisted))

    summary = apply_guarded_auto(provider, report, GuardedAutoPolicy(mode="guarded-auto"))

    assert summary["applied"] == []
    assert summary["errors"][0]["code"] == "provider_rejected"
    assert provider._qdrant.payload_updates == []
    assert provider._qdrant.deleted_ids == []


def test_guarded_auto_rejects_tampered_report_identity(tmp_path):
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
    report, proposal = _persist_duplicate_report(provider)
    artifact = _report_artifact(tmp_path, report["report_id"])
    persisted = json.loads(artifact.read_text())
    persisted["report_id"] = "different-report"
    artifact.write_text(json.dumps(persisted))

    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_consolidation_apply",
            {"report_id": report["report_id"], "proposal_id": proposal["proposal_id"], "action": "merge", "dry_run": False, "approve": True},
        )
    )

    assert "error" in result
    assert "report metadata changed" in result["error"]
    assert provider._qdrant.payload_updates == []
    assert provider._qdrant.deleted_ids == []


def test_guarded_auto_quarantine_is_idempotent_against_same_report(tmp_path):
    class StatefulFakeQdrant(FakeQdrant):
        def update_payload(self, name, point_id, payload):
            super().update_payload(name, point_id, payload)
            for point in self.by_collection.get(name, []):
                if str(point.get("id")) == str(point_id):
                    point["payload"].update(payload)

    provider = _provider(tmp_path)
    old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    provider._qdrant = StatefulFakeQdrant(
        {"memory": [_point("m1", "old weak memory", source_type="conversation", importance=1, confidence=0.3, access_count=0, created_at=old)], "learnings": []}
    )
    report, proposal = _persist_stale_report(provider)
    args = {"report_id": report["report_id"], "proposal_id": proposal["proposal_id"], "action": "quarantine", "dry_run": False, "approve": True}

    first = json.loads(provider.handle_tool_call("qdrant_memory_consolidation_apply", args))
    second = json.loads(provider.handle_tool_call("qdrant_memory_consolidation_apply", args))

    assert first["applied"] is True
    assert "error" in second
    assert "fresh report" in second["error"]
    assert len(provider._qdrant.payload_updates) == 1

# ===========================================================================
# W0: provider apply refuses proposals selecting structural records
# ===========================================================================

def test_apply_refuses_structural_lineage_points_selected_by_old_proposal(tmp_path):
    """Structural lineage records retrieved by exact ID must never reach the
    apply plan or any mutation, even when an old proposal selects them."""
    provider = _provider(tmp_path)
    report_payload = {
        "report_id": "consolidation-aaaaaaaaaaaa",
        "report_type": "consolidation_report",
        "profile_id": "architect",
        "proposals": [
            {
                "proposal_id": "proposal-1",
                "proposal_type": "heading_noise",
                "collection_name": "memory",
                "affected_ids": ["structural-1"],
                "suggested_action": "delete_review_only",
                "confidence": 0.7,
                "risk": "medium",
                "evidence": [{"id": "structural-1", "reason": "heading noise"}],
                "requires_explicit_approval": True,
            }
        ],
    }
    artifact = _report_artifact(tmp_path, "consolidation-aaaaaaaaaaaa")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(report_payload), encoding="utf-8")
    structural_point = {
        "id": "structural-1",
        "payload": {"text": "file-source-abc123", "lineage_record": True},
    }
    provider._qdrant = FakeQdrant(by_collection={"memory": [structural_point]})

    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_consolidation_apply",
            {
                "report_id": "consolidation-aaaaaaaaaaaa",
                "proposal_id": "proposal-1",
                "dry_run": True,
            },
        )
    )
    assert "error" in result
    assert "structural lineage" in result["error"]
    assert provider._qdrant.upserts == []
    assert provider._qdrant.deleted_ids == []
    assert provider._qdrant.payload_updates == []


# ===========================================================================
# W2 delta-4 review: G3 one lock-refusal contract, G5 caller-agnostic
# structural refusal, G6 profile-scoped fence, G7 no out-of-scope ids
# ===========================================================================

LINEAGE_LOCK_FIXED_TEXT = "lineage collection lock unavailable"


def test_all_lineage_lock_refusals_share_one_fixed_inner_text(tmp_path):
    """G3: fence, forget and the locked writers report the identical refusal.

    The fence and forget return the fixed text bare; the locked-upsert writers
    keep their operation prefix but must not invent a second lock text.
    ``qdrant_memory_learning_approve`` is deliberately absent: it calls
    ``LearningStore.store`` directly and never takes the lineage lock, so it has
    no lock refusal to normalise (verified: ``_lineage_locked_upsert`` has
    exactly three call sites).
    """
    from qdrant_memory.source_extraction import extract_source_candidates_from_text
    from qdrant_memory.raptor import RaptorBuilder
    from qdrant_memory.raptor.apply import persist_manifest_report, wrap_manifest

    fixed = LINEAGE_LOCK_FIXED_TEXT
    old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    observed: dict[str, str] = {}

    fence = _provider(tmp_path / "fence")
    fence._config.update({
        "lineage_mode": "off",
        "lineage_lock_dir": str(tmp_path / "fence" / "locks"),
        "lineage_lock_timeout_seconds": 0.15,
    })
    fence._qdrant = FakeQdrant({
        "memory": [
            _point(
                "m1", "old weak memory", source_type="conversation", importance=1,
                confidence=0.3, access_count=0, created_at=old,
            ),
        ],
        "learnings": [],
    })
    report, proposal = _persist_stale_report(fence)
    observed["consolidation_fence"] = _invoke_writer_while_configured_lock_is_held(
        fence,
        lambda: fence.handle_tool_call(
            "qdrant_memory_consolidation_apply",
            {
                "report_id": report["report_id"], "proposal_id": proposal["proposal_id"],
                "action": "delete", "dry_run": False, "approve": True,
            },
        ),
    )["error"]

    forget = _provider(tmp_path / "forget")
    forget._config.update({
        "lineage_lock_dir": str(tmp_path / "forget" / "locks"),
        "lineage_lock_timeout_seconds": 0.15,
    })
    forget._qdrant = FakeQdrant({
        "memory": [_point("root", "root", source_type="manual")], "learnings": [],
    })
    observed["forget"] = _invoke_writer_while_configured_lock_is_held(
        forget,
        lambda: forget.handle_tool_call(
            "qdrant_memory_forget", {"ids": ["root"], "dry_run": False},
        ),
    )["error"]

    extraction = _provider(tmp_path / "extraction")
    extraction._qdrant = FakeQdrant({"memory": [], "learnings": []})
    extraction._embeddings = FakeEmbedding()
    extraction._config.update({
        "source_extraction_enabled": True,
        "source_extraction_mode": "preview",
        "source_extraction_min_confidence": 0.65,
        "source_extraction_max_candidates_per_session": 8,
        "lineage_lock_dir": str(tmp_path / "extraction" / "locks"),
        "lineage_lock_timeout_seconds": 0.15,
    })
    extraction_candidate = extract_source_candidates_from_text(
        "Decision: keep extraction approvals behind the configured collection lock.",
        source_uri="session://lineage-lock/extraction",
    )[0]
    extraction._pending_extraction_candidates[extraction_candidate.candidate_id] = extraction_candidate
    assert json.loads(extraction.handle_tool_call(
        "qdrant_memory_extraction_approve",
        {"candidate_id": extraction_candidate.candidate_id},
    ))["dry_run"] is True
    observed["extraction_approval"] = _invoke_writer_while_configured_lock_is_held(
        extraction,
        lambda: extraction.handle_tool_call(
            "qdrant_memory_extraction_approve",
            {
                "candidate_id": extraction_candidate.candidate_id,
                "dry_run": False, "approve": True,
            },
        ),
    )["error"]

    improve = _provider(tmp_path / "improve")
    improve._qdrant = FakeQdrant({"memory": [], "learnings": []})
    improve._embeddings = FakeEmbedding()
    improve._config.update({
        "lineage_lock_dir": str(tmp_path / "improve" / "locks"),
        "lineage_lock_timeout_seconds": 0.15,
    })
    improve_preview = json.loads(improve.handle_tool_call(
        "qdrant_memory_improve_preview",
        {
            "source_text": "Graph entity: project: LockedImprove",
            "source_uri": "session://lineage-lock/improve",
        },
    ))
    improve_candidate = next(item for item in improve_preview["candidates"] if item["would_store"])
    assert json.loads(improve.handle_tool_call(
        "qdrant_memory_improve_apply",
        {"report_id": improve_preview["report_id"], "candidate_id": improve_candidate["candidate_id"]},
    ))["dry_run"] is True
    observed["improve_apply"] = _invoke_writer_while_configured_lock_is_held(
        improve,
        lambda: improve.handle_tool_call(
            "qdrant_memory_improve_apply",
            {
                "report_id": improve_preview["report_id"],
                "candidate_id": improve_candidate["candidate_id"],
                "dry_run": False, "approve": True,
            },
        ),
    )["error"]

    raptor = _provider(tmp_path / "raptor")
    raptor._profile_id = "default"
    raptor._qdrant = FakeQdrant({"memory": [], "learnings": []})
    raptor._embeddings = FakeEmbedding()
    raptor._config.update({
        "embedding_model": "test-model",
        "lineage_lock_dir": str(tmp_path / "raptor" / "locks"),
        "lineage_lock_timeout_seconds": 0.15,
    })
    manifest = RaptorBuilder().build([
        _point(
            "leaf-1", "alpha note about RAPTOR", source="test://leaf1",
            source_uri="test://leaf1", source_type="manual", profile_id="default",
        ),
        _point(
            "leaf-2", "beta note about RAPTOR", source="test://leaf2",
            source_uri="test://leaf2", source_type="manual", profile_id="default",
        ),
    ]).to_dict()
    raptor_report_id = f"raptor-{manifest['manifest_digest'][:12]}"
    raptor_wrapper = wrap_manifest(manifest, report_id=raptor_report_id)
    persist_manifest_report(raptor_wrapper, hermes_home=raptor._hermes_home)
    raptor._pending_raptor_manifests[raptor_report_id] = raptor_wrapper
    raptor_args = {
        "report_id": raptor_report_id,
        "build_id": manifest["build_id"],
        "manifest_digest": manifest["manifest_digest"],
    }
    assert json.loads(raptor.handle_tool_call(
        "qdrant_memory_raptor_apply", raptor_args,
    ))["dry_run"] is True
    observed["raptor_apply"] = _invoke_writer_while_configured_lock_is_held(
        raptor,
        lambda: raptor.handle_tool_call(
            "qdrant_memory_raptor_apply", {**raptor_args, "dry_run": False, "approve": True},
        ),
    )["error"]

    assert observed == {
        "consolidation_fence": fixed,
        "forget": fixed,
        "extraction_approval": f"Extraction approval failed: {fixed}",
        "improve_apply": f"Improve apply failed: {fixed}",
        "raptor_apply": f"RAPTOR apply failed: {fixed}",
    }


def test_lock_directory_os_text_never_reaches_a_tool_response(tmp_path):
    """G3: an unusable lock directory must not leak its OS error text.

    A regular file where the lock directory should be makes
    ``collection_write_lock`` raise ``RuntimeError('lineage lock directory
    unavailable: [Errno 20] Not a directory: <path>')``. Every refusal surface
    must collapse that to the single fixed text.
    """
    from qdrant_memory.source_extraction import extract_source_candidates_from_text

    fixed = LINEAGE_LOCK_FIXED_TEXT
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x", encoding="utf-8")
    unusable_lock_dir = str(blocker / "locks")
    old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()

    def assert_no_os_text(text: str) -> None:
        assert unusable_lock_dir not in text
        assert "not-a-directory" not in text
        assert "Errno" not in text
        assert "Not a directory" not in text
        assert "lock directory unavailable" not in text

    extraction = _provider(tmp_path / "extraction")
    extraction._qdrant = FakeQdrant({"memory": [], "learnings": []})
    extraction._embeddings = FakeEmbedding()
    extraction._config.update({
        "source_extraction_enabled": True,
        "source_extraction_mode": "preview",
        "source_extraction_min_confidence": 0.65,
        "source_extraction_max_candidates_per_session": 8,
        "lineage_lock_dir": unusable_lock_dir,
    })
    candidate = extract_source_candidates_from_text(
        "Decision: keep extraction approvals behind the configured collection lock.",
        source_uri="session://lineage-lock/extraction",
    )[0]
    extraction._pending_extraction_candidates[candidate.candidate_id] = candidate
    assert json.loads(extraction.handle_tool_call(
        "qdrant_memory_extraction_approve",
        {"candidate_id": candidate.candidate_id},
    ))["dry_run"] is True
    writer_text = extraction.handle_tool_call(
        "qdrant_memory_extraction_approve",
        {"candidate_id": candidate.candidate_id, "dry_run": False, "approve": True},
    )
    assert json.loads(writer_text) == {"error": f"Extraction approval failed: {fixed}"}
    assert_no_os_text(writer_text)
    assert extraction._qdrant.upserts == []
    assert candidate.candidate_id in extraction._pending_extraction_candidates

    forget = _provider(tmp_path / "forget")
    forget._config["lineage_lock_dir"] = unusable_lock_dir
    forget._qdrant = FakeQdrant({
        "memory": [_point("root", "root", source_type="manual")], "learnings": [],
    })
    forget_text = forget.handle_tool_call(
        "qdrant_memory_forget", {"ids": ["root"], "dry_run": False},
    )
    assert json.loads(forget_text) == {"error": fixed}
    assert_no_os_text(forget_text)
    assert forget._qdrant.deleted_ids == []

    fence = _provider(tmp_path / "fence")
    fence._config.update({
        "lineage_mode": "off",
        "lineage_lock_dir": unusable_lock_dir,
    })
    fence._qdrant = FakeQdrant({
        "memory": [
            _point(
                "m1", "old weak memory", source_type="conversation", importance=1,
                confidence=0.3, access_count=0, created_at=old,
            ),
        ],
        "learnings": [],
    })
    report, proposal = _persist_stale_report(fence)
    fence_text = fence.handle_tool_call(
        "qdrant_memory_consolidation_apply",
        {
            "report_id": report["report_id"], "proposal_id": proposal["proposal_id"],
            "action": "delete", "dry_run": False, "approve": True,
        },
    )
    assert json.loads(fence_text) == {"error": fixed}
    assert_no_os_text(fence_text)
    assert fence._qdrant.deleted_ids == []
    assert fence._qdrant.upserts == []
    assert fence._qdrant.payload_updates == []


def test_structural_lineage_refusal_wording_is_caller_agnostic(tmp_path):
    """G5: one refusal string that is accurate for both callers.

    The consolidation apply path and ``qdrant_memory_forget`` share
    ``_retrieve_consolidation_points``; the forget caller has no proposal, so
    the message must not claim one. Both refusals must be write-free.
    """
    expected = "refusing to touch structural lineage records: structural-1"

    forget = _provider(tmp_path / "forget")
    forget._config["lineage_lock_dir"] = str(tmp_path / "forget" / "locks")
    forget._qdrant = FakeQdrant({
        "memory": [_point("structural-1", "file-source-abc123", lineage_record=True)],
        "learnings": [],
    })
    forget_text = forget.handle_tool_call(
        "qdrant_memory_forget", {"ids": ["structural-1"], "dry_run": False},
    )
    assert json.loads(forget_text) == {"error": expected}
    assert "proposal" not in forget_text
    assert forget._qdrant.deleted_ids == []
    assert forget._qdrant.upserts == []
    assert forget._qdrant.payload_updates == []

    apply_provider = _provider(tmp_path / "apply")
    apply_provider._config["lineage_lock_dir"] = str(tmp_path / "apply" / "locks")
    report_id = "consolidation-aaaaaaaaaaaa"
    artifact = _report_artifact(tmp_path / "apply", report_id)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps({
        "report_id": report_id,
        "report_type": "consolidation_report",
        "profile_id": "architect",
        "proposals": [
            {
                "proposal_id": "proposal-1",
                "proposal_type": "heading_noise",
                "collection_name": "memory",
                "affected_ids": ["structural-1"],
                "suggested_action": "delete_review_only",
                "confidence": 0.7,
                "risk": "medium",
                "evidence": [{"id": "structural-1", "reason": "heading noise"}],
                "requires_explicit_approval": True,
            },
        ],
    }), encoding="utf-8")
    apply_provider._qdrant = FakeQdrant({
        "memory": [{"id": "structural-1", "payload": {"text": "file-source-abc123", "lineage_record": True}}],
    })
    apply_text = apply_provider.handle_tool_call(
        "qdrant_memory_consolidation_apply",
        {"report_id": report_id, "proposal_id": "proposal-1", "dry_run": True},
    )
    assert json.loads(apply_text) == {"error": expected}
    assert "proposal" not in apply_text
    assert apply_provider._qdrant.upserts == []
    assert apply_provider._qdrant.deleted_ids == []
    assert apply_provider._qdrant.payload_updates == []


@pytest.mark.parametrize("mode", ["off", "capture", "reconcile"])
def test_foreign_profile_dependent_does_not_block_forget(mode, tmp_path):
    """G6: the dependency fence is profile-scoped; that boundary is pinned.

    A dependency edge whose ``profile_id`` belongs to another profile is
    invisible to ``find_direct_dependents``, so the forget proceeds and the
    out-of-profile edge survives pointing at the deleted target. This is a
    recorded boundary decision (docs/SAFETY.md section 22), not a regression.
    """
    class MutableFakeQdrant(FakeQdrant):
        def delete_ids(self, name, ids):
            super().delete_ids(name, ids)
            wanted = {str(point_id) for point_id in ids}
            self.by_collection[name] = [
                point for point in self.by_collection.get(name, [])
                if str(point.get("id")) not in wanted
            ]

    provider = _provider(tmp_path)
    provider._config["lineage_mode"] = mode
    provider._config["lineage_lock_dir"] = str(tmp_path / "locks")
    provider._qdrant = MutableFakeQdrant({
        "memory": [
            _point("root", "root without lineage markers", source_type="manual"),
            _point(
                "other-profile-dependent", "derived in another profile",
                source_type="conversation", profile_id="someone-else",
            ),
            _point(
                "other-profile-edge", "", profile_id="someone-else",
                memory_kind="graph_edge", source_point_id="other-profile-dependent",
                target_point_id="root", relation_type="DERIVED_FROM",
            ),
            _point(
                "unrelated", "must survive", source_type="manual",
            ),
        ],
        "learnings": [],
    })

    deleted = json.loads(provider.handle_tool_call(
        "qdrant_memory_forget", {"ids": ["root"], "dry_run": False},
    ))

    assert deleted == {"dry_run": False, "ids": ["root"], "deleted": 1}
    assert provider._qdrant.deleted_ids == [("memory", ["root"])]
    remaining = {str(point["id"]) for point in provider._qdrant.by_collection["memory"]}
    assert remaining == {"other-profile-dependent", "other-profile-edge", "unrelated"}
    assert provider._qdrant.upserts == []
    assert provider._qdrant.payload_updates == []


@pytest.mark.parametrize("mode", ["off", "capture", "reconcile"])
def test_foreign_profile_dependent_is_invisible_to_the_consolidation_apply(mode, tmp_path):
    """G6: same boundary for the consolidation consumer of the same lookup.

    Under off/capture the fence reports clear and the destructive apply
    proceeds. Under reconcile the fence is replaced by the lineage-impact
    snapshot, so the boundary shows up as an empty ``dependent_ids`` and no
    dependency error; note that a destructive consolidation apply in reconcile
    mode is refused anyway by the pre-existing, unrelated
    ``ordinary_root_transition_cause_unratified`` gate, because the persisted
    snapshot is always built without a ratified event id.
    """
    provider = _provider(tmp_path)
    provider._config["lineage_mode"] = mode
    provider._config["lineage_lock_dir"] = str(tmp_path / "locks")
    old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    provider._qdrant = FakeQdrant({
        "memory": [
            _point(
                "m1", "old weak memory", source_type="conversation", importance=1,
                confidence=0.3, access_count=0, created_at=old,
            ),
            _point(
                "other-profile-dependent", "derived in another profile",
                source_type="conversation", profile_id="someone-else",
            ),
            _point(
                "other-profile-edge", "", profile_id="someone-else",
                memory_kind="graph_edge", source_point_id="other-profile-dependent",
                target_point_id="m1", relation_type="DERIVED_FROM",
            ),
        ],
        "learnings": [],
    })
    report, proposal = _persist_stale_report(provider)

    result_text = provider.handle_tool_call(
        "qdrant_memory_consolidation_apply",
        {
            "report_id": report["report_id"], "proposal_id": proposal["proposal_id"],
            "action": "delete", "dry_run": False, "approve": True,
        },
    )
    result = json.loads(result_text)

    if mode == "reconcile":
        assert proposal["lineage_impact"]["dependent_ids"] == []
        assert proposal["lineage_impact"]["errors"] == ["ordinary_root_transition_cause_unratified"]
        assert "dependents" not in result["error"]
        assert "scope mismatch" not in result_text
        assert provider._qdrant.deleted_ids == []
    else:
        assert proposal["lineage_fence"]["status"] == "clear"
        assert proposal["lineage_fence"]["dependent_count"] == 0
        assert result.get("error") is None, result
        assert result["applied"] is True
        assert provider._qdrant.deleted_ids == [("memory", ["m1"])]


def test_default_lineage_lock_dir_honours_the_environment_override(tmp_path, monkeypatch):
    """G8: HERMES_QDRANT_MEMORY_LINEAGE_LOCK_DIR supplies the default lock dir.

    The override composes with the documented precedence
    (``qdrant_memory/config.py``): it reaches ``load_config`` for the provider
    path, and it also supplies the default at the point of use inside
    ``collection_write_lock`` for callers that pass an empty ``lock_dir``. An
    explicit ``lock_dir`` still wins.
    """
    from qdrant_memory.config import load_config
    from qdrant_memory.lineage import collection_write_lock

    env_dir = tmp_path / "env-locks"
    monkeypatch.setenv("HERMES_QDRANT_MEMORY_LINEAGE_LOCK_DIR", str(env_dir))

    assert load_config(hermes_home=str(tmp_path), hermes_config={})["lineage_lock_dir"] == str(env_dir)

    with collection_write_lock(collection_name="env-default") as path:
        assert path.parent == env_dir
    assert [entry.name for entry in env_dir.iterdir()]

    explicit_dir = tmp_path / "explicit-locks"
    with collection_write_lock(collection_name="explicit-wins", lock_dir=str(explicit_dir)) as path:
        assert path.parent == explicit_dir
    assert len(list(env_dir.iterdir())) == 1


def test_reconcile_persisted_impact_errors_drop_out_of_scope_point_ids(tmp_path):
    """G7: the persisted reconcile report must not carry a foreign point id.

    A dependency edge in the caller's profile but outside the caller's chat
    scope is returned by the graph scroll and then rejected by the scope check.
    The error keeps its category and must not echo the edge id into the
    persisted report.
    """
    provider = _provider(tmp_path)
    provider._config["lineage_mode"] = "reconcile"
    provider._config["lineage_lock_dir"] = str(tmp_path / "locks")
    old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    provider._qdrant = FakeQdrant({
        "memory": [
            _point(
                "m1", "old weak memory", source_type="conversation", importance=1,
                confidence=0.3, access_count=0, created_at=old,
            ),
            _point(
                "dependent", "derived memory", source_type="conversation",
                profile_id=provider._profile_id, user_id_hash=provider._user_id_hash,
                chat_id_hash=provider._chat_id_hash,
            ),
            _point(
                "out-of-scope-edge", "", profile_id=provider._profile_id,
                user_id_hash=provider._user_id_hash, chat_id_hash="foreign-chat",
                memory_kind="graph_edge", source_point_id="dependent",
                target_point_id="m1", relation_type="DERIVED_FROM",
            ),
        ],
        "learnings": [],
    })

    report, proposal = _persist_stale_report(provider)
    persisted = json.loads(_report_artifact(tmp_path, report["report_id"]).read_text(encoding="utf-8"))
    persisted_proposal = next(
        item for item in persisted["proposals"] if item["proposal_id"] == proposal["proposal_id"]
    )
    impact = persisted_proposal["lineage_impact"]

    assert "dependency edge scope mismatch" in impact["errors"]
    assert not any(
        "out-of-scope-edge" in str(error) for error in impact["errors"]
    )
    assert "out-of-scope-edge" not in json.dumps(impact)

    result_text = provider.handle_tool_call(
        "qdrant_memory_consolidation_apply",
        {
            "report_id": report["report_id"], "proposal_id": proposal["proposal_id"],
            "action": "delete", "dry_run": False, "approve": True,
        },
    )
    assert "out-of-scope-edge" not in result_text
    assert "lineage impact is incomplete" in json.loads(result_text)["error"]
    assert provider._qdrant.deleted_ids == []
    assert provider._qdrant.upserts == []
    assert provider._qdrant.payload_updates == []


def test_dependency_scope_mismatch_errors_keep_their_category_without_ids(tmp_path):
    """G7: all four scope-mismatch sites keep the category and drop the id."""
    from qdrant_memory.lineage import find_direct_dependents, plan_reconciliation

    qdrant = FakeQdrant({
        "memory": [
            {"id": "target", "vector": [0.1], "payload": {
                "text": "target", "profile_id": "architect", "chat_id_hash": "",
            }},
            {"id": "foreign-edge", "vector": [0.1], "payload": {
                "text": "", "memory_kind": "graph_edge", "profile_id": "architect",
                "chat_id_hash": "foreign-chat", "source_point_id": "source",
                "target_point_id": "target", "relation_type": "DERIVED_FROM",
            }},
            {"id": "valid-edge", "vector": [0.1], "payload": {
                "text": "", "memory_kind": "graph_edge", "profile_id": "architect",
                "chat_id_hash": "", "source_point_id": "foreign-source",
                "target_point_id": "target", "relation_type": "DERIVED_FROM",
            }},
            {"id": "foreign-source", "vector": [0.1], "payload": {
                "text": "source", "profile_id": "architect", "chat_id_hash": "foreign-chat",
            }},
            {"id": "foreign-inline", "vector": [0.1], "payload": {
                "text": "inline", "profile_id": "architect", "chat_id_hash": "foreign-chat",
                "derived_from": [{"point_id": "target", "relation_type": "DERIVED_FROM"}],
            }},
            {"id": "owned-foreign", "vector": [0.1], "payload": {
                "text": "owned chunk", "memory_kind": "source_chunk",
                "profile_id": "architect", "chat_id_hash": "foreign-chat",
                "file_sha256": "a" * 64, "chunk_index": 0,
            }},
        ],
        "learnings": [],
    })

    dependents = find_direct_dependents(
        qdrant=qdrant, collection_name="memory", target_point_id=["target"],
        profile_id="architect", user_id_hash="", chat_id_hash="",
    )

    assert dependents["complete"] is False
    assert "dependency edge scope mismatch" in dependents["errors"]
    assert "dependency point scope mismatch" in dependents["errors"]
    assert "inline dependency scope mismatch" in dependents["errors"]
    assert not any(
        point_id in error
        for error in dependents["errors"]
        for point_id in ("foreign-edge", "foreign-source", "foreign-inline")
    )

    plan = plan_reconciliation(
        qdrant=qdrant, collection_name="memory", profile_id="architect",
        user_id_hash="", chat_id_hash="", file_path="/notes/a.md",
        manifest=None, chunks=[], existing_points=[qdrant.by_collection["memory"][-1]],
        existing_source=None,
    )
    assert "owned inventory scope mismatch" in plan["errors"]
    assert not any("owned-foreign" in error for error in plan["errors"])


# ===========================================================================
# W2 closure review (delta 5) N1/N2: the lock-refusal contract at the tool
# boundary, and the acquisition-only scope of the writer normalization.
# ===========================================================================

def test_index_tool_response_collapses_lineage_lock_refusals(monkeypatch, tmp_path):
    """N1: the index tool must not forward raw lock text or the lock path.

    `FileIndexer` is an internal component that reports precisely; the tool
    response is the operator contract. Redaction therefore happens at the
    boundary, and the raw text is kept in the server log.
    """
    provider = _provider(tmp_path)
    provider._qdrant = FakeQdrant({"memory": [], "learnings": []})
    provider._embeddings = FakeEmbedding()
    raw_errors = [
        "lineage capture failed: lineage lock directory unavailable: "
        "[Errno 20] Not a directory: '/tmp/conf/locks'",
        "lineage capture failed: lineage collection lock acquisition timed out",
        "lineage reconciliation failed: lineage lock directory must not be a symlink",
    ]
    expected = [
        "lineage capture failed: lineage collection lock unavailable",
        "lineage capture failed: lineage collection lock unavailable",
        "lineage reconciliation failed: lineage collection lock unavailable",
    ]

    def fake_index(self, paths, **kwargs):
        return {
            "dry_run": False,
            "errors": [
                {"file_path": "/notes/a.md", "error": raw} for raw in raw_errors
            ],
            "lineage_blocked_files": [],
        }

    monkeypatch.setattr("__init__.FileIndexer.index", fake_index)
    text = provider.handle_tool_call(
        "qdrant_memory_index", {"paths": ["/notes/a.md"], "dry_run": False, "force": True},
    )
    payload = json.loads(text)

    assert [entry["error"] for entry in payload["errors"]] == expected
    for forbidden in (
        "Errno", "Not a directory", "symlink", "lock directory unavailable",
        "timed out", "/tmp/conf/locks", "lineage collection lock acquisition",
    ):
        assert forbidden not in text


def test_index_tool_response_redacts_a_lock_failure_raised_out_of_the_indexer(monkeypatch, tmp_path):
    """N1: the `except` branch of the index tool is a second way to leak raw text."""
    provider = _provider(tmp_path)
    provider._qdrant = FakeQdrant({"memory": [], "learnings": []})
    provider._embeddings = FakeEmbedding()

    def failing_index(self, paths, **kwargs):
        raise RuntimeError(
            "lineage lock directory unavailable: [Errno 20] Not a directory: '/tmp/conf/locks'"
        )

    monkeypatch.setattr("__init__.FileIndexer.index", failing_index)
    text = provider.handle_tool_call(
        "qdrant_memory_index", {"paths": ["/notes/a.md"], "dry_run": False, "force": True},
    )

    assert json.loads(text) == {
        "error": "Index failed: lineage collection lock unavailable",
    }
    assert "Errno" not in text
    assert "/tmp/conf/locks" not in text


@pytest.mark.parametrize("body_error", [RuntimeError, TimeoutError])
def test_locked_writer_body_failure_is_not_reported_as_a_lock_refusal(tmp_path, body_error):
    """N2/I1: the normalization wraps the acquisition only, never the guarded body.

    Both exception types a lock handler catches are covered: a body failure of either
    type must surface as itself, never as a lock refusal.
    """
    from qdrant_memory.source_extraction import extract_source_candidates_from_text

    class BodyFailsOnUpsert(FakeQdrant):
        def upsert(self, name, points):
            raise body_error("store exploded inside the locked body")

    provider = _provider(tmp_path)
    provider._qdrant = BodyFailsOnUpsert({"memory": [], "learnings": []})
    provider._embeddings = FakeEmbedding()
    provider._config.update({
        "source_extraction_enabled": True,
        "source_extraction_mode": "preview",
        "source_extraction_min_confidence": 0.65,
        "source_extraction_max_candidates_per_session": 8,
        "lineage_lock_dir": str(tmp_path / "locks"),
    })
    candidate = extract_source_candidates_from_text(
        "Decision: keep extraction approvals behind the configured collection lock.",
        source_uri="session://lineage-lock/extraction",
    )[0]
    provider._pending_extraction_candidates[candidate.candidate_id] = candidate
    assert json.loads(provider.handle_tool_call(
        "qdrant_memory_extraction_approve",
        {"candidate_id": candidate.candidate_id},
    ))["dry_run"] is True

    text = provider.handle_tool_call(
        "qdrant_memory_extraction_approve",
        {"candidate_id": candidate.candidate_id, "dry_run": False, "approve": True},
    )

    assert json.loads(text) == {
        "error": "Extraction approval failed: store exploded inside the locked body",
    }
    assert LINEAGE_LOCK_FIXED_TEXT not in text
    assert candidate.candidate_id in provider._pending_extraction_candidates


def test_lock_refusal_redactor_is_idempotent_and_leaves_other_text_alone():
    """N1: the redactor is a boundary transform, not a blanket string rewrite."""
    from qdrant_memory.lineage import redact_lock_refusals

    fixed = "lineage capture failed: lineage collection lock unavailable"
    assert redact_lock_refusals(fixed) == fixed
    assert redact_lock_refusals("ordinary index summary text") == "ordinary index summary text"
    assert redact_lock_refusals(7) == 7
    assert redact_lock_refusals({"a": [{"b": fixed}]}) == {"a": [{"b": fixed}]}
    assert redact_lock_refusals(
        "lineage capture failed: lineage lock directory must not be a symlink"
    ) == fixed


# ===========================================================================
# W2 closure review (delta 6) F1/F2/F3 + I1: OS-level lock failures, the
# per-clause log, and the clause boundary of the redactor.
# ===========================================================================

def _locked_provider(tmp_path, *, mode="off"):
    """Provider whose writers lock inside tmp_path, never in the shared default."""
    provider = _provider(tmp_path)
    provider._qdrant = FakeQdrant({"memory": [], "learnings": []})
    provider._embeddings = FakeEmbedding()
    lock_dir = Path(tmp_path) / "locks"
    lock_dir.mkdir(mode=0o700, exist_ok=True)
    provider._config["lineage_lock_dir"] = str(lock_dir)
    provider._config["lineage_mode"] = mode
    return provider, lock_dir


def _lock_file(lock_dir, collection="memory"):
    from qdrant_memory.lineage import _collection_lock_digest

    return Path(lock_dir) / f"{_collection_lock_digest(collection, os.getuid())}.lock"


def test_lock_file_os_failure_is_normalized_by_the_helper(tmp_path):
    """F1: a bare OSError must not leave `collection_write_lock`."""
    from qdrant_memory.lineage import collection_write_lock

    lock_dir = Path(tmp_path) / "locks"
    lock_dir.mkdir(mode=0o700)
    os.symlink(str(tmp_path / "missing-target"), str(_lock_file(lock_dir)))

    with pytest.raises(RuntimeError) as excinfo:
        with collection_write_lock(collection_name="memory", timeout=0.2, lock_dir=str(lock_dir)):
            raise AssertionError("the lock must not be acquired through a symlinked lock file")

    # RuntimeError is what the callers' normalization catches; a bare OSError is not.
    assert isinstance(excinfo.value, (TimeoutError, RuntimeError))
    assert str(excinfo.value).startswith("lineage lock file unavailable:")
    assert "Errno" in str(excinfo.value)  # direct callers keep the OS detail


def test_lock_file_os_failure_never_reaches_the_tool_response(tmp_path):
    """F1: the forget path takes the lock unconditionally, so it is reachable."""
    provider, lock_dir = _locked_provider(tmp_path)
    victim = _lock_file(lock_dir)
    os.symlink(str(tmp_path / "missing-target"), str(victim))

    text = provider.handle_tool_call(
        "qdrant_memory_forget",
        {"ids": ["00000000-0000-0000-0000-000000000001"], "dry_run": False},
    )

    assert json.loads(text) == {"error": LINEAGE_LOCK_FIXED_TEXT}
    assert "Errno" not in text
    assert str(victim) not in text


def test_lock_file_os_failure_never_reaches_the_index_capture_response(tmp_path):
    """F1: lineage capture is the other tool surface reachable with a broken lock file."""
    provider, lock_dir = _locked_provider(tmp_path, mode="capture")
    victim = _lock_file(lock_dir)
    os.symlink(str(tmp_path / "missing-target"), str(victim))
    note = Path(tmp_path) / "notes" / "a.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("# Note\n\nOrdinary text for the indexer.\n", encoding="utf-8")

    text = provider.handle_tool_call(
        "qdrant_memory_index", {"paths": [str(note)], "dry_run": False, "force": True},
    )
    payload = json.loads(text)

    assert [entry["error"] for entry in payload["errors"]] == [
        "lineage capture failed: lineage collection lock unavailable",
    ]
    assert payload["chunks_upserted"] == 0  # fail-closed: nothing was written
    assert "Errno" not in text
    assert str(victim) not in text


def test_raw_lock_refusal_is_logged_as_its_own_clause(monkeypatch, tmp_path, caplog):
    """F2: the diagnosis must survive a payload whose errors[] falls past a 600-char cut."""
    provider, lock_dir = _locked_provider(tmp_path)
    long_root = "/" + "/".join(["workspaces", "qdrant-lineage", "vault", "notes"] * 20)
    refusal = (
        "lineage capture failed: lineage lock directory unavailable: "
        f"[Errno 20] Not a directory: '{lock_dir}'"
    )

    def fake_index(self, paths, **kwargs):
        return {
            "dry_run": False,
            "files_seen": 1,
            "files_indexed": 1,
            "directory_roots_checked": [long_root],
            "deleted_file_paths": [],
            "deleted_file_ids": [],
            "delete_mode": "none",
            "errors": [{"file_path": f"{long_root}/note.md", "error": refusal}],
            "lineage_blocked_files": [],
        }

    monkeypatch.setattr("__init__.FileIndexer.index", fake_index)
    with caplog.at_level(logging.WARNING, logger="__init__"):
        text = provider.handle_tool_call(
            "qdrant_memory_index", {"paths": ["/notes/a.md"], "dry_run": False, "force": True},
        )

    payload = json.loads(text)
    assert [entry["error"] for entry in payload["errors"]] == [
        "lineage capture failed: lineage collection lock unavailable",
    ]
    assert str(lock_dir) not in text  # the response stays clean
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "Not a directory" in logged  # ... and the log keeps the diagnosis
    assert str(lock_dir) in logged


def test_a_path_that_spells_a_marker_is_data_not_a_refusal(tmp_path, caplog):
    """F3: a directory named like a marker must not be truncated out of a summary."""
    provider, _ = _locked_provider(tmp_path)
    spelled = Path(tmp_path) / "lineage lock directory unavailable"
    spelled.mkdir()
    (spelled / "a.md").write_text("# Note\n\nOrdinary text for the indexer.\n", encoding="utf-8")
    provider._config["index_dirs"] = [str(spelled)]

    with caplog.at_level(logging.WARNING, logger="__init__"):
        text = provider.handle_tool_call("qdrant_memory_index", {"dry_run": True})
    payload = json.loads(text)

    assert payload["directory_roots_checked"] == [str(spelled)]
    assert payload["paths"] == [str(spelled)]
    assert "lineage collection lock unavailable" not in text
    assert not [record for record in caplog.records if "redacted" in record.getMessage()]


def test_redactor_collapses_every_marker_and_keeps_marker_shaped_data():
    """F3/I1: the clause rule in both directions, over the whole marker list."""
    from qdrant_memory.lineage import LINEAGE_LOCK_REFUSAL_MARKERS, redact_lock_refusals

    fixed = "lineage capture failed: lineage collection lock unavailable"
    # Presence is part of the contract: a marker dropped from the tuple stops
    # redacting that shape on every surface at once. A superset, not equality: the
    # contract is "every wording the helper can emit is redacted", so a ninth wording
    # must be allowed to arrive with its own test rather than forcing an edit here.
    assert set(LINEAGE_LOCK_REFUSAL_MARKERS) >= {
        "lineage collection lock acquisition timed out",
        "lineage lock directory unavailable",
        "lineage lock directory must not be a symlink",
        "lineage lock directory has unsafe ownership, type, or permissions",
        "lineage lock file unavailable",
        "lineage lock file has unsafe ownership, type, or permissions",
        "lineage locking is unsupported on this platform",
        "lineage locking requires O_NOFOLLOW support",
    }
    for marker in LINEAGE_LOCK_REFUSAL_MARKERS:
        assert redact_lock_refusals(marker) == "lineage collection lock unavailable"
        assert redact_lock_refusals(
            f"lineage capture failed: {marker}: [Errno 5] Input/output error: '/locks/x.lock'"
        ) == fixed
    # A refusal inside a joined list keeps everything before its own clause, which
    # includes the operation prefix that belongs to it.
    assert redact_lock_refusals(
        "a: lineage capture failed: lineage lock directory must not be a symlink; b"
    ) == "a: lineage capture failed: lineage collection lock unavailable"
    # Markers glued to a non-space character are paths or identifiers, not clauses.
    for data in (
        "/home/ops/lineage lock directory unavailable/notes/a.md",
        "/home/ops/pre-lineage lock file unavailable-suffix/notes/a.md",
        "notes/lineage lock file has unsafe ownership, type, or permissions.md",
    ):
        assert redact_lock_refusals(data) == data


def test_collect_lock_refusals_walks_the_payload_and_skips_data_markers():
    """F2: the walker feeds the log, so it must find clauses anywhere and nothing else."""
    from qdrant_memory.lineage import collect_lock_refusals

    raw = (
        "lineage capture failed: lineage lock file unavailable: "
        "[Errno 40] Too many levels of symbolic links: '/locks/x.lock'"
    )
    payload = {
        "a": [{"b": raw}, "/path/lineage lock directory unavailable/file.md"],
        "c": raw,
        "d": "ordinary text",
    }

    assert collect_lock_refusals(payload) == [raw, raw]


# ===========================================================================
# W2 closure review (delta 7) H1/H2/H3 + I2: the lock-directory lstat, the
# clause log cap, NUL-byte paths and the guard sites that had no pin.
# ===========================================================================

def test_lock_directory_lstat_failure_never_reaches_the_tool_response(tmp_path):
    """H1: `Path.is_symlink` re-raises EACCES/ENAMETOOLONG, so it must sit in the guard."""
    unsearchable = tmp_path / "unsearchable-parent"
    unsearchable.mkdir()
    cases = [unsearchable / "locks", tmp_path / ("x" * 300) / "locks"]
    try:
        os.chmod(unsearchable, 0o000)
        for lock_dir in cases:
            provider, _ = _locked_provider(tmp_path)
            provider._config["lineage_lock_dir"] = str(lock_dir)
            try:
                text = provider.handle_tool_call(
                    "qdrant_memory_forget",
                    {"ids": ["00000000-0000-0000-0000-000000000001"], "dry_run": False},
                )
            finally:
                if lock_dir == cases[0]:
                    os.chmod(unsearchable, 0o700)
            assert json.loads(text) == {"error": LINEAGE_LOCK_FIXED_TEXT}
            assert "Errno" not in text
            assert str(lock_dir) not in text
    finally:
        os.chmod(unsearchable, 0o700)


def test_nul_byte_in_the_lock_dir_is_a_refusal_not_a_value_error(tmp_path):
    """H3: a NUL byte raises ValueError from the path calls; it is still a refusal."""
    provider, _ = _locked_provider(tmp_path)
    provider._config["lineage_lock_dir"] = f"{tmp_path}/locks\x00tail"

    text = provider.handle_tool_call(
        "qdrant_memory_forget",
        {"ids": ["00000000-0000-0000-0000-000000000001"], "dry_run": False},
    )

    assert json.loads(text) == {"error": LINEAGE_LOCK_FIXED_TEXT}
    assert "embedded null byte" not in text


def test_every_lock_file_guard_site_is_normalized_and_releases_the_fd(tmp_path, monkeypatch):
    """I2: only the `os.open` path had a pin; fdopen and fstat had none."""
    import errno

    import qdrant_memory.lineage as lineage_module
    from qdrant_memory.lineage import collection_write_lock

    lock_dir = Path(tmp_path) / "locks"
    lock_dir.mkdir(mode=0o700)
    handed_out = []

    class OsWithFstatBoom:
        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

        def fstat(self, *args, **kwargs):
            raise OSError(errno.EIO, "Input/output error")

    real_fdopen = os.fdopen

    def fdopen_boom(fd, *args, **kwargs):
        handed_out.append(fd)
        raise OSError(errno.EIO, "Input/output error")

    # fdopen failure: the descriptor must be closed on the way out.
    monkeypatch.setattr(os, "fdopen", fdopen_boom)
    with pytest.raises(RuntimeError) as excinfo:
        with collection_write_lock(collection_name="memory", timeout=0.2, lock_dir=str(lock_dir)):
            raise AssertionError("the lock must not be acquired when fdopen fails")
    monkeypatch.setattr(os, "fdopen", real_fdopen)
    assert str(excinfo.value).startswith("lineage lock file unavailable:")
    assert handed_out, "the fdopen guard did not run"
    with pytest.raises(OSError):
        os.fstat(handed_out[-1])  # closed, not leaked

    # fstat failure: normalized like every other lock-file failure.
    monkeypatch.setattr(lineage_module, "os", OsWithFstatBoom(os))
    with pytest.raises(RuntimeError) as excinfo:
        with collection_write_lock(collection_name="memory", timeout=0.2, lock_dir=str(lock_dir)):
            raise AssertionError("the lock must not be acquired when fstat fails")
    assert str(excinfo.value).startswith("lineage lock file unavailable:")


def test_flock_failure_is_normalized_and_cleanup_does_not_strand_the_lock(tmp_path, monkeypatch):
    """I2: the non-blocking flock errno path and the swallowed cleanup, pinned together."""
    import errno
    import fcntl

    from qdrant_memory.lineage import collection_write_lock

    lock_dir = Path(tmp_path) / "locks"
    lock_dir.mkdir(mode=0o700)
    real_flock = fcntl.flock

    def flock_enolck(fd, operation):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(fcntl, "flock", flock_enolck)
    with pytest.raises(RuntimeError) as excinfo:
        with collection_write_lock(collection_name="memory", timeout=0.2, lock_dir=str(lock_dir)):
            raise AssertionError("the lock must not be acquired when flock fails")
    assert str(excinfo.value).startswith("lineage lock file unavailable:")

    # Cleanup: unlock fails, close succeeds. The context must exit cleanly and the
    # kernel must release the flock with the closed description.
    def unlock_boom(fd, operation):
        if operation == fcntl.LOCK_UN:
            raise OSError(errno.EIO, "Input/output error")
        return real_flock(fd, operation)

    monkeypatch.setattr(fcntl, "flock", unlock_boom)
    with collection_write_lock(collection_name="memory", timeout=0.2, lock_dir=str(lock_dir)):
        pass  # no exception out of the context despite the failed unlock
    monkeypatch.setattr(fcntl, "flock", real_flock)

    handle = os.open(_lock_file(lock_dir), os.O_RDWR | os.O_CREAT)
    try:
        real_flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)  # would raise if still held
        real_flock(handle, fcntl.LOCK_UN)
    finally:
        os.close(handle)


def test_a_long_lock_path_survives_the_clause_log(monkeypatch, tmp_path, caplog):
    """H2: the lock path is the last thing in the clause, so a tight cap eats it."""
    provider, _ = _locked_provider(tmp_path)
    long_lock_dir = str(Path(tmp_path) / ("deep-" + "d" * 550))
    provider._config["lineage_lock_dir"] = long_lock_dir
    refusal = (
        "lineage capture failed: lineage lock directory unavailable: "
        f"[Errno 20] Not a directory: '{long_lock_dir}'"
    )

    def fake_index(self, paths, **kwargs):
        return {
            "dry_run": False,
            "files_seen": 1,
            "directory_roots_checked": ["/root"],
            "errors": [{"file_path": "/root/note.md", "error": refusal}],
        }

    monkeypatch.setattr("__init__.FileIndexer.index", fake_index)
    with caplog.at_level(logging.WARNING, logger="__init__"):
        text = provider.handle_tool_call(
            "qdrant_memory_index", {"paths": ["/notes/a.md"], "dry_run": False, "force": True},
        )

    assert long_lock_dir not in text
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert len(long_lock_dir) > 550
    assert f"'{long_lock_dir}'" in logged  # the path, complete, in the log


def test_the_redactor_and_the_walker_agree_on_tuples():
    """I2: the walker already recursed into tuples; the redactor did not."""
    from qdrant_memory.lineage import collect_lock_refusals, redact_lock_refusals

    raw = "lineage capture failed: lineage lock file unavailable: [Errno 37] No locks available"
    payload = {"a": (raw, "plain"), "b": [raw]}

    assert collect_lock_refusals(payload) == [raw, raw]
    redacted = redact_lock_refusals(payload)
    assert redacted == {
        "a": ("lineage capture failed: lineage collection lock unavailable", "plain"),
        "b": ["lineage capture failed: lineage collection lock unavailable"],
    }
    assert isinstance(redacted["a"], tuple)




