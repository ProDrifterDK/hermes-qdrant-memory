from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from qdrant_memory.config import load_config
from qdrant_memory.tools import CONSOLIDATE_SCHEMA, TOOL_SCHEMAS


def _provider():
    from __init__ import QdrantMemoryProvider

    provider = QdrantMemoryProvider()
    provider._active = True
    provider._profile_id = "architect"
    provider._platform = "cli"
    provider._session_id = "s1"
    provider._config = load_config(hermes_home="/tmp/nonexistent-hermes", hermes_config={})
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

    def upsert(self, name, points):
        self.upserts.append((name, points))

    def update_payload(self, name, point_id, payload):
        self.payload_updates.append((name, point_id, payload))

    def delete_ids(self, name, ids):
        self.deleted_ids.append((name, ids))

    def delete_filter(self, name, filter):
        self.deleted_filters.append((name, filter))

    def search(self, *args, **kwargs):
        self.searches.append((args, kwargs))
        return []


def _point(point_id, text, **payload):
    data = {"id": point_id, "payload": {"text": text, **payload}}
    return data


def test_consolidate_schema_is_exposed_as_report_only_tool():
    names = {schema["name"] for schema in TOOL_SCHEMAS}
    assert CONSOLIDATE_SCHEMA["name"] == "qdrant_memory_consolidate"
    assert "qdrant_memory_consolidate" in names
    assert CONSOLIDATE_SCHEMA["parameters"]["additionalProperties"] is False
    assert "Dry-run" in CONSOLIDATE_SCHEMA["description"] or "dry-run" in CONSOLIDATE_SCHEMA["description"]


def test_consolidate_report_scrolls_without_mutation():
    provider = _provider()
    provider._qdrant = FakeQdrant(
        {
            "memory": [
                _point("m1", "Use dry-run before live indexing", source_type="manual", importance=7),
                _point("m2", "Always dry-run before live indexing", source_type="manual", importance=7),
            ],
            "learnings": [_point("l1", "Keep approvals dry-run first", source_type="learning", learning_type="workflow_lesson")],
        }
    )

    result = json.loads(provider.handle_tool_call("qdrant_memory_consolidate", {"include_examples": True}))

    assert result["dry_run"] is True
    assert result["report_only"] is True
    assert result["mutations_performed"] is False
    assert result["collections"]["memory"] == "memory"
    assert result["collections"]["learning"] == "learnings"
    assert result["analyzed"]["memory_points"] == 2
    assert result["analyzed"]["learning_points"] == 1
    assert len(provider._qdrant.scrolls) == 2
    assert provider._qdrant.upserts == []
    assert provider._qdrant.payload_updates == []
    assert provider._qdrant.deleted_ids == []
    assert provider._qdrant.deleted_filters == []
    assert provider._qdrant.searches == []


def test_consolidate_rejects_live_mode():
    provider = _provider()
    provider._qdrant = FakeQdrant({"memory": [_point("m1", "hello", source_type="manual")]})

    result = json.loads(provider.handle_tool_call("qdrant_memory_consolidate", {"dry_run": False}))

    assert "error" in result
    assert "report-only" in result["error"]
    assert provider._qdrant.scrolls == []
    assert provider._qdrant.upserts == []
    assert provider._qdrant.payload_updates == []
    assert provider._qdrant.deleted_ids == []


def test_consolidate_rejects_string_false_live_mode():
    provider = _provider()
    provider._qdrant = FakeQdrant({"memory": [_point("m1", "hello", source_type="manual")]})

    result = json.loads(provider.handle_tool_call("qdrant_memory_consolidate", {"dry_run": "false"}))

    assert "error" in result
    assert "report-only" in result["error"]
    assert provider._qdrant.scrolls == []


def test_consolidate_respects_scope_filters_for_memory_and_learning():
    provider = _provider()
    provider._qdrant = FakeQdrant({"memory": [], "learnings": []})

    json.loads(provider.handle_tool_call("qdrant_memory_consolidate", {"scope": "both"}))

    memory_filter = provider._qdrant.scrolls[0]["filter"]
    learning_filter = provider._qdrant.scrolls[1]["filter"]
    assert {"key": "profile_id", "match": {"value": "architect"}} in memory_filter["must"]
    assert {"key": "profile_id", "match": {"value": "architect"}} in learning_filter["must"]
    assert {"key": "source_type", "match": {"value": "learning"}} in learning_filter["must"]
    assert {"key": "source_type", "match": {"value": "learning"}} not in memory_filter["must"]


def test_consolidate_finds_duplicate_clusters_without_writing():
    provider = _provider()
    provider._qdrant = FakeQdrant(
        {
            "memory": [
                _point("m1", "Always run dry-run before live vault indexing", source_type="manual"),
                _point("m2", "Always run dry-run before live vault indexing", source_type="conversation"),
                _point("m3", "Unrelated memory about TeamForge", source_type="manual"),
            ],
            "learnings": [],
        }
    )

    result = json.loads(provider.handle_tool_call("qdrant_memory_consolidate", {"scope": "memory", "include_examples": True}))

    duplicates = [p for p in result["proposals"] if p["proposal_type"] == "duplicate_cluster"]
    assert duplicates
    assert set(duplicates[0]["affected_ids"]) == {"m1", "m2"}
    assert duplicates[0]["suggested_action"] == "merge_review_only"
    assert duplicates[0]["requires_explicit_approval"] is True
    assert provider._qdrant.upserts == []
    assert provider._qdrant.deleted_ids == []


def test_consolidate_finds_stale_low_value_candidates():
    provider = _provider()
    old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    provider._qdrant = FakeQdrant(
        {
            "memory": [
                _point("m1", "old weak memory", source_type="conversation", importance=1, confidence=0.3, access_count=0, created_at=old),
            ],
            "learnings": [],
        }
    )

    result = json.loads(provider.handle_tool_call("qdrant_memory_consolidate", {"scope": "memory"}))

    stale = [p for p in result["proposals"] if p["proposal_type"] == "stale_low_value"]
    assert stale
    assert stale[0]["affected_ids"] == ["m1"]
    assert stale[0]["suggested_action"] == "delete_review_only"
    assert stale[0]["requires_explicit_approval"] is True


def test_consolidate_finds_learning_promotion_candidates():
    provider = _provider()
    provider._qdrant = FakeQdrant(
        {
            "memory": [],
            "learnings": [
                _point(
                    "l1",
                    "Durable workflow lesson worth turning into a skill",
                    source_type="learning",
                    learning_type="workflow_lesson",
                    confidence=0.95,
                    importance=9,
                    promote_to_skill_candidate=True,
                )
            ],
        }
    )

    result = json.loads(provider.handle_tool_call("qdrant_memory_consolidate", {"scope": "learning"}))

    promotions = [p for p in result["proposals"] if p["proposal_type"] == "learning_promotion_candidate"]
    assert promotions
    assert promotions[0]["affected_ids"] == ["l1"]
    assert promotions[0]["collection_name"] == "learnings"


def test_consolidate_secret_warning_does_not_echo_secret():
    provider = _provider()
    secret = "Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456"
    provider._qdrant = FakeQdrant({"memory": [_point("m1", f"bad memory {secret}", source_type="manual")], "learnings": []})

    result_text = provider.handle_tool_call("qdrant_memory_consolidate", {"scope": "memory", "include_examples": True})
    result = json.loads(result_text)

    warnings = [p for p in result["proposals"] if p["proposal_type"] == "quality_warning"]
    assert warnings
    assert warnings[0]["affected_ids"] == ["m1"]
    assert secret not in result_text
    assert "Bearer" not in result_text


def test_status_includes_consolidation_report_flags():
    provider = _provider()
    provider._qdrant = None

    result = json.loads(provider.handle_tool_call("qdrant_memory_status", {}))

    assert result["consolidation_enabled"] is False
    assert result["reconsolidation_enabled"] is False
    assert result["consolidation_report_only"] is True
