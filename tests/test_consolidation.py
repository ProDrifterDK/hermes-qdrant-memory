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
    assert duplicates[0]["match_kind"] == "exact_normalized"
    assert duplicates[0]["guarded_auto_eligible"] is True
    assert duplicates[0]["confidence"] >= 0.98


def test_consolidate_blocks_guarded_auto_for_profile_duplicate_facts():
    provider = _provider()
    provider._qdrant = FakeQdrant(
        {
            "memory": [
                _point("u1", "Alan prefers explicit reboot warnings", source_type="user_profile"),
                _point("u2", "Alan prefers explicit reboot warnings", source_type="user_profile"),
            ],
            "learnings": [],
        }
    )

    result = json.loads(provider.handle_tool_call("qdrant_memory_consolidate", {"scope": "memory"}))

    duplicates = [p for p in result["proposals"] if p["proposal_type"] == "duplicate_cluster"]
    assert duplicates
    assert duplicates[0]["match_kind"] == "exact_normalized"
    assert duplicates[0]["guarded_auto_eligible"] is False
    assert duplicates[0]["manual_review_required"] is True
    assert "preauthorized_policy" not in duplicates[0]


def test_consolidate_blocks_guarded_auto_for_stale_secret_and_profile_facts():
    provider = _provider()
    old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    secret_like = "".join(["s", "k", "-", "abcdefghijklmnopqrstuvwxyz"])
    provider._qdrant = FakeQdrant(
        {
            "memory": [
                _point("secret", f"old weak memory containing {secret_like}", source_type="conversation", importance=1, confidence=0.3, access_count=0, created_at=old),
                _point("profile", "old weak profile fact", source_type="user_profile", importance=1, confidence=0.3, access_count=0, created_at=old),
            ],
            "learnings": [],
        }
    )

    result = json.loads(provider.handle_tool_call("qdrant_memory_consolidate", {"scope": "memory"}))

    stale_by_id = {p["affected_ids"][0]: p for p in result["proposals"] if p["proposal_type"] == "stale_low_value"}
    assert set(stale_by_id) == {"secret", "profile"}
    for proposal in stale_by_id.values():
        assert proposal["guarded_auto_eligible"] is False
        assert proposal["suggested_action"] == "quarantine_review_only"
        assert proposal["manual_review_required"] is True
        assert "preauthorized_policy" not in proposal
    assert stale_by_id["secret"]["contains_secret_text"]


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
    assert stale[0]["suggested_action"] == "quarantine_guarded_auto_eligible"
    assert stale[0]["requires_explicit_approval"] is True
    assert stale[0]["guarded_auto_eligible"] is True
    assert stale[0]["preauthorized_policy"] == "guarded-auto:stale-low-value-quarantine"


def test_consolidate_finds_heading_noise_candidates_for_guarded_auto_cleanup():
    provider = _provider()
    provider._qdrant = FakeQdrant(
        {
            "memory": [
                _point("m1", "## Tareas", source_type="vault_note"),
                _point("m2", "Real durable memory body with enough content", source_type="manual"),
            ],
            "learnings": [],
        }
    )

    result = json.loads(provider.handle_tool_call("qdrant_memory_consolidate", {"scope": "memory", "include_examples": True}))

    headings = [p for p in result["proposals"] if p["proposal_type"] == "heading_noise"]
    assert headings
    assert headings[0]["affected_ids"] == ["m1"]
    assert headings[0]["guarded_auto_eligible"] is True
    assert headings[0]["suggested_action"] == "delete_guarded_auto_eligible"


def test_consolidate_reports_generic_headings_but_blocks_guarded_auto_cleanup():
    provider = _provider()
    provider._qdrant = FakeQdrant(
        {
            "memory": [
                _point("m1", "# Project Phoenix", source_type="manual"),
            ],
            "learnings": [],
        }
    )

    result = json.loads(provider.handle_tool_call("qdrant_memory_consolidate", {"scope": "memory", "include_examples": True}))

    headings = [p for p in result["proposals"] if p["proposal_type"] == "heading_noise"]
    assert headings
    assert headings[0]["affected_ids"] == ["m1"]
    assert headings[0]["guarded_auto_eligible"] is False
    assert headings[0]["suggested_action"] == "delete_review_only"
    assert headings[0]["manual_review_required"] is True


def test_consolidate_skips_already_quarantined_stale_candidates():
    provider = _provider()
    old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    provider._qdrant = FakeQdrant(
        {
            "memory": [
                _point("m1", "old weak memory", source_type="conversation", importance=1, confidence=0.3, access_count=0, created_at=old, consolidation_quarantined=True),
            ],
            "learnings": [],
        }
    )

    result = json.loads(provider.handle_tool_call("qdrant_memory_consolidate", {"scope": "memory"}))

    assert [p for p in result["proposals"] if p["proposal_type"] == "stale_low_value"] == []


def test_consolidate_skips_already_promoted_learning_candidates():
    provider = _provider()
    provider._qdrant = FakeQdrant(
        {
            "memory": [],
            "learnings": [
                _point(
                    "l1",
                    "Durable workflow lesson already drafted",
                    source_type="learning",
                    learning_type="workflow_lesson",
                    confidence=0.95,
                    importance=9,
                    promote_to_skill_candidate=True,
                    promoted_to_skill_draft=True,
                )
            ],
        }
    )

    result = json.loads(provider.handle_tool_call("qdrant_memory_consolidate", {"scope": "learning"}))

    assert [p for p in result["proposals"] if p["proposal_type"] == "learning_promotion_candidate"] == []


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
    sensitive_text = "Authorization: " + "Bearer " + "".join(["abc", "def", "123", "456"])
    provider._qdrant = FakeQdrant({"memory": [_point("m1", f"bad memory {sensitive_text}", source_type="manual")], "learnings": []})

    result_text = provider.handle_tool_call("qdrant_memory_consolidate", {"scope": "memory", "include_examples": True})
    result = json.loads(result_text)

    warnings = [p for p in result["proposals"] if p["proposal_type"] == "quality_warning"]
    assert warnings
    assert warnings[0]["affected_ids"] == ["m1"]
    assert sensitive_text not in result_text
    assert "Bearer" not in result_text


def test_consolidate_ignores_token_budget_language_as_secret_warning():
    provider = _provider()
    provider._qdrant = FakeQdrant(
        {
            "memory": [
                _point("m1", "Token Budget Enforcement keeps summaries under STRUCTURED_SUMMARY_TOKEN_BUDGET", source_type="vault_note"),
                _point("m2", "Token Counting Cache improves tokenizer estimates", source_type="vault_note"),
                _point("m3", "JWT validation strategy uses HS256 for POC and RS256 for prod", source_type="vault_note"),
            ],
            "learnings": [],
        }
    )

    result = json.loads(provider.handle_tool_call("qdrant_memory_consolidate", {"scope": "memory"}))

    assert [p for p in result["proposals"] if p["proposal_type"] == "quality_warning"] == []


def test_consolidate_ignores_task_ids_as_openai_style_secret_warning():
    provider = _provider()
    task_id = "".join(["ta", "sk", "-", "1778977497560258115"])
    provider._qdrant = FakeQdrant(  # type: ignore[assignment]
        {
            "memory": [
                _point("m1", f"Use request_restart without task_id for global restarts; affected ticket was {task_id}.", source_type="conversation"),
            ],
            "learnings": [],
        }
    )

    result = json.loads(provider.handle_tool_call("qdrant_memory_consolidate", {"scope": "memory"}))

    assert [p for p in result["proposals"] if p["proposal_type"] == "quality_warning"] == []


def test_consolidate_still_flags_openai_style_secret_warning():
    provider = _provider()
    key = "".join(["s", "k", "-", "abcdefghijklmnopqrstuvwxyz"])
    provider._qdrant = FakeQdrant({"memory": [_point("m1", f"bad memory {key}", source_type="manual")], "learnings": []})  # type: ignore[assignment]

    result = json.loads(provider.handle_tool_call("qdrant_memory_consolidate", {"scope": "memory"}))

    warnings = [p for p in result["proposals"] if p["proposal_type"] == "quality_warning"]
    assert warnings
    assert warnings[0]["affected_ids"] == ["m1"]


def test_status_includes_consolidation_report_flags():
    provider = _provider()
    provider._qdrant = None

    result = json.loads(provider.handle_tool_call("qdrant_memory_status", {}))

    assert result["consolidation_enabled"] is False
    assert result["reconsolidation_enabled"] is False
    assert result["consolidation_report_only"] is False
    assert result["consolidation_persist_reports"] is True
    assert result["consolidation_apply_enabled"] is True
    assert result["consolidation_supported_actions"] == ["merge", "delete", "quarantine", "promote_to_skill", "draft_review"]
    assert result["reconsolidation_report_only"] is True
    assert result["reconsolidation_supported_actions"] == ["draft_review"]
