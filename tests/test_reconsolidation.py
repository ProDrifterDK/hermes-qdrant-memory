from __future__ import annotations

import json
from pathlib import Path

from qdrant_memory.config import load_config
from qdrant_memory.tools import CONSOLIDATE_SCHEMA, TOOL_SCHEMAS


def _provider(tmp_path: Path):
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

    def search(self, *args, **kwargs):
        self.searches.append((args, kwargs))
        return []


def _point(point_id, text, **payload):
    return {"id": point_id, "payload": {"text": text, **payload}}


def _persist_reconsolidation_report(provider):
    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_consolidate",
            {"scope": "memory", "include_reconsolidation": True, "persist": True},
        )
    )
    proposal = next(p for p in result["proposals"] if p["proposal_type"] == "reconsolidation_candidate")
    return result, proposal


def test_reconsolidation_schema_and_status_flags(tmp_path):
    schemas = {schema["name"]: schema for schema in TOOL_SCHEMAS}

    assert "include_reconsolidation" in CONSOLIDATE_SCHEMA["parameters"]["properties"]
    assert "reconsolidation_max_candidates" in CONSOLIDATE_SCHEMA["parameters"]["properties"]
    assert "draft_review" in schemas["qdrant_memory_consolidation_apply"]["parameters"]["properties"]["action"]["enum"]

    provider = _provider(tmp_path)
    provider._qdrant = None
    status = json.loads(provider.handle_tool_call("qdrant_memory_status", {}))
    assert status["reconsolidation_enabled"] is False
    assert status["reconsolidation_report_only"] is True
    assert status["reconsolidation_supported_actions"] == ["draft_review"]


def test_reconsolidation_ignores_weak_topic_only_groups(tmp_path):
    provider = _provider(tmp_path)
    provider._qdrant = FakeQdrant(
        {
            "memory": [
                _point("m1", "General project notes", source_type="vault_note", topic="Notes", importance=5, confidence=0.95),
                _point("m2", "Unrelated meeting notes", source_type="vault_note", topic="Notes", importance=5, confidence=0.95),
            ],
            "learnings": [],
        }
    )

    result = json.loads(provider.handle_tool_call("qdrant_memory_consolidate", {"scope": "memory", "include_reconsolidation": True}))

    assert [p for p in result["proposals"] if p["proposal_type"] == "reconsolidation_candidate"] == []


def test_reconsolidation_detects_conflicting_fact_candidates_without_mutation(tmp_path):
    provider = _provider(tmp_path)
    provider._qdrant = FakeQdrant(
        {
            "memory": [
                _point("m1", "TeamForge MCP binary is teamforge-mcp", source_type="manual", fact_key="teamforge.mcp.binary", importance=8, confidence=0.95),
                _point("m2", "TeamForge MCP binary is teamforge-mcp-hermes", source_type="conversation", fact_key="teamforge.mcp.binary", importance=5, confidence=0.7),
                _point("m3", "Unrelated memory", source_type="manual", fact_key="other", importance=5, confidence=0.8),
            ],
            "learnings": [],
        }
    )

    result = json.loads(provider.handle_tool_call("qdrant_memory_consolidate", {"scope": "memory", "include_reconsolidation": True, "include_examples": True}))

    candidates = [p for p in result["proposals"] if p["proposal_type"] == "reconsolidation_candidate"]
    assert candidates
    candidate = candidates[0]
    assert set(candidate["affected_ids"]) == {"m1", "m2"}
    assert candidate["suggested_action"] == "reconsolidate_review_only"
    assert candidate["manual_review_required"] is True
    assert candidate["requires_explicit_approval"] is True
    assert candidate["risk"] == "high"
    assert provider._qdrant.upserts == []
    assert provider._qdrant.payload_updates == []
    assert provider._qdrant.deleted_ids == []
    assert provider._qdrant.deleted_filters == []
    assert provider._qdrant.searches == []


def test_reconsolidation_is_opt_in(tmp_path):
    provider = _provider(tmp_path)
    provider._qdrant = FakeQdrant(
        {
            "memory": [
                _point("m1", "The API endpoint is /api/v1/old", source_type="manual", fact_key="api.endpoint", importance=7),
                _point("m2", "The API endpoint is /api/v2/new", source_type="conversation", fact_key="api.endpoint", importance=7),
            ],
            "learnings": [],
        }
    )

    result = json.loads(provider.handle_tool_call("qdrant_memory_consolidate", {"scope": "memory"}))

    assert [p for p in result["proposals"] if p["proposal_type"] == "reconsolidation_candidate"] == []


def test_reconsolidation_secret_candidate_is_redacted_in_response_and_artifact(tmp_path):
    provider = _provider(tmp_path)
    auth_header = " ".join(["Authorization:", "Bearer", "".join(["abc", "def", "ghi", "jkl", "mno", "pqr", "stu", "vwx", "yz"])])
    provider._qdrant = FakeQdrant(
        {
            "memory": [
                _point("m1", f"Current token is {auth_header}", source_type="manual", fact_key="token", importance=9),
                _point("m2", "Current token is rotated", source_type="conversation", fact_key="token", importance=5),
            ],
            "learnings": [],
        }
    )

    result_text = provider.handle_tool_call("qdrant_memory_consolidate", {"scope": "memory", "include_reconsolidation": True, "include_examples": True, "persist": True})
    result = json.loads(result_text)
    artifact_path = tmp_path / "qdrant_memory" / "consolidation" / f"report-{result['report_id']}.json"
    artifact_text = artifact_path.read_text()

    candidates = [p for p in result["proposals"] if p["proposal_type"] == "reconsolidation_candidate"]
    assert candidates
    assert candidates[0]["manual_review_required"] is True
    assert "Bearer" not in result_text
    assert "".join(["abc", "def", "ghi", "jkl", "mno", "pqr", "stu", "vwx", "yz"]) not in result_text
    assert "Bearer" not in artifact_text
    assert "".join(["abc", "def", "ghi", "jkl", "mno", "pqr", "stu", "vwx", "yz"]) not in artifact_text


def test_reconsolidation_draft_review_dry_run_has_no_mutation_or_file(tmp_path):
    provider = _provider(tmp_path)
    provider._qdrant = FakeQdrant(
        {
            "memory": [
                _point("m1", "Feature flag is enabled", source_type="manual", fact_key="feature.flag", importance=8),
                _point("m2", "Feature flag is disabled", source_type="conversation", fact_key="feature.flag", importance=5),
            ],
            "learnings": [],
        }
    )
    report, proposal = _persist_reconsolidation_report(provider)

    result = json.loads(provider.handle_tool_call("qdrant_memory_consolidation_apply", {"report_id": report["report_id"], "proposal_id": proposal["proposal_id"], "action": "draft_review"}))

    assert result["dry_run"] is True
    assert result["would_apply"] is True
    assert result["action"] == "draft_review"
    assert result["proposal_draft_path"]
    assert result["reconsolidation_draft_path"] == result["proposal_draft_path"]
    assert result["write_decision"]["decision"] == "draft_review"
    assert Path(result["proposal_draft_path"]).parent == tmp_path / "qdrant_memory" / "proposals"
    assert not Path(result["reconsolidation_draft_path"]).exists()
    assert provider._qdrant.upserts == []
    assert provider._qdrant.payload_updates == []
    assert provider._qdrant.deleted_ids == []
    assert provider._qdrant.deleted_filters == []


def test_reconsolidation_draft_review_live_writes_only_local_review_artifact(tmp_path):
    provider = _provider(tmp_path)
    provider._qdrant = FakeQdrant(
        {
            "memory": [
                _point("m1", "Project status is active", source_type="manual", fact_key="project.status", importance=8),
                _point("m2", "Project status is archived", source_type="conversation", fact_key="project.status", importance=5),
            ],
            "learnings": [],
        }
    )
    report, proposal = _persist_reconsolidation_report(provider)

    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_consolidation_apply",
            {"report_id": report["report_id"], "proposal_id": proposal["proposal_id"], "action": "draft_review", "dry_run": False, "approve": True},
        )
    )

    assert result["applied"] is True
    assert result["action"] == "draft_review"
    draft_path = Path(result["reconsolidation_draft_path"])
    assert draft_path.exists()
    draft_text = draft_path.read_text()
    assert "# Reconsolidation review draft" in draft_text
    assert "Project status" in draft_text
    assert provider._qdrant.upserts == []
    assert provider._qdrant.payload_updates == []
    assert provider._qdrant.deleted_ids == []
    assert provider._qdrant.deleted_filters == []
    applications = list((tmp_path / "qdrant_memory" / "consolidation" / "applications").glob("*.json"))
    assert applications


def test_reconsolidation_draft_review_live_redacts_secret_artifact(tmp_path):
    provider = _provider(tmp_path)
    auth_header = " ".join(["Authorization:", "Bearer", "".join(["abc", "def", "ghi", "jkl", "mno", "pqr", "stu", "vwx", "yz"])])
    provider._qdrant = FakeQdrant(
        {
            "memory": [
                _point("m1", f"Current token is {auth_header}", source_type="manual", fact_key="token", importance=9),
                _point("m2", "Current token is rotated", source_type="conversation", fact_key="token", importance=5),
            ],
            "learnings": [],
        }
    )
    report, proposal = _persist_reconsolidation_report(provider)

    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_consolidation_apply",
            {"report_id": report["report_id"], "proposal_id": proposal["proposal_id"], "action": "draft_review", "dry_run": False, "approve": True},
        )
    )

    draft_text = Path(result["reconsolidation_draft_path"]).read_text()
    assert "Bearer" not in draft_text
    assert "".join(["abc", "def", "ghi", "jkl", "mno", "pqr", "stu", "vwx", "yz"]) not in draft_text
    assert "redacted" in draft_text
    assert provider._qdrant.upserts == []
    assert provider._qdrant.payload_updates == []
    assert provider._qdrant.deleted_ids == []
    assert provider._qdrant.deleted_filters == []


def test_reconsolidation_rejects_destructive_action_mismatch(tmp_path):
    provider = _provider(tmp_path)
    provider._qdrant = FakeQdrant(
        {
            "memory": [
                _point("m1", "Endpoint is /old", source_type="manual", fact_key="endpoint", importance=8),
                _point("m2", "Endpoint is /new", source_type="conversation", fact_key="endpoint", importance=5),
            ],
            "learnings": [],
        }
    )
    report, proposal = _persist_reconsolidation_report(provider)

    result = json.loads(provider.handle_tool_call("qdrant_memory_consolidation_apply", {"report_id": report["report_id"], "proposal_id": proposal["proposal_id"], "action": "delete", "dry_run": False, "approve": True}))

    assert "error" in result
    assert "mismatch" in result["error"]
    assert provider._qdrant.deleted_ids == []
    assert provider._qdrant.deleted_filters == []
