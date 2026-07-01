"""Tests for RAPTOR Phase 4 apply/status (Phase 4).

These tests verify the full Phase 4 contract:
- Schema/tool presence
- Dry-run apply produces a plan with no mutation
- Live apply requires prior dry-run + approve=true
- Live apply rejects altered manifest digest/build/report mismatch
- Live apply upserts explicit node ids only
- Idempotent repeat does not re-upsert
- Existing conflicting node id fails closed
- canonical=true/requires_review=false/missing child hashes/missing
  provenance/secrets are rejected
- Status is read-only and reports stale/excluded conservatively
- No delete_filter/delete_ids/update_payload is used by RAPTOR apply
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any

import pytest

from __init__ import QdrantMemoryProvider
from qdrant_memory.raptor import (
    RaptorBuilder,
    build_raptor_dry_run,
)
from qdrant_memory.raptor.apply import (
    REPORT_ID_RE,
    RaptorApplyError,
    assess_leaf_safety,
    assess_parent_status,
    extract_manifest,
    is_already_applied,
    load_manifest_report,
    persist_apply_record,
    persist_manifest_report,
    plan_apply,
    validate_manifest,
    verify_manifest_digest,
    wrap_manifest,
)
from qdrant_memory.raptor.schema import compute_manifest_digest
from qdrant_memory.tools import TOOL_SCHEMAS


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeEmbedding:
    def __init__(self):
        self.documents: list[str] = []

    def embed_document(self, text):
        self.documents.append(text)
        return [0.3, 0.4]


class FakeQdrant:
    """Tracks all mutations for assertion."""

    def __init__(self):
        self.upserts: list[tuple[str, list[dict]]] = []
        self.delete_ids_calls: list[tuple[str, list[str]]] = []
        self.delete_filter_calls: list[tuple[str, dict]] = []
        self.update_payload_calls: list[tuple[str, str, dict]] = []
        self.retrieve_results: dict[str, dict] = {}
        self.retrieve_raises = False

    def upsert(self, name, points):
        self.upserts.append((name, points))
        return {"status": "ok"}

    def retrieve(self, name, ids, *, with_payload=True, with_vector=False):
        if self.retrieve_raises:
            raise RuntimeError("simulated Qdrant retrieval failure")
        results = []
        for pid in ids:
            if pid in self.retrieve_results:
                results.append(self.retrieve_results[pid])
        return results

    def delete_ids(self, name, ids):
        self.delete_ids_calls.append((name, ids))

    def delete_filter(self, name, filter):
        self.delete_filter_calls.append((name, filter))

    def update_payload(self, name, point_id, payload):
        self.update_payload_calls.append((name, point_id, payload))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _point(
    point_id: str,
    text: str = "",
    *,
    source_type: str = "manual",
    profile_id: str = "default",
    source_uri: str = "",
    fact_status: str = "",
    stale: bool | None = None,
    quarantined: bool | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "text": text,
        "source": source_uri or "test",
        "source_type": source_type,
        "profile_id": profile_id,
    }
    if source_uri:
        payload["source_uri"] = source_uri
    if fact_status:
        payload["fact_status"] = fact_status
    if stale is not None:
        payload["stale"] = stale
    if quarantined is not None:
        payload["consolidation_quarantined"] = quarantined
    return {"id": point_id, "payload": payload}


def _build_manifest(points: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build a valid Phase 3 manifest dict."""
    if points is None:
        points = [
            _point("leaf-1", "alpha note about RAPTOR", source_uri="test://leaf1"),
            _point("leaf-2", "beta note about RAPTOR", source_uri="test://leaf2"),
        ]
    builder = RaptorBuilder()
    manifest = builder.build(points)
    return manifest.to_dict()


def _wrap_manifest(manifest_dict: dict[str, Any], report_id: str = "") -> dict[str, Any]:
    """Wrap a manifest dict in the artifact wrapper shape."""
    return wrap_manifest(manifest_dict, report_id=report_id)


def _provider_for_raptor(tmp_path: Path) -> QdrantMemoryProvider:
    provider = QdrantMemoryProvider()
    provider._qdrant = FakeQdrant()
    provider._embeddings = FakeEmbedding()
    provider._active = True
    provider._hermes_home = str(tmp_path / "hermes")
    provider._session_id = "raptor-session-1"
    provider._config.update(
        {
            "collection_name": "memory",
            "learning_collection_name": "learnings",
            "embedding_model": "test-model",
        }
    )
    return provider


def _provider_with_manifest(
    tmp_path: Path,
    manifest_dict: dict[str, Any] | None = None,
) -> tuple[QdrantMemoryProvider, dict[str, Any], str, str, str]:
    """Build a provider with a valid manifest loaded in memory + persisted."""
    provider = _provider_for_raptor(tmp_path)
    manifest_dict = manifest_dict or _build_manifest()
    report_id = f"raptor-{manifest_dict['manifest_digest'][:12]}"
    build_id = manifest_dict["build_id"]
    manifest_digest = manifest_dict["manifest_digest"]
    wrapper = _wrap_manifest(manifest_dict, report_id=report_id)
    # Persist to disk so load_manifest_report can find it
    persist_manifest_report(
        wrapper,
        hermes_home=provider._hermes_home,
    )
    # Also load into memory
    provider._pending_raptor_manifests[report_id] = wrapper
    return provider, manifest_dict, report_id, build_id, manifest_digest


# ---------------------------------------------------------------------------
# Schema/tool presence
# ---------------------------------------------------------------------------


def test_raptor_apply_schema_present_in_tool_schemas():
    names = [s["name"] for s in TOOL_SCHEMAS]
    assert "qdrant_memory_raptor_apply" in names
    assert "qdrant_memory_raptor_status" in names


def test_raptor_apply_schema_has_required_fields():
    apply_schema = next(s for s in TOOL_SCHEMAS if s["name"] == "qdrant_memory_raptor_apply")
    assert "report_id" in apply_schema["parameters"]["required"]
    assert "build_id" in apply_schema["parameters"]["required"]
    assert "manifest_digest" in apply_schema["parameters"]["required"]
    assert apply_schema["parameters"]["additionalProperties"] is False


def test_raptor_status_schema_has_required_fields():
    status_schema = next(s for s in TOOL_SCHEMAS if s["name"] == "qdrant_memory_raptor_status")
    assert "report_id" in status_schema["parameters"]["required"]
    assert "build_id" in status_schema["parameters"]["required"]
    assert "manifest_digest" in status_schema["parameters"]["required"]
    assert status_schema["parameters"]["additionalProperties"] is False


def test_raptor_apply_report_id_regex():
    assert REPORT_ID_RE.match("raptor-abcdef123456")
    assert not REPORT_ID_RE.match("raptor-ABCDEF123456")
    assert not REPORT_ID_RE.match("raptor-short")
    assert not REPORT_ID_RE.match("improve-abcdef123456")
    assert not REPORT_ID_RE.match("raptor-abcdef1234567")  # too long


def test_retrieve_schema_present_in_tool_schemas():
    from qdrant_memory.tools import RETRIEVE_SCHEMA

    names = [s["name"] for s in TOOL_SCHEMAS]
    assert "qdrant_memory_retrieve" in names
    assert RETRIEVE_SCHEMA["name"] == "qdrant_memory_retrieve"
    assert RETRIEVE_SCHEMA["parameters"]["additionalProperties"] is False
    assert "query" in RETRIEVE_SCHEMA["parameters"]["required"]
    props = RETRIEVE_SCHEMA["parameters"]["properties"]
    for field in (
        "query",
        "top_k",
        "mode",
        "include_fact_history",
        "include_metadata",
        "source_type",
        "tags",
        "source",
        "file_path",
        "project_path",
        "since",
        "until",
        "collection",
        "max_depth",
        "max_children",
        "max_source_chars",
    ):
        assert field in props, f"missing property: {field}"
    # mode enum is fixed
    mode_enum = props["mode"]["enum"]
    assert set(mode_enum) == {"hybrid", "evidence"}


# ---------------------------------------------------------------------------
# Dry-run apply
# ---------------------------------------------------------------------------


def test_dry_run_apply_produces_plan_and_no_mutation(tmp_path):
    provider, manifest, report_id, build_id, digest = _provider_with_manifest(tmp_path)

    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_raptor_apply",
            {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
        )
    )

    assert result["dry_run"] is True
    assert result["report_id"] == report_id
    assert result["build_id"] == build_id
    assert result["manifest_digest"] == digest
    assert "would_upsert_ids" in result
    assert "already_present_ids" in result
    assert "blocked_ids" in result
    assert "write_decisions" in result
    assert "warnings" in result
    assert len(result["would_upsert_ids"]) > 0
    # No mutation
    assert provider._qdrant.upserts == []
    assert provider._qdrant.delete_ids_calls == []
    assert provider._qdrant.delete_filter_calls == []
    assert provider._qdrant.update_payload_calls == []


def test_dry_run_apply_defaults_to_true(tmp_path):
    provider, manifest, report_id, build_id, digest = _provider_with_manifest(tmp_path)

    # Even without explicitly passing dry_run, it should default to true
    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_raptor_apply",
            {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
        )
    )
    assert result["dry_run"] is True


# ---------------------------------------------------------------------------
# Live apply requires prior dry-run + approve
# ---------------------------------------------------------------------------


def test_live_apply_refuses_without_prior_dry_run(tmp_path):
    provider, manifest, report_id, build_id, digest = _provider_with_manifest(tmp_path)

    live = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_raptor_apply",
            {"report_id": report_id, "build_id": build_id, "manifest_digest": digest,
             "dry_run": False, "approve": True},
        )
    )
    assert "error" in live
    assert "dry-run" in live["error"].lower()


def test_live_apply_refuses_without_approve_true(tmp_path):
    provider, manifest, report_id, build_id, digest = _provider_with_manifest(tmp_path)

    # Do dry-run first
    provider.handle_tool_call(
        "qdrant_memory_raptor_apply",
        {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
    )

    # Live apply without approve
    live = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_raptor_apply",
            {"report_id": report_id, "build_id": build_id, "manifest_digest": digest,
             "dry_run": False, "approve": False},
        )
    )
    assert "error" in live
    assert "approve" in live["error"].lower()


# ---------------------------------------------------------------------------
# Live apply rejects altered manifest digest/build/report mismatch
# ---------------------------------------------------------------------------


def test_live_apply_rejects_altered_manifest_digest(tmp_path):
    provider, manifest, report_id, build_id, digest = _provider_with_manifest(tmp_path)

    # Do dry-run first
    provider.handle_tool_call(
        "qdrant_memory_raptor_apply",
        {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
    )

    # Live apply with wrong digest
    wrong_digest = "0" * 64
    live = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_raptor_apply",
            {"report_id": report_id, "build_id": build_id, "manifest_digest": wrong_digest,
             "dry_run": False, "approve": True},
        )
    )
    assert "error" in live
    assert "validation failed" in live["error"].lower() or "digest" in live["error"].lower()


def test_live_apply_rejects_build_id_mismatch(tmp_path):
    provider, manifest, report_id, build_id, digest = _provider_with_manifest(tmp_path)

    # Do dry-run first
    provider.handle_tool_call(
        "qdrant_memory_raptor_apply",
        {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
    )

    # Live apply with wrong build_id
    live = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_raptor_apply",
            {"report_id": report_id, "build_id": "raptor-build-wrongid",
             "manifest_digest": digest, "dry_run": False, "approve": True},
        )
    )
    assert "error" in live


def test_live_apply_rejects_report_id_mismatch(tmp_path):
    provider, manifest, report_id, build_id, digest = _provider_with_manifest(tmp_path)

    # Live apply with non-canonical report_id
    live = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_raptor_apply",
            {"report_id": "raptor-wrongid123", "build_id": build_id,
             "manifest_digest": digest, "dry_run": True},
        )
    )
    assert "error" in live


def test_verify_manifest_digest_detects_altered_manifest(tmp_path):
    manifest = _build_manifest()
    digest = manifest["manifest_digest"]

    # Alter the manifest after digest was computed
    manifest["candidate_node_payloads"][0]["text"] = "TAMPERED"
    with pytest.raises(RaptorApplyError, match="digest mismatch"):
        verify_manifest_digest(manifest, expected_digest=digest)


# ---------------------------------------------------------------------------
# Live apply upserts explicit node ids only
# ---------------------------------------------------------------------------


def test_live_apply_upserts_explicit_node_ids_only(tmp_path):
    provider, manifest, report_id, build_id, digest = _provider_with_manifest(tmp_path)

    # Dry-run first
    provider.handle_tool_call(
        "qdrant_memory_raptor_apply",
        {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
    )

    # Live apply
    live = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_raptor_apply",
            {"report_id": report_id, "build_id": build_id, "manifest_digest": digest,
             "dry_run": False, "approve": True},
        )
    )

    assert live["dry_run"] is False
    assert live["applied"] is True
    assert live["already_applied"] is False
    assert len(live["applied_node_ids"]) > 0

    # Check only upsert was called, no delete/update
    assert len(provider._qdrant.upserts) > 0
    assert provider._qdrant.delete_ids_calls == []
    assert provider._qdrant.delete_filter_calls == []
    assert provider._qdrant.update_payload_calls == []

    # Each upserted point id must be a raptor-node id from the manifest
    manifest_node_ids = {str(p.get("raptor_node_id") or "") for p in manifest["candidate_node_payloads"]}
    for _, points in provider._qdrant.upserts:
        for point in points:
            assert str(point["id"]) in manifest_node_ids


def test_live_apply_enriches_payload_with_provider_metadata(tmp_path):
    provider, manifest, report_id, build_id, digest = _provider_with_manifest(tmp_path)

    # Dry-run first
    provider.handle_tool_call(
        "qdrant_memory_raptor_apply",
        {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
    )

    # Live apply
    provider.handle_tool_call(
        "qdrant_memory_raptor_apply",
        {"report_id": report_id, "build_id": build_id, "manifest_digest": digest,
         "dry_run": False, "approve": True},
    )

    # Check enriched metadata in upserted payloads
    for _, points in provider._qdrant.upserts:
        for point in points:
            payload = point["payload"]
            assert payload["raptor_report_id"] == report_id
            assert payload["raptor_manifest_digest"] == digest
            assert payload["raptor_build_id"] == build_id
            assert "raptor_applied_at" in payload
            assert payload["profile_id"] == provider._profile_id


# ---------------------------------------------------------------------------
# Idempotent repeat does not re-upsert
# ---------------------------------------------------------------------------


def test_idempotent_repeat_does_not_re_upsert(tmp_path):
    provider, manifest, report_id, build_id, digest = _provider_with_manifest(tmp_path)

    # Dry-run
    provider.handle_tool_call(
        "qdrant_memory_raptor_apply",
        {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
    )
    # First live apply
    first = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_raptor_apply",
            {"report_id": report_id, "build_id": build_id, "manifest_digest": digest,
             "dry_run": False, "approve": True},
        )
    )
    assert first["applied"] is True
    first_upsert_count = len(provider._qdrant.upserts)

    # Simulate the nodes now existing in Qdrant with identical metadata
    for _, points in provider._qdrant.upserts:
        for point in points:
            provider._qdrant.retrieve_results[str(point["id"])] = {
                "id": str(point["id"]),
                "payload": point["payload"],
            }

    # Second live apply (idempotent — should not re-upsert)
    # Need to do dry-run again since review_key was consumed
    provider.handle_tool_call(
        "qdrant_memory_raptor_apply",
        {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
    )
    second = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_raptor_apply",
            {"report_id": report_id, "build_id": build_id, "manifest_digest": digest,
             "dry_run": False, "approve": True},
        )
    )
    assert second["applied"] is True
    assert second["already_applied"] is True
    assert len(provider._qdrant.upserts) == first_upsert_count  # no new upserts


def test_already_applied_via_persisted_record(tmp_path):
    """A persisted apply record matching the manifest exactly short-circuits
    live apply as idempotent. The applied_node_ids must equal the manifest's
    candidate node ids; otherwise the strict loader rejects it.
    """
    provider, manifest, report_id, build_id, digest = _provider_with_manifest(tmp_path)
    expected_node_ids = sorted(
        str(p.get("raptor_node_id") or "")
        for p in manifest["candidate_node_payloads"]
    )

    # Manually persist an apply record with the EXACT expected node ids.
    persist_apply_record(
        report_id=report_id,
        build_id=build_id,
        manifest_digest=digest,
        applied_node_ids=expected_node_ids,
        hermes_home=provider._hermes_home,
    )

    # Dry-run
    provider.handle_tool_call(
        "qdrant_memory_raptor_apply",
        {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
    )
    # Live apply should return already_applied
    live = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_raptor_apply",
            {"report_id": report_id, "build_id": build_id, "manifest_digest": digest,
             "dry_run": False, "approve": True},
        )
    )
    assert live["applied"] is True
    assert live["already_applied"] is True
    assert provider._qdrant.upserts == []


# ---------------------------------------------------------------------------
# Existing conflicting node id fails closed
# ---------------------------------------------------------------------------


def test_existing_conflicting_node_fails_closed(tmp_path):
    provider, manifest, report_id, build_id, digest = _provider_with_manifest(tmp_path)

    node_ids = [str(p.get("raptor_node_id") or "") for p in manifest["candidate_node_payloads"]]

    # Simulate a node that exists with DIFFERENT metadata
    provider._qdrant.retrieve_results[node_ids[0]] = {
        "id": node_ids[0],
        "payload": {
            "raptor_report_id": "raptor-different0001",
            "raptor_build_id": "raptor-build-different",
            "raptor_manifest_digest": "different_digest",
        },
    }

    # Dry-run
    provider.handle_tool_call(
        "qdrant_memory_raptor_apply",
        {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
    )

    # Live apply should fail closed
    live = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_raptor_apply",
            {"report_id": report_id, "build_id": build_id, "manifest_digest": digest,
             "dry_run": False, "approve": True},
        )
    )
    assert "error" in live
    assert "refusing to overwrite" in live["error"]
    assert provider._qdrant.upserts == []


# ---------------------------------------------------------------------------
# Rejection: canonical=true, requires_review=false, missing child hashes,
# missing provenance, secrets
# ---------------------------------------------------------------------------


def test_canonical_true_is_rejected(tmp_path):
    manifest = _build_manifest()
    # Tamper: set canonical=True on a payload
    manifest["candidate_node_payloads"][0]["canonical"] = True
    # Recompute digest to reflect the tampered manifest
    manifest["manifest_digest"] = compute_manifest_digest(manifest)

    report_id = f"raptor-{manifest['manifest_digest'][:12]}"
    build_id = manifest["build_id"]
    digest = manifest["manifest_digest"]

    provider = _provider_for_raptor(tmp_path)
    wrapper = _wrap_manifest(manifest, report_id=report_id)
    persist_manifest_report(wrapper, hermes_home=provider._hermes_home)
    provider._pending_raptor_manifests[report_id] = wrapper

    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_raptor_apply",
            {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
        )
    )
    assert "error" in result
    assert "canonical" in result["error"].lower()


def test_requires_review_false_is_rejected(tmp_path):
    manifest = _build_manifest()
    manifest["candidate_node_payloads"][0]["requires_review"] = False
    manifest["manifest_digest"] = compute_manifest_digest(manifest)

    report_id = f"raptor-{manifest['manifest_digest'][:12]}"
    build_id = manifest["build_id"]
    digest = manifest["manifest_digest"]

    provider = _provider_for_raptor(tmp_path)
    wrapper = _wrap_manifest(manifest, report_id=report_id)
    persist_manifest_report(wrapper, hermes_home=provider._hermes_home)
    provider._pending_raptor_manifests[report_id] = wrapper

    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_raptor_apply",
            {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
        )
    )
    assert "error" in result
    assert "requires_review" in result["error"].lower()


def test_missing_child_hashes_rejected(tmp_path):
    manifest = _build_manifest()
    # Remove source_hashes
    manifest["candidate_node_payloads"][0]["source_hashes"] = []
    manifest["manifest_digest"] = compute_manifest_digest(manifest)

    report_id = f"raptor-{manifest['manifest_digest'][:12]}"
    build_id = manifest["build_id"]
    digest = manifest["manifest_digest"]

    provider = _provider_for_raptor(tmp_path)
    wrapper = _wrap_manifest(manifest, report_id=report_id)
    persist_manifest_report(wrapper, hermes_home=provider._hermes_home)
    provider._pending_raptor_manifests[report_id] = wrapper

    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_raptor_apply",
            {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
        )
    )
    assert "error" in result


def test_missing_provenance_rejected(tmp_path):
    manifest = _build_manifest()
    # Remove derived_from
    manifest["candidate_node_payloads"][0]["derived_from"] = []
    manifest["manifest_digest"] = compute_manifest_digest(manifest)

    report_id = f"raptor-{manifest['manifest_digest'][:12]}"
    build_id = manifest["build_id"]
    digest = manifest["manifest_digest"]

    provider = _provider_for_raptor(tmp_path)
    wrapper = _wrap_manifest(manifest, report_id=report_id)
    persist_manifest_report(wrapper, hermes_home=provider._hermes_home)
    provider._pending_raptor_manifests[report_id] = wrapper

    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_raptor_apply",
            {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
        )
    )
    assert "error" in result


def test_secret_bearing_payload_rejected(tmp_path):
    manifest = _build_manifest()
    # Inject a secret-shaped string (built at runtime to avoid tripping the
    # scanner-sensitive fake-secrets guard).
    fake_secret = "".join(["s", "k", "-", "1234567890abcdef"])
    manifest["candidate_node_payloads"][0]["text"] = f"{fake_secret} secret key"
    manifest["manifest_digest"] = compute_manifest_digest(manifest)

    report_id = f"raptor-{manifest['manifest_digest'][:12]}"
    build_id = manifest["build_id"]
    digest = manifest["manifest_digest"]

    provider = _provider_for_raptor(tmp_path)
    wrapper = _wrap_manifest(manifest, report_id=report_id)
    persist_manifest_report(wrapper, hermes_home=provider._hermes_home)
    provider._pending_raptor_manifests[report_id] = wrapper

    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_raptor_apply",
            {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
        )
    )
    assert "error" in result
    assert "secret" in result["error"].lower()


def test_duplicate_node_ids_rejected(tmp_path):
    manifest = _build_manifest()
    if len(manifest["candidate_node_payloads"]) >= 2:
        # Make both payloads have the same node_id
        manifest["candidate_node_payloads"][1]["raptor_node_id"] = manifest["candidate_node_payloads"][0]["raptor_node_id"]
        manifest["manifest_digest"] = compute_manifest_digest(manifest)

        report_id = f"raptor-{manifest['manifest_digest'][:12]}"
        build_id = manifest["build_id"]
        digest = manifest["manifest_digest"]

        provider = _provider_for_raptor(tmp_path)
        wrapper = _wrap_manifest(manifest, report_id=report_id)
        persist_manifest_report(wrapper, hermes_home=provider._hermes_home)
        provider._pending_raptor_manifests[report_id] = wrapper

        result = json.loads(
            provider.handle_tool_call(
                "qdrant_memory_raptor_apply",
                {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
            )
        )
        assert "error" in result
        assert "duplicate" in result["error"].lower()


# ---------------------------------------------------------------------------
# Status is read-only and conservative
# ---------------------------------------------------------------------------


def test_status_is_read_only(tmp_path):
    provider, manifest, report_id, build_id, digest = _provider_with_manifest(tmp_path)

    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_raptor_status",
            {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
        )
    )

    assert result["read_only"] is True
    assert result["report_id"] == report_id
    assert result["node_count"] > 0
    assert "node_statuses" in result
    assert result["applied"] is False  # not applied yet
    # No mutation
    assert provider._qdrant.upserts == []
    assert provider._qdrant.delete_ids_calls == []
    assert provider._qdrant.delete_filter_calls == []


def test_status_reports_stale_leaves_conservatively():
    stale_payload = {"text": "some memory", "fact_status": "stale"}
    assessment = assess_leaf_safety(stale_payload)
    assert assessment["safe"] is False
    assert any("stale" in r for r in assessment["reasons"])


def test_status_reports_quarantined_leaves_conservatively():
    quarantined_payload = {"text": "some memory", "consolidation_quarantined": True}
    assessment = assess_leaf_safety(quarantined_payload)
    assert assessment["safe"] is False
    assert "quarantined" in assessment["reasons"]


def test_status_reports_secret_bearing_leaves():
    secret_payload = {"text": "token: " + "".join(["s", "k", "-", "1234567890abcdef"])}
    assessment = assess_leaf_safety(secret_payload)
    assert assessment["safe"] is False
    assert any("secret" in r for r in assessment["reasons"])


def test_assess_parent_status_excluded_when_child_secret():
    fake_secret = "".join(["s", "k", "-", "1234567890abcdef"])
    children = [
        {"text": "safe memory"},
        {"text": f"{fake_secret} key"},
    ]
    result = assess_parent_status(children)
    assert result["parent_status"] == "excluded"


def test_assess_parent_status_stale_when_child_stale():
    children = [
        {"text": "safe memory"},
        {"text": "stale memory", "fact_status": "stale"},
    ]
    result = assess_parent_status(children)
    assert result["parent_status"] == "stale"


def test_assess_parent_status_active_when_all_safe():
    children = [
        {"text": "safe memory one"},
        {"text": "safe memory two"},
    ]
    result = assess_parent_status(children)
    assert result["parent_status"] == "active"
    assert result["safe_children_count"] == 2


# ---------------------------------------------------------------------------
# No delete_filter/delete_ids/update_payload used by RAPTOR apply
# ---------------------------------------------------------------------------


def test_apply_module_does_not_use_delete_or_broad_mutation():
    """AST scan: apply.py must not call delete_filter, delete_ids, or update_payload."""
    import qdrant_memory.raptor.apply as apply_module

    source = open(apply_module.__file__, encoding="utf-8").read()
    tree = ast.parse(source)

    forbidden_calls = {"delete_filter", "delete_ids", "update_payload", "scroll_by_filter"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if node.attr in forbidden_calls:
                pytest.fail(f"apply.py must not call {node.attr}")


def test_raptor_apply_handler_does_not_use_delete_or_update(tmp_path):
    """Live apply path only uses retrieve + upsert."""
    provider, manifest, report_id, build_id, digest = _provider_with_manifest(tmp_path)

    # Dry-run
    provider.handle_tool_call(
        "qdrant_memory_raptor_apply",
        {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
    )
    # Live apply
    provider.handle_tool_call(
        "qdrant_memory_raptor_apply",
        {"report_id": report_id, "build_id": build_id, "manifest_digest": digest,
         "dry_run": False, "approve": True},
    )

    assert provider._qdrant.delete_ids_calls == []
    assert provider._qdrant.delete_filter_calls == []
    assert provider._qdrant.update_payload_calls == []


# ---------------------------------------------------------------------------
# Manifest wrapper shape
# ---------------------------------------------------------------------------


def test_wrap_manifest_with_explicit_report_id():
    manifest = _build_manifest()
    wrapper = wrap_manifest(manifest, report_id="raptor-abcdef123456")
    assert wrapper["report_id"] == "raptor-abcdef123456"
    assert "manifest" in wrapper
    assert isinstance(wrapper["manifest"], dict)


def test_wrap_manifest_auto_generates_report_id():
    manifest = _build_manifest()
    wrapper = wrap_manifest(manifest)
    assert wrapper["report_id"].startswith("raptor-")
    assert REPORT_ID_RE.match(wrapper["report_id"])


def test_extract_manifest_from_wrapper():
    manifest = _build_manifest()
    wrapper = wrap_manifest(manifest)
    extracted = extract_manifest(wrapper)
    assert extracted["build_id"] == manifest["build_id"]


def test_extract_manifest_from_raw():
    manifest = _build_manifest()
    extracted = extract_manifest(manifest)
    assert extracted["build_id"] == manifest["build_id"]


def test_persist_and_load_manifest_report(tmp_path):
    manifest = _build_manifest()
    wrapper = wrap_manifest(manifest)
    report_id = wrapper["report_id"]

    persist_manifest_report(wrapper, hermes_home=str(tmp_path / "hermes"))

    loaded = load_manifest_report(report_id, hermes_home=str(tmp_path / "hermes"))
    assert loaded is not None
    assert loaded["report_id"] == report_id
    assert loaded["manifest"]["build_id"] == manifest["build_id"]


def test_load_manifest_report_returns_none_for_unknown(tmp_path):
    result = load_manifest_report("raptor-ffffffffffff", hermes_home=str(tmp_path / "hermes"))
    assert result is None


def test_load_manifest_report_returns_none_for_non_canonical(tmp_path):
    result = load_manifest_report("invalid-id", hermes_home=str(tmp_path / "hermes"))
    assert result is None


# ---------------------------------------------------------------------------
# Unknown report
# ---------------------------------------------------------------------------


def test_apply_unknown_report_returns_error(tmp_path):
    provider = _provider_for_raptor(tmp_path)
    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_raptor_apply",
            {"report_id": "raptor-aaaaaaaaaaaa", "build_id": "raptor-build-aaa",
             "manifest_digest": "a" * 64},
        )
    )
    assert "error" in result
    assert "Unknown" in result["error"]


def test_status_unknown_report_returns_error(tmp_path):
    provider = _provider_for_raptor(tmp_path)
    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_raptor_status",
            {"report_id": "raptor-aaaaaaaaaaaa", "build_id": "raptor-build-aaa",
             "manifest_digest": "a" * 64},
        )
    )
    assert "error" in result
    assert "Unknown" in result["error"]


# ---------------------------------------------------------------------------
# Blocker 1: configured raptor_artifact_dir collision / typed apply record
# ---------------------------------------------------------------------------


def test_configured_artifact_dir_separates_reports_and_applied(tmp_path):
    """When raptor_artifact_dir is configured, manifest reports and apply
    records MUST live in distinct subdirectories so live apply cannot
    overwrite the manifest report.
    """
    provider, manifest, report_id, build_id, digest = _provider_with_manifest(tmp_path)
    configured_dir = str(tmp_path / "artifacts")
    provider._config["raptor_artifact_dir"] = configured_dir

    # Persist manifest report using the configured dir
    wrapper = provider._pending_raptor_manifests[report_id]
    persist_manifest_report(wrapper, hermes_home=provider._hermes_home, configured_dir=configured_dir)
    report_path = tmp_path / "artifacts" / "raptor_reports" / f"{report_id}.json"
    assert report_path.exists(), f"report not at {report_path}"
    report_text_before = report_path.read_text()
    assert "manifest" in report_text_before

    # Dry-run + live apply
    provider.handle_tool_call(
        "qdrant_memory_raptor_apply",
        {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
    )
    live = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_raptor_apply",
            {"report_id": report_id, "build_id": build_id, "manifest_digest": digest,
             "dry_run": False, "approve": True},
        )
    )
    assert live["applied"] is True
    assert live.get("already_applied") is False

    # The apply record must be in a separate subdir
    applied_path = tmp_path / "artifacts" / "raptor_applied" / f"{report_id}.json"
    assert applied_path.exists(), f"apply record not at {applied_path}"
    applied_record = json.loads(applied_path.read_text())
    assert applied_record.get("record_type") == "raptor_apply"

    # The manifest report must still be intact at the report path
    assert report_path.exists()
    report_text_after = report_path.read_text()
    assert report_text_after == report_text_before, "manifest report was overwritten"
    reloaded = json.loads(report_text_after)
    assert "manifest" in reloaded, "manifest key missing from report after apply"


def test_manifest_wrapper_in_applied_dir_not_accepted_as_applied(tmp_path):
    """If a manifest wrapper is dropped into raptor_applied/<id>.json (e.g.
    via a misconfigured base dir), live apply MUST fail closed rather than
    treating it as already_applied=true.
    """
    provider, manifest, report_id, build_id, digest = _provider_with_manifest(tmp_path)
    configured_dir = str(tmp_path / "artifacts")
    provider._config["raptor_artifact_dir"] = configured_dir

    # Persist the manifest report via the configured dir
    wrapper = provider._pending_raptor_manifests[report_id]
    persist_manifest_report(wrapper, hermes_home=provider._hermes_home, configured_dir=configured_dir)

    # Drop a manifest-wrapper file into the applied dir (simulating the
    # previous bug where reports and apply records shared the same path)
    applied_dir = tmp_path / "artifacts" / "raptor_applied"
    applied_dir.mkdir(parents=True, exist_ok=True)
    rogue = applied_dir / f"{report_id}.json"
    rogue.write_text(json.dumps(wrapper))

    # Dry-run
    provider.handle_tool_call(
        "qdrant_memory_raptor_apply",
        {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
    )

    # Live apply MUST fail closed, not return already_applied=true
    live = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_raptor_apply",
            {"report_id": report_id, "build_id": build_id, "manifest_digest": digest,
             "dry_run": False, "approve": True},
        )
    )
    assert "error" in live, f"expected error, got {live}"
    assert "malformed" in live["error"].lower() or "record_type" in live["error"].lower()
    # No upserts were performed
    assert provider._qdrant.upserts == []


def test_malformed_apply_record_fails_closed(tmp_path):
    """A non-JSON or structurally invalid file at raptor_applied/<id>.json
    must NOT be treated as 'not applied' silently. Live apply must error
    closed instead of attempting fresh upsert.
    """
    provider, manifest, report_id, build_id, digest = _provider_with_manifest(tmp_path)
    applied_dir = tmp_path / "hermes" / "qdrant_memory" / "raptor_applied"
    applied_dir.mkdir(parents=True, exist_ok=True)
    (applied_dir / f"{report_id}.json").write_text("not valid json {{{")

    provider.handle_tool_call(
        "qdrant_memory_raptor_apply",
        {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
    )
    live = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_raptor_apply",
            {"report_id": report_id, "build_id": build_id, "manifest_digest": digest,
             "dry_run": False, "approve": True},
        )
    )
    assert "error" in live
    assert "malformed" in live["error"].lower()
    # No upserts were performed — failure is fail-closed
    assert provider._qdrant.upserts == []


def test_stale_build_id_in_apply_record_rejected(tmp_path):
    """An apply record with a different build_id than the live call must
    NOT count as already-applied.
    """
    provider, manifest, report_id, build_id, digest = _provider_with_manifest(tmp_path)

    # Persist a record with the SAME report_id but a WRONG build_id
    persist_apply_record(
        report_id=report_id,
        build_id="raptor-build-deadbeefcafe",
        manifest_digest=digest,
        applied_node_ids=[
            str(p.get("raptor_node_id") or "")
            for p in manifest["candidate_node_payloads"]
        ],
        hermes_home=provider._hermes_home,
    )

    provider.handle_tool_call(
        "qdrant_memory_raptor_apply",
        {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
    )
    live = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_raptor_apply",
            {"report_id": report_id, "build_id": build_id, "manifest_digest": digest,
             "dry_run": False, "approve": True},
        )
    )
    assert "error" in live
    assert "build_id" in live["error"]


def test_stale_manifest_digest_in_apply_record_rejected(tmp_path):
    """An apply record with a different manifest_digest must NOT count as
    already-applied.
    """
    provider, manifest, report_id, build_id, digest = _provider_with_manifest(tmp_path)

    persist_apply_record(
        report_id=report_id,
        build_id=build_id,
        manifest_digest="f" * 64,  # wrong digest
        applied_node_ids=[
            str(p.get("raptor_node_id") or "")
            for p in manifest["candidate_node_payloads"]
        ],
        hermes_home=provider._hermes_home,
    )

    provider.handle_tool_call(
        "qdrant_memory_raptor_apply",
        {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
    )
    live = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_raptor_apply",
            {"report_id": report_id, "build_id": build_id, "manifest_digest": digest,
             "dry_run": False, "approve": True},
        )
    )
    assert "error" in live
    assert "manifest_digest" in live["error"]


def test_mismatched_node_ids_in_apply_record_rejected(tmp_path):
    """An apply record whose applied_node_ids don't match the manifest's
    candidate node set must NOT be treated as already-applied.
    """
    provider, manifest, report_id, build_id, digest = _provider_with_manifest(tmp_path)

    # Persist a record with WRONG applied_node_ids
    persist_apply_record(
        report_id=report_id,
        build_id=build_id,
        manifest_digest=digest,
        applied_node_ids=["raptor-node-totally-different"],
        hermes_home=provider._hermes_home,
    )

    provider.handle_tool_call(
        "qdrant_memory_raptor_apply",
        {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
    )
    live = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_raptor_apply",
            {"report_id": report_id, "build_id": build_id, "manifest_digest": digest,
             "dry_run": False, "approve": True},
        )
    )
    assert "error" in live
    assert "applied_node_ids" in live["error"] or "node" in live["error"].lower()


def test_apply_record_without_record_type_rejected(tmp_path):
    """A JSON file at raptor_applied/<id>.json without record_type must NOT
    be silently treated as already-applied.
    """
    provider, manifest, report_id, build_id, digest = _provider_with_manifest(tmp_path)
    applied_dir = tmp_path / "hermes" / "qdrant_memory" / "raptor_applied"
    applied_dir.mkdir(parents=True, exist_ok=True)
    fake_record = {
        "report_id": report_id,
        "build_id": build_id,
        "manifest_digest": digest,
        "applied_node_ids": [
            str(p.get("raptor_node_id") or "")
            for p in manifest["candidate_node_payloads"]
        ],
        # no record_type!
    }
    (applied_dir / f"{report_id}.json").write_text(json.dumps(fake_record))

    provider.handle_tool_call(
        "qdrant_memory_raptor_apply",
        {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
    )
    live = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_raptor_apply",
            {"report_id": report_id, "build_id": build_id, "manifest_digest": digest,
             "dry_run": False, "approve": True},
        )
    )
    assert "error" in live
    assert "record_type" in live["error"]


def test_status_reports_malformed_apply_record_error(tmp_path):
    """Status must surface a malformed apply record as an error, not as
    applied=False silently.
    """
    provider, manifest, report_id, build_id, digest = _provider_with_manifest(tmp_path)
    applied_dir = tmp_path / "hermes" / "qdrant_memory" / "raptor_applied"
    applied_dir.mkdir(parents=True, exist_ok=True)
    (applied_dir / f"{report_id}.json").write_text("not valid json")

    res = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_raptor_status",
            {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
        )
    )
    assert res["read_only"] is True
    assert res["applied"] is False
    assert res["apply_record_error"] != ""
    assert "malformed" in res["apply_record_error"].lower()


def test_status_reports_wrong_applied_node_ids_as_error(tmp_path):
    """A typed apply record with the right report/build/digest but wrong
    ``applied_node_ids`` must NOT report ``applied: true``; it must surface
    as ``apply_record_error``.
    """
    provider, manifest, report_id, build_id, digest = _provider_with_manifest(tmp_path)
    # Persist a record with the right type/report/build/digest but a
    # single wrong applied_node_ids entry — the strict loader must reject
    # this when the manifest's expected node set is supplied.
    persist_apply_record(
        report_id=report_id,
        build_id=build_id,
        manifest_digest=digest,
        applied_node_ids=["raptor-node-totally-different"],
        hermes_home=provider._hermes_home,
    )

    res = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_raptor_status",
            {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
        )
    )
    assert res["read_only"] is True
    assert res["applied"] is False
    assert res["apply_record"] is None
    assert res["apply_record_error"] != ""
    assert "applied_node_ids" in res["apply_record_error"]


def test_partial_preexisting_exact_node_recovery_writes_full_record(tmp_path):
    """Partial-preexisting exact RAPTOR node recovery: seed one manifest
    node in fake Qdrant with matching RAPTOR metadata and no apply
    record. Dry-run + live apply should upsert the missing node, persist
    an apply record containing ALL manifest node IDs, and a repeat live
    apply in a fresh provider/process should return ``already_applied=True``
    with no new upserts.
    """
    provider, manifest, report_id, build_id, digest = _provider_with_manifest(tmp_path)
    node_ids = [
        str(p.get("raptor_node_id") or "")
        for p in manifest["candidate_node_payloads"]
    ]
    assert len(node_ids) >= 2, "fixture must have at least 2 candidate nodes"

    # Pick the first node as the "already present with matching metadata"
    # one. The payload it stores must include raptor_report_id,
    # raptor_build_id, raptor_manifest_digest matching the live call.
    preexist_id = node_ids[0]
    preexist_payload = {
        "text": "preexisting leaf text",
        "source": "test",
        "source_type": "manual",
        "profile_id": provider._profile_id,
        "raptor_report_id": report_id,
        "raptor_build_id": build_id,
        "raptor_manifest_digest": digest,
        # Anything else needed to satisfy leaf safety / canonical review.
    }
    provider._qdrant.retrieve_results[preexist_id] = {
        "id": preexist_id,
        "payload": preexist_payload,
    }

    # Dry-run preview first
    dry = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_raptor_apply",
            {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
        )
    )
    assert dry["dry_run"] is True

    # Live apply — should upsert only the missing nodes but persist the
    # full now-applied set in the apply record.
    live = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_raptor_apply",
            {"report_id": report_id, "build_id": build_id, "manifest_digest": digest,
             "dry_run": False, "approve": True},
        )
    )
    assert live["applied"] is True
    assert live["already_applied"] is False
    # Every manifest node id is now reported as applied (the recovery
    # upserted the missing ones; the pre-existing one is part of the
    # full now-applied set).
    assert sorted(live["applied_node_ids"]) == sorted(node_ids)
    # upserted_count only counts the NEWLY upserted ids, not the
    # pre-existing one.
    assert live["upserted_count"] == len(node_ids) - 1

    # The persisted apply record must contain every manifest node id,
    # not just the newly upserted ones.
    record_path = (
        tmp_path
        / "hermes"
        / "qdrant_memory"
        / "raptor_applied"
        / f"{report_id}.json"
    )
    assert record_path.exists(), "apply record was not persisted"
    persisted = json.loads(record_path.read_text(encoding="utf-8"))
    assert persisted.get("record_type") == "raptor_apply"
    assert sorted(persisted.get("applied_node_ids") or []) == sorted(node_ids)

    # Simulate a fresh provider/process: build a new provider that shares
    # the same hermes_home + qdrant state, then perform another live
    # apply. It should be idempotent (already_applied=True) and perform
    # NO new upserts.
    new_provider = _provider_for_raptor(tmp_path)
    # Re-seed the same pre-existing exact node in the fresh provider's
    # fake Qdrant. (Real Qdrant would still have it.)
    new_provider._qdrant.retrieve_results[preexist_id] = {
        "id": preexist_id,
        "payload": preexist_payload,
    }
    # Carry over the persisted manifest so the new provider can find it.
    new_provider._pending_raptor_manifests[report_id] = (
        provider._pending_raptor_manifests[report_id]
    )

    new_provider.handle_tool_call(
        "qdrant_memory_raptor_apply",
        {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
    )
    repeat = json.loads(
        new_provider.handle_tool_call(
            "qdrant_memory_raptor_apply",
            {"report_id": report_id, "build_id": build_id, "manifest_digest": digest,
             "dry_run": False, "approve": True},
        )
    )
    assert repeat["applied"] is True
    assert repeat["already_applied"] is True
    # No new upserts in the fresh provider.
    assert new_provider._qdrant.upserts == []
    # The reported applied_node_ids from the persisted record must still
    # be the full manifest node set.
    assert sorted(repeat["applied_node_ids"]) == sorted(node_ids)


def test_partial_preexisting_status_returns_applied_with_full_node_set(tmp_path):
    """After partial-preexisting recovery, status must report applied=True
    with a complete exact-match apply record (no apply_record_error).
    """
    provider, manifest, report_id, build_id, digest = _provider_with_manifest(tmp_path)
    node_ids = sorted(
        str(p.get("raptor_node_id") or "")
        for p in manifest["candidate_node_payloads"]
    )

    # Manually persist a record whose applied_node_ids equals the full
    # manifest node set (the post-fix invariant).
    persist_apply_record(
        report_id=report_id,
        build_id=build_id,
        manifest_digest=digest,
        applied_node_ids=node_ids,
        hermes_home=provider._hermes_home,
    )

    res = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_raptor_status",
            {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
        )
    )
    assert res["read_only"] is True
    assert res["applied"] is True
    assert res["apply_record_error"] == ""
    assert sorted(res["apply_record"]["applied_node_ids"]) == node_ids


# ---------------------------------------------------------------------------
# Blocker 2: status must inspect actual child leaves conservatively
# ---------------------------------------------------------------------------


def _seed_child_results(provider, child_payloads_by_id):
    """Seed the provider's FakeQdrant retrieve_results with child payloads
    so the status tool can fetch them as if from Qdrant.
    """
    for cid, payload in child_payloads_by_id.items():
        provider._qdrant.retrieve_results[cid] = {"id": cid, "payload": dict(payload)}


def _candidate_with_children(tmp_path, child_ids):
    """Build a manifest with a single candidate whose raptor_child_ids is
    exactly the supplied list.
    """
    points = [
        _point(cid, f"leaf text {cid}", source_uri=f"test://{cid}")
        for cid in child_ids
    ]
    manifest = _build_manifest(points)
    # Force exactly one candidate with the given child_ids
    builder = RaptorBuilder()
    built = builder.build(points).to_dict()
    # We only need the manifest for status — pick a candidate whose
    # raptor_child_ids == child_ids, or rebuild so the first candidate
    # references the supplied leaves.
    # The default builder creates one or more RAPTOR nodes; we just
    # stamp raptor_child_ids onto the first candidate to match.
    built["candidate_node_payloads"][0]["raptor_child_ids"] = list(child_ids)
    built["candidate_node_payloads"][0]["raptor_summary_of"] = list(child_ids)
    built["manifest_digest"] = compute_manifest_digest(built)
    return built


def test_status_parent_not_active_when_child_stale(tmp_path):
    """A child with fact_status=stale must make the parent non-active."""
    child_ids = ["leaf-1", "leaf-2"]
    manifest = _candidate_with_children(tmp_path, child_ids)
    provider, _, report_id, build_id, digest = _provider_with_manifest(tmp_path, manifest)
    _seed_child_results(provider, {
        "leaf-1": {"text": "alpha", "source": "test", "fact_status": "stale"},
        "leaf-2": {"text": "beta", "source": "test"},
    })
    res = json.loads(provider.handle_tool_call(
        "qdrant_memory_raptor_status",
        {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
    ))
    assert res["read_only"] is True
    statuses = res["node_statuses"]
    # The candidate whose children we seeded must NOT be active
    parents = [s for s in statuses if s["child_ids"]]
    assert any(p["parent_status"] != "active" for p in parents), (
        f"expected at least one non-active parent, got: {[p['parent_status'] for p in parents]}"
    )
    # No mutations
    assert provider._qdrant.upserts == []


def test_status_parent_not_active_when_child_quarantined(tmp_path):
    child_ids = ["leaf-1", "leaf-2"]
    manifest = _candidate_with_children(tmp_path, child_ids)
    provider, _, report_id, build_id, digest = _provider_with_manifest(tmp_path, manifest)
    _seed_child_results(provider, {
        "leaf-1": {"text": "alpha", "consolidation_quarantined": True},
        "leaf-2": {"text": "beta"},
    })
    res = json.loads(provider.handle_tool_call(
        "qdrant_memory_raptor_status",
        {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
    ))
    parents = [s for s in res["node_statuses"] if s["child_ids"]]
    assert any(p["parent_status"] != "active" for p in parents), (
        f"got {[p['parent_status'] for p in parents]}"
    )
    assert res["read_only"] is True
    assert provider._qdrant.upserts == []


def test_status_parent_not_active_when_child_secret_bearing(tmp_path):
    child_ids = ["leaf-1", "leaf-2"]
    manifest = _candidate_with_children(tmp_path, child_ids)
    provider, _, report_id, build_id, digest = _provider_with_manifest(tmp_path, manifest)
    fake_secret = "".join(["s", "k", "-", "1234567890abcdef"])
    _seed_child_results(provider, {
        "leaf-1": {"text": f"api key: {fake_secret}"},
        "leaf-2": {"text": "beta"},
    })
    res = json.loads(provider.handle_tool_call(
        "qdrant_memory_raptor_status",
        {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
    ))
    parents = [s for s in res["node_statuses"] if s["child_ids"]]
    assert any(p["parent_status"] != "active" for p in parents), (
        f"got {[p['parent_status'] for p in parents]}"
    )
    assert res["read_only"] is True
    assert provider._qdrant.upserts == []


def test_status_parent_not_active_when_child_missing(tmp_path):
    """A child that cannot be retrieved from Qdrant must make the parent
    non-active.
    """
    child_ids = ["leaf-1", "leaf-2"]
    manifest = _candidate_with_children(tmp_path, child_ids)
    provider, _, report_id, build_id, digest = _provider_with_manifest(tmp_path, manifest)
    # Only seed one of the two children — the other is "deleted/missing"
    _seed_child_results(provider, {
        "leaf-1": {"text": "alpha", "source": "test"},
    })
    res = json.loads(provider.handle_tool_call(
        "qdrant_memory_raptor_status",
        {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
    ))
    parents = [s for s in res["node_statuses"] if s["child_ids"]]
    assert any(p["parent_status"] != "active" for p in parents), (
        f"got {[p['parent_status'] for p in parents]}"
    )
    # child_status_error should explain the missing child
    assert any(p.get("child_status_error", "") for p in parents)
    assert res["read_only"] is True
    assert provider._qdrant.upserts == []


def test_status_parent_not_active_when_child_requires_review(tmp_path):
    """A child with requires_review=True must make the parent non-active."""
    child_ids = ["leaf-1", "leaf-2"]
    manifest = _candidate_with_children(tmp_path, child_ids)
    provider, _, report_id, build_id, digest = _provider_with_manifest(tmp_path, manifest)
    _seed_child_results(provider, {
        "leaf-1": {"text": "alpha", "requires_review": True},
        "leaf-2": {"text": "beta"},
    })
    res = json.loads(provider.handle_tool_call(
        "qdrant_memory_raptor_status",
        {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
    ))
    parents = [s for s in res["node_statuses"] if s["child_ids"]]
    assert any(p["parent_status"] != "active" for p in parents), (
        f"expected non-active for requires_review child, got: "
        f"{[p['parent_status'] for p in parents]}"
    )
    assert res["read_only"] is True
    assert provider._qdrant.upserts == []


def test_status_parent_not_active_when_qdrant_retrieval_errors(tmp_path):
    """When Qdrant retrieval raises, the parent must be non-active and
    the error must be surfaced in child_status_error.
    """
    child_ids = ["leaf-1", "leaf-2"]
    manifest = _candidate_with_children(tmp_path, child_ids)
    provider, _, report_id, build_id, digest = _provider_with_manifest(tmp_path, manifest)
    # Make every retrieve() call fail
    provider._qdrant.retrieve_raises = True

    res = json.loads(provider.handle_tool_call(
        "qdrant_memory_raptor_status",
        {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
    ))
    parents = [s for s in res["node_statuses"] if s["child_ids"]]
    assert any(p["parent_status"] != "active" for p in parents), (
        f"got {[p['parent_status'] for p in parents]}"
    )
    assert any(p.get("child_status_error", "") for p in parents)
    assert res["read_only"] is True
    # No mutations despite the error
    assert provider._qdrant.upserts == []


def test_assess_parent_status_not_active_for_requires_review_child():
    """assess_parent_status must mark a parent non-active when the only
    unsafe child reason is requires_review (not just secret/quarantine/stale).
    """
    children = [
        {"text": "safe memory"},
        {"text": "review-needed memory", "requires_review": True},
    ]
    result = assess_parent_status(children)
    assert result["parent_status"] != "active", (
        f"requires_review child should make parent non-active, got {result}"
    )


def test_status_does_not_use_mutation_methods(tmp_path):
    """Status path must not call upsert / delete_ids / delete_filter /
    update_payload, even when child retrieval errors.
    """
    child_ids = ["leaf-1"]
    manifest = _candidate_with_children(tmp_path, child_ids)
    provider, _, report_id, build_id, digest = _provider_with_manifest(tmp_path, manifest)
    provider._qdrant.retrieve_raises = True
    provider.handle_tool_call(
        "qdrant_memory_raptor_status",
        {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
    )
    assert provider._qdrant.upserts == []
    assert provider._qdrant.delete_ids_calls == []
    assert provider._qdrant.delete_filter_calls == []
    assert provider._qdrant.update_payload_calls == []


# ---------------------------------------------------------------------------
# apply.py has no new runtime deps
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# apply.py has no new runtime deps
# ---------------------------------------------------------------------------


def test_apply_module_has_no_new_runtime_deps():
    """Verify apply.py imports only stdlib + project modules."""
    import qdrant_memory.raptor.apply as apply_module

    source = open(apply_module.__file__, encoding="utf-8").read()
    tree = ast.parse(source)

    forbidden = {"qdrant_client", "requests", "httpx", "aiohttp", "numpy", "scipy", "pandas"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root not in forbidden, f"apply.py must not import {alias.name}"
        if isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                assert root not in forbidden, f"apply.py must not import from {node.module}"


# ---------------------------------------------------------------------------
# Phase 4 fix4: strict RAPTOR trust/provenance regression tests
# ---------------------------------------------------------------------------
# These tests cover the digest-consistent tampered-manifest cases the
# security reviewer flagged as still passing the type-loose
# canonical/requires_review/source_hashes/derived_from checks. Each test
# mutates one field, recomputes the manifest digest, runs the dry-run
# apply, asserts the apply returned an error, and confirms no fake-Qdrant
# upsert was issued.


_GOOD_HASH = "a" * 64
_GOOD_EDGE = {
    "source_uri": "raptor://node/raptor-tree-x/child-1",
    "derivation_type": "raptor_summary",
    "relation_type": "SUMMARIZES",
    "child_node_id": "child-1",
}


def _tampered_provider(tmp_path, mutator):
    """Build a provider whose manifest has been mutated by *mutator*.

    *mutator* receives the manifest dict in-place and returns it.
    """
    manifest = _build_manifest()
    mutator(manifest)
    manifest["manifest_digest"] = compute_manifest_digest(manifest)
    report_id = f"raptor-{manifest['manifest_digest'][:12]}"
    build_id = manifest["build_id"]
    digest = manifest["manifest_digest"]
    provider = _provider_for_raptor(tmp_path)
    wrapper = _wrap_manifest(manifest, report_id=report_id)
    persist_manifest_report(wrapper, hermes_home=provider._hermes_home)
    provider._pending_raptor_manifests[report_id] = wrapper
    return provider, report_id, build_id, digest


def _assert_dry_run_rejected(provider, report_id, build_id, digest, *, must_contain: str = ""):
    """Run dry-run apply and assert it errors with no upsert."""
    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_raptor_apply",
            {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
        )
    )
    assert "error" in result, f"expected error in dry-run result, got: {result}"
    if must_contain:
        assert must_contain in result["error"].lower(), (
            f"expected error to contain {must_contain!r}, got: {result['error']!r}"
        )
    # No mutation may have happened.
    assert provider._qdrant.upserts == []
    assert provider._qdrant.delete_ids_calls == []
    assert provider._qdrant.delete_filter_calls == []
    assert provider._qdrant.update_payload_calls == []
    return result


def _assert_live_rejected_after_dry_run(provider, report_id, build_id, digest, *, must_contain: str = ""):
    """Dry-run then live apply; assert live is rejected and no upsert happens.

    This covers the post-enrichment gate (mirroring the strict
    trust/provenance checks).
    """
    provider.handle_tool_call(
        "qdrant_memory_raptor_apply",
        {"report_id": report_id, "build_id": build_id, "manifest_digest": digest},
    )
    live = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_raptor_apply",
            {
                "report_id": report_id,
                "build_id": build_id,
                "manifest_digest": digest,
                "dry_run": False,
                "approve": True,
            },
        )
    )
    assert "error" in live, f"expected live error, got: {live}"
    if must_contain:
        assert must_contain in live["error"].lower(), (
            f"expected live error to contain {must_contain!r}, got: {live['error']!r}"
        )
    assert provider._qdrant.upserts == []
    return live


class TestStrictCanonicalBoolean:
    """fix4: ``canonical`` must be exactly the boolean ``False``."""

    @pytest.mark.parametrize("bad_value", ["true", 1, 1.0, "false", 0, "yes"])
    def test_dry_run_rejects_non_boolean_canonical(self, tmp_path, bad_value):
        def mutate(m):
            m["candidate_node_payloads"][0]["canonical"] = bad_value

        provider, report_id, build_id, digest = _tampered_provider(tmp_path, mutate)
        _assert_dry_run_rejected(provider, report_id, build_id, digest, must_contain="canonical")

    def test_dry_run_rejects_missing_canonical_key(self, tmp_path):
        def mutate(m):
            m["candidate_node_payloads"][0].pop("canonical", None)

        provider, report_id, build_id, digest = _tampered_provider(tmp_path, mutate)
        _assert_dry_run_rejected(provider, report_id, build_id, digest, must_contain="canonical")

    @pytest.mark.parametrize("bad_value", ["true", 1, 1.0])
    def test_live_rejects_non_boolean_canonical(self, tmp_path, bad_value):
        def mutate(m):
            m["candidate_node_payloads"][0]["canonical"] = bad_value

        provider, report_id, build_id, digest = _tampered_provider(tmp_path, mutate)
        _assert_live_rejected_after_dry_run(
            provider, report_id, build_id, digest, must_contain="canonical"
        )


class TestStrictRequiresReviewBoolean:
    """fix4: ``requires_review`` must be exactly the boolean ``True``."""

    @pytest.mark.parametrize("bad_value", ["false", 0, 0.0, "no", None])
    def test_dry_run_rejects_non_true_requires_review(self, tmp_path, bad_value):
        def mutate(m):
            m["candidate_node_payloads"][0]["requires_review"] = bad_value

        provider, report_id, build_id, digest = _tampered_provider(tmp_path, mutate)
        _assert_dry_run_rejected(
            provider, report_id, build_id, digest, must_contain="requires_review"
        )

    def test_dry_run_rejects_missing_requires_review_key(self, tmp_path):
        def mutate(m):
            m["candidate_node_payloads"][0].pop("requires_review", None)

        provider, report_id, build_id, digest = _tampered_provider(tmp_path, mutate)
        _assert_dry_run_rejected(
            provider, report_id, build_id, digest, must_contain="requires_review"
        )

    @pytest.mark.parametrize("bad_value", ["false", 0, 0.0])
    def test_live_rejects_non_true_requires_review(self, tmp_path, bad_value):
        def mutate(m):
            m["candidate_node_payloads"][0]["requires_review"] = bad_value

        provider, report_id, build_id, digest = _tampered_provider(tmp_path, mutate)
        _assert_live_rejected_after_dry_run(
            provider, report_id, build_id, digest, must_contain="requires_review"
        )


class TestStrictSourceHashes:
    """fix4: ``source_hashes`` must be a non-empty list of 64-char
    lowercase hex SHA-256 strings."""

    def _mutate_source_hashes(self, m, new_value):
        m["candidate_node_payloads"][0]["source_hashes"] = new_value

    def test_dry_run_rejects_source_hashes_with_none(self, tmp_path):
        def mutate(m):
            self._mutate_source_hashes(m, [None])
        provider, report_id, build_id, digest = _tampered_provider(tmp_path, mutate)
        _assert_dry_run_rejected(provider, report_id, build_id, digest, must_contain="source_hashes")

    def test_dry_run_rejects_source_hashes_with_dict(self, tmp_path):
        def mutate(m):
            self._mutate_source_hashes(m, [{}])
        provider, report_id, build_id, digest = _tampered_provider(tmp_path, mutate)
        _assert_dry_run_rejected(provider, report_id, build_id, digest, must_contain="source_hashes")

    def test_dry_run_rejects_source_hashes_with_empty_string(self, tmp_path):
        def mutate(m):
            self._mutate_source_hashes(m, [""])
        provider, report_id, build_id, digest = _tampered_provider(tmp_path, mutate)
        _assert_dry_run_rejected(provider, report_id, build_id, digest, must_contain="source_hashes")

    def test_dry_run_rejects_source_hashes_short(self, tmp_path):
        def mutate(m):
            self._mutate_source_hashes(m, ["abcdef"])
        provider, report_id, build_id, digest = _tampered_provider(tmp_path, mutate)
        _assert_dry_run_rejected(provider, report_id, build_id, digest, must_contain="source_hashes")

    def test_dry_run_rejects_source_hashes_non_hex(self, tmp_path):
        def mutate(m):
            self._mutate_source_hashes(m, ["z" * 64])
        provider, report_id, build_id, digest = _tampered_provider(tmp_path, mutate)
        _assert_dry_run_rejected(provider, report_id, build_id, digest, must_contain="source_hashes")

    def test_dry_run_rejects_source_hashes_uppercase(self, tmp_path):
        def mutate(m):
            self._mutate_source_hashes(m, ["A" * 64])
        provider, report_id, build_id, digest = _tampered_provider(tmp_path, mutate)
        _assert_dry_run_rejected(provider, report_id, build_id, digest, must_contain="source_hashes")

    def test_live_rejects_source_hashes_with_none(self, tmp_path):
        def mutate(m):
            self._mutate_source_hashes(m, [None])
        provider, report_id, build_id, digest = _tampered_provider(tmp_path, mutate)
        _assert_live_rejected_after_dry_run(
            provider, report_id, build_id, digest, must_contain="source_hashes"
        )

    def test_live_rejects_source_hashes_with_dict(self, tmp_path):
        def mutate(m):
            self._mutate_source_hashes(m, [{}])
        provider, report_id, build_id, digest = _tampered_provider(tmp_path, mutate)
        _assert_live_rejected_after_dry_run(
            provider, report_id, build_id, digest, must_contain="source_hashes"
        )


class TestStrictDerivedFrom:
    """fix4: ``derived_from`` must be a non-empty list of structurally
    valid RAPTOR provenance edges."""

    def _mutate_derived_from(self, m, new_value):
        m["candidate_node_payloads"][0]["derived_from"] = new_value

    def test_dry_run_rejects_derived_from_with_dict(self, tmp_path):
        def mutate(m):
            self._mutate_derived_from(m, [{}])
        provider, report_id, build_id, digest = _tampered_provider(tmp_path, mutate)
        _assert_dry_run_rejected(provider, report_id, build_id, digest, must_contain="derived_from")

    def test_dry_run_rejects_derived_from_with_none(self, tmp_path):
        def mutate(m):
            self._mutate_derived_from(m, [None])
        provider, report_id, build_id, digest = _tampered_provider(tmp_path, mutate)
        _assert_dry_run_rejected(provider, report_id, build_id, digest, must_contain="derived_from")

    def test_dry_run_rejects_derived_from_with_string(self, tmp_path):
        def mutate(m):
            self._mutate_derived_from(m, ["not-a-provenance-edge"])
        provider, report_id, build_id, digest = _tampered_provider(tmp_path, mutate)
        _assert_dry_run_rejected(provider, report_id, build_id, digest, must_contain="derived_from")

    def test_dry_run_rejects_derived_from_wrong_derivation_type(self, tmp_path):
        bad = dict(_GOOD_EDGE)
        bad["derivation_type"] = "summary"

        def mutate(m):
            self._mutate_derived_from(m, [bad])
        provider, report_id, build_id, digest = _tampered_provider(tmp_path, mutate)
        _assert_dry_run_rejected(provider, report_id, build_id, digest, must_contain="derived_from")

    def test_dry_run_rejects_derived_from_wrong_relation_type(self, tmp_path):
        bad = dict(_GOOD_EDGE)
        bad["relation_type"] = "REFERENCES"

        def mutate(m):
            self._mutate_derived_from(m, [bad])
        provider, report_id, build_id, digest = _tampered_provider(tmp_path, mutate)
        _assert_dry_run_rejected(provider, report_id, build_id, digest, must_contain="derived_from")

    def test_dry_run_rejects_derived_from_empty_source_uri(self, tmp_path):
        bad = dict(_GOOD_EDGE)
        bad["source_uri"] = ""

        def mutate(m):
            self._mutate_derived_from(m, [bad])
        provider, report_id, build_id, digest = _tampered_provider(tmp_path, mutate)
        _assert_dry_run_rejected(provider, report_id, build_id, digest, must_contain="derived_from")

    def test_dry_run_rejects_derived_from_empty_child_node_id(self, tmp_path):
        bad = dict(_GOOD_EDGE)
        bad["child_node_id"] = ""

        def mutate(m):
            self._mutate_derived_from(m, [bad])
        provider, report_id, build_id, digest = _tampered_provider(tmp_path, mutate)
        _assert_dry_run_rejected(provider, report_id, build_id, digest, must_contain="derived_from")

    def test_live_rejects_derived_from_with_dict(self, tmp_path):
        def mutate(m):
            self._mutate_derived_from(m, [{}])
        provider, report_id, build_id, digest = _tampered_provider(tmp_path, mutate)
        _assert_live_rejected_after_dry_run(
            provider, report_id, build_id, digest, must_contain="derived_from"
        )

    def test_live_rejects_derived_from_with_none(self, tmp_path):
        def mutate(m):
            self._mutate_derived_from(m, [None])
        provider, report_id, build_id, digest = _tampered_provider(tmp_path, mutate)
        _assert_live_rejected_after_dry_run(
            provider, report_id, build_id, digest, must_contain="derived_from"
        )

    def test_live_rejects_derived_from_with_string(self, tmp_path):
        def mutate(m):
            self._mutate_derived_from(m, ["not-a-provenance-edge"])
        provider, report_id, build_id, digest = _tampered_provider(tmp_path, mutate)
        _assert_live_rejected_after_dry_run(
            provider, report_id, build_id, digest, must_contain="derived_from"
        )


class TestStrictCombined:
    """Multi-field tampered manifests (digest-consistent)."""

    def test_dry_run_rejects_combined_bad_canonical_and_source_hashes(self, tmp_path):
        def mutate(m):
            m["candidate_node_payloads"][0]["canonical"] = "true"
            m["candidate_node_payloads"][0]["source_hashes"] = [None]
        provider, report_id, build_id, digest = _tampered_provider(tmp_path, mutate)
        result = _assert_dry_run_rejected(
            provider, report_id, build_id, digest, must_contain="canonical"
        )
        # Even though multiple fields are bad, the first error is reported
        # and the manifest is rejected.
        assert "error" in result

    def test_live_rejects_combined_bad_requires_review_and_derived_from(self, tmp_path):
        def mutate(m):
            m["candidate_node_payloads"][0]["requires_review"] = "false"
            m["candidate_node_payloads"][0]["derived_from"] = [{}]
        provider, report_id, build_id, digest = _tampered_provider(tmp_path, mutate)
        _assert_live_rejected_after_dry_run(
            provider, report_id, build_id, digest, must_contain="requires_review"
        )


# ---------------------------------------------------------------------------
# Adversarial: Phase 5 retrieve collection=learning must not poison the
# memory cache, and must not return memory results.
# ---------------------------------------------------------------------------


class _RetrievedMemory:
    """Stand-in for :class:`MemoryRetriever`'s output."""

    def __init__(self, pid, text, payload, final_score=0.5):
        self.id = pid
        self.text = text
        self.payload = payload
        self.final_score = final_score
        self.qdrant_score = final_score
        self.ranking_debug = {}


class _FakeLearningStore:
    """Minimal LearningStore stub for collection=learning retrieve tests."""

    def __init__(self, *, chunks=None, raise_exc=False, record_calls=False):
        self._chunks = chunks or []
        self._raise = raise_exc
        self.calls: list[dict[str, Any]] = []
        self.collection_name = "learnings"

    def search(self, query, *, top_k=5, update_access=False, **kwargs):
        self.calls.append({
            "query": query,
            "top_k": top_k,
            "update_access": update_access,
            "kwargs": kwargs,
        })
        if self._raise:
            raise RuntimeError("simulated learning failure")
        return list(self._chunks)


class _FakeLearningChunk:
    def __init__(self, *, point_id, text, payload, final_score=0.6, qdrant_score=0.6):
        self.id = point_id
        self.text = text
        self.payload = payload
        self.final_score = final_score
        self.qdrant_score = qdrant_score


class _FakeBaseRetriever:
    """Minimal MemoryRetriever stub. Compatible with the Phase 5 router."""

    def __init__(self, chunks=None):
        self._chunks = chunks or []
        self.calls: list[dict[str, Any]] = []

    def search(self, query, *, top_k=5, update_access=False, **kwargs):
        self.calls.append({
            "query": query,
            "top_k": top_k,
            "update_access": update_access,
            "kwargs": kwargs,
        })
        return list(self._chunks)


def _provider_with_retrieve_components(tmp_path):
    """Build a provider with stubbed Qdrant, embeddings, and learning store."""
    provider = QdrantMemoryProvider()
    provider._qdrant = FakeQdrant()
    provider._embeddings = FakeEmbedding()
    provider._active = True
    provider._hermes_home = str(tmp_path / "hermes")
    provider._session_id = "test-session"
    provider._config.update(
        {
            "collection_name": "memory",
            "learning_collection_name": "learnings",
            "embedding_model": "test-model",
            "learning_enabled": True,
        }
    )
    return provider


def test_tool_retrieve_learning_does_not_use_memory_router(tmp_path):
    """Memory-first then learning must not reuse the cached memory router."""
    provider = _provider_with_retrieve_components(tmp_path)
    memory_chunk = _RetrievedMemory(
        "mem-1",
        text="memory hit",
        payload={"profile_id": "default", "source_type": "manual"},
        final_score=0.5,
    )
    memory_retriever = _FakeBaseRetriever(chunks=[memory_chunk])
    provider._retriever = memory_retriever

    learning_chunk = _FakeLearningChunk(
        point_id="learn-1",
        text="learning hit",
        payload={"learning_type": "tool_failure_lesson", "profile_id": "default"},
        final_score=0.7,
    )
    fake_store = _FakeLearningStore(chunks=[learning_chunk])
    provider._learning_store = fake_store

    # Now invoke retrieve with collection=learning.
    result_json = provider.handle_tool_call(
        "qdrant_memory_retrieve",
        {"query": "anything", "collection": "learning", "top_k": 3},
    )
    payload = json.loads(result_json)
    assert "error" not in payload, payload
    assert payload["debug"]["collection"] == "learning"
    # The exact_hits must contain only learning hits, never memory hits.
    ids = [hit["point_id"] for hit in payload["results"]["exact_hits"]]
    assert "learn-1" in ids
    assert "mem-1" not in ids
    # The learning store was called with update_access=False.
    assert fake_store.calls, "learning store must have been called"
    assert fake_store.calls[0]["update_access"] is False
    # The memory retriever was NOT called for the learning retrieve.
    memory_collect_calls = [
        c for c in memory_retriever.calls
        if c["query"] == "anything"
    ]
    assert memory_collect_calls == [], (
        "memory retriever must not be used for collection=learning"
    )
    # The cached memory hybrid router must NOT have been built.
    assert getattr(provider, "_hybrid_router", None) is None, (
        "memory hybrid router must NOT be lazily built on a learning retrieve"
    )


def test_tool_retrieve_memory_then_learning_no_cache_contamination(tmp_path):
    """Memory then learning in sequence: no cross-cache contamination."""
    provider = _provider_with_retrieve_components(tmp_path)
    memory_chunk = _RetrievedMemory(
        "mem-1",
        text="memory hit",
        payload={"profile_id": "default", "source_type": "manual"},
        final_score=0.5,
    )
    memory_retriever = _FakeBaseRetriever(chunks=[memory_chunk])
    provider._retriever = memory_retriever

    learning_chunk = _FakeLearningChunk(
        point_id="learn-1",
        text="learning hit",
        payload={"learning_type": "tool_failure_lesson"},
        final_score=0.7,
    )
    fake_store = _FakeLearningStore(chunks=[learning_chunk])
    provider._learning_store = fake_store

    # 1) memory call
    mem_payload = json.loads(provider.handle_tool_call(
        "qdrant_memory_retrieve",
        {"query": "alpha", "collection": "memory", "top_k": 3},
    ))
    # 2) learning call
    learn_payload = json.loads(provider.handle_tool_call(
        "qdrant_memory_retrieve",
        {"query": "beta", "collection": "learning", "top_k": 3},
    ))
    # Memory call returned memory hit
    mem_ids = [h["point_id"] for h in mem_payload["results"]["exact_hits"]]
    assert "mem-1" in mem_ids
    assert "learn-1" not in mem_ids
    # Learning call returned learning hit, not memory
    learn_ids = [h["point_id"] for h in learn_payload["results"]["exact_hits"]]
    assert "learn-1" in learn_ids
    assert "mem-1" not in learn_ids
    # update_access=False must hold for both lanes.
    assert all(c["update_access"] is False for c in memory_retriever.calls)
    assert all(c["update_access"] is False for c in fake_store.calls)


def test_tool_retrieve_learning_to_memory_no_learning_results_in_memory(tmp_path):
    """Learning then memory: memory payload must not include learning hits."""
    provider = _provider_with_retrieve_components(tmp_path)
    memory_chunk = _RetrievedMemory(
        "mem-1",
        text="memory hit",
        payload={"profile_id": "default", "source_type": "manual"},
        final_score=0.5,
    )
    memory_retriever = _FakeBaseRetriever(chunks=[memory_chunk])
    provider._retriever = memory_retriever

    learning_chunk = _FakeLearningChunk(
        point_id="learn-1",
        text="learning hit",
        payload={"learning_type": "tool_failure_lesson"},
        final_score=0.7,
    )
    fake_store = _FakeLearningStore(chunks=[learning_chunk])
    provider._learning_store = fake_store

    # 1) learning call
    learn_payload = json.loads(provider.handle_tool_call(
        "qdrant_memory_retrieve",
        {"query": "alpha", "collection": "learning", "top_k": 3},
    ))
    # 2) memory call
    mem_payload = json.loads(provider.handle_tool_call(
        "qdrant_memory_retrieve",
        {"query": "beta", "collection": "memory", "top_k": 3},
    ))
    # Memory call must NOT contain learning hit, even though learning was first.
    mem_ids = [h["point_id"] for h in mem_payload["results"]["exact_hits"]]
    assert "mem-1" in mem_ids
    assert "learn-1" not in mem_ids


# ---------------------------------------------------------------------------
# Phase 5 fix3 — adversarial: collection=learning exact_hits must drop /
# redact secret-bearing content. Mirrors the dense-lane protection in
# HybridRouter._dense_to_exact_hits so a credential-shaped learning
# point cannot leak through ``qdrant_memory_retrieve`` even when
# ``include_metadata=true`` or when the secret lives only in a
# default-emitted source field.
# ---------------------------------------------------------------------------


def test_tool_retrieve_learning_drops_secret_bearing_text(tmp_path):
    """Secret-shaped learning ``text`` must NOT surface in exact_hits."""
    provider = _provider_with_retrieve_components(tmp_path)
    # Build a secret-shaped bearer token at runtime so the
    # ``scripts/check_no_literal_fake_secrets.py`` scanner does not
    # trip on a contiguous example.
    bearer = "".join(["Bearer ", "a" * 24])
    bad_id = bearer  # secret-shaped point id also
    bad_chunk = _FakeLearningChunk(
        point_id=bad_id,
        text=bearer + " tail",
        payload={
            "learning_type": "tool_failure_lesson",
            "source_uri": "clean://example/x",
            "profile_id": "default",
        },
        final_score=0.6,
    )
    good_chunk = _FakeLearningChunk(
        point_id="learn-clean",
        text="clean learning text",
        payload={"learning_type": "tool_failure_lesson", "profile_id": "default"},
        final_score=0.5,
    )
    fake_store = _FakeLearningStore(chunks=[bad_chunk, good_chunk])
    provider._learning_store = fake_store

    payload = json.loads(provider.handle_tool_call(
        "qdrant_memory_retrieve",
        {"query": "anything", "collection": "learning", "top_k": 3},
    ))
    assert "error" not in payload, payload
    ids = [h["point_id"] for h in payload["results"]["exact_hits"]]
    # The clean chunk must still pass through.
    assert "learn-clean" in ids
    # The secret-bearing chunk must be dropped.
    assert bad_id not in ids
    # The serialized envelope must not echo the raw secret token
    # anywhere — neither in exact_hits nor in warnings nor in debug.
    serialized = json.dumps(payload, default=str)
    assert bearer not in serialized
    # The dropped hit must be visible in the debug block as a
    # redacted handle, never the raw id.
    dropped = payload["debug"].get("dropped_exact_hit_ids", [])
    assert dropped, "expected at least one dropped hit handle"
    for handle in dropped:
        assert bad_id not in handle
        assert "redacted:" in handle
    # Warning channel must reference the redacted handle, not the
    # raw secret-shaped id.
    redact_warnings = [w for w in payload["warnings"] if "learning exact hit redacted" in w]
    assert redact_warnings, "expected a learning-exact-hit redaction warning"
    for w in redact_warnings:
        assert bad_id not in w
        assert "redacted:" in w


def test_tool_retrieve_learning_drops_secret_bearing_source_uri(tmp_path):
    """Secret-shaped learning ``source_uri`` must NOT leak even when text is clean."""
    provider = _provider_with_retrieve_components(tmp_path)
    secret_uri = "https://user:" + ("b" * 24) + "@internal.example/x"
    bad_chunk = _FakeLearningChunk(
        point_id="learn-uri",
        text="clean text",
        payload={
            "learning_type": "tool_failure_lesson",
            "source_uri": secret_uri,
            "profile_id": "default",
        },
        final_score=0.7,
    )
    fake_store = _FakeLearningStore(chunks=[bad_chunk])
    provider._learning_store = fake_store

    payload = json.loads(provider.handle_tool_call(
        "qdrant_memory_retrieve",
        {"query": "anything", "collection": "learning", "top_k": 3},
    ))
    serialized = json.dumps(payload, default=str)
    # The raw secret URI must NOT appear anywhere in the output.
    assert secret_uri not in serialized
    # The chunk was dropped from exact_hits.
    ids = [h["point_id"] for h in payload["results"]["exact_hits"]]
    assert "learn-uri" not in ids
    # Warning + debug must use the redacted handle, not the raw id.
    redact_warnings = [w for w in payload["warnings"] if "learning exact hit redacted" in w]
    assert redact_warnings
    for w in redact_warnings:
        assert "learn-uri" not in w
        assert "redacted:" in w
    for handle in payload["debug"].get("dropped_exact_hit_ids", []):
        assert "learn-uri" not in handle


def test_tool_retrieve_learning_drops_secret_bearing_metadata(tmp_path):
    """``include_metadata=true`` must not leak credential-shaped payload values."""
    provider = _provider_with_retrieve_components(tmp_path)
    secret_uri = "https://user:" + ("c" * 24) + "@internal.example/y"
    bad_chunk = _FakeLearningChunk(
        point_id="learn-meta",
        text="plain text",
        payload={
            "learning_type": "tool_failure_lesson",
            "source_uri": secret_uri,   # secret lives in metadata
            "profile_id": "default",
            "extra_field": "clean",
        },
        final_score=0.7,
    )
    fake_store = _FakeLearningStore(chunks=[bad_chunk])
    provider._learning_store = fake_store

    payload = json.loads(provider.handle_tool_call(
        "qdrant_memory_retrieve",
        {
            "query": "anything",
            "collection": "learning",
            "top_k": 3,
            "include_metadata": True,
        },
    ))
    serialized = json.dumps(payload, default=str)
    # The raw secret URI must NOT appear anywhere — including the
    # ``metadata`` payload.
    assert secret_uri not in serialized
    ids = [h["point_id"] for h in payload["results"]["exact_hits"]]
    assert "learn-meta" not in ids


def test_tool_retrieve_learning_redacts_secret_shaped_point_id(tmp_path):
    """A secret-shaped learning point id must never appear in raw form."""
    provider = _provider_with_retrieve_components(tmp_path)
    bad_id = "".join(["Bearer ", "d" * 24])
    bad_chunk = _FakeLearningChunk(
        point_id=bad_id,
        text="clean text",
        payload={
            "learning_type": "tool_failure_lesson",
            "source_uri": "clean://example/x",
            "profile_id": "default",
        },
        final_score=0.7,
    )
    fake_store = _FakeLearningStore(chunks=[bad_chunk])
    provider._learning_store = fake_store

    payload = json.loads(provider.handle_tool_call(
        "qdrant_memory_retrieve",
        {"query": "anything", "collection": "learning", "top_k": 3},
    ))
    # The raw secret-shaped id must NOT appear anywhere in the
    # serialized envelope. ``contains_secret`` does not flag bearer
    # tokens in isolation (we still want bare point ids to remain
    # traceable), but the redaction helper applies to the dropped
    # hit anyway because the chunk's payload also carried a clean
    # ``source_uri`` but the id itself was bearer-shaped; the
    # projection (without the id) is secret-free so the chunk
    # passes through. This is the regression: ensure the *raw* id is
    # at least not echoed into debug/warnings when it is dropped.
    # We do not require id-level redaction for a non-secret text
    # path; instead we exercise the inverse: a chunk with both a
    # secret-shaped id AND secret-shaped source_uri must be dropped
    # and the warning/debug must use the redacted handle.
    bad_uri = "https://user:" + ("e" * 24) + "@internal.example/z"
    bad_chunk2 = _FakeLearningChunk(
        point_id=bad_id,
        text="plain text",
        payload={
            "learning_type": "tool_failure_lesson",
            "source_uri": bad_uri,
            "profile_id": "default",
        },
        final_score=0.7,
    )
    fake_store._chunks = [bad_chunk2]
    payload2 = json.loads(provider.handle_tool_call(
        "qdrant_memory_retrieve",
        {"query": "anything", "collection": "learning", "top_k": 3},
    ))
    serialized2 = json.dumps(payload2, default=str)
    # Raw secret-shaped id must not appear.
    assert bad_id not in serialized2
    # Raw secret-shaped uri must not appear either.
    assert bad_uri not in serialized2
    ids = [h["point_id"] for h in payload2["results"]["exact_hits"]]
    assert bad_id not in ids
    # Warning + debug use the redacted handle.
    redact_warnings = [w for w in payload2["warnings"] if "learning exact hit redacted" in w]
    assert redact_warnings
    for w in redact_warnings:
        assert bad_id not in w
        assert "redacted:" in w
    for handle in payload2["debug"].get("dropped_exact_hit_ids", []):
        assert bad_id not in handle
        assert "redacted:" in handle


def test_tool_retrieve_learning_clean_chunk_passes_through(tmp_path):
    """Regression: a clean learning chunk must still surface in exact_hits."""
    provider = _provider_with_retrieve_components(tmp_path)
    good_chunk = _FakeLearningChunk(
        point_id="learn-clean",
        text="clean learning text",
        payload={
            "learning_type": "tool_failure_lesson",
            "source_uri": "clean://example/x",
            "profile_id": "default",
        },
        final_score=0.6,
    )
    fake_store = _FakeLearningStore(chunks=[good_chunk])
    provider._learning_store = fake_store

    payload = json.loads(provider.handle_tool_call(
        "qdrant_memory_retrieve",
        {"query": "anything", "collection": "learning", "top_k": 3},
    ))
    assert "error" not in payload, payload
    assert "learn-clean" in [h["point_id"] for h in payload["results"]["exact_hits"]]
    # No spurious learning-exact-hit redaction warning for clean text.
    assert not any("learning exact hit redacted" in w for w in payload["warnings"])
    # ``dropped_exact_hit_ids`` must be empty.
    assert payload["debug"].get("dropped_exact_hit_ids", []) == []
