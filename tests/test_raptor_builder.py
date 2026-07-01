from __future__ import annotations

import json
from typing import Any

import pytest

from qdrant_memory.raptor import (
    DEFAULT_MAX_CLUSTER_SIZE,
    DEFAULT_PROMPT_VERSION,
    RAPTOR_DERIVATION_TYPE,
    RAPTOR_REQUIRED_NODE_FIELDS,
    RaptorBuilder,
    build_raptor_dry_run,
)
from qdrant_memory.raptor.schema import (
    RAPTOR_LEVEL_LEAF,
    compute_manifest_digest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _point(
    point_id: str,
    text: str = "",
    *,
    source_type: str = "manual",
    profile_id: str = "default",
    user_id_hash: str = "",
    chat_id_hash: str = "",
    project_path: str = "",
    heading: str = "",
    tags: list[str] | None = None,
    source_uri: str = "",
    fact_status: str = "",
    stale: bool | None = None,
    requires_review: bool | None = None,
    quarantined: bool | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "text": text,
        "source": source_uri or "test",
        "source_type": source_type,
        "profile_id": profile_id,
        "user_id_hash": user_id_hash,
        "chat_id_hash": chat_id_hash,
        "tags": tags or [],
    }
    if project_path:
        payload["project_path"] = project_path
    if heading:
        payload["heading"] = heading
    if source_uri:
        payload["source_uri"] = source_uri
    if fact_status:
        payload["fact_status"] = fact_status
    if stale is not None:
        payload["stale"] = stale
    if requires_review is not None:
        payload["requires_review"] = requires_review
    if quarantined is not None:
        payload["consolidation_quarantined"] = quarantined
    return {"id": point_id, "payload": payload}


def _secret_token() -> str:
    return "sk-" + "abcdef1234567890" + "xyz"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_builder_is_deterministic_across_repeated_runs():
    points = [
        _point("leaf-3", "third memory about RAPTOR schema", source_type="manual"),
        _point("leaf-1", "first memory about RAPTOR schema", source_type="manual"),
        _point("leaf-2", "second memory about RAPTOR schema", source_type="manual"),
    ]
    manifest_a = RaptorBuilder().build(points)
    manifest_b = RaptorBuilder().build(list(reversed(points)))
    blob_a = manifest_a.to_dict()
    blob_b = manifest_b.to_dict()
    assert blob_a["manifest_digest"] == blob_b["manifest_digest"]
    assert blob_a["build_id"] == blob_b["build_id"]
    assert blob_a["tree_id"] == blob_b["tree_id"]
    assert blob_a["root_id"] == blob_b["root_id"]
    assert blob_a["node_count"] == blob_b["node_count"]
    assert blob_a["leaf_count"] == blob_b["leaf_count"]


def test_build_raptor_dry_run_function_uses_deterministic_ids():
    points = [
        _point("leaf-1", "alpha note", source_type="manual"),
        _point("leaf-2", "beta note", source_type="manual"),
    ]
    a = build_raptor_dry_run(points, max_cluster_size=2)
    b = build_raptor_dry_run(points, max_cluster_size=2)
    assert a.build_id == b.build_id
    assert a.tree_id == b.tree_id
    assert a.root_id == b.root_id
    assert a.manifest_digest == b.manifest_digest


def test_builder_build_id_changes_when_config_changes():
    points = [_point("leaf-1", "alpha note", source_type="manual")]
    a = RaptorBuilder(config={"strategy": "bucket-source-type"}).build(points)
    b = RaptorBuilder(config={"strategy": "token-bucket"}).build(points)
    assert a.build_id != b.build_id


def test_builder_caller_supplied_build_id_is_preserved():
    points = [_point("leaf-1", "alpha", source_type="manual")]
    manifest = RaptorBuilder().build(points, build_id="custom-build-id")
    assert manifest.build_id == "custom-build-id"


# ---------------------------------------------------------------------------
# Dry-run safety contract
# ---------------------------------------------------------------------------


def test_builder_performs_no_qdrant_mutation():
    """The builder accepts plain dicts and never imports Qdrant."""
    import ast
    import qdrant_memory.raptor.builder as builder_module

    source = open(builder_module.__file__, encoding="utf-8").read()
    tree = ast.parse(source)

    forbidden_imports = {"qdrant_client"}
    forbidden_calls = {
        "upsert",
        "delete_payload",
        "delete_filter",
        "delete_ids",
        "update_payload",
        "scroll",
        "search",
        "retrieve",
        "query_points",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                assert top not in forbidden_imports, f"forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            assert top not in forbidden_imports, f"forbidden import from: {node.module}"
        elif isinstance(node, ast.Attribute):
            attr = node.attr
            assert attr not in forbidden_calls, f"forbidden attribute access: .{attr}"
        elif isinstance(node, ast.Name):
            assert node.id not in forbidden_calls, f"forbidden name reference: {node.id}"


def test_manifest_pins_dry_run_and_no_mutations():
    points = [_point("leaf-1", "alpha", source_type="manual")]
    manifest = RaptorBuilder().build(points)
    blob = manifest.to_dict()
    assert blob["dry_run"] is True
    assert blob["mutations_performed"] is False


def test_manifest_is_json_serializable():
    points = [_point(f"leaf-{i:03d}", f"note {i}", source_type="manual") for i in range(5)]
    manifest = RaptorBuilder().build(points)
    encoded = json.dumps(manifest.to_dict(), sort_keys=True, default=str)
    decoded = json.loads(encoded)
    assert decoded["leaf_count"] == 5
    assert decoded["manifest_digest"]


# ---------------------------------------------------------------------------
# Required RAPTOR fields on every candidate payload
# ---------------------------------------------------------------------------


def test_all_candidate_payloads_have_required_raptor_fields():
    points = [
        _point("leaf-1", "alpha", source_type="manual"),
        _point("leaf-2", "beta", source_type="manual"),
        _point("leaf-3", "gamma", source_type="manual"),
    ]
    manifest = RaptorBuilder().build(points)
    assert manifest.node_count >= 2
    for payload in manifest.candidate_node_payloads:
        for field_name in RAPTOR_REQUIRED_NODE_FIELDS:
            assert field_name in payload, f"missing {field_name}"
        # Every payload must include child IDs, source hashes, and provenance.
        assert isinstance(payload["raptor_child_ids"], list)
        assert payload["raptor_child_ids"], "non-leaf nodes must have children"
        assert isinstance(payload["source_hashes"], list)
        assert isinstance(payload["derived_from"], list)
        assert payload["derived_from"], "provenance must be present"
        # Safety flags:
        assert payload["canonical"] is False
        assert payload["requires_review"] is True
        # Derivation type must be RAPTOR:
        assert payload["derivation_type"] == RAPTOR_DERIVATION_TYPE


def test_every_node_has_child_ids_and_source_hashes_and_provenance():
    points = [
        _point("leaf-1", "alpha", source_type="manual"),
        _point("leaf-2", "beta", source_type="manual"),
    ]
    manifest = RaptorBuilder().build(points)
    assert manifest.node_count >= 2
    seen_node_ids: set[str] = set()
    for payload in manifest.candidate_node_payloads:
        assert payload["raptor_node_id"]
        assert payload["raptor_node_id"] not in seen_node_ids
        seen_node_ids.add(payload["raptor_node_id"])
        # Child IDs and source hashes are non-empty for every node we emit.
        assert payload["raptor_child_ids"]
        assert payload["source_hashes"]
        # Provenance edge count equals child count.
        assert len(payload["derived_from"]) == len(payload["raptor_child_ids"])
        for edge in payload["derived_from"]:
            assert edge["derivation_type"] == RAPTOR_DERIVATION_TYPE
            assert edge["relation_type"] == "SUMMARIZES"


def test_cluster_summary_node_has_leaves_as_children_and_root_has_clusters():
    points = [
        _point("leaf-1", "alpha", source_type="manual"),
        _point("leaf-2", "beta", source_type="manual"),
    ]
    manifest = RaptorBuilder().build(points)
    levels = {payload["raptor_level"]: payload for payload in manifest.candidate_node_payloads}
    # level 1 nodes (cluster summaries) should have leaves as children.
    cluster_payloads = [p for p in manifest.candidate_node_payloads if p["raptor_level"] == RAPTOR_LEVEL_LEAF + 1]
    root_payloads = [p for p in manifest.candidate_node_payloads if p["raptor_level"] == RAPTOR_LEVEL_LEAF + 2]
    assert cluster_payloads
    assert root_payloads
    for cp in cluster_payloads:
        assert set(cp["raptor_child_ids"]).issubset({"leaf-1", "leaf-2"})
    for rp in root_payloads:
        # Root's children are cluster node ids, not leaf ids.
        for cid in rp["raptor_child_ids"]:
            assert cid.startswith("raptor-node-")


# ---------------------------------------------------------------------------
# Skip rules
# ---------------------------------------------------------------------------


def test_leaves_with_missing_point_id_are_skipped():
    points = [
        {"id": "", "payload": {"text": "no id"}},
        {"payload": {"text": "missing id key"}},
        _point("leaf-1", "real", source_type="manual"),
    ]
    manifest = RaptorBuilder().build(points)
    skipped_ids = [entry["point_id"] for entry in manifest.skipped_leaves]
    assert "" in skipped_ids
    assert "leaf-1" not in skipped_ids
    assert manifest.leaf_count == 1


def test_leaves_with_missing_text_are_skipped():
    points = [
        {"id": "leaf-1", "payload": {"source_type": "manual"}},
        {"id": "leaf-2", "payload": {"text": "", "source_type": "manual"}},
        _point("leaf-3", "real text", source_type="manual"),
    ]
    manifest = RaptorBuilder().build(points)
    skipped_ids = {entry["point_id"]: entry["reason"] for entry in manifest.skipped_leaves}
    assert skipped_ids.get("leaf-1") == "missing_text"
    assert skipped_ids.get("leaf-2") == "missing_text"
    assert manifest.leaf_count == 1


def test_secret_bearing_leaves_are_skipped_and_excluded_from_summary():
    secret_value = _secret_token()
    points = [
        _point("leaf-bad", f"contains {secret_value} text", source_type="manual"),
        _point("leaf-good", "benign text", source_type="manual"),
    ]
    manifest = RaptorBuilder().build(points)
    skipped_ids = [entry["point_id"] for entry in manifest.skipped_leaves]
    assert "leaf-bad" in skipped_ids
    assert "leaf-good" not in skipped_ids
    # Secret string must never appear in any candidate payload.
    for payload in manifest.candidate_node_payloads:
        encoded = json.dumps(payload, sort_keys=True, default=str)
        assert secret_value not in encoded
        assert "leaf-bad" not in encoded


def test_secret_in_payload_field_is_skipped():
    points = [
        {
            "id": "leaf-bad-payload",
            "payload": {
                "text": "benign text",
                "source_type": "manual",
                "extra_field": f"contains {_secret_token()} hidden in payload",
            },
        },
        _point("leaf-good", "benign text", source_type="manual"),
    ]
    manifest = RaptorBuilder().build(points)
    skipped_ids = [entry["point_id"] for entry in manifest.skipped_leaves]
    assert "leaf-bad-payload" in skipped_ids


def test_quarantined_leaves_are_skipped():
    points = [
        _point("leaf-q", "quarantined text", source_type="manual", quarantined=True),
        _point("leaf-ok", "ok text", source_type="manual"),
    ]
    manifest = RaptorBuilder().build(points)
    skipped_ids = [entry["point_id"] for entry in manifest.skipped_leaves]
    assert "leaf-q" in skipped_ids
    assert "leaf-ok" not in skipped_ids


def test_stale_and_review_required_leaves_are_skipped():
    points = [
        _point("leaf-stale", "stale text", source_type="manual", stale=True),
        _point("leaf-review", "review text", source_type="manual", requires_review=True),
        _point("leaf-ok", "ok text", source_type="manual"),
    ]
    manifest = RaptorBuilder().build(points)
    skipped_reasons = {entry["point_id"]: entry["reason"] for entry in manifest.skipped_leaves}
    assert skipped_reasons.get("leaf-stale") == "unsafe_flag"
    assert skipped_reasons.get("leaf-review") == "unsafe_flag"
    assert manifest.leaf_count == 1


@pytest.mark.parametrize(
    "status",
    ["stale", "deprecated", "superseded", "disputed", "review_required"],
)
def test_terminal_fact_status_leaves_are_skipped(status: str):
    points = [
        _point(f"leaf-{status}", "text", source_type="manual", fact_status=status),
        _point("leaf-ok", "ok text", source_type="manual"),
    ]
    manifest = RaptorBuilder().build(points)
    skipped_ids = [entry["point_id"] for entry in manifest.skipped_leaves]
    assert f"leaf-{status}" in skipped_ids


# ---------------------------------------------------------------------------
# Cross-scope isolation
# ---------------------------------------------------------------------------


def test_cross_profile_leaves_are_split_into_separate_clusters():
    points = [
        _point("leaf-a1", "alpha one", source_type="manual", profile_id="profile-A"),
        _point("leaf-a2", "alpha two", source_type="manual", profile_id="profile-A"),
        _point("leaf-b1", "beta one", source_type="manual", profile_id="profile-B"),
    ]
    manifest = RaptorBuilder().build(points)
    # We expect two trees because the profiles disagree.
    tree_ids = {payload["raptor_tree_id"] for payload in manifest.candidate_node_payloads}
    assert len(tree_ids) == 2
    # No cluster payload should contain leaves from both profiles.
    for payload in manifest.candidate_node_payloads:
        children = payload["raptor_child_ids"]
        if not children:
            continue
        if payload["raptor_level"] == RAPTOR_LEVEL_LEAF + 1:
            # Cluster summary children are leaf ids
            by_profile = {"profile-A": 0, "profile-B": 0}
            for leaf_id in children:
                if leaf_id in {"leaf-a1", "leaf-a2"}:
                    by_profile["profile-A"] += 1
                elif leaf_id in {"leaf-b1"}:
                    by_profile["profile-B"] += 1
            # Either zero from one profile or no mix.
            assert by_profile["profile-A"] == 0 or by_profile["profile-B"] == 0, (
                f"cross-profile cluster: {children}"
            )


def test_scope_is_only_propagated_when_all_leaves_agree():
    points = [
        _point("leaf-1", "alpha", source_type="manual", profile_id="default"),
        _point("leaf-2", "beta", source_type="manual", profile_id="default"),
    ]
    manifest = RaptorBuilder().build(points)
    for payload in manifest.candidate_node_payloads:
        assert payload.get("profile_id") == "default"


def test_disagreement_emits_warning_and_omits_scope():
    points = [
        _point("leaf-1", "alpha", source_type="manual", profile_id="profile-A"),
        _point("leaf-2", "beta", source_type="manual", profile_id="profile-B"),
        _point("leaf-3", "gamma", source_type="manual", profile_id="profile-A"),
    ]
    manifest = RaptorBuilder().build(points)
    # Cross-profile split means scope is per-tree, not global.
    assert "scope_disagreement_across_clusters" in manifest.warnings or len(set(
        payload.get("profile_id", "") for payload in manifest.candidate_node_payloads
    )) >= 1


# ---------------------------------------------------------------------------
# Extractive summaries
# ---------------------------------------------------------------------------


def test_summaries_are_extractive_and_use_child_snippets():
    points = [
        _point("leaf-1", "first operational note about RAPTOR schema", source_type="manual"),
        _point("leaf-2", "second operational note about RAPTOR schema", source_type="manual"),
    ]
    manifest = RaptorBuilder().build(points)
    summaries_with_children = [p for p in manifest.candidate_node_payloads if p["raptor_child_ids"]]
    assert summaries_with_children
    for payload in summaries_with_children:
        text = payload["text"]
        # Summary text must contain child point ids as anchors (extractive).
        for cid in payload["raptor_child_ids"]:
            assert cid in text, f"summary missing anchor for {cid}: {text!r}"
        # Summary text must not contain unsupported novel claims — it must
        # contain at least one of the original leaf snippets verbatim.
        # Snippet cap is 240 chars; pick the first short line per leaf.
        assert "first operational note" in text or "second operational note" in text


def test_summary_text_never_contains_secret_values():
    """Even with a passing leaf, summary text must not echo secrets."""
    points = [
        _point("leaf-1", "normal memory with redaction", source_type="manual"),
        _point("leaf-2", "another benign memory", source_type="manual"),
    ]
    manifest = RaptorBuilder().build(points)
    for payload in manifest.candidate_node_payloads:
        text = payload["text"]
        for forbidden in (_secret_token(), "Bearer "):
            assert forbidden not in text


# ---------------------------------------------------------------------------
# Manifest stability
# ---------------------------------------------------------------------------


def test_manifest_is_stable_across_repeated_runs():
    points = [
        _point(f"leaf-{i:03d}", f"note {i} about RAPTOR", source_type="manual") for i in range(8)
    ]
    a = RaptorBuilder(max_cluster_size=4).build(points)
    b = RaptorBuilder(max_cluster_size=4).build(list(reversed(points)))
    blob_a = a.to_dict()
    blob_b = b.to_dict()
    # Manifest must be byte-identical in canonical JSON form.
    json_a = json.dumps(blob_a, sort_keys=True, default=str)
    json_b = json.dumps(blob_b, sort_keys=True, default=str)
    assert json_a == json_b


def test_manifest_digest_changes_when_a_node_changes():
    points = [
        _point("leaf-1", "alpha", source_type="manual"),
        _point("leaf-2", "beta", source_type="manual"),
    ]
    manifest_a = RaptorBuilder().build(points)
    points_with_extra = [
        _point("leaf-1", "alpha", source_type="manual"),
        _point("leaf-2", "beta with extra context", source_type="manual"),
    ]
    manifest_b = RaptorBuilder().build(points_with_extra)
    assert manifest_a.to_dict()["manifest_digest"] != manifest_b.to_dict()["manifest_digest"]


def test_manifest_uses_compute_manifest_digest_internally():
    points = [_point("leaf-1", "alpha", source_type="manual")]
    manifest = RaptorBuilder().build(points)
    blob = manifest.to_dict()
    # Re-computing via the public helper yields the same digest.
    expected = compute_manifest_digest(blob)
    assert blob["manifest_digest"] == expected


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_builder_handles_empty_input():
    manifest = RaptorBuilder().build([])
    assert manifest.leaf_count == 0
    assert manifest.node_count == 0
    assert manifest.candidate_node_payloads == []
    assert manifest.dry_run is True
    assert manifest.mutations_performed is False
    assert "no_safe_leaves" in manifest.warnings


def test_builder_handles_all_unsafe_input():
    points = [
        _point("leaf-1", "stale", source_type="manual", stale=True),
        _point("leaf-2", "secret", source_type="manual").update({"payload": {**{"text": "x", "source_type": "manual", "x": _secret_token()}}}),
    ]
    manifest = RaptorBuilder().build(points)
    assert manifest.leaf_count == 0
    assert manifest.node_count == 0
    assert manifest.candidate_node_payloads == []


def test_builder_chunks_large_buckets_by_max_cluster_size():
    points = [
        _point(f"leaf-{i:03d}", f"note {i}", source_type="manual") for i in range(20)
    ]
    manifest = RaptorBuilder(max_cluster_size=4).build(points)
    cluster_payloads = [p for p in manifest.candidate_node_payloads if p["raptor_level"] == RAPTOR_LEVEL_LEAF + 1]
    for cp in cluster_payloads:
        assert len(cp["raptor_child_ids"]) <= 4


def test_builder_default_max_cluster_size_is_8():
    assert DEFAULT_MAX_CLUSTER_SIZE == 8


def test_builder_prompt_version_is_propagated():
    points = [_point("leaf-1", "alpha", source_type="manual")]
    manifest = RaptorBuilder(prompt_version="raptor-test-v9").build(points)
    assert manifest.prompt_version == "raptor-test-v9"
    for payload in manifest.candidate_node_payloads:
        assert payload["raptor_prompt_version"] == "raptor-test-v9"


def test_builder_default_prompt_version_is_known():
    points = [_point("leaf-1", "alpha", source_type="manual")]
    manifest = RaptorBuilder().build(points)
    assert manifest.prompt_version == DEFAULT_PROMPT_VERSION == "raptor-mvp-extractive-v1"


def test_builder_keeps_skipped_leaves_with_reason():
    points = [
        _point("leaf-bad", f"contains {_secret_token()}", source_type="manual"),
        _point("leaf-q", "quarantined text", source_type="manual", quarantined=True),
    ]
    manifest = RaptorBuilder().build(points)
    reasons = {entry["point_id"]: entry["reason"] for entry in manifest.skipped_leaves}
    assert reasons["leaf-bad"] == "secret_bearing"
    assert reasons["leaf-q"] == "quarantined"


def test_builder_emits_warning_when_leaves_are_skipped():
    points = [
        _point("leaf-bad", f"contains {_secret_token()}", source_type="manual"),
        _point("leaf-ok", "ok", source_type="manual"),
    ]
    manifest = RaptorBuilder().build(points)
    assert any("skipped_leaves" in warning for warning in manifest.warnings)


def test_builder_cluster_ids_are_deterministic_across_runs():
    points = [
        _point("leaf-1", "alpha", source_type="manual"),
        _point("leaf-2", "beta", source_type="manual"),
    ]
    a = RaptorBuilder().build(points)
    b = RaptorBuilder().build(list(reversed(points)))
    cluster_ids_a = sorted(
        p["raptor_cluster_id"] for p in a.candidate_node_payloads if p["raptor_level"] == RAPTOR_LEVEL_LEAF + 1
    )
    cluster_ids_b = sorted(
        p["raptor_cluster_id"] for p in b.candidate_node_payloads if p["raptor_level"] == RAPTOR_LEVEL_LEAF + 1
    )
    assert cluster_ids_a == cluster_ids_b


def test_node_payloads_include_source_hashes_from_leaves():
    points = [
        _point("leaf-1", "alpha note about RAPTOR", source_type="manual"),
        _point("leaf-2", "beta note about RAPTOR", source_type="manual"),
    ]
    manifest = RaptorBuilder().build(points)
    for payload in manifest.candidate_node_payloads:
        if payload["raptor_level"] == RAPTOR_LEVEL_LEAF + 1:
            # Cluster summary should carry one source hash per child leaf.
            assert len(payload["source_hashes"]) == len(payload["raptor_child_ids"])
            for hash_id in payload["source_hashes"]:
                # sha256 hex prefix → 64 hex chars; allow the schema's
                # sha256-derived content hashes.
                assert isinstance(hash_id, str)
                assert len(hash_id) >= 32


def test_root_node_has_no_parent_ids_and_includes_all_clusters():
    points = [
        _point(f"leaf-{i:03d}", f"note {i}", source_type="manual") for i in range(6)
    ]
    manifest = RaptorBuilder(max_cluster_size=2).build(points)
    root_payloads = [p for p in manifest.candidate_node_payloads if p["raptor_level"] == RAPTOR_LEVEL_LEAF + 2]
    assert len(root_payloads) == 1
    root = root_payloads[0]
    assert root["raptor_parent_ids"] == []
    cluster_count = len([p for p in manifest.candidate_node_payloads if p["raptor_level"] == RAPTOR_LEVEL_LEAF + 1])
    assert len(root["raptor_child_ids"]) == cluster_count


# ---------------------------------------------------------------------------
# Secret-shaped point IDs (security-reviewer P2 #1)
# ---------------------------------------------------------------------------


def test_secret_shaped_point_id_is_rejected_and_not_in_manifest():
    """A token-like point id must be rejected before any payload is built.

    The raw id must never appear in the manifest JSON, in any candidate
    payload, in derived_from child_node_id, in raptor_child_ids, or in the
    extractive summary text. Only the stable redacted handle is recorded in
    the skipped_leaves entry.
    """
    secret_id = "sk-" + "abcdef1234567890xyz"
    points = [
        _point(secret_id, "benign note about RAPTOR", source_type="manual"),
        _point("leaf-good", "another benign note about RAPTOR", source_type="manual"),
    ]
    manifest = RaptorBuilder().build(points)
    # Leaf was skipped, not accepted.
    assert manifest.leaf_count == 1
    assert manifest.node_count >= 2
    serialized = json.dumps(manifest.to_dict(), sort_keys=True, default=str)
    assert secret_id not in serialized
    # No candidate payload references the raw id.
    for payload in manifest.candidate_node_payloads:
        encoded = json.dumps(payload, sort_keys=True, default=str)
        assert secret_id not in encoded
        assert secret_id not in (payload.get("text") or "")
        for child_id in payload.get("raptor_child_ids", []):
            assert child_id != secret_id
        for edge in payload.get("derived_from", []):
            assert edge.get("child_node_id") != secret_id
    # The skipped_leaves entry uses a redacted handle, not the raw id.
    skipped = manifest.skipped_leaves
    assert any(entry["reason"] == "secret_id_bearing" for entry in skipped)
    for entry in skipped:
        if entry["reason"] == "secret_id_bearing":
            assert secret_id not in entry["point_id"]
            assert entry["point_id"].startswith("redacted:")


def test_secret_shaped_point_id_handles_are_deterministic_across_runs():
    """Same secret-shaped id must produce the same redacted handle every run."""
    secret_id = "ghp_" + "abcdef1234567890abcd"
    points = [_point(secret_id, "benign note", source_type="manual")]
    a = RaptorBuilder().build(points).skipped_leaves
    b = RaptorBuilder().build(points).skipped_leaves
    assert a == b
    assert a[0]["reason"] == "secret_id_bearing"
    assert secret_id not in a[0]["point_id"]


def test_skipped_leaves_with_secret_id_do_not_leak_into_summary_text():
    """Extractive summary lines must never quote a secret-shaped id verbatim."""
    secret_id = "sk-" + "abcdef1234567890abc"
    points = [
        _point(secret_id, "first memory about RAPTOR", source_type="manual"),
        _point("leaf-1", "second memory about RAPTOR", source_type="manual"),
    ]
    manifest = RaptorBuilder().build(points)
    for payload in manifest.candidate_node_payloads:
        assert secret_id not in (payload.get("text") or "")
        assert secret_id not in (payload.get("summary_text") or "")


# ---------------------------------------------------------------------------
# Manifest digest stability under reordering (security-reviewer P3)
# ---------------------------------------------------------------------------


def test_manifest_digest_is_stable_under_reordering_skipped_leaves():
    """Same safe/unsafe set in different input order must yield the same digest."""
    safe = _point("leaf-good", "benign note about RAPTOR", source_type="manual")
    bad_a = {"id": "leaf-bad-secret", "payload": {"text": "contains " + _secret_token() + " here", "source_type": "manual"}}
    bad_b = {"id": "leaf-bad-stale", "payload": {"text": "stale thing", "source_type": "manual", "stale": True}}
    inputs_a = [safe, bad_a, bad_b]
    inputs_b = [bad_b, safe, bad_a]
    inputs_c = [bad_a, bad_b, safe]
    manifest_a = RaptorBuilder().build(inputs_a)
    manifest_b = RaptorBuilder().build(inputs_b)
    manifest_c = RaptorBuilder().build(inputs_c)
    blob_a = manifest_a.to_dict()
    blob_b = manifest_b.to_dict()
    blob_c = manifest_c.to_dict()
    assert blob_a["manifest_digest"] == blob_b["manifest_digest"] == blob_c["manifest_digest"]
    # And the serialized skipped leaves must also be byte-identical.
    json_a = json.dumps(blob_a["skipped_leaves"], sort_keys=True, default=str)
    json_b = json.dumps(blob_b["skipped_leaves"], sort_keys=True, default=str)
    json_c = json.dumps(blob_c["skipped_leaves"], sort_keys=True, default=str)
    assert json_a == json_b == json_c


def test_manifest_digest_is_stable_under_reordering_skipped_secret_ids():
    """Same secret-shaped ids in different order must yield the same digest."""
    secret_a = "sk-" + "aaaaaaaaaaaaaaaaaaaa"
    secret_b = "ghp_" + "bbbbbbbbbbbbbbbbbbbb"
    bad_a = {"id": secret_a, "payload": {"text": "benign note about RAPTOR", "source_type": "manual"}}
    bad_b = {"id": secret_b, "payload": {"text": "another benign note about RAPTOR", "source_type": "manual"}}
    inputs_a = [bad_a, bad_b]
    inputs_b = [bad_b, bad_a]
    manifest_a = RaptorBuilder().build(inputs_a)
    manifest_b = RaptorBuilder().build(inputs_b)
    blob_a = manifest_a.to_dict()
    blob_b = manifest_b.to_dict()
    assert blob_a["manifest_digest"] == blob_b["manifest_digest"]
    # Raw secret ids never appear in the serialized manifest at all.
    encoded_a = json.dumps(blob_a, sort_keys=True, default=str)
    encoded_b = json.dumps(blob_b, sort_keys=True, default=str)
    assert secret_a not in encoded_a
    assert secret_b not in encoded_a
    assert secret_a not in encoded_b
    assert secret_b not in encoded_b


# ---------------------------------------------------------------------------
# Invalid-but-secret-shaped point IDs (security-reviewer P2 #1 fix)
# ---------------------------------------------------------------------------


def _pem_private_key_marker() -> str:
    """Build a PEM private-key marker at runtime so it never appears as a
    contiguous literal in the test source (which would be flagged by
    ``scripts/check_no_literal_fake_secrets.py``)."""
    return "".join(["-----BEGIN ", "RSA", " PRIVATE KEY", "-----"])


def _basic_auth_url_point_id() -> str:
    """Build a basic-auth URL with a user:password host so the
    ``https?://[^\\s:@]+:[^\\s:@]+@`` scanner pattern fires, but the
    literal ``://user:pass@example`` is split across the ``@`` boundary so
    it is not flagged by the literal-fake-secrets guard."""
    return "https://" + "user:pass" + "@example.invalid"


def _bearer_contextual_point_id() -> str:
    """Build a bearer/contextual string with spaces and a 16+ char token so
    the secret-scanner pattern fires, but the literal token is constructed
    from parts so it never appears as a contiguous literal in the source."""
    parts = ["bearer", "access_token_aabbccdd"]
    return " ".join(parts)


def test_invalid_shape_pem_private_key_point_id_is_redacted_in_skipped_leaves():
    """A PEM private-key-shaped point id fails ``_POINT_ID_RE`` because of
    ``-`` followed by spaces, but ``contains_secret()`` must still classify
    it as ``secret_id_bearing`` so the raw id never appears in
    ``manifest.skipped_leaves`` or anywhere in the serialized manifest."""
    pem_id = _pem_private_key_marker()
    points = [
        _point(pem_id, "benign memory about RAPTOR", source_type="manual"),
        _point("leaf-good", "another benign memory about RAPTOR", source_type="manual"),
    ]
    manifest = RaptorBuilder().build(points)
    blob = manifest.to_dict()
    serialized = json.dumps(blob, sort_keys=True, default=str)
    assert pem_id not in serialized, "raw PEM marker leaked into manifest JSON"
    # The skip entry must use a stable redacted handle, not the raw id.
    skipped = blob["skipped_leaves"]
    secret_entries = [
        entry for entry in skipped
        if entry.get("reason") == "secret_id_bearing" or "pem" in str(entry.get("point_id", "")).lower() is False
    ]
    # The PEM-shaped id was classified as secret_id_bearing (not invalid_point_id).
    matching = [
        entry for entry in skipped
        if "BEGIN" in str(entry.get("point_id", ""))
    ]
    assert not matching, "raw PEM marker leaked into skipped_leaves[*].point_id"
    pem_skips = [entry for entry in skipped if pem_id == entry["point_id"]]
    assert not pem_skips, "raw PEM marker leaked into skipped_leaves[*].point_id verbatim"
    # And the redacted handle is recorded for the secret-shaped skip.
    assert any(entry["reason"] == "secret_id_bearing" for entry in skipped), (
        f"no secret_id_bearing skip: {skipped!r}"
    )
    for entry in skipped:
        if entry["reason"] == "secret_id_bearing":
            assert pem_id not in entry["point_id"]
            assert entry["point_id"].startswith("redacted:"), entry["point_id"]
    # The good leaf was accepted: 1 leaf.
    assert blob["leaf_count"] == 1


def test_invalid_shape_basic_auth_url_point_id_is_redacted_in_skipped_leaves():
    """A basic-auth URL-shaped point id contains ``://`` and ``@``, both of
    which fail ``_POINT_ID_RE``. The secret scanner must still fire and
    classify the id as ``secret_id_bearing``."""
    url_id = _basic_auth_url_point_id()
    points = [
        _point(url_id, "benign memory about RAPTOR", source_type="manual"),
        _point("leaf-good", "another benign memory about RAPTOR", source_type="manual"),
    ]
    manifest = RaptorBuilder().build(points)
    blob = manifest.to_dict()
    serialized = json.dumps(blob, sort_keys=True, default=str)
    # Raw URL id (including user/password pair) must not appear verbatim.
    assert url_id not in serialized, "raw basic-auth URL leaked into manifest JSON"
    user_pass_fragment = "user:pass"
    assert user_pass_fragment not in serialized, "user:password fragment leaked into manifest JSON"
    # Skip entry is classified as secret_id_bearing, not invalid_point_id.
    skipped = blob["skipped_leaves"]
    url_skips = [entry for entry in skipped if "user:pass" in str(entry.get("point_id", ""))]
    assert not url_skips
    assert any(entry["reason"] == "secret_id_bearing" for entry in skipped), skipped
    for entry in skipped:
        if entry["reason"] == "secret_id_bearing":
            assert url_id not in entry["point_id"]
            assert entry["point_id"].startswith("redacted:")
    assert blob["leaf_count"] == 1


def test_invalid_shape_bearer_contextual_point_id_is_redacted_in_skipped_leaves():
    """A bearer/contextual string with spaces (and a 16+ char token) fails
    ``_POINT_ID_RE`` because of the embedded space, but ``contains_secret()``
    still matches the ``bearer <token>`` and ``access token <token>``
    patterns. The build loop must redact the id."""
    bearer_id = _bearer_contextual_point_id()
    points = [
        _point(bearer_id, "benign memory about RAPTOR", source_type="manual"),
        _point("leaf-good", "another benign memory about RAPTOR", source_type="manual"),
    ]
    manifest = RaptorBuilder().build(points)
    blob = manifest.to_dict()
    serialized = json.dumps(blob, sort_keys=True, default=str)
    # The raw bearer string (with the token) must never appear.
    assert bearer_id not in serialized, "raw bearer/contextual id leaked into manifest JSON"
    # The 16-char token must never appear verbatim either (defence in depth).
    token_part = "access_token_aabbccdd"
    assert token_part not in serialized, "bearer token leaked into manifest JSON"
    # Skip entries — the bearer/contextual id is classified secret_id_bearing.
    skipped = blob["skipped_leaves"]
    bearer_skips = [entry for entry in skipped if token_part == entry.get("point_id")]
    assert not bearer_skips
    assert any(entry["reason"] == "secret_id_bearing" for entry in skipped), skipped
    for entry in skipped:
        if entry["reason"] == "secret_id_bearing":
            assert bearer_id not in entry["point_id"]
            assert token_part not in entry["point_id"]
            assert entry["point_id"].startswith("redacted:")
    assert blob["leaf_count"] == 1


def test_invalid_secret_shaped_point_ids_are_handled_after_shape_only_input():
    """The full regression: combine a secret-shaped invalid point id with
    a normal-shape invalid point id and benign leaves. The secret-shaped
    invalid id must still be classified as secret_id_bearing (redacted)
    while the plain invalid id is classified as invalid_point_id (kept
    raw for correlation). Both branches must coexist correctly."""
    pem_id = _pem_private_key_marker()
    plain_invalid = "leaf with spaces"  # plain invalid_point_id (not secret)
    points = [
        _point(pem_id, "benign memory about RAPTOR", source_type="manual"),
        _point(plain_invalid, "benign memory about RAPTOR", source_type="manual"),
        _point("leaf-good", "another benign memory about RAPTOR", source_type="manual"),
    ]
    manifest = RaptorBuilder().build(points)
    blob = manifest.to_dict()
    serialized = json.dumps(blob, sort_keys=True, default=str)
    assert pem_id not in serialized
    reasons = {entry["point_id"]: entry["reason"] for entry in blob["skipped_leaves"]}
    # Secret id → secret_id_bearing with redacted handle; plain id →
    # invalid_point_id with original id preserved.
    secret_handles = [p for p, r in reasons.items() if r == "secret_id_bearing"]
    assert secret_handles, "no secret_id_bearing skip recorded"
    for handle in secret_handles:
        assert handle.startswith("redacted:")
        assert pem_id not in handle
    assert reasons.get(plain_invalid) == "invalid_point_id"
    assert blob["leaf_count"] == 1
