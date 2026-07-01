from __future__ import annotations

import json
from typing import Any

import pytest

from qdrant_memory.raptor.schema import (
    DEFAULT_PROMPT_VERSION,
    RAPTOR_DERIVATION_TYPE,
    RAPTOR_LEVEL_LEAF,
    RAPTOR_REQUIRED_NODE_FIELDS,
    RaptorBuildManifest,
    RaptorCluster,
    RaptorNode,
    RaptorScope,
    RaptorTree,
    _safe_extra,
    _sorted_unique,
    compute_build_id,
    compute_manifest_digest,
    compute_node_id,
    compute_root_id,
    compute_tree_id,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_raptor_required_fields_complete():
    expected = {
        "raptor_tree_id",
        "raptor_node_id",
        "raptor_level",
        "raptor_parent_ids",
        "raptor_child_ids",
        "raptor_cluster_id",
        "raptor_summary_of",
        "raptor_root_id",
        "raptor_build_id",
        "raptor_prompt_version",
        "source_hashes",
    }
    assert expected.issubset(set(RAPTOR_REQUIRED_NODE_FIELDS))


def test_raptor_derivation_type_constant():
    assert RAPTOR_DERIVATION_TYPE == "raptor_summary"
    assert RAPTOR_LEVEL_LEAF == 0
    assert DEFAULT_PROMPT_VERSION == "raptor-mvp-extractive-v1"


# ---------------------------------------------------------------------------
# Deterministic ID helpers
# ---------------------------------------------------------------------------


def test_compute_tree_id_is_deterministic_and_distinct():
    a = compute_tree_id(build_id="b1", prompt_version="v1", root_id="r1")
    b = compute_tree_id(build_id="b1", prompt_version="v1", root_id="r1")
    c = compute_tree_id(build_id="b1", prompt_version="v1", root_id="r2")
    d = compute_tree_id(build_id="b2", prompt_version="v1", root_id="r1")
    assert a == b
    assert a != c
    assert a != d
    assert a.startswith("raptor-tree-")


def test_compute_root_id_is_deterministic_and_distinct():
    a = compute_root_id(build_id="b1", cluster_id="c1")
    b = compute_root_id(build_id="b1", cluster_id="c1")
    c = compute_root_id(build_id="b1", cluster_id="c2")
    assert a == b
    assert a != c
    assert a.startswith("raptor-root-")


def test_compute_node_id_is_stable_with_parent_order():
    a = compute_node_id(tree_id="t1", level=1, cluster_id="c1", parent_ids=("p1", "p2"))
    b = compute_node_id(tree_id="t1", level=1, cluster_id="c1", parent_ids=("p2", "p1"))
    c = compute_node_id(tree_id="t1", level=1, cluster_id="c1", parent_ids=("p1", "p2"))
    # Order must not change the resulting node id.
    assert a == b == c
    d = compute_node_id(tree_id="t1", level=2, cluster_id="c1", parent_ids=("p1", "p2"))
    assert d != a


def test_compute_build_id_is_stable_under_leaf_order_and_config_independent():
    leaves_a = [{"id": "x"}, {"id": "y"}, {"id": "z"}]
    leaves_b = [{"id": "z"}, {"id": "x"}, {"id": "y"}]
    bid_a = compute_build_id(prompt_version="v1", leaves=leaves_a)
    bid_b = compute_build_id(prompt_version="v1", leaves=leaves_b)
    assert bid_a == bid_b


def test_compute_build_id_uses_config_when_provided():
    leaves = [{"id": "x"}]
    bid_a = compute_build_id(prompt_version="v1", leaves=leaves, config={"k": 1})
    bid_b = compute_build_id(prompt_version="v1", leaves=leaves, config={"k": 1})
    bid_c = compute_build_id(prompt_version="v1", leaves=leaves, config={"k": 2})
    assert bid_a == bid_b
    assert bid_a != bid_c


# ---------------------------------------------------------------------------
# RaptorScope
# ---------------------------------------------------------------------------


def test_scope_matches_fail_closed():
    scope = RaptorScope(profile_id="default", user_id_hash="u1", chat_id_hash="c1")
    assert scope.matches({"profile_id": "default", "user_id_hash": "u1", "chat_id_hash": "c1"})
    # Missing profile_id must fail closed when scope expects it.
    assert not scope.matches({"user_id_hash": "u1", "chat_id_hash": "c1"})
    # Mismatched chat must fail closed.
    assert not scope.matches({"profile_id": "default", "user_id_hash": "u1", "chat_id_hash": "c2"})


def test_scope_empty_matches_anything():
    scope = RaptorScope()
    assert scope.is_empty()
    assert scope.matches({})
    assert scope.matches({"profile_id": "anything"})


# ---------------------------------------------------------------------------
# RaptorNode.to_payload — required fields and safety
# ---------------------------------------------------------------------------


def _make_node(**overrides: Any) -> RaptorNode:
    defaults: dict[str, Any] = dict(
        node_id="raptor-node-1",
        tree_id="raptor-tree-1",
        root_id="raptor-root-1",
        build_id="raptor-build-1",
        cluster_id="raptor-cluster-1",
        level=1,
        parent_ids=[],
        child_ids=["leaf-1", "leaf-2"],
        summary_of=["leaf-1", "leaf-2"],
        source_hashes=["h1", "h2"],
        prompt_version=DEFAULT_PROMPT_VERSION,
        scope=RaptorScope(profile_id="default"),
        bucket_key="docs",
        summary_text="extractive summary",
    )
    defaults.update(overrides)
    return RaptorNode(**defaults)


def test_node_payload_includes_all_required_fields():
    node = _make_node()
    payload = node.to_payload()
    for field_name in RAPTOR_REQUIRED_NODE_FIELDS:
        assert field_name in payload, f"missing required field: {field_name}"
    assert payload["raptor_tree_id"] == "raptor-tree-1"
    assert payload["raptor_node_id"] == "raptor-node-1"
    assert payload["raptor_level"] == 1
    assert payload["raptor_child_ids"] == ["leaf-1", "leaf-2"]
    assert payload["raptor_summary_of"] == ["leaf-1", "leaf-2"]
    assert payload["raptor_root_id"] == "raptor-root-1"
    assert payload["raptor_build_id"] == "raptor-build-1"
    assert payload["raptor_prompt_version"] == DEFAULT_PROMPT_VERSION
    assert payload["source_hashes"] == ["h1", "h2"]
    assert payload["derivation_type"] == RAPTOR_DERIVATION_TYPE


def test_node_payload_is_non_canonical_and_review_required():
    node = _make_node()
    payload = node.to_payload()
    assert payload["canonical"] is False
    assert payload["requires_review"] is True
    assert payload["raptor_review_status"] == "review_required"


def test_node_payload_sorts_child_ids_and_summary_of():
    node = _make_node(
        child_ids=["leaf-2", "leaf-1", "leaf-3"],
        summary_of=["leaf-3", "leaf-1"],
        source_hashes=["z", "a", "m"],
    )
    payload = node.to_payload()
    assert payload["raptor_child_ids"] == ["leaf-1", "leaf-2", "leaf-3"]
    assert payload["raptor_summary_of"] == ["leaf-1", "leaf-3"]
    assert payload["source_hashes"] == ["a", "m", "z"]


def test_node_payload_dedupes_repeated_ids():
    node = _make_node(
        child_ids=["leaf-1", "leaf-1", "leaf-2"],
        summary_of=["leaf-2", "leaf-2", "leaf-3"],
    )
    payload = node.to_payload()
    assert payload["raptor_child_ids"] == ["leaf-1", "leaf-2"]
    assert payload["raptor_summary_of"] == ["leaf-2", "leaf-3"]


def test_node_payload_derived_from_has_provenance_for_each_child():
    node = _make_node(child_ids=["leaf-1", "leaf-2"])
    payload = node.to_payload()
    edges = payload["derived_from"]
    assert isinstance(edges, list)
    assert len(edges) == 2
    child_ids_in_edges = {edge["child_node_id"] for edge in edges}
    assert child_ids_in_edges == {"leaf-1", "leaf-2"}
    for edge in edges:
        assert edge["derivation_type"] == RAPTOR_DERIVATION_TYPE
        assert edge["relation_type"] == "SUMMARIZES"
        assert edge["source_uri"].startswith("raptor://node/")


def test_node_payload_omits_scope_when_empty():
    node = _make_node(scope=RaptorScope())
    payload = node.to_payload()
    assert "profile_id" not in payload
    assert "user_id_hash" not in payload
    assert "chat_id_hash" not in payload


def test_node_payload_includes_scope_when_non_empty():
    node = _make_node(scope=RaptorScope(profile_id="default", user_id_hash="u1"))
    payload = node.to_payload()
    assert payload["profile_id"] == "default"
    assert payload["user_id_hash"] == "u1"


def test_node_payload_strips_unsafe_extra_keys():
    secret_key_a = "sk-" + "abcdef1234567890"
    secret_key_b = "ghp_" + "abcdef1234567890"
    node = _make_node(
        extra={
            "safe_meta": "ok",
            "authorization": "Bearer something",
            "api_key": "key-1",
            secret_key_a: "value",
            "password": "hunter2 hunter2",
            secret_key_b: "w",
        }
    )
    payload = node.to_payload()
    assert payload.get("safe_meta") == "ok"
    assert "authorization" not in payload
    assert "api_key" not in payload
    assert secret_key_a not in payload
    assert "password" not in payload  # reserved key
    assert secret_key_b not in payload


def test_node_payload_strips_secret_shaped_string_values():
    """Even if the key itself is benign, secret-shaped values must be dropped."""
    secret_token = "sk-" + "abcdef1234567890xyz"
    node = _make_node(extra={"note": f"looks like {secret_token} token"})
    payload = node.to_payload()
    assert "note" not in payload


def test_node_payload_json_serializable():
    node = _make_node()
    payload = node.to_payload()
    encoded = json.dumps(payload, sort_keys=True, default=str)
    decoded = json.loads(encoded)
    assert decoded["raptor_node_id"] == "raptor-node-1"


# ---------------------------------------------------------------------------
# _safe_extra helper
# ---------------------------------------------------------------------------


def test_safe_extra_drops_reserved_and_secret_keys():
    secret_key_a = "sk-" + "abcdef1234567890"
    secret_key_b = "ghp_" + "abcdef1234567890"
    out = _safe_extra(
        {
            "ok": "value",
            "authorization": "Bearer x",
            "bearer": "y",
            secret_key_a: "z",
            secret_key_b: "w",
            "empty": "",
        }
    )
    assert "ok" in out
    assert "authorization" not in out
    assert "bearer" not in out
    assert secret_key_a not in out
    assert secret_key_b not in out
    assert "empty" not in out


def test_safe_extra_handles_non_dict_input():
    assert _safe_extra(None) == {}
    assert _safe_extra("string") == {}


# ---------------------------------------------------------------------------
# RaptorBuildManifest digest
# ---------------------------------------------------------------------------


def _make_manifest(**overrides) -> RaptorBuildManifest:
    defaults = dict(
        build_id="raptor-build-1",
        prompt_version=DEFAULT_PROMPT_VERSION,
        tree_id="raptor-tree-1",
        root_id="raptor-root-1",
        config={"k": 1},
        leaf_count=3,
        node_count=2,
        skipped_leaves=[],
        warnings=[],
        candidate_node_payloads=[{"raptor_node_id": "n1"}],
        scope={"profile_id": "default"},
    )
    defaults.update(overrides)
    return RaptorBuildManifest(**defaults)


def test_manifest_dry_run_and_no_mutations():
    manifest = _make_manifest()
    assert manifest.dry_run is True
    assert manifest.mutations_performed is False
    blob = manifest.to_dict()
    assert blob["dry_run"] is True
    assert blob["mutations_performed"] is False


def test_manifest_digest_is_deterministic_and_excludes_volatile_fields():
    manifest_a = _make_manifest(timestamp="2026-07-01T00:00:00Z")
    manifest_b = _make_manifest(timestamp="2099-01-01T12:34:56Z")
    blob_a = manifest_a.to_dict()
    blob_b = manifest_b.to_dict()
    # Volatile timestamps must not be in the digest input set, and the
    # digest must therefore be identical regardless of timestamp.
    assert blob_a["manifest_digest"] == blob_b["manifest_digest"]


def test_manifest_digest_changes_when_payloads_change():
    manifest_a = _make_manifest(candidate_node_payloads=[{"raptor_node_id": "n1"}])
    manifest_b = _make_manifest(candidate_node_payloads=[{"raptor_node_id": "n2"}])
    assert manifest_a.to_dict()["manifest_digest"] != manifest_b.to_dict()["manifest_digest"]


def test_manifest_to_dict_is_json_serializable():
    manifest = _make_manifest()
    encoded = json.dumps(manifest.to_dict(), sort_keys=True, default=str)
    decoded = json.loads(encoded)
    assert decoded["build_id"] == "raptor-build-1"
    assert decoded["manifest_digest"]


def test_compute_manifest_digest_direct():
    blob = {"build_id": "x", "node_payloads": [{"raptor_node_id": "n1"}]}
    digest = compute_manifest_digest(blob)
    assert isinstance(digest, str)
    assert len(digest) == 64  # sha256 hex
    # Same inputs must produce same digest.
    assert compute_manifest_digest(blob) == digest


# ---------------------------------------------------------------------------
# _sorted_unique
# ---------------------------------------------------------------------------


def test_sorted_unique_dedupes_and_sorts():
    assert _sorted_unique(["b", "a", "b", "c", "a"]) == ["a", "b", "c"]


def test_sorted_unique_handles_blank_values():
    assert _sorted_unique(["a", "", "  ", "b"]) == ["a", "b"]
    assert _sorted_unique([]) == []


# ---------------------------------------------------------------------------
# Dataclass smoke tests
# ---------------------------------------------------------------------------


def test_raptor_cluster_to_dict():
    cluster = RaptorCluster(
        cluster_id="c1",
        level=1,
        bucket_key="docs",
        leaf_ids=["a", "b"],
        parent_ids=[],
        scope=RaptorScope(profile_id="default"),
    )
    blob = cluster.to_dict()
    assert blob["cluster_id"] == "c1"
    assert blob["level"] == 1
    assert blob["bucket_key"] == "docs"
    assert blob["leaf_ids"] == ["a", "b"]
    assert blob["scope"] == {"profile_id": "default", "user_id_hash": "", "chat_id_hash": ""}


def test_raptor_tree_to_dict():
    tree = RaptorTree(
        tree_id="t1",
        root_id="r1",
        build_id="b1",
        prompt_version="v1",
        scope=RaptorScope(profile_id="default"),
        leaf_ids=["a", "b"],
        cluster_ids=["c1"],
        node_ids=["n1", "n2"],
    )
    blob = tree.to_dict()
    assert blob["tree_id"] == "t1"
    assert blob["leaf_ids"] == ["a", "b"]
    assert blob["cluster_ids"] == ["c1"]
    assert blob["node_ids"] == ["n1", "n2"]


def test_raptor_node_payload_node_role_for_leaf_and_summary():
    leaf = _make_node(level=RAPTOR_LEVEL_LEAF)
    summary = _make_node(level=2)
    assert leaf.to_payload()["raptor_node_role"] == "leaf_ref"
    assert summary.to_payload()["raptor_node_role"] == "summary"


# ---------------------------------------------------------------------------
# Reserved-key denylist (security-reviewer P2 #2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden_key",
    [
        "fact_status",
        "profile_id",
        "user_id_hash",
        "chat_id_hash",
        "schema",
        "schema_version",
        "version",
        "source_uri",
        "source_type",
        "locator",
        "content_hash",
        "source_modified_at",
        "derived_from",
        "evidence",
        "raptor_tree_id",
        "raptor_node_id",
        "raptor_level",
        "raptor_parent_ids",
        "raptor_child_ids",
        "raptor_cluster_id",
        "raptor_summary_of",
        "raptor_root_id",
        "raptor_build_id",
        "raptor_prompt_version",
        "raptor_bucket_key",
        "raptor_node_role",
        "raptor_review_status",
        "canonical",
        "requires_review",
        "confidence",
        "truth_confidence",
        "usefulness_weight",
    ],
)
def test_node_payload_extra_cannot_inject_reserved_keys(forbidden_key: str):
    """Even when the base payload omits the key, ``extra`` must not inject it.

    The denylist covers status/scope/provenance/schema/trust fields and
    all RAPTOR-owned structural fields. ``extra`` may only contribute
    auxiliary metadata that the schema does not own.
    """
    node = _make_node(scope=RaptorScope(), extra={forbidden_key: "injected_value"})
    payload = node.to_payload()
    # Either the key is absent from the payload, or — if it is present as
    # a base-payload-owned field — its value is whatever the schema sets
    # (not the caller's injected string).
    assert payload.get(forbidden_key) != "injected_value", (
        f"{forbidden_key} leaked into the payload via extra"
    )


def test_node_payload_extra_cannot_override_present_reserved_keys():
    """``extra`` must not override a reserved key the base payload owns."""
    node = _make_node(
        scope=RaptorScope(profile_id="default"),
        extra={"profile_id": "injected", "fact_status": "active", "raptor_node_id": "evil"},
    )
    payload = node.to_payload()
    assert payload.get("profile_id") == "default"
    assert payload.get("raptor_node_id") != "evil"
    assert payload.get("fact_status") != "active"


def test_safe_extra_drops_all_reserved_keys():
    """The reserved denylist must strip every status/scope/provenance key."""
    out = _safe_extra(
        {
            "fact_status": "active",
            "profile_id": "injected",
            "user_id_hash": "injected",
            "chat_id_hash": "injected",
            "schema_version": "9",
            "version": "2",
            "derived_from": "injected",
            "evidence": "injected",
            "canonical": True,
            "requires_review": False,
            "raptor_node_id": "evil",
            "safe_extra": "kept",
        }
    )
    for forbidden in (
        "fact_status",
        "profile_id",
        "user_id_hash",
        "chat_id_hash",
        "schema_version",
        "version",
        "derived_from",
        "evidence",
        "canonical",
        "requires_review",
        "raptor_node_id",
    ):
        assert forbidden not in out, f"{forbidden} leaked through _safe_extra"
    assert out.get("safe_extra") == "kept"
