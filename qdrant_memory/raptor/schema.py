"""RAPTOR schema — nodes, trees, and build manifest dataclasses.

Phase 3 is dry-run only: the schema describes what a RAPTOR candidate node
*would* look like in Qdrant, but the builder never upserts anything. All
serialization helpers stay stdlib-only and deterministic so repeated runs
over the same input/config produce byte-identical manifests.

Required fields for every RAPTOR summary node payload (Phase 3 acceptance):

- ``raptor_tree_id``
- ``raptor_node_id``
- ``raptor_level``           (0 = leaf ref, 1..N = cluster summaries)
- ``raptor_parent_ids``      (list of parent node IDs; empty for level-1 root)
- ``raptor_child_ids``       (sorted, deterministic, leaves for non-leaf nodes)
- ``raptor_cluster_id``
- ``raptor_summary_of``      (list of leaf IDs this node summarizes)
- ``raptor_root_id``
- ``raptor_build_id``
- ``raptor_prompt_version``
- ``source_hashes``          (sorted, de-duplicated)
- ``derived_from``           (provenance edges)
- ``canonical``              (always ``False``)
- ``requires_review``        (always ``True``)
- ``derivation_type``        (always ``raptor_summary``)

The manifest exposes:

- ``build_id``, ``manifest_digest``
- ``dry_run=True``, ``mutations_performed=False``
- ``node_count``, ``leaf_count``, ``skipped_leaves``
- ``warnings``
- ``candidate_node_payloads`` (sorted by node_id, safe, JSON-serializable)

The manifest digest deliberately excludes volatile timestamps. Callers can
inject a deterministic ``timestamp`` value (e.g. ``"2026-07-01T00:00:00Z"``)
if they want a fingerprint that includes wall-clock information — otherwise
only the structural inputs participate in the digest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from qdrant_memory.lesson_extractor import contains_secret


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAPTOR_DERIVATION_TYPE = "raptor_summary"
RAPTOR_LEVEL_LEAF = 0
RAPTOR_LEVEL_ROOT = 1
DEFAULT_PROMPT_VERSION = "raptor-mvp-extractive-v1"

RAPTOR_REQUIRED_NODE_FIELDS: tuple[str, ...] = (
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
    "derived_from",
    "derivation_type",
    "canonical",
    "requires_review",
)

# RAPTOR-owned payload keys. Any unknown key the caller tries to inject must
# be filtered through ``_safe_extra()``; secret-shaped / reserved keys are
# rejected before they reach the candidate payload. The denylist covers
# RAPTOR-owned structural fields, status/review flags, scope/provenance,
# schema/version markers, and obvious secret-shape names — even keys that
# the base payload happens to omit on a given call (since ``to_payload``
# uses ``setdefault``, callers cannot override present fields anyway).
_RAPTOR_RESERVED_KEYS: frozenset[str] = frozenset(
    {
        # RAPTOR-owned structural fields
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
        # Status / review / trust flags
        "fact_status",
        "requires_review",
        "canonical",
        "confidence",
        "truth_confidence",
        "usefulness_weight",
        # Scope / ownership fields
        "profile_id",
        "user_id_hash",
        "chat_id_hash",
        # Provenance / lineage
        "derived_from",
        "evidence",
        "source_uri",
        "source_type",
        "locator",
        "content_hash",
        "source_modified_at",
        # Schema / version markers
        "schema",
        "schema_version",
        "version",
        # Qdrant / write gate / collection keys we must never let
        # ``extra`` reintroduce.
        "authorization",
        "api_key",
        "api-key",
        "apikey",
        "bearer",
        "password",
        "passwd",
        "secret",
        "token",
        "credential",
        "credentials",
        "private_key",
        "private-key",
        "client_secret",
    }
)

# Scope fields that must agree across all leaves in a cluster. If they
# disagree the cluster builder must omit the scope and add a warning.
_SCOPE_KEYS: tuple[str, ...] = ("profile_id", "user_id_hash", "chat_id_hash")


# ---------------------------------------------------------------------------
# ID helpers (deterministic, sha256-truncated)
# ---------------------------------------------------------------------------


def _sha256_id(*parts: str, prefix: str = "", hex_len: int = 32) -> str:
    """Stable id from a join of *parts*. Pure stdlib."""
    raw = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{prefix}{digest[:hex_len]}"


def compute_tree_id(*, build_id: str, prompt_version: str, root_id: str) -> str:
    return _sha256_id("raptor-tree", build_id, prompt_version, root_id, prefix="raptor-tree-")


def compute_root_id(*, build_id: str, cluster_id: str) -> str:
    return _sha256_id("raptor-root", build_id, cluster_id, prefix="raptor-root-")


def compute_node_id(
    *,
    tree_id: str,
    level: int,
    cluster_id: str,
    parent_ids: Iterable[str] = (),
) -> str:
    sorted_parents = sorted(str(p) for p in parent_ids if p)
    return _sha256_id(
        "raptor-node",
        tree_id,
        str(int(level)),
        cluster_id,
        "|".join(sorted_parents),
        prefix="raptor-node-",
    )


def compute_build_id(
    *,
    prompt_version: str,
    leaves: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any] | None = None,
) -> str:
    """Build id is deterministic from prompt + leaf inputs + cluster config.

    Only structural inputs participate (no timestamps, no caller identity
    unless explicitly added to *config*).
    """
    leaf_keys: list[str] = []
    for leaf in leaves:
        if not isinstance(leaf, Mapping):
            continue
        leaf_keys.append(str(leaf.get("id") or ""))
    leaf_keys.sort()
    config_blob = json.dumps(dict(config or {}), sort_keys=True, default=str)
    return _sha256_id(
        "raptor-build",
        prompt_version,
        config_blob,
        ",".join(leaf_keys),
        prefix="raptor-build-",
    )


def compute_manifest_digest(manifest: Mapping[str, Any]) -> str:
    """Deterministic digest of the manifest, excluding volatile fields.

    Volatile fields (timestamps, build wall-clock) are intentionally NOT
    included in the digest unless the caller injects a deterministic
    ``timestamp`` value via :class:`RaptorBuildManifest`.
    """
    blob = {
        "build_id": manifest.get("build_id"),
        "prompt_version": manifest.get("prompt_version"),
        "tree_id": manifest.get("tree_id"),
        "root_id": manifest.get("root_id"),
        "config": manifest.get("config"),
        "leaf_count": manifest.get("leaf_count"),
        "node_count": manifest.get("node_count"),
        "skipped_leaves": manifest.get("skipped_leaves"),
        "warnings": manifest.get("warnings"),
        "node_payloads": manifest.get("candidate_node_payloads"),
    }
    encoded = json.dumps(blob, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Field validation helpers (used by the builder + tests)
# ---------------------------------------------------------------------------


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    return str(value).strip()


def _is_safe_string(value: Any) -> bool:
    """Reserved key / secret-shape detector for ``extra`` payload fields."""
    text = _stringify(value).lower()
    if not text:
        return False
    if text in _RAPTOR_RESERVED_KEYS:
        return False
    # Reject anything that looks like a credential shape.
    credential_markers = (
        "sk-",
        "ghp_",
        "gho_",
        "ghu_",
        "ghs_",
        "ghr_",
        "akia",
        "bearer ",
        "-----begin ",
        "eyj",
    )
    return not any(text.startswith(marker) for marker in credential_markers)


def _safe_extra(extra: Mapping[str, Any] | None) -> dict[str, Any]:
    """Filter ``extra`` to JSON-serializable, secret-free, safe-key entries.

    Unknown keys are dropped, secret-shaped keys/values are dropped, and
    non-scalar values are stringified. The output is always a dict.
    """
    if not isinstance(extra, Mapping):
        return {}
    safe: dict[str, Any] = {}
    for key, value in extra.items():
        key_text = _stringify(key)
        if not key_text:
            continue
        if key_text.lower() in _RAPTOR_RESERVED_KEYS:
            continue
        if not _is_safe_string(key_text):
            continue
        if isinstance(value, (dict, list, tuple, set)):
            value_text = json.dumps(value, sort_keys=True, default=str)
            if not _is_safe_string(value_text) or contains_secret(value_text):
                continue
            safe[key_text] = value
        else:
            value_text = _stringify(value)
            if not value_text or not _is_safe_string(value_text) or contains_secret(value_text):
                continue
            safe[key_text] = value_text
    return safe


def _sorted_unique(values: Iterable[str]) -> list[str]:
    """Deterministic, de-duplicated list of strings."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = _stringify(value)
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return sorted(out)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RaptorScope:
    """Scope tuple required to agree across all leaves in a single cluster."""

    profile_id: str = ""
    user_id_hash: str = ""
    chat_id_hash: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "profile_id": self.profile_id,
            "user_id_hash": self.user_id_hash,
            "chat_id_hash": self.chat_id_hash,
        }

    def is_empty(self) -> bool:
        return not any(self.as_dict().values())

    def matches(self, payload: Mapping[str, Any]) -> bool:
        """Fail-closed: any non-empty scope key must match the payload."""
        for key in _SCOPE_KEYS:
            expected = getattr(self, key)
            actual = _stringify(payload.get(key))
            if expected and actual != expected:
                return False
        return True


@dataclass
class RaptorCluster:
    """A deterministic cluster of leaves that will become one summary node."""

    cluster_id: str
    level: int
    bucket_key: str
    leaf_ids: list[str] = field(default_factory=list)
    parent_ids: list[str] = field(default_factory=list)
    scope: RaptorScope | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "level": int(self.level),
            "bucket_key": self.bucket_key,
            "leaf_ids": list(self.leaf_ids),
            "parent_ids": list(self.parent_ids),
            "scope": self.scope.as_dict() if isinstance(self.scope, RaptorScope) else {},
        }


@dataclass
class RaptorNode:
    """Candidate RAPTOR summary node — never persisted in Phase 3."""

    node_id: str
    tree_id: str
    root_id: str
    build_id: str
    cluster_id: str
    level: int
    parent_ids: list[str] = field(default_factory=list)
    child_ids: list[str] = field(default_factory=list)
    summary_of: list[str] = field(default_factory=list)
    source_hashes: list[str] = field(default_factory=list)
    prompt_version: str = DEFAULT_PROMPT_VERSION
    scope: RaptorScope | None = None
    bucket_key: str = ""
    summary_text: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Return the candidate Qdrant-style payload for this node.

        Always non-canonical and review-required. Provenance edges are
        derived from child leaves; ``source_hashes`` come from the leaves.
        """
        derived_from: list[dict[str, Any]] = []
        for child_id in self.child_ids:
            derived_from.append(
                {
                    "source_uri": f"raptor://node/{self.tree_id}/{child_id}",
                    "derivation_type": RAPTOR_DERIVATION_TYPE,
                    "relation_type": "SUMMARIZES",
                    "child_node_id": child_id,
                }
            )

        payload: dict[str, Any] = {
            "text": self.summary_text,
            "memory_kind": "summary",
            "source": "raptor_dry_run",
            "source_type": "raptor_summary",
            "derivation_type": RAPTOR_DERIVATION_TYPE,
            "canonical": False,
            "requires_review": True,
            "raptor_tree_id": self.tree_id,
            "raptor_node_id": self.node_id,
            "raptor_level": int(self.level),
            "raptor_parent_ids": _sorted_unique(self.parent_ids),
            "raptor_child_ids": _sorted_unique(self.child_ids),
            "raptor_cluster_id": self.cluster_id,
            "raptor_summary_of": _sorted_unique(self.summary_of),
            "raptor_root_id": self.root_id,
            "raptor_build_id": self.build_id,
            "raptor_prompt_version": self.prompt_version,
            "source_hashes": _sorted_unique(self.source_hashes),
            "derived_from": derived_from,
        }
        if isinstance(self.scope, RaptorScope) and not self.scope.is_empty():
            payload["profile_id"] = self.scope.profile_id
            payload["user_id_hash"] = self.scope.user_id_hash
            payload["chat_id_hash"] = self.scope.chat_id_hash
        if self.bucket_key:
            payload["raptor_bucket_key"] = self.bucket_key
        payload["raptor_node_role"] = "leaf_ref" if int(self.level) <= RAPTOR_LEVEL_LEAF else "summary"
        payload["raptor_review_status"] = "review_required"

        # Only attach caller-supplied extras that pass the safety filter.
        for key, value in _safe_extra(self.extra).items():
            payload.setdefault(key, value)
        return payload


@dataclass
class RaptorTree:
    """A single RAPTOR tree produced by the dry-run builder."""

    tree_id: str
    root_id: str
    build_id: str
    prompt_version: str
    scope: RaptorScope
    leaf_ids: list[str]
    cluster_ids: list[str]
    node_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tree_id": self.tree_id,
            "root_id": self.root_id,
            "build_id": self.build_id,
            "prompt_version": self.prompt_version,
            "scope": self.scope.as_dict(),
            "leaf_ids": list(self.leaf_ids),
            "cluster_ids": list(self.cluster_ids),
            "node_ids": list(self.node_ids),
        }


@dataclass
class RaptorBuildManifest:
    """Top-level dry-run manifest returned by the builder.

    Always: ``dry_run=True`` and ``mutations_performed=False``.
    ``manifest_digest`` is deterministic and excludes volatile timestamps.
    """

    build_id: str
    prompt_version: str
    tree_id: str
    root_id: str
    config: dict[str, Any] = field(default_factory=dict)
    leaf_count: int = 0
    node_count: int = 0
    skipped_leaves: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    candidate_node_payloads: list[dict[str, Any]] = field(default_factory=list)
    scope: dict[str, str] = field(default_factory=dict)
    timestamp: str = ""
    dry_run: bool = True
    mutations_performed: bool = False
    manifest_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # ``manifest_digest`` is always recomputed on serialization to keep it
        # in sync with the payload list, even if a caller mutates the manifest.
        data["manifest_digest"] = compute_manifest_digest(data)
        # The serialized form must never include timestamps in the digest
        # unless the caller injected a deterministic value.
        if not self.timestamp:
            data.pop("timestamp", None)
        return data


__all__ = [
    "DEFAULT_PROMPT_VERSION",
    "RAPTOR_DERIVATION_TYPE",
    "RAPTOR_LEVEL_LEAF",
    "RAPTOR_LEVEL_ROOT",
    "RAPTOR_REQUIRED_NODE_FIELDS",
    "RaptorBuildManifest",
    "RaptorCluster",
    "RaptorNode",
    "RaptorScope",
    "RaptorTree",
    "compute_build_id",
    "compute_manifest_digest",
    "compute_node_id",
    "compute_root_id",
    "compute_tree_id",
]