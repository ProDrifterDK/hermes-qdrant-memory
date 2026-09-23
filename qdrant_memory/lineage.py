"""Lineage identity, frozen W0 validation, W1 capture, and W2 reconciliation.

The W0 identity/evidence builders and validators remain the frozen admission
surface. W2 change events and transition orchestration use their own strict
builders below; they do not widen the W0 write gate.

Identity conventions (approved plan, section 4):

- Identity material is canonical JSON (sorted keys, compact separators,
  UTF-8) hashed with SHA-256; the domain tag is the first element of the
  identity material list, so every digest is domain-separated.
- ``entity-*`` / ``edge-*`` strings are logical handles only. Actual Qdrant
  point IDs for structural graph records use
  ``graph_schema.make_graph_point_id(logical_id)`` (``graph-record-v1``).
- Exact case-sensitive file paths are hashed before any slugifying identity
  helper sees them; ``make_entity_id`` collapses case/punctuation and must
  never be applied to a raw path.
- Structural records store their full identity digest
  (``lineage_identity_digest``); a logical handle whose truncated hex does
  not match that digest is a collision and a hard error, never an overwrite.
"""

from __future__ import annotations

import hashlib
import os
import stat
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from qdrant_memory.graph_schema import (
    KNOWN_ENTITY_TYPES,
    LINEAGE_OPERATIONS,
    LINEAGE_OBSERVATIONS,
    LINEAGE_ROLES,
    build_edge_payload,
    build_entity_payload,
    canonical_json,
    entity_identity_digest,
    is_uuid_string,
    make_entity_id,
    make_graph_point_id,
    make_point_id,
    w0_content_hash_problems,
    w0_file_path_problems,
    w0_is_omitted,
    w0_logical_handle_problems,
    w0_locator_problems,
    w0_sha256_hex_problems,
    w0_uri_problems,
    w0_uuid_problems,
    w0_vocabulary_problems,
)
from qdrant_memory.lesson_extractor import contains_secret

# Backward-compatible name: the single shared locator-shape definition moved
# to the W0 validation surface in ``graph_schema`` (used by both the gate and
# the builder paths).
file_chunk_locator_problems = w0_locator_problems

LINEAGE_SCHEMA_VERSION = 1
MECHANICAL_EDGE_CLASS = "mechanical"

# Mechanical structural relations (section 4 direction table). Claim-level
# relations are never mechanical, whatever ``edge_class`` claims.
STRUCTURAL_MECHANICAL_RELATIONS = ("DERIVED_FROM", "PART_OF", "SUPERSEDES")
# Approved-citation materialization relations (approved summary -> cited child,
# approved derived point -> cited source point).
APPROVED_CITATION_RELATIONS = ("SUMMARIZES", "EXTRACTED_FROM")
# Every relation a mechanical edge payload may carry.
MECHANICAL_EDGE_RELATIONS = STRUCTURAL_MECHANICAL_RELATIONS + APPROVED_CITATION_RELATIONS
CLAIM_RELATIONS = ("SUPPORTS", "CONTRADICTS")
NON_DEPENDENCY_RELATIONS = frozenset({
    "SUPPORTS", "CONTRADICTS", "REFERENCES", "PART_OF", "SUPERSEDES",
})

# Operation -> permitted mechanical relation types (claims stay forbidden).
OPERATION_RELATION_MATRIX: dict[str, tuple[str, ...]] = {
    "index_capture": ("DERIVED_FROM", "PART_OF"),
    "index_reconcile": ("DERIVED_FROM", "PART_OF", "SUPERSEDES"),
    "approved_citation": ("SUMMARIZES", "EXTRACTED_FROM", "DERIVED_FROM"),
    "source_revalidation": ("SUPERSEDES",),
    "restore": ("DERIVED_FROM", "PART_OF", "SUPERSEDES"),
}

# Exact-token anchors: Python re '$' also matches immediately before a
# trailing newline, so a '\n'-suffixed digest would pass a '$'-anchored
# pattern. Every W0 shape domain (hash, content hash, UUID reference token,
# logical handle, controlled vocabulary, locator, URI) is defined ONCE in the
# shared validation surface (``graph_schema`` 'W0 structural shape domains')
# and both the gate below and the W0-capable builder paths call those same
# functions, so the two paths cannot disagree about a documented shape.


def is_sha256_hex(value: Any) -> bool:
    """Return True iff *value* is a 64-char lowercase hex SHA-256 digest."""
    return not w0_sha256_hex_problems(value)

# ``canonical_json`` is imported from ``graph_schema`` above: the single
# shared canonical-JSON definition used by the lineage identity helpers and
# the bounded-locator serialization check.


def lineage_digest(material: Any) -> str:
    """Domain-separated SHA-256 over canonical JSON identity material."""
    return hashlib.sha256(canonical_json(material).encode("utf-8")).hexdigest()


def make_scope_key(
    *,
    collection_name: str,
    profile_id: str,
    user_id_hash: str = "",
    chat_id_hash: str = "",
) -> str:
    """Digest of collection plus the exact profile/user/chat ownership tuple.

    Missing optional user/chat scope means empty string, never a wildcard.
    """
    return lineage_digest(
        [
            "memory-scope-v1",
            str(collection_name or ""),
            str(profile_id or ""),
            str(user_id_hash or ""),
            str(chat_id_hash or ""),
        ]
    )


def make_source_key(*, scope_key: str, resolved_file_path: str) -> str:
    """Digest of ``["file-source-v1", scope_key, resolved_file_path]``.

    ``resolved_file_path`` must already be the exact host path (case and
    spelling preserved).
    """
    return lineage_digest(["file-source-v1", str(scope_key or ""), str(resolved_file_path or "")])


def make_version_identity_digest(*, source_key: str, file_sha256: str) -> str:
    """Digest of ``["file-version-v1", source_key, file_sha256]``."""
    return lineage_digest(["file-version-v1", str(source_key or ""), str(file_sha256 or "")])


def memory_point_endpoint_logical_id(*, scope_key: str, point_id: str, profile_id: str) -> str:
    """Logical lineage endpoint handle for an ordinary memory point.

    Section 4: ``make_entity_id("memory_point", digest(["point-endpoint-v1",
    scope_key, exact_point_id]), profile_id=profile_id)``. The endpoint's
    identity is derived from the exact scoped point ID, so the gate can
    independently recompute the handle an edge claims for an ordinary point.
    """
    return make_entity_id(
        "memory_point",
        lineage_digest(["point-endpoint-v1", str(scope_key or ""), str(point_id or "")]),
        profile_id=profile_id,
    )


def edge_identity_digest(*, profile_id: str, source_entity_id: str, relation_type: str, target_entity_id: str) -> str:
    """Full SHA-256 identity digest of an edge (matches ``make_edge_id``)."""
    return hashlib.sha256(
        f"edge|{profile_id}|{source_entity_id}|{relation_type}|{target_entity_id}".encode("utf-8")
    ).hexdigest()


def source_logical_id(*, source_key: str, profile_id: str) -> str:
    """Logical entity handle for a file source node."""
    return make_entity_id("source", str(source_key or ""), profile_id=profile_id)


def version_logical_id(*, version_identity_digest: str, profile_id: str) -> str:
    """Logical entity handle for a file version node (bare digest key)."""
    return make_entity_id("source", str(version_identity_digest or ""), profile_id=profile_id)


def storage_point_id(logical_id: str) -> str:
    """Qdrant storage ID (UUID) for a structural graph record."""
    return make_graph_point_id(logical_id)


def logical_handle_hex_part(logical_id: str) -> str:
    """Return the 16-hex truncation of an ``entity-*`` / ``edge-*`` handle."""
    text = str(logical_id or "")
    if text.startswith(("entity-", "edge-")):
        return text.split("-", 1)[1]
    return ""


def logical_id_matches_digest(logical_id: str, identity_digest: str) -> bool:
    """Collision check: the handle must be the truncation of the full digest."""
    if not is_sha256_hex(identity_digest):
        return False
    return logical_handle_hex_part(logical_id) == identity_digest[:16]


# See ``graph_schema.w0_locator_problems`` — the single shared locator-shape
# definition (aliased above as ``file_chunk_locator_problems`` for backward
# compatibility).


def expected_file_uri(resolved_file_path: str) -> str | None:
    """Exact file URI of an evidenced resolved host path.

    Same stdlib ``PurePath.as_uri`` encoding the indexer's ``file_uri``
    produces; case and spelling are preserved and only URI-unsafe characters
    are percent-encoded. Returns ``None`` when the token is not a resolvable
    absolute path — callers fail closed in that case instead of guessing.
    """
    try:
        return Path(str(resolved_file_path or "")).as_uri()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Typed internal evidence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LineageEvidence:
    """Independent evidence supplied by the authorized mechanical caller.

    The gate binds this evidence to the final payload fields; a payload whose
    lineage claims diverge from the evidence is rejected. This is an internal
    type: it is never constructed from user tool input.
    """

    operation: str
    collection_name: str
    profile_id: str
    user_id_hash: str = ""
    chat_id_hash: str = ""
    scope_key: str = ""
    source_key: str = ""
    source_uri: str = ""
    file_path: str = ""
    file_sha256: str = ""
    content_hash: str = ""
    source_content_hash: str = ""
    target_content_hash: str = ""
    file_version_id: str = ""
    locator: dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    relation_type: str = ""
    source_point_id: str = ""
    target_point_id: str = ""
    source_entity_type: str = ""
    target_entity_type: str = ""
    approval_ref: str = ""
    predecessor_event_id: str = ""
    event_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evidence_scope_key(evidence: LineageEvidence) -> str:
    """Recompute the exact write-scope digest from the evidence's own tuple.

    The scope binding is never taken from a claimed digest; it is derived
    from the collection plus the exact ownership tuple the gate checked.
    Missing optional user/chat scope means empty string, never a wildcard.
    """
    return make_scope_key(
        collection_name=evidence.collection_name,
        profile_id=evidence.profile_id,
        user_id_hash=evidence.user_id_hash,
        chat_id_hash=evidence.chat_id_hash,
    )


def evidence_source_key(evidence: LineageEvidence, scope_key: str) -> str:
    """Recompute the source-key digest from the recomputed scope and path."""
    return make_source_key(scope_key=scope_key, resolved_file_path=evidence.file_path)


def evidence_internal_mismatches(evidence: LineageEvidence) -> list[str]:
    """Fail closed when claimed evidence digests or URIs contradict the
    evidence's own collection/ownership/path material."""
    mismatches: list[str] = []
    expected_scope = evidence_scope_key(evidence)
    if str(evidence.scope_key or "") and evidence.scope_key != expected_scope:
        mismatches.append("evidence_scope_key")
    if str(evidence.file_path or ""):
        expected_source = evidence_source_key(evidence, expected_scope)
        if str(evidence.source_key or "") and evidence.source_key != expected_source:
            mismatches.append("evidence_source_key")
    # An evidenced file URI is bound to the exact evidenced resolved path:
    # whitespace-bearing tokens, other schemes, or a URI for a different
    # path are different tokens and fail closed. This binds edge evidence
    # URIs exactly like entity evidence URIs.
    if str(evidence.file_path or "") and str(evidence.source_uri or ""):
        expected_uri = expected_file_uri(evidence.file_path)
        if expected_uri is None or evidence.source_uri != expected_uri:
            mismatches.append("evidence_source_uri")
    return mismatches


def evidence_shape_problems(evidence: LineageEvidence) -> list[str]:
    """Raw-type/domain validation of the supplied evidence record itself.

    The identical shape rule that applies to payload fields applies to the
    evidence values the authorized caller supplies: an optional evidence
    field is omitted ONLY by its sentinel (empty string; empty locator
    mapping), and any present value must be the exact documented token from
    the shared W0 shape domains. Called by the gate BEFORE the comparison
    functions, so equal-malformed payload/evidence pairs are still a
    refusal.
    """
    problems: list[str] = []
    for field_name in ("scope_key", "source_key", "file_sha256"):
        value = getattr(evidence, field_name)
        if not w0_is_omitted(value):
            problems.extend(w0_sha256_hex_problems(value, f"evidence {field_name}"))
    for field_name in ("content_hash", "source_content_hash", "target_content_hash"):
        value = getattr(evidence, field_name)
        if not w0_is_omitted(value):
            problems.extend(w0_content_hash_problems(value, f"evidence {field_name}"))
    for field_name in ("source_point_id", "target_point_id", "file_version_id", "event_id", "predecessor_event_id"):
        value = getattr(evidence, field_name)
        if not w0_is_omitted(value):
            problems.extend(w0_uuid_problems(value, f"evidence {field_name}"))
    if not w0_is_omitted(evidence.source_uri):
        problems.extend(w0_uri_problems(evidence.source_uri, "evidence source_uri"))
    if not w0_is_omitted(evidence.file_path):
        problems.extend(w0_file_path_problems(evidence.file_path, "evidence file_path"))
    if evidence.locator != {}:
        # Only the documented empty-dict sentinel omits the evidence locator;
        # [], False, 0, or None are present, wrong-typed values and are held
        # to the shared locator domain like any other present evidence value.
        problems.extend(w0_locator_problems(evidence.locator))
    if not w0_is_omitted(evidence.observation):
        problems.extend(w0_vocabulary_problems(evidence.observation, "evidence observation", LINEAGE_OBSERVATIONS))
    if not w0_is_omitted(evidence.relation_type):
        problems.extend(
            w0_vocabulary_problems(evidence.relation_type, "evidence relation_type", MECHANICAL_EDGE_RELATIONS)
        )
    for field_name in ("source_entity_type", "target_entity_type"):
        value = getattr(evidence, field_name)
        if not w0_is_omitted(value):
            problems.extend(w0_vocabulary_problems(value, f"evidence {field_name}", KNOWN_ENTITY_TYPES))
    return problems


def ownership_matches(payload: dict[str, Any], evidence: LineageEvidence) -> bool:
    """Exact write-ownership check over the full scope tuple.

    The payload carries no collection field; the collection is bound at the
    gate (explicit ``collection_name`` argument) and through the recomputed
    scope digest every structural payload must carry. Missing optional
    ``user_id_hash`` / ``chat_id_hash`` on either side means empty string.
    Global retrieval scope semantics never apply to writes.
    """
    if not isinstance(payload, dict) or not isinstance(evidence, LineageEvidence):
        return False
    return (
        str(payload.get("profile_id") or "") == str(evidence.profile_id or "")
        and str(payload.get("user_id_hash") or "") == str(evidence.user_id_hash or "")
        and str(payload.get("chat_id_hash") or "") == str(evidence.chat_id_hash or "")
    )


# ---------------------------------------------------------------------------
# Structural payload builders
# ---------------------------------------------------------------------------

def build_source_node_payload(
    *,
    source_key: str,
    scope_key: str,
    profile_id: str,
    file_path: str,
    source_uri: str,
    lineage_operation: str = "index_capture",
    observation: str = "read_bytes",
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build the file-source graph entity payload (no raw title/text).

    The logical handle is derived from the bare source key digest; the display
    label is ``file-source-<source_key>`` and is not an identity input.
    """
    logical = source_logical_id(source_key=source_key, profile_id=profile_id)
    digest = entity_identity_digest("source", str(source_key or ""), profile_id=profile_id)
    payload = build_entity_payload(
        entity_type="source",
        label=f"file-source-{source_key}",
        logical_entity_id=logical,
        profile_id=profile_id,
        source_uri=source_uri,
        created_at=created_at,
        file_path=file_path,
        lineage_record=True,
        lineage_schema_version=LINEAGE_SCHEMA_VERSION,
        lineage_operation=lineage_operation,
        lineage_identity_digest=digest,
        lineage_source_key=source_key,
        lineage_scope_key=scope_key,
        lineage_observation=observation,
        lineage_role="file_source",
    )
    payload["user_id_hash"] = ""
    payload["chat_id_hash"] = ""
    return payload


def build_version_node_payload(
    *,
    source_key: str,
    scope_key: str,
    profile_id: str,
    file_path: str,
    file_sha256: str,
    file_size: int = 0,
    file_mtime: str = "",
    source_uri: str = "",
    lineage_operation: str = "index_capture",
    observation: str = "read_bytes",
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build the file-version graph entity payload.

    Identity is ``["file-version-v1", source_key, file_sha256]``; the label is
    ``file-version-<digest>``. ``content_hash`` records the raw-file hash on
    this version record only.
    """
    # Raw-type validation at the builder boundary (shared domain): a wrong-
    # typed digest (e.g. a 64-digit int) must refuse here, never be coerced
    # through str() into a look-alike token that identity, content_hash, and
    # string evidence would then agree on.
    problems = w0_sha256_hex_problems(file_sha256, "file_sha256")
    if problems:
        raise ValueError(problems[0])
    version_digest = make_version_identity_digest(source_key=source_key, file_sha256=file_sha256)
    logical = version_logical_id(version_identity_digest=version_digest, profile_id=profile_id)
    handle_digest = entity_identity_digest("source", version_digest, profile_id=profile_id)
    extra: dict[str, Any] = {}
    if file_size:
        extra["file_size"] = int(file_size)
    if file_mtime:
        extra["file_mtime"] = str(file_mtime)
    payload = build_entity_payload(
        entity_type="source",
        label=f"file-version-{version_digest}",
        logical_entity_id=logical,
        profile_id=profile_id,
        content_hash=f"sha256:{file_sha256}",
        source_uri=source_uri,
        created_at=created_at,
        file_path=file_path,
        lineage_record=True,
        lineage_schema_version=LINEAGE_SCHEMA_VERSION,
        lineage_operation=lineage_operation,
        lineage_identity_digest=handle_digest,
        lineage_source_key=source_key,
        lineage_scope_key=scope_key,
        lineage_observation=observation,
        lineage_role="file_version",
    )
    payload["file_sha256"] = file_sha256
    payload["user_id_hash"] = ""
    payload["chat_id_hash"] = ""
    payload.update(extra)
    return payload


def build_mechanical_edge_payload(
    *,
    relation_type: str,
    source_entity_id: str,
    target_entity_id: str,
    profile_id: str,
    source_point_id: str,
    target_point_id: str,
    source_entity_type: str,
    target_entity_type: str,
    lineage_operation: str,
    lineage_source_key: str = "",
    lineage_scope_key: str = "",
    observation: str = "indexed_payload",
    source_content_hash: str = "",
    target_content_hash: str = "",
    file_version_id: str = "",
    file_path: str = "",
    file_sha256: str = "",
    locator: dict[str, Any] | None = None,
    lineage_event_id: str = "",
    provenance_content_hash: str = "",
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a mechanical structural edge payload (``edge_class="mechanical"``).

    ``source_point_id`` / ``target_point_id`` are the actual exact Qdrant IDs
    (UUIDs for structural endpoints). The full edge identity digest is stored
    for collision detection; the logical ``edge-*`` handle stays in the payload.
    """
    if not w0_is_omitted(provenance_content_hash):
        # W0 shared content-hash domain, exact token, BEFORE the legacy
        # generic sanitizer: ``sanitize_content_hash`` strips and lax-matches,
        # which would normalize whitespace-padded or newline-suffixed digests
        # into storable tokens. Mechanical provenance must be the exact
        # documented token or the build refuses.
        problems = w0_content_hash_problems(provenance_content_hash, "provenance_content_hash")
        if problems:
            raise ValueError(problems[0])
    digest = edge_identity_digest(
        profile_id=profile_id,
        source_entity_id=source_entity_id,
        relation_type=relation_type,
        target_entity_id=target_entity_id,
    )
    payload = build_edge_payload(
        source_entity_id=source_entity_id,
        target_entity_id=target_entity_id,
        relation_type=relation_type,
        profile_id=profile_id,
        content_hash=provenance_content_hash,
        created_at=created_at,
        lineage_record=True,
        lineage_schema_version=LINEAGE_SCHEMA_VERSION,
        lineage_operation=lineage_operation,
        lineage_identity_digest=digest,
        lineage_source_key=lineage_source_key,
        lineage_scope_key=lineage_scope_key,
        lineage_observation=observation,
        lineage_event_id=lineage_event_id,
        edge_class=MECHANICAL_EDGE_CLASS,
        source_point_id=source_point_id,
        target_point_id=target_point_id,
        source_entity_type=source_entity_type,
        target_entity_type=target_entity_type,
        source_content_hash=source_content_hash,
        target_content_hash=target_content_hash,
        file_version_id=file_version_id,
        file_path=file_path,
        file_sha256=file_sha256,
        locator=locator,
    )
    payload["user_id_hash"] = ""
    payload["chat_id_hash"] = ""
    return payload


# ---------------------------------------------------------------------------
# Strict payload validation
# ---------------------------------------------------------------------------

def validate_lineage_payload(payload: Any) -> list[str]:
    """Strictly validate a structural lineage payload.

    Returns a list of problems; an empty list means the payload is a valid
    structural record. Every failure here is fail-closed at the gate.
    """
    problems: list[str] = []
    if not isinstance(payload, dict):
        return ["payload must be a dict"]

    memory_kind = payload.get("memory_kind")
    if memory_kind not in ("graph_entity", "graph_edge"):
        problems.append("memory_kind must be graph_entity or graph_edge")
        return problems

    if payload.get("lineage_record") is not True:
        problems.append("lineage_record must be exactly True")
    pending = payload.get("lineage_pending", False)
    if not isinstance(pending, bool):
        problems.append("lineage_pending must be a boolean when present")

    schema_version = payload.get("lineage_schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        problems.append("lineage_schema_version must be an integer (booleans are invalid)")
    elif schema_version != LINEAGE_SCHEMA_VERSION:
        problems.append(f"lineage_schema_version must be {LINEAGE_SCHEMA_VERSION}")

    operation = payload.get("lineage_operation")
    if operation not in LINEAGE_OPERATIONS:
        problems.append(f"unsupported lineage_operation: {operation!r}")

    scope_key = payload.get("lineage_scope_key")
    if not is_sha256_hex(scope_key):
        problems.append("lineage_scope_key must be a non-empty 64 lowercase hex scope digest")
    # Shared W0 shape domains: every interpreted structural field is
    # validated by raw type and domain whenever its key is present, BEFORE
    # any relation-specific requirement and before evidence comparison. Key
    # presence is presence: omission is the key's absence — an explicit
    # ""/None/0/False/{} value is a failed validation, never a silent
    # "absent". Builders share the exact same domain functions.
    if "lineage_source_key" in payload:
        problems.extend(w0_sha256_hex_problems(payload.get("lineage_source_key"), "lineage_source_key"))
    observation = payload.get("lineage_observation")
    if not observation or observation not in LINEAGE_OBSERVATIONS:
        problems.append(f"lineage_observation must be one of {LINEAGE_OBSERVATIONS}")

    identity_digest = payload.get("lineage_identity_digest")
    if not is_sha256_hex(identity_digest):
        problems.append("lineage_identity_digest must be 64 lowercase hex chars")

    logical_id = payload.get("entity_id") if memory_kind == "graph_entity" else payload.get("edge_id")
    # Exact logical-handle domain (shared validator, no stripping): a padded
    # or newline-suffixed handle is a different token and is refused here,
    # never normalized into the clean spelling.
    problems.extend(
        w0_logical_handle_problems(
            logical_id,
            field="entity_id" if memory_kind == "graph_entity" else "edge_id",
            handle_kind="entity" if memory_kind == "graph_entity" else "edge",
        )
    )
    if isinstance(identity_digest, str) and logical_id and not logical_id_matches_digest(str(logical_id), identity_digest):
        problems.append("logical handle does not match lineage_identity_digest truncation (collision)")

    # Trust flags: structural persistence never implies canonical truth or
    # review exemption.
    if payload.get("canonical") is not False:
        problems.append("structural records must have canonical=False")
    if payload.get("requires_review") is not True:
        problems.append("structural records must have requires_review=True")

    # Exact ownership keys must be present (absent optional scope == empty).
    for key in ("profile_id", "user_id_hash", "chat_id_hash"):
        if key not in payload or not isinstance(payload.get(key), str):
            problems.append(f"ownership field {key} must be present as a string")
    if not str(payload.get("profile_id") or "").strip():
        problems.append("profile_id must not be empty")

    if memory_kind == "graph_edge":
        edge_class = payload.get("edge_class")
        if edge_class != MECHANICAL_EDGE_CLASS:
            problems.append(f"structural edges must carry edge_class={MECHANICAL_EDGE_CLASS!r}")
        relation = payload.get("relation_type")
        if relation in CLAIM_RELATIONS:
            problems.append(f"claim relation {relation!r} is forbidden on mechanical edges")
        elif relation not in MECHANICAL_EDGE_RELATIONS:
            problems.append(f"unsupported mechanical relation: {relation!r}")
        if relation == "SUPERSEDES":
            # SUPERSEDES only ever runs between observed file versions; a
            # claim-level replacement between memory points is rejected here.
            for key in ("source_entity_type", "target_entity_type"):
                if payload.get(key) != "source":
                    problems.append(f"SUPERSEDES {key} must be 'source' (file version)")
        for key in ("source_point_id", "target_point_id"):
            value = payload.get(key)
            if value is None or not is_uuid_string(value):
                problems.append(f"{key} must be an exact UUID point ID")
        # Endpoint reference handles are logical handles: exact shared domain,
        # presence-based (a present non-handle is a failed validation).
        for key in ("source_entity_id", "target_entity_id"):
            if key in payload:
                problems.extend(
                    w0_logical_handle_problems(payload.get(key), field=key, handle_kind="entity")
                )
        # Endpoint types are controlled vocabulary (shared domain): present
        # values must already be the exact known token.
        for key in ("source_entity_type", "target_entity_type"):
            if key in payload:
                problems.extend(w0_vocabulary_problems(payload.get(key), key, KNOWN_ENTITY_TYPES))
        # lineage_event_id is a UUID reference token whenever the key is
        # present, on every relation — presence is presence, truthiness is
        # not omission (shared domain, enforced before any relation rule).
        if "lineage_event_id" in payload:
            problems.extend(w0_uuid_problems(payload.get("lineage_event_id"), "lineage_event_id"))
        # Optional observed-provenance fields: present keys must be in their
        # documented domains (shared validators).
        if "file_sha256" in payload:
            problems.extend(w0_sha256_hex_problems(payload.get("file_sha256"), "file_sha256"))
        if "file_path" in payload:
            problems.extend(w0_file_path_problems(payload.get("file_path"), "file_path"))
        if "source_uri" in payload:
            problems.extend(w0_uri_problems(payload.get("source_uri"), "source_uri"))
        if "locator" in payload:
            problems.extend(w0_locator_problems(payload.get("locator")))
        # A structural field only entity records interpret must not ride on
        # an edge record (the kind-domain mirror of the entity guard below).
        for key in ("entity_id", "entity_type", "lineage_role"):
            if key in payload:
                problems.append(f"{key} is not a valid field on a structural edge record")
        # file_version_id: key presence is presence. An explicit empty or
        # None value is not an omission — it is a binding claim that fails
        # validation instead of silently meaning "absent".
        if "file_version_id" in payload:
            version_value = payload.get("file_version_id")
            if not isinstance(version_value, str) or not is_uuid_string(version_value):
                problems.append(
                    "file_version_id must be a storage UUID point ID (logical handles live in the *entity_id reference fields)"
                )
        file_binding = any(str(payload.get(k) or "") for k in ("file_version_id", "file_path", "file_sha256"))
        if file_binding and not is_sha256_hex(payload.get("lineage_source_key")):
            problems.append("edges bound to a file require lineage_source_key")
        # Relation-specific endpoint roles (section 4 direction table). The
        # mechanical path only ever writes the documented directions; an
        # endpoint claiming another role is a forged binding.
        src_etype = str(payload.get("source_entity_type") or "")
        tgt_etype = str(payload.get("target_entity_type") or "")
        if relation == "PART_OF" and (src_etype != "source" or tgt_etype != "source"):
            problems.append("PART_OF must run file version -> file source (both endpoints entity_type='source')")
        elif relation == "DERIVED_FROM":
            if operation == "approved_citation":
                if "source" in (src_etype, tgt_etype):
                    problems.append("approved_citation DERIVED_FROM runs between ordinary memory points, not structural file nodes")
            elif tgt_etype != "source" or src_etype == "source":
                problems.append("index DERIVED_FROM must run ordinary point -> file version (target entity_type='source')")
        elif relation in ("SUMMARIZES", "EXTRACTED_FROM") and "source" in (src_etype, tgt_etype):
            problems.append(f"{relation} must run between ordinary memory points, not structural file nodes")
        # Endpoint content hashes are in the documented hash domain
        # (sha256:<64 lowercase hex>) whenever their keys are present — for
        # both endpoint positions and WHATEVER the relation (this replaces
        # the old relation-conditional shape check, which let PART_OF store a
        # malformed target hash). Shape validation runs BEFORE any evidence
        # comparison: equality with an equally malformed evidence value
        # proves nothing.
        for hash_key in ("source_content_hash", "target_content_hash"):
            if hash_key in payload:
                problems.extend(w0_content_hash_problems(payload.get(hash_key), hash_key))
        # Endpoint snapshot-hash LEGALITY is derived from the documented
        # relation/endpoint role table, not enumerated per instance: a
        # file_source endpoint designates the exact path and never carries a
        # snapshot digest, so PART_OF's target_content_hash refuses even when
        # the evidence supplies an equal value (validation precedes every
        # comparison). file_version and memory_point endpoints are the only
        # snapshot-bearing roles; their required hashes stay governed by the
        # relation requirements below.
        for position, hash_key in (("source", "source_content_hash"), ("target", "target_content_hash")):
            if hash_key in payload:
                endpoint_role = _endpoint_role(relation_type=relation, operation=operation, position=position)
                if endpoint_role == "file_source":
                    problems.append(
                        f"{hash_key} is not valid on a file-source endpoint: a file-source node designates the path, not a content snapshot"
                    )
        # The edge's own provenance digest (``content_hash``) is in the same
        # documented hash domain whenever the payload carries it — whatever
        # the relation. Key presence is presence: an explicit empty string,
        # None, or a non-string is a failed validation, not an omission the
        # guard can skip. Binding to the evidence (below) never excuses a
        # value outside the domain, including a digest only an attacker
        # coined.
        if "content_hash" in payload:
            problems.extend(w0_content_hash_problems(payload.get("content_hash"), "content_hash"))
        if relation == "SUPERSEDES":
            # SUPERSEDES runs only between two distinct observed file versions:
            # both endpoints are content-bearing versions with different hashes
            # (the shape of a present hash is enforced above; this adds the
            # relation requirement that both are actually present and distinct).
            for hash_key in ("source_content_hash", "target_content_hash"):
                if hash_key not in payload:
                    problems.append(
                        f"SUPERSEDES requires {hash_key} (both endpoints must be content-bearing file versions)"
                    )
            src_hash = str(payload.get("source_content_hash") or "")
            tgt_hash = str(payload.get("target_content_hash") or "")
            if src_hash and src_hash == tgt_hash:
                problems.append("SUPERSEDES requires two distinct version hashes")
            if not payload.get("lineage_event_id"):
                problems.append("SUPERSEDES requires a lineage_event_id UUID reference token")
    else:
        role = payload.get("lineage_role")
        if role not in LINEAGE_ROLES:
            problems.append(f"lineage_role must be one of {LINEAGE_ROLES}")
        entity_type = payload.get("entity_type")
        if entity_type != "source" and role in ("file_source", "file_version"):
            problems.append("file_source/file_version records must use entity_type='source'")
        # Kind-domain guard: a structural field only a mechanical EDGE
        # interprets must not ride on an entity record. Key presence is
        # presence — the field would otherwise escape its documented domain
        # (e.g. a lineage_event_id on a file_source record).
        for key in (
            "edge_class",
            "edge_id",
            "relation_type",
            "source_entity_id",
            "target_entity_id",
            "source_point_id",
            "target_point_id",
            "source_entity_type",
            "target_entity_type",
            "source_content_hash",
            "target_content_hash",
            "file_version_id",
            "lineage_event_id",
            "locator",
        ):
            if key in payload:
                problems.append(f"{key} is not a valid field on a structural entity record")
        # Observed file fields on entities: present keys must be in their
        # documented domains (shared validators). Record-role validity: a
        # snapshot digest exists ONLY on the file_version record — a
        # file_source node designates the exact path, never a content
        # snapshot, so it may not carry a file digest whatever the value or
        # evidence (mirrors the content_hash prohibition).
        if "file_sha256" in payload:
            problems.extend(w0_sha256_hex_problems(payload.get("file_sha256"), "file_sha256"))
            if role != "file_version":
                problems.append(
                    "file_sha256 is not a valid field on a structural entity record that is not file_version: only the file-version record carries the snapshot digest"
                )
        if "file_path" in payload:
            problems.extend(w0_file_path_problems(payload.get("file_path"), "file_path"))
        if "source_uri" in payload:
            problems.extend(w0_uri_problems(payload.get("source_uri"), "source_uri"))
        # A structural entity record that carries its own content hash (the
        # file-version record does) carries it in the documented shared
        # domain — shape is validated here, whatever the role. Key presence
        # is presence: an explicit empty string, None, or non-string fails
        # validation instead of meaning "absent". A file_source record must
        # not carry one at all: the source node designates the exact path,
        # not a content snapshot, and no evidence record observes content
        # for it — any digest it carried would be unbindable provenance
        # (R3: a well-formed but unrelated digest on a file_source record
        # used to store).
        if "content_hash" in payload:
            shape_problems = w0_content_hash_problems(payload.get("content_hash"), "content_hash")
            if shape_problems:
                problems.extend(shape_problems)
            elif role == "file_source":
                problems.append(
                    "file_source records must not carry content_hash: the source node designates the path, not a content snapshot (file_version records carry the evidenced digest)"
                )
        if role == "change_event":
            # W0 has no typed change-event identity (that arrives with the
            # reconciliation wave), so change_event structural records fail
            # closed rather than persisting under an unvalidated identity.
            problems.append(
                "change_event structural records are not writable in W0: typed event identity arrives in a later wave (fail closed)"
            )
        if role in ("file_source", "file_version") and not is_sha256_hex(payload.get("lineage_source_key")):
            problems.append("file_source/file_version records require lineage_source_key")
        label = str(payload.get("label") or "")
        text = str(payload.get("text") or "")
        if role == "file_source":
            if not label.startswith("file-source-") or text != label:
                problems.append("file_source label must be 'file-source-<source_key>' with no extra text")
            if "source_uri" not in payload or not str(payload.get("source_uri") or ""):
                problems.append("file_source requires source_uri")
            if "file_path" not in payload or not str(payload.get("file_path") or ""):
                problems.append("file_source requires file_path")
        elif role == "file_version":
            if not label.startswith("file-version-") or text != label:
                problems.append("file_version label must be 'file-version-<digest>' with no extra text")
            file_sha256 = payload.get("file_sha256")
            if not is_sha256_hex(file_sha256):
                problems.append("file_version requires file_sha256 as 64 lowercase hex chars")
            if payload.get("content_hash") != f"sha256:{file_sha256}":
                problems.append("file_version content_hash must be 'sha256:' + file_sha256")
            if "file_path" not in payload or not str(payload.get("file_path") or ""):
                problems.append("file_version requires file_path")

    if contains_secret(canonical_json(payload)):
        problems.append("possible_secret")
    return problems


# Payload field -> evidence twin for the independent-binding class: whenever
# the payload carries an interpreted evidence/provenance field, the evidence
# twin must be present (non-sentinel) or the record fails closed BEFORE any
# comparison. Documented independence exceptions (fields whose validity is
# already established without the evidence twin):
# - ``lineage_scope_key``: bound by RECOMPUTATION from the gate's collection
#   plus the evidence ownership tuple; the mismatch layer compares the payload
#   digest to that recomputation exactly, so a claimed evidence digest is never
#   needed (present-and-wrong still refuses via the internal-evidence check).
# - ``content_hash`` on EDGE records: bound by the documented anchors rule
#   (a present payload digest must equal the evidence digest or one of the
#   independently required endpoint/file digests). On ENTITY records there is
#   no anchors path, so the evidence twin is required.
_EVIDENCE_BINDING_FIELDS: tuple[tuple[str, str], ...] = (
    ("lineage_scope_key", "scope_key"),
    ("lineage_source_key", "source_key"),
    ("content_hash", "content_hash"),
    ("file_sha256", "file_sha256"),
    ("file_version_id", "file_version_id"),
    ("source_content_hash", "source_content_hash"),
    ("target_content_hash", "target_content_hash"),
    ("file_path", "file_path"),
    ("source_uri", "source_uri"),
    ("locator", "locator"),
    ("lineage_event_id", "event_id"),
    ("relation_type", "relation_type"),
    ("source_entity_type", "source_entity_type"),
    ("target_entity_type", "target_entity_type"),
    ("source_point_id", "source_point_id"),
    ("target_point_id", "target_point_id"),
)


def evidence_missing_requirements(payload: dict[str, Any], evidence: LineageEvidence) -> list[str]:
    """Independent evidence every record kind must carry before comparison.

    A payload must not certify its own provenance: the fields its identity is
    derived from must exist on the evidence record first. Missing independent
    evidence fails closed — it is never treated as "nothing to compare".
    """
    missing: list[str] = []

    def _require(condition: Any, label: str) -> None:
        if not condition:
            missing.append(label)

    _require(bool(str(evidence.observation or "")), "observation")
    kind = payload.get("memory_kind")
    if kind == "graph_entity":
        role = payload.get("lineage_role")
        if role in ("file_source", "file_version"):
            _require(bool(str(evidence.file_path or "")), "file_path")
        if role == "file_source":
            _require(bool(str(evidence.source_uri or "")), "source_uri")
        if role == "file_version":
            _require(is_sha256_hex(evidence.file_sha256), "file_sha256")
            # Both file-node kinds bind the exact file URI of the evidenced
            # resolved path (validated against the path in
            # ``evidence_internal_mismatches``).
            _require(bool(str(evidence.source_uri or "")), "source_uri")
    elif kind == "graph_edge":
        relation = str(payload.get("relation_type") or "")
        _require(bool(str(evidence.relation_type or "")), "relation_type")
        _require(is_uuid_string(evidence.source_point_id), "source_point_id")
        _require(is_uuid_string(evidence.target_point_id), "target_point_id")
        _require(bool(str(evidence.source_entity_type or "")), "source_entity_type")
        _require(bool(str(evidence.target_entity_type or "")), "target_entity_type")
        file_sha = str(evidence.file_sha256 or "")
        if relation == "PART_OF" or (relation == "DERIVED_FROM" and str(evidence.target_entity_type or "") == "source") or relation == "SUPERSEDES":
            # File-bound structural edges: the file identity the endpoints
            # are derived from must exist independently.
            _require(bool(str(evidence.file_path or "")), "file_path")
            _require(is_sha256_hex(file_sha), "file_sha256")
        if relation == "PART_OF":
            _require(str(evidence.source_content_hash or "") == f"sha256:{file_sha}", "source_content_hash")
        elif relation == "DERIVED_FROM":
            if str(evidence.target_entity_type or "") == "source":
                # chunk -> file version: the version endpoint is content-bearing
                # and must be exactly the version of the observed file hash.
                _require(str(evidence.target_content_hash or "") == f"sha256:{file_sha}", "target_content_hash")
                # The chunk itself needs independent provenance: its own
                # validated content hash plus a bounded locator, and the URI
                # of the file version it derives from. A version binding with
                # no chunk provenance is exactly what the direction table
                # refuses ("locator and chunk hash match the prepared
                # snapshot").
                _require(
                    not w0_content_hash_problems(evidence.source_content_hash, "source_content_hash"),
                    "source_content_hash",
                )
                # The locator is evidence only in the exact real indexer
                # shape: presence alone proves nothing.
                _require(not file_chunk_locator_problems(evidence.locator), "locator")
                _require(bool(str(evidence.source_uri or "")), "source_uri")
            else:
                # approved derived point -> cited source point: both cited
                # content snapshots must exist AND be in the documented hash
                # domain. String truthiness would let an equally malformed
                # evidence value authorize an equally malformed payload.
                _require(
                    not w0_content_hash_problems(evidence.source_content_hash, "source_content_hash"),
                    "source_content_hash must be a documented sha256:<64 lowercase hex> content hash",
                )
                _require(
                    not w0_content_hash_problems(evidence.target_content_hash, "target_content_hash"),
                    "target_content_hash must be a documented sha256:<64 lowercase hex> content hash",
                )
        elif relation == "SUPERSEDES":
            # The new version is the one the evidence observed on disk.
            _require(str(evidence.source_content_hash or "") == f"sha256:{file_sha}", "source_content_hash")
            # The predecessor version's hash is a documented sha256 digest
            # too: truthiness alone would accept any non-empty spelling, and
            # the payload-side shape check must not be the only exact-token
            # enforcement in the chain.
            _require(
                not w0_content_hash_problems(evidence.target_content_hash, "target_content_hash"),
                "target_content_hash must be a documented sha256:<64 lowercase hex> content hash",
            )
        elif relation in ("SUMMARIZES", "EXTRACTED_FROM"):
            # Both cited content snapshots must exist AND be in the documented
            # hash domain (sha256:<64 lowercase hex>) — for both endpoint
            # positions. Truthiness alone accepted 'not-a-content-hash' as
            # long as the payload carried the same malformation.
            _require(
                not w0_content_hash_problems(evidence.source_content_hash, "source_content_hash"),
                "source_content_hash must be a documented sha256:<64 lowercase hex> content hash",
            )
            _require(
                not w0_content_hash_problems(evidence.target_content_hash, "target_content_hash"),
                "target_content_hash must be a documented sha256:<64 lowercase hex> content hash",
            )
    # Independent-binding class (systematic): whenever the payload carries an
    # interpreted evidence/provenance field, its evidence twin must be present
    # (non-sentinel) — omission of the twin fails closed here, BEFORE the
    # equality/domain comparison layers; when the twin is present, the
    # existing comparison handles alteration. Sentinels: None/"" for scalar
    # evidence fields, the empty mapping ONLY for the evidence locator.
    for payload_key, evidence_field in _EVIDENCE_BINDING_FIELDS:
        if payload_key not in payload:
            continue
        if payload_key == "lineage_scope_key":
            # Documented exception: bound by recomputation (see map above).
            continue
        if payload_key == "content_hash" and kind != "graph_entity":
            # Documented exception: edge provenance digest bound by the
            # anchors rule (see map above).
            continue
        evidence_value = getattr(evidence, evidence_field)
        twin_omitted = evidence_value == {} if evidence_field == "locator" else w0_is_omitted(evidence_value)
        if twin_omitted:
            missing.append(
                f"{payload_key} is present on the payload but evidence {evidence_field} is omitted (unbound provenance)"
            )
    return missing


def evidence_payload_mismatches(payload: dict[str, Any], evidence: LineageEvidence) -> list[str]:
    """Bind independently supplied evidence to the final payload fields.

    Altered provenance after enrichment shows up here: a sanitized-away or
    swapped path/hash/scope cannot be replaced by an unrelated surviving ID.
    The authorized operation, the recomputed write scope, and (when the
    evidence binds a file) the recomputed source key are REQUIRED on the
    final payload and must match exactly; a missing required binding is a
    mismatch, not an omission.
    """
    mismatches: list[str] = []

    def _check(payload_value: Any, evidence_value: Any, label: str) -> None:
        # Compare only fields the payload actually carries: legitimate omissions
        # are handled by schema validation, silent alteration is caught here.
        ev = str(evidence_value or "")
        if not ev or payload_value is None:
            return
        if str(payload_value) != ev:
            mismatches.append(label)

    def _check_required(payload_value: Any, evidence_value: Any, label: str) -> None:
        if str(evidence_value or "") and str(payload_value or "") != str(evidence_value):
            mismatches.append(label)

    def _check_bound(payload_value: Any, evidence_value: Any, label: str) -> None:
        # Bidirectional optional-provenance binding: a payload may carry an
        # optional provenance field only when the evidence observed exactly
        # that value. Payload-only provenance (the evidence value is absent)
        # is as unbound as an altered value, and a sanitized-away field whose
        # evidence still carries a value is equally refused. Absent on both
        # sides stays intentionally optional.
        ev = str(evidence_value or "")
        pv_absent = payload_value is None or (isinstance(payload_value, str) and not payload_value)
        if pv_absent:
            if ev:
                mismatches.append(label)
            return
        if not ev or str(payload_value) != ev:
            mismatches.append(label)

    # The authorized operation must be the operation the final payload claims.
    _check_required(payload.get("lineage_operation"), evidence.operation, "lineage_operation")

    # Scope is recomputed from the evidence's own collection plus ownership
    # tuple; the payload must carry exactly that digest.
    expected_scope = evidence_scope_key(evidence)
    if str(payload.get("lineage_scope_key") or "") != expected_scope:
        mismatches.append("lineage_scope_key")

    # When the evidence binds a file path, the source key is recomputed from
    # the recomputed scope and that exact path (never taken on trust).
    if str(evidence.file_path or ""):
        expected_source = evidence_source_key(evidence, expected_scope)
        if str(payload.get("lineage_source_key") or "") != expected_source:
            mismatches.append("lineage_source_key")
    else:
        _check_required(payload.get("lineage_source_key"), evidence.source_key, "lineage_source_key")

    _check_required(payload.get("lineage_observation"), evidence.observation, "lineage_observation")
    _check(payload.get("file_path"), evidence.file_path, "file_path")
    _check(payload.get("file_sha256"), evidence.file_sha256, "file_sha256")

    # A current transition event claimed by the payload must be the event the
    # evidence recorded; SUPERSEDES requires one (enforced at the gate), every
    # other record kind merely refuses an unevidenced or altered event id.
    payload_event = str(payload.get("lineage_event_id") or "")
    evidence_event = str(evidence.event_id or "")
    if payload_event or evidence_event:
        if payload_event != evidence_event:
            mismatches.append("lineage_event_id")

    if payload.get("memory_kind") == "graph_entity":
        role = payload.get("lineage_role")
        payload_uri = str(payload.get("source_uri") or "")
        if role == "file_version":
            # The version record's snapshot digest must be exactly the
            # content digest the evidence observed. The binding layer already
            # fails an omitted twin; an unrelated WELL-FORMED twin must fail
            # here — without this comparison a valid version payload stored
            # against any unrelated sha256 evidence digest.
            _check_required(payload.get("content_hash"), evidence.content_hash, "content_hash")
        if role in ("file_source", "file_version"):
            # File-node records must carry the URI the evidence observed —
            # which itself must be the exact URI of the evidenced resolved
            # path. A dropped URI is as unbound as an altered one; a payload
            # URI the evidence never observed is equally unbound provenance.
            if payload_uri != str(evidence.source_uri or ""):
                mismatches.append("source_uri")
        elif str(evidence.source_uri or "") and payload_uri != str(evidence.source_uri):
            mismatches.append("source_uri")
        elif payload_uri and not str(evidence.source_uri or ""):
            mismatches.append("source_uri")
    else:
        # Edges must bind the evidence-supplied endpoint identity completely:
        # a missing binding is as suspicious as an altered one.
        for key in (
            "relation_type",
            "source_point_id",
            "target_point_id",
            "source_entity_type",
            "target_entity_type",
            "source_content_hash",
            "target_content_hash",
            # file_version_id is evidence-bound when the evidence records it
            # and, for file-bound edges, independently bound by the identity
            # recomputation (it must designate a derived version endpoint),
            # so it is not observation-bound here.
            "file_version_id",
        ):
            _check_required(payload.get(key), getattr(evidence, key), key)
        # Observation-bound optional provenance fields bind bidirectionally:
        # payload-only provenance the evidence never observed must not be
        # acceptable for any of them.
        for key in ("file_path", "file_sha256"):
            _check_bound(payload.get(key), getattr(evidence, key), key)
        # ``content_hash`` carries the edge's own provenance digest and is
        # bound, never compare-when-both: when the evidence records the
        # digest, the payload must carry exactly that value (a
        # sanitized-away digest is a mismatch, not an omission). A
        # payload-only digest must designate independently evidenced
        # content — never an arbitrary value. (The shape of a present
        # payload value is enforced in ``validate_lineage_payload``;
        # structural entity records are bound independently — a
        # file-version record's digest must equal its evidenced file hash —
        # and a file-source record legitimately carries none, so this
        # edge-only rule does not extend to them.)
        payload_hash = payload.get("content_hash")
        payload_hash_text = str(payload_hash or "")
        evidence_hash = str(evidence.content_hash or "")
        if evidence_hash:
            if payload_hash is None or payload_hash_text != evidence_hash:
                mismatches.append("content_hash")
        elif payload_hash_text:
            # A payload-only digest is bound when it designates content the
            # evidence independently observed: the evidenced file digest
            # (file-bound edges) or the evidenced source-endpoint snapshot
            # (approved-citation edges; for PART_OF the source hash IS the
            # file digest, and for chunk DERIVED_FROM the source hash is the
            # chunk digest the edge is about). Anything else is unbound
            # provenance the payload coined for itself.
            anchors = {str(evidence.source_content_hash or "")}
            file_sha = str(evidence.file_sha256 or "")
            if file_sha:
                anchors.add(f"sha256:{file_sha}")
            if payload_hash_text not in anchors:
                mismatches.append("content_hash")
        payload_locator = payload.get("locator")
        if evidence.locator:
            # Python dict equality treats True == 1 and 1.0 == 1 as equal, so
            # a type-drifted payload locator can compare equal to valid
            # evidence. The final payload locator must itself be valid real
            # indexer shape, independent of equality.
            if payload_locator != evidence.locator or file_chunk_locator_problems(payload_locator):
                mismatches.append("locator")
        elif payload_locator:
            # A locator the evidence never observed is unbound provenance.
            mismatches.append("locator")
        # Edge payloads do not carry the file URI; the evidence URI itself is
        # bound to the exact evidenced path (internal mismatch check above).
        # Defensively, an edge payload that does carry one may only carry the
        # evidenced URI — a payload-only URI (empty evidence URI) is unevidenced
        # provenance and is refused exactly like a contradictory one.
        payload_uri = str(payload.get("source_uri") or "")
        if payload_uri and payload_uri != str(evidence.source_uri or ""):
            mismatches.append("source_uri")
    return mismatches


def _endpoint_role(*, relation_type: str, operation: str, position: str) -> str:
    """Role one edge endpoint must play for this relation (section 4 table)."""
    if relation_type == "PART_OF":
        # file version -> file source, never version -> version.
        return "file_version" if position == "source" else "file_source"
    if relation_type == "SUPERSEDES":
        return "file_version"
    if relation_type == "DERIVED_FROM" and operation != "approved_citation":
        # indexed chunk -> file version.
        return "memory_point" if position == "source" else "file_version"
    # SUMMARIZES / EXTRACTED_FROM and approved-citation DERIVED_FROM run
    # between ordinary memory points.
    return "memory_point"


def _expected_endpoint_binding(
    *,
    relation_type: str,
    operation: str,
    entity_type: str,
    position: str,
    evidence: LineageEvidence,
    scope_key: str,
    source_key: str,
) -> tuple[str | None, str | None, str, str]:
    """Derive ``(logical_handle, storage_point_id, role, problem)`` for one edge endpoint.

    Endpoint roles are relation-specific, never inferred from a generic
    ``entity_type`` + content-hash-presence rule: PART_OF's target is ALWAYS
    the file-source node (a target content hash on the evidence cannot turn
    it into a second version), and the indexed DERIVED_FROM source is always
    an ordinary memory point. A structural file-node endpoint is derived from
    the evidence's recomputed source key plus the endpoint's own version
    hash; a memory-point endpoint is derived from the exact scoped point ID.
    ``storage_point_id`` is None for ordinary endpoints: their Qdrant point
    ID is the ordinary point's own ID, supplied by the evidence and bound by
    ``evidence_payload_mismatches``.
    """
    profile = str(evidence.profile_id or "")
    role = _endpoint_role(relation_type=relation_type, operation=operation, position=position)
    expected_type = "memory_point" if role == "memory_point" else "source"
    if str(entity_type or "") != expected_type:
        return None, None, role, f"{position}_entity_type"
    if role == "memory_point":
        point_id = evidence.source_point_id if position == "source" else evidence.target_point_id
        if not is_uuid_string(point_id):
            return None, None, role, f"{position}_point_id"
        handle = memory_point_endpoint_logical_id(scope_key=scope_key, point_id=point_id, profile_id=profile)
        return handle, None, role, ""
    if not is_sha256_hex(source_key):
        return None, None, role, "lineage_source_key"
    if role == "file_source":
        handle = make_entity_id("source", source_key, profile_id=profile)
        return handle, storage_point_id(handle), role, ""
    # file_version endpoint: bound to the version hash the evidence observed.
    # The version on disk is evidence.file_sha256; SUPERSEDES' previous
    # version is the target endpoint's own content hash.
    file_sha = str(evidence.file_sha256 or "")
    if relation_type == "SUPERSEDES" and position == "target":
        content_hash = str(evidence.target_content_hash or "")
        file_sha = content_hash[7:] if content_hash.startswith("sha256:") else ""
    if not is_sha256_hex(file_sha):
        return None, None, role, "file_sha256"
    version_digest = make_version_identity_digest(source_key=source_key, file_sha256=file_sha)
    handle = make_entity_id("source", version_digest, profile_id=profile)
    return handle, storage_point_id(handle), role, ""


def structural_identity_mismatches(payload: dict[str, Any], evidence: LineageEvidence) -> list[str]:
    """Independently recompute the full structural identity and compare.

    Logical handles are 16-hex truncations of SHA-256; two distinct full
    identities can share a truncation, so truncation agreement alone proves
    nothing. The gate therefore recomputes the complete identity digest from
    the evidence's own binding material (recomputed scope, source key, raw
    hash, declared endpoint material) and refuses any differing full identity
    — a same-prefix / different-suffix collision included.

    For edges, BOTH endpoint logical handles (and, for structural endpoints,
    their storage UUIDs) are derived from the evidence, and the edge identity
    digest is recomputed from the DERIVED handles — never from the handles the
    payload declares for itself. For file-version nodes the identity hash is
    the evidence's independently observed hash, not the payload's claim.

    Compare-before-overwrite against the digest already stored under an
    existing storage ID is a persistence-time duty (the W1 writer's read-back
    repair path) on top of this check; W0 has no writer.
    """
    mismatches: list[str] = []
    profile = str(evidence.profile_id or "")
    scope = evidence_scope_key(evidence)
    source_key = evidence_source_key(evidence, scope) if str(evidence.file_path or "") else ""
    if payload.get("memory_kind") == "graph_entity":
        role = payload.get("lineage_role")
        if role == "file_source":
            expected_digest = entity_identity_digest("source", source_key, profile_id=profile)
            if payload.get("lineage_identity_digest") != expected_digest:
                mismatches.append("lineage_identity_digest")
            if payload.get("entity_id") != make_entity_id("source", source_key, profile_id=profile):
                mismatches.append("entity_id")
            if payload.get("label") != f"file-source-{source_key}":
                mismatches.append("label")
        elif role == "file_version":
            file_sha = str(evidence.file_sha256 or "")
            if not is_sha256_hex(file_sha):
                mismatches.append("file_sha256")
            else:
                if str(payload.get("file_sha256") or "") != file_sha:
                    mismatches.append("file_sha256")
                version_digest = make_version_identity_digest(source_key=source_key, file_sha256=file_sha)
                expected_digest = entity_identity_digest("source", version_digest, profile_id=profile)
                if payload.get("lineage_identity_digest") != expected_digest:
                    mismatches.append("lineage_identity_digest")
                if payload.get("entity_id") != make_entity_id("source", version_digest, profile_id=profile):
                    mismatches.append("entity_id")
                if payload.get("label") != f"file-version-{version_digest}":
                    mismatches.append("label")
    else:
        relation = str(payload.get("relation_type") or "")
        operation = str(payload.get("lineage_operation") or "")
        src_handle, src_storage, src_role, src_problem = _expected_endpoint_binding(
            relation_type=relation,
            operation=operation,
            entity_type=str(payload.get("source_entity_type") or ""),
            position="source",
            evidence=evidence,
            scope_key=scope,
            source_key=source_key,
        )
        tgt_handle, tgt_storage, tgt_role, tgt_problem = _expected_endpoint_binding(
            relation_type=relation,
            operation=operation,
            entity_type=str(payload.get("target_entity_type") or ""),
            position="target",
            evidence=evidence,
            scope_key=scope,
            source_key=source_key,
        )
        for problem in (src_problem, tgt_problem):
            if problem:
                mismatches.append(problem)
        if src_handle and payload.get("source_entity_id") != src_handle:
            mismatches.append("source_entity_id")
        if tgt_handle and payload.get("target_entity_id") != tgt_handle:
            mismatches.append("target_entity_id")
        if src_storage and payload.get("source_point_id") != src_storage:
            mismatches.append("source_point_id")
        if tgt_storage and payload.get("target_point_id") != tgt_storage:
            mismatches.append("target_point_id")
        if src_handle and tgt_handle:
            expected = edge_identity_digest(
                profile_id=str(payload.get("profile_id") or ""),
                source_entity_id=src_handle,
                relation_type=relation,
                target_entity_id=tgt_handle,
            )
            if payload.get("lineage_identity_digest") != expected:
                mismatches.append("lineage_identity_digest")
            if payload.get("edge_id") != f"edge-{expected[:16]}":
                mismatches.append("edge_id")
        # file_version_id must designate a file-version endpoint exactly.
        # The file-source node's UUID is a different record: accepting it as a
        # version binding would let an edge claim a version identity that is
        # actually the source node. An id with no version endpoint behind it
        # is equally unbindable.
        version_storages = {
            storage
            for role, storage in ((src_role, src_storage), (tgt_role, tgt_storage))
            if role == "file_version" and storage
        }
        payload_version_id = str(payload.get("file_version_id") or "")
        if payload_version_id and payload_version_id not in version_storages:
            mismatches.append("file_version_id")
    return mismatches


# W1 additive capture helpers. W0 functions above intentionally stay unchanged.
CHUNKER_VERSION = "text-markdown-v1"


def make_versioned_file_chunk_id(
    *, scope_key: str, resolved_file_path: str, file_sha256: str,
    chunker_version: str, chunk_index: int, chunk_hash: str,
) -> str:
    return make_point_id(
        "indexed-file-v2",
        canonical_json([scope_key, resolved_file_path, file_sha256, chunker_version, chunk_index, chunk_hash]),
    )


def _capture_evidence(
    *, collection_name: str, profile_id: str, user_id_hash: str,
    chat_id_hash: str, scope_key: str, source_key: str, source_uri: str,
    file_path: str, file_sha256: str, observation: str,
    content_hash: str = "", source_content_hash: str = "",
    target_content_hash: str = "", file_version_id: str = "",
    locator: dict[str, Any] | None = None, relation_type: str = "",
    source_point_id: str = "", target_point_id: str = "",
    source_entity_type: str = "", target_entity_type: str = "",
) -> LineageEvidence:
    return LineageEvidence(
        operation="index_capture", collection_name=collection_name,
        profile_id=profile_id, user_id_hash=user_id_hash,
        chat_id_hash=chat_id_hash, scope_key=scope_key, source_key=source_key,
        source_uri=source_uri, file_path=file_path, file_sha256=file_sha256,
        content_hash=content_hash, source_content_hash=source_content_hash,
        target_content_hash=target_content_hash, file_version_id=file_version_id,
        locator=dict(locator or {}), observation=observation,
        relation_type=relation_type, source_point_id=source_point_id,
        target_point_id=target_point_id, source_entity_type=source_entity_type,
        target_entity_type=target_entity_type,
    )


def build_file_source(
    *, collection_name: str, profile_id: str, user_id_hash: str,
    chat_id_hash: str, scope_key: str, source_key: str, file_path: str,
    source_uri: str, current_version_id: str, created_at: str | None = None,
) -> tuple[str, dict[str, Any], LineageEvidence]:
    if not is_uuid_string(current_version_id):
        raise ValueError("current_version_id must be a UUID")
    payload = build_source_node_payload(
        source_key=source_key, scope_key=scope_key, profile_id=profile_id,
        file_path=file_path, source_uri=source_uri, created_at=created_at,
    )
    payload.update(user_id_hash=user_id_hash, chat_id_hash=chat_id_hash,
                   current_version_id=current_version_id, source_deleted=False)
    point_id = storage_point_id(str(payload["entity_id"]))
    evidence = _capture_evidence(
        collection_name=collection_name, profile_id=profile_id,
        user_id_hash=user_id_hash, chat_id_hash=chat_id_hash,
        scope_key=scope_key, source_key=source_key, source_uri=source_uri,
        file_path=file_path, file_sha256="", observation="read_bytes",
    )
    return point_id, payload, evidence


def build_file_version(
    *, collection_name: str, profile_id: str, user_id_hash: str,
    chat_id_hash: str, scope_key: str, source_key: str, file_path: str,
    source_uri: str, file_sha256: str, file_size: int, file_mtime: str,
    file_mtime_ns: int, created_at: str | None = None,
) -> tuple[str, dict[str, Any], LineageEvidence]:
    payload = build_version_node_payload(
        source_key=source_key, scope_key=scope_key, profile_id=profile_id,
        file_path=file_path, file_sha256=file_sha256, file_size=file_size,
        file_mtime=file_mtime, source_uri=source_uri, created_at=created_at,
    )
    payload.update(user_id_hash=user_id_hash, chat_id_hash=chat_id_hash,
                   file_mtime_ns=file_mtime_ns, lineage_history_complete=False)
    point_id = storage_point_id(str(payload["entity_id"]))
    evidence = _capture_evidence(
        collection_name=collection_name, profile_id=profile_id,
        user_id_hash=user_id_hash, chat_id_hash=chat_id_hash,
        scope_key=scope_key, source_key=source_key, source_uri=source_uri,
        file_path=file_path, file_sha256=file_sha256,
        content_hash=f"sha256:{file_sha256}", observation="read_bytes",
    )
    return point_id, payload, evidence


def build_chunk_derivation(
    *, collection_name: str, profile_id: str, user_id_hash: str,
    chat_id_hash: str, scope_key: str, source_key: str, file_path: str,
    source_uri: str, file_sha256: str, chunk_point_id: str,
    chunk_entity_id: str, chunk_hash: str, locator: dict[str, Any],
    version_point_id: str, version_entity_id: str,
) -> tuple[str, dict[str, Any], LineageEvidence]:
    source_hash, target_hash = f"sha256:{chunk_hash}", f"sha256:{file_sha256}"
    payload = build_mechanical_edge_payload(
        relation_type="DERIVED_FROM", source_entity_id=chunk_entity_id,
        target_entity_id=version_entity_id, profile_id=profile_id,
        source_point_id=chunk_point_id, target_point_id=version_point_id,
        source_entity_type="memory_point", target_entity_type="source",
        lineage_operation="index_capture", lineage_source_key=source_key,
        lineage_scope_key=scope_key, observation="indexed_payload",
        source_content_hash=source_hash, target_content_hash=target_hash,
        file_version_id=version_point_id, file_path=file_path,
        file_sha256=file_sha256, locator=locator,
    )
    payload.update(user_id_hash=user_id_hash, chat_id_hash=chat_id_hash)
    point_id = storage_point_id(str(payload["edge_id"]))
    evidence = _capture_evidence(
        collection_name=collection_name, profile_id=profile_id,
        user_id_hash=user_id_hash, chat_id_hash=chat_id_hash,
        scope_key=scope_key, source_key=source_key, source_uri=source_uri,
        file_path=file_path, file_sha256=file_sha256,
        source_content_hash=source_hash, target_content_hash=target_hash,
        file_version_id=version_point_id, locator=locator,
        observation="indexed_payload", relation_type="DERIVED_FROM",
        source_point_id=chunk_point_id, target_point_id=version_point_id,
        source_entity_type="memory_point", target_entity_type="source",
    )
    return point_id, payload, evidence


def _capture_fingerprint(point: dict[str, Any]) -> str:
    payload = point.get("payload") if isinstance(point, dict) else None
    stable_payload = {
        key: value for key, value in (payload or {}).items()
        if key not in {"access_count", "last_accessed"}
    }
    return lineage_digest(["lineage-reread-v1", str(point.get("id", "")), stable_payload])


def _missing_bindings(payload: dict[str, Any]) -> list[str]:
    missing = [key for key in ("file_sha256", "source_uri", "chunk_hash", "content_hash") if not payload.get(key)]
    if not isinstance(payload.get("locator"), dict) or not payload["locator"]:
        missing.append("locator")
    return missing


def _w1_chunk_binding_problems(payload: dict[str, Any]) -> list[str]:
    problems = _missing_bindings(payload)
    if problems:
        return problems
    problems.extend(w0_sha256_hex_problems(payload.get("file_sha256"), "file_sha256"))
    problems.extend(w0_sha256_hex_problems(payload.get("chunk_hash"), "chunk_hash"))
    problems.extend(w0_content_hash_problems(payload.get("content_hash"), "content_hash"))
    problems.extend(w0_uri_problems(payload.get("source_uri"), "source_uri"))
    problems.extend(w0_locator_problems(payload.get("locator")))
    if payload.get("content_hash") != f"sha256:{payload.get('chunk_hash')}":
        problems.append("content_hash must bind the chunk_hash")
    return problems


def plan_file_lineage(
    *, collection_name: str, profile_id: str, user_id_hash: str,
    chat_id_hash: str, manifest: dict[str, Any], chunks: list[Any],
    existing_points: list[dict[str, Any]],
    existing_source: dict[str, Any] | None = None,
    existing_version: dict[str, Any] | None = None,
    ownership_errors: list[str] | None = None,
) -> dict[str, Any]:
    """Purely classify one snapshot and build deterministic W1 records."""
    file_path, file_sha256 = str(manifest["file_path"]), str(manifest["file_sha256"])
    source_uri = str(manifest["source_uri"])
    scope_key = make_scope_key(collection_name=collection_name, profile_id=profile_id,
                               user_id_hash=user_id_hash, chat_id_hash=chat_id_hash)
    source_key = make_source_key(scope_key=scope_key, resolved_file_path=file_path)
    source_entity = source_logical_id(source_key=source_key, profile_id=profile_id)
    version_digest = make_version_identity_digest(source_key=source_key, file_sha256=file_sha256)
    version_entity = version_logical_id(version_identity_digest=version_digest, profile_id=profile_id)
    source_id, version_id = storage_point_id(source_entity), storage_point_id(version_entity)
    plan: dict[str, Any] = {
        "file_path": file_path, "lineage_baseline_basis": "new_file",
        "lineage_missing_fields": [], "lineage_blocked_reason": "",
        "lineage_history_complete": False, "lineage_entity_ids": [source_entity, version_entity],
        "lineage_edge_ids": [], "lineage_point_ids": [source_id, version_id],
        "lineage_existing_ids": [], "lineage_repair_ids": [],
        "structural_records": [], "chunk_patches": [],
        "old_ids": [str(p["id"]) for p in existing_points if p.get("id") is not None],
        "old_fingerprints": {str(p["id"]): _capture_fingerprint(p) for p in existing_points if p.get("id") is not None},
        "source_point_id": source_id, "version_point_id": version_id,
        "source_expected": existing_source is not None,
        "source_fingerprint": _capture_fingerprint(existing_source) if existing_source else "",
        "capturable": False, "is_new": not existing_points and existing_source is None,
    }

    if ownership_errors:
        plan.update(lineage_baseline_basis="incomplete_indexed_provenance",
                    lineage_missing_fields=sorted(set(ownership_errors)),
                    lineage_blocked_reason="legacy_baseline_requires_reconcile")
        return plan

    for point in existing_points:
        payload = point.get("payload") if isinstance(point, dict) else None
        if not isinstance(payload, dict) or payload.get("profile_id") != profile_id or any(
            payload.get(key, "") != expected
            for key, expected in (("user_id_hash", user_id_hash), ("chat_id_hash", chat_id_hash))
        ):
            plan.update(lineage_baseline_basis="incomplete_indexed_provenance",
                        lineage_missing_fields=["ownership_scope"],
                        lineage_blocked_reason="legacy_baseline_requires_reconcile")
            return plan

    hashes: set[str] = set()
    present = 0
    for point in existing_points:
        value = (point.get("payload") or {}).get("file_sha256")
        if value in (None, ""):
            continue
        present += 1
        if not is_sha256_hex(value):
            plan.update(lineage_baseline_basis="incomplete_indexed_provenance",
                        lineage_missing_fields=["malformed_file_sha256"],
                        lineage_blocked_reason="legacy_baseline_requires_reconcile")
            return plan
        hashes.add(value)
    if existing_points and not present:
        missing = sorted({field for point in existing_points for field in _missing_bindings(point.get("payload") or {})})
        plan.update(lineage_baseline_basis="missing_indexed_file_sha256",
                    lineage_missing_fields=missing or ["file_sha256"],
                    lineage_blocked_reason="legacy_baseline_requires_reconcile")
        return plan
    if existing_points and present != len(existing_points):
        plan.update(lineage_baseline_basis="incomplete_indexed_provenance",
                    lineage_missing_fields=["file_sha256"],
                    lineage_blocked_reason="legacy_baseline_requires_reconcile")
        return plan
    if len(hashes) > 1:
        plan.update(lineage_baseline_basis="incomplete_indexed_provenance",
                    lineage_missing_fields=["ambiguous_file_sha256"],
                    lineage_blocked_reason="legacy_baseline_requires_reconcile")
        return plan
    if existing_source:
        source_payload = existing_source.get("payload") or {}
        if source_payload.get("head_event_id") or source_payload.get("pending_event_id"):
            plan.update(lineage_baseline_basis="incomplete_indexed_provenance",
                        lineage_missing_fields=["event_managed_source"],
                        lineage_blocked_reason="legacy_baseline_requires_reconcile")
            return plan
        current_version = source_payload.get("current_version_id")
        if current_version != version_id:
            plan.update(lineage_baseline_basis="indexed_file_sha256",
                        lineage_missing_fields=[] if current_version else ["current_version_id"],
                        lineage_blocked_reason="retirement_requires_reconcile" if current_version else "legacy_baseline_requires_reconcile")
            return plan
        version_payload = (existing_version or {}).get("payload") or {}
        version_durable = (
            str((existing_version or {}).get("id") or "") == version_id
            and version_payload.get("lineage_role") == "file_version"
            and version_payload.get("lineage_identity_digest") == entity_identity_digest(
                "source", version_digest, profile_id=profile_id)
            and version_payload.get("entity_id") == version_entity
            and version_payload.get("file_sha256") == file_sha256
            and version_payload.get("lineage_source_key") == source_key
            and version_payload.get("profile_id") == profile_id
        )
        if not version_durable:
            plan.update(lineage_baseline_basis="incomplete_indexed_provenance",
                        lineage_missing_fields=["file_version_record"],
                        lineage_blocked_reason="incomplete_capture")
            return plan
        if not existing_points:
            plan["is_new"] = True  # interrupted new capture or managed empty file
            plan["lineage_baseline_basis"] = "indexed_file_sha256"

    if existing_points:
        if next(iter(hashes)) != file_sha256:
            plan.update(lineage_baseline_basis="indexed_file_sha256",
                        lineage_blocked_reason="retirement_requires_reconcile")
            return plan
        by_index, missing = {}, set()
        for point in existing_points:
            payload = point.get("payload") or {}
            missing.update(_missing_bindings(payload))
            index = payload.get("chunk_index")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in by_index:
                missing.add("chunk_inventory")
            else:
                by_index[index] = point
        expected_ids = {
            int(chunk.chunk_index): make_versioned_file_chunk_id(
                scope_key=scope_key, resolved_file_path=file_path,
                file_sha256=file_sha256, chunker_version=chunk.chunker_version,
                chunk_index=int(chunk.chunk_index), chunk_hash=chunk.chunk_hash,
            )
            for chunk in chunks
        }
        incomplete_managed = bool(existing_source) and (
            len(by_index) != len(chunks)
            or set(by_index) != {int(chunk.chunk_index) for chunk in chunks}
        )
        if incomplete_managed:
            surviving_ids = {str(point.get("id")) for point in existing_points}
            if (missing or not surviving_ids.issubset(set(expected_ids.values()))
                    or any(
                        (point.get("payload") or {}).get("file_version_id") != version_id
                        or str(point.get("id")) != expected_ids.get(int((point.get("payload") or {}).get("chunk_index", -1)))
                        for point in existing_points
                    )):
                plan.update(lineage_baseline_basis="incomplete_indexed_provenance",
                            lineage_missing_fields=sorted(missing),
                            lineage_blocked_reason="incomplete_capture")
                return plan
            for chunk in chunks:
                chunk.id = expected_ids[int(chunk.chunk_index)]
                if chunk.id in surviving_ids:
                    plan["lineage_existing_ids"].append(chunk.id)
            plan["is_new"] = True
            plan["lineage_baseline_basis"] = "indexed_file_sha256"
        else:
            if len(by_index) != len(chunks) or set(by_index) != {int(chunk.chunk_index) for chunk in chunks}:
                plan.update(lineage_baseline_basis="indexed_file_sha256",
                            lineage_blocked_reason="retirement_requires_reconcile")
                return plan
            for chunk in chunks:
                point, locator = by_index[int(chunk.chunk_index)], chunk.locator()
                payload = point.get("payload") or {}
                if payload.get("chunk_count") != len(chunks):
                    missing.add("chunk_count")
                if payload.get("chunk_hash") and payload.get("content_hash") and (
                    payload.get("chunk_hash") != chunk.chunk_hash
                    or payload.get("content_hash") != f"sha256:{chunk.chunk_hash}"
                ):
                    plan.update(lineage_baseline_basis="indexed_file_sha256",
                                lineage_blocked_reason="retirement_requires_reconcile")
                    return plan
                if payload.get("source_uri") != source_uri:
                    missing.add("source_uri")
                if payload.get("locator") != locator:
                    missing.add("locator")
                chunk.id = str(point["id"])
                plan["lineage_existing_ids"].append(chunk.id)
            if missing:
                plan.update(lineage_baseline_basis="incomplete_indexed_provenance",
                            lineage_missing_fields=sorted(missing),
                            lineage_blocked_reason="legacy_baseline_requires_reconcile")
                return plan
            plan["lineage_baseline_basis"] = "indexed_file_sha256"

    source_created = ((existing_source or {}).get("payload") or {}).get("created_at")
    source_id, source_payload, source_ev = build_file_source(
        collection_name=collection_name, profile_id=profile_id,
        user_id_hash=user_id_hash, chat_id_hash=chat_id_hash,
        scope_key=scope_key, source_key=source_key, file_path=file_path,
        source_uri=source_uri, current_version_id=version_id, created_at=source_created,
    )
    version_id, version_payload, version_ev = build_file_version(
        collection_name=collection_name, profile_id=profile_id,
        user_id_hash=user_id_hash, chat_id_hash=chat_id_hash,
        scope_key=scope_key, source_key=source_key, file_path=file_path,
        source_uri=source_uri, file_sha256=file_sha256,
        file_size=int(manifest["file_size"]), file_mtime=str(manifest["file_mtime_iso"]),
        file_mtime_ns=int(manifest["file_mtime_ns"]),
    )
    source_hash = f"sha256:{file_sha256}"
    part_payload = build_mechanical_edge_payload(
        relation_type="PART_OF", source_entity_id=version_entity,
        target_entity_id=source_entity, profile_id=profile_id,
        source_point_id=version_id, target_point_id=source_id,
        source_entity_type="source", target_entity_type="source",
        lineage_operation="index_capture", lineage_source_key=source_key,
        lineage_scope_key=scope_key, observation="read_bytes",
        source_content_hash=source_hash, file_version_id=version_id,
        file_path=file_path, file_sha256=file_sha256,
    )
    part_payload.update(user_id_hash=user_id_hash, chat_id_hash=chat_id_hash)
    part_id = storage_point_id(str(part_payload["edge_id"]))
    part_ev = _capture_evidence(
        collection_name=collection_name, profile_id=profile_id,
        user_id_hash=user_id_hash, chat_id_hash=chat_id_hash,
        scope_key=scope_key, source_key=source_key, source_uri=source_uri,
        file_path=file_path, file_sha256=file_sha256,
        source_content_hash=source_hash, file_version_id=version_id,
        observation="read_bytes", relation_type="PART_OF",
        source_point_id=version_id, target_point_id=source_id,
        source_entity_type="source", target_entity_type="source",
    )
    records = [
        {"id": version_id, "payload": version_payload, "evidence": version_ev},
        {"id": source_id, "payload": source_payload, "evidence": source_ev},
        {"id": part_id, "payload": part_payload, "evidence": part_ev},
    ]
    plan["lineage_edge_ids"].append(str(part_payload["edge_id"]))
    plan["lineage_point_ids"].append(part_id)
    for chunk in chunks:
        if plan["is_new"]:
            chunk.id = make_versioned_file_chunk_id(
                scope_key=scope_key, resolved_file_path=file_path,
                file_sha256=file_sha256, chunker_version=chunk.chunker_version,
                chunk_index=chunk.chunk_index, chunk_hash=chunk.chunk_hash,
            )
        chunk_entity = memory_point_endpoint_logical_id(
            scope_key=scope_key, point_id=chunk.id, profile_id=profile_id)
        chunk.lineage_schema_version = LINEAGE_SCHEMA_VERSION
        chunk.lineage_entity_id, chunk.file_version_id = chunk_entity, version_id
        chunk.file_version_entity_id, chunk.lineage_pending = version_entity, False
        if plan["is_new"]:
            chunk.manifest_version = 2
        else:
            prior_manifest = (by_index[int(chunk.chunk_index)].get("payload") or {}).get("manifest_version")
            chunk.manifest_version = 2 if prior_manifest == 2 else 1
        derivation = {
            "point_id": version_id, "source_uri": f"memory://point/{version_id}",
            "relation_type": "DERIVED_FROM", "derivation_type": "indexed_chunk",
            "content_hash": source_hash,
        }
        prior_derivations: list[dict[str, Any]] = []
        if not plan["is_new"]:
            prior = by_index[int(chunk.chunk_index)].get("payload") or {}
            if isinstance(prior.get("derived_from"), list):
                prior_derivations = [item for item in prior["derived_from"] if isinstance(item, dict)]
        chunk.derived_from = [item for item in prior_derivations
                              if not (item.get("point_id") == version_id and item.get("relation_type") == "DERIVED_FROM")]
        chunk.derived_from.append(derivation)
        edge_id, edge_payload, edge_ev = build_chunk_derivation(
            collection_name=collection_name, profile_id=profile_id,
            user_id_hash=user_id_hash, chat_id_hash=chat_id_hash,
            scope_key=scope_key, source_key=source_key, file_path=file_path,
            source_uri=source_uri, file_sha256=file_sha256,
            chunk_point_id=chunk.id, chunk_entity_id=chunk_entity,
            chunk_hash=chunk.chunk_hash, locator=chunk.locator(),
            version_point_id=version_id, version_entity_id=version_entity,
        )
        records.append({"id": edge_id, "payload": edge_payload, "evidence": edge_ev})
        plan["lineage_entity_ids"].append(chunk_entity)
        plan["lineage_edge_ids"].append(str(edge_payload["edge_id"]))
        plan["lineage_point_ids"].append(edge_id)
        plan["chunk_patches"].append(chunk)
    plan["structural_records"], plan["capturable"] = records, True
    return plan


def _collection_lock_digest(collection_name: str, user_id: int) -> str:
    return hashlib.sha256(f"{int(user_id)}\n{collection_name}".encode()).hexdigest()


LINEAGE_LOCK_UNAVAILABLE = "lineage collection lock unavailable"


class LineageLockUnavailable(RuntimeError):
    """Fixed-text refusal raised when the lineage collection lock is unavailable.

    Subclasses RuntimeError so every existing ``except (TimeoutError, RuntimeError)``
    lock handler keeps working, while carrying only the single fixed contract text.
    """


# Raw refusal texts the lock helper and its callers emit. A tool response must not
# forward them verbatim: they carry OS detail (errno, filesystem paths) and a second
# wording for one condition. Direct callers keep the raw text, because frozen W1
# tests pin those messages; the server log keeps it as well.
LINEAGE_LOCK_REFUSAL_MARKERS = (
    "lineage collection lock acquisition timed out",
    "lineage lock directory unavailable",
    "lineage lock directory must not be a symlink",
    "lineage lock directory has unsafe ownership, type, or permissions",
    "lineage lock file unavailable",
    "lineage lock file has unsafe ownership, type, or permissions",
    "lineage locking is unsupported on this platform",
    "lineage locking requires O_NOFOLLOW support",
)


LINEAGE_MANAGED_RETIREMENT_REFUSED = "lineage-managed points require the reviewed retirement path"
LINEAGE_REVIEW_STATE_REFUSAL = "lineage review state requires the reviewed transition path"

# Lineage identity fields an ordinary content point carries once it belongs to a
# captured file. Value semantics: an explicit empty value is absence, exactly as
# the W1 capture predicate reads them. Structural records (``lineage_record`` /
# ``lineage_pending``) are a separate write class and are detected by
# ``is_structural_lineage_payload``; a point can be managed without being
# structural, and that is precisely the shape a bare exact-ID delete would
# orphan.
LINEAGE_MANAGED_FIELDS = (
    "file_version_id",
    "lineage_entity_id",
    "lineage_schema_version",
    "lineage_scope_key",
    "lineage_source_key",
    "lineage_role",
)

# ``lineage_review_event_ids`` is history, not identity: the demotion patch is its
# only writer, and it marks ordinary dependents a transition *touched* (carrying
# ``requires_review`` / ``fact_status=review_required``), not points bound to a
# version record. Whether it counts as protection is decided per **effect**, not
# per field:
#
#   - A retirement (forget, destructive consolidation) removes the point. Widening
#     the retirement predicate with the marker would make a demoted ordinary point
#     permanently unretirable in every mode with no reviewed path available to it,
#     so retirement reads ``LINEAGE_MANAGED_FIELDS`` alone.
#   - An overwrite (store at the content-derived id, extraction approval at the
#     candidate id) replaces the payload while keeping the id. The unretirable
#     argument does not apply, and the damage is worse than the retirement it
#     resembles: the payload loses ``requires_review`` / ``fact_status`` and the
#     recorded causes, and ``requires_review`` is the gate the retriever and
#     auto-recall read — a fact W2 flagged as stale-pending-review would be
#     re-served as active.
#
# Overwrite therefore reads ``LINEAGE_OVERWRITE_PROTECTED_FIELDS``, which is the
# identity set plus exactly the review-state set.
LINEAGE_REVIEW_STATE_FIELDS = ("lineage_review_event_ids",)

LINEAGE_OVERWRITE_PROTECTED_FIELDS = LINEAGE_MANAGED_FIELDS + LINEAGE_REVIEW_STATE_FIELDS


def lineage_managed_reasons(payload: Any) -> list[str]:
    """Reasons a payload participates in lineage bookkeeping without being structural.

    Retirement of such a point has to go through the reviewed path: it is bound to
    a file version or entity record whose bookkeeping the delete would leave
    dangling, and no reviewed transition exists for it yet. Callers refuse the
    destructive route rather than degrading to a legacy delete.
    """
    if not isinstance(payload, dict):
        return []
    return [key for key in LINEAGE_MANAGED_FIELDS if payload.get(key) not in (None, "", [], {})]


def lineage_overwrite_protected_reasons(payload: Any) -> list[str]:
    """Reasons a payload may not be replaced in place at its own id.

    Broader than the retirement predicate by exactly the transition-history marker,
    for the reasons recorded above the constants: an overwrite keeps the id and
    destroys the review state the retriever gates on. Value semantics match the
    identity predicate, so an explicit empty value is absence.
    """
    if not isinstance(payload, dict):
        return []
    return [key for key in LINEAGE_OVERWRITE_PROTECTED_FIELDS if payload.get(key) not in (None, "", [], {})]


def lineage_overwrite_refusal(payload: Any) -> str | None:
    """Refusal text for replacing ``payload`` in place, or ``None`` when allowed.

    Identity wins the text when a point carries both: the reviewed retirement path
    is the one that can actually release such a point, while the review-state text
    points at the transition that flagged it.
    """
    reasons = lineage_overwrite_protected_reasons(payload)
    if not reasons:
        return None
    if any(key in LINEAGE_MANAGED_FIELDS for key in reasons):
        return LINEAGE_MANAGED_RETIREMENT_REFUSED
    return LINEAGE_REVIEW_STATE_REFUSAL


def _refusal_offset(value: str) -> int | None:
    """Offset of the first marker that reads as a refusal clause, or ``None``.

    A refusal is a standalone clause: the marker either starts the string or follows
    whitespace, because every operation prefix the callers add ends in ``": "``. A
    marker glued to a non-space character is part of a path, a directory name or an
    identifier — the operator's own data — and rewriting it would corrupt a summary
    field that has nothing to do with locking. Plain substring matching cannot tell
    those two apart, so the clause boundary is what decides.
    """
    best: int | None = None
    for marker in LINEAGE_LOCK_REFUSAL_MARKERS:
        start = 0
        while True:
            found = value.find(marker, start)
            if found < 0:
                break
            if found == 0 or value[found - 1].isspace():
                if best is None or found < best:
                    best = found
                break
            start = found + 1
    return best


def redact_lock_refusals(value: Any) -> Any:
    """Collapse a raw lineage lock refusal to the single fixed contract text.

    Every string that carries a refusal clause is truncated at the clause and suffixed
    with ``LINEAGE_LOCK_UNAVAILABLE``, so the operation prefix survives and the OS
    detail does not. Strings already carrying the fixed text, strings where a marker
    appears as data rather than as a clause, and non-string values pass through
    untouched, which makes the transform idempotent.
    """
    if isinstance(value, str):
        offset = _refusal_offset(value)
        if offset is None:
            return value
        return value[:offset] + LINEAGE_LOCK_UNAVAILABLE
    if isinstance(value, dict):
        return {key: redact_lock_refusals(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        # Tuples are walked by collect_lock_refusals, so they have to be redacted
        # too: an asymmetry between the two would log a refusal it never removes.
        redacted = [redact_lock_refusals(item) for item in value]
        return tuple(redacted) if isinstance(value, tuple) else redacted
    return value


def collect_lock_refusals(value: Any) -> list[str]:
    """Raw refusal clauses a response boundary is about to rewrite, in order.

    The tool response must not carry them; the server log must, or a misconfigured
    ``lineage_lock_dir`` becomes undiagnosable once its errno and path are dropped.
    Logging the clauses themselves (not a prefix of the whole payload) is what keeps
    the detail reachable, independently of how long the rest of the summary is.
    """
    found: list[str] = []

    def walk(item: Any) -> None:
        if isinstance(item, str):
            if _refusal_offset(item) is not None:
                found.append(item)
            return
        if isinstance(item, dict):
            for nested in item.values():
                walk(nested)
            return
        if isinstance(item, (list, tuple)):
            for nested in item:
                walk(nested)

    walk(value)
    return found


@contextmanager
def collection_write_lock(
    *, collection_name: str, timeout: float = 5.0, lock_dir: str = "",
) -> Iterator[Path]:
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("lineage locking is unsupported on this platform") from exc
    uid = os.getuid()
    # Same naming/reading convention as the plugin's other
    # HERMES_QDRANT_MEMORY_<KEY> overrides (qdrant_memory/config.py): the
    # environment value only supplies the default, an explicit lock_dir wins.
    resolved_lock_dir = str(lock_dir or "").strip() or os.environ.get(
        "HERMES_QDRANT_MEMORY_LINEAGE_LOCK_DIR", ""
    ).strip()
    directory = (
        Path(resolved_lock_dir) if resolved_lock_dir
        else Path(f"/tmp/hermes-qdrant-lineage-{uid}")
    )
    try:
        # The symlink check runs inside the guard on purpose. `Path.is_symlink`
        # swallows the ignored errnos *and* `ValueError` (a non-encodable path), but
        # re-raises everything else, so EACCES on an unsearchable parent component and
        # ENAMETOOLONG for an over-long one would leave this helper as a bare OSError
        # and reach a tool response with the errno and the lock-directory path. The
        # `ValueError` half of this guard is for `mkdir`/`stat`, which raise it for a
        # path carrying a NUL byte or an unencodable surrogate: a path that cannot
        # exist is a refusal, not a crash.
        if directory.is_symlink():
            raise RuntimeError("lineage lock directory must not be a symlink")
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = directory.stat(follow_symlinks=False)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"lineage lock directory unavailable: {exc}") from exc
    if (info.st_uid != uid or not stat.S_ISDIR(info.st_mode)
            or stat.S_IMODE(info.st_mode) & 0o077):
        raise RuntimeError("lineage lock directory has unsafe ownership, type, or permissions")
    path = directory / f"{_collection_lock_digest(collection_name, uid)}.lock"
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("lineage locking requires O_NOFOLLOW support")
    flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW
    # Every OS-level failure of the lock file itself has to leave this helper as a
    # RuntimeError carrying a refusal marker. A bare OSError from open/fdopen/fstat/
    # flock is neither TimeoutError nor RuntimeError, so it would bypass the callers'
    # normalization *and* the response redaction at the same time, forwarding the
    # errno and the lock-file path into a tool response.
    try:
        fd = os.open(path, flags, 0o600)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"lineage lock file unavailable: {exc}") from exc
    try:
        handle = os.fdopen(fd, "a+")
    except (OSError, ValueError) as exc:
        try:
            os.close(fd)
        except OSError:  # pragma: no cover - best effort while already failing
            pass
        raise RuntimeError(f"lineage lock file unavailable: {exc}") from exc
    try:
        try:
            info = os.fstat(handle.fileno())
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"lineage lock file unavailable: {exc}") from exc
        if info.st_uid != uid or not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077:
            raise RuntimeError("lineage lock file has unsafe ownership, type, or permissions")
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("lineage collection lock acquisition timed out")
                time.sleep(0.05)
            except (OSError, ValueError) as exc:
                # Any other errno (ENOLCK on NFS, EINVAL, EBADF) is a refusal too.
                raise RuntimeError(f"lineage lock file unavailable: {exc}") from exc
        yield path
    finally:
        # Cleanup is best-effort on purpose: the kernel releases the flock when the
        # file description closes, so a failed unlock or close must not mask the
        # body's exception nor turn a completed write into a lock refusal.
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:  # pragma: no cover - close below releases the lock
            pass
        try:
            handle.close()
        except OSError:  # pragma: no cover - nothing left to release
            pass


def read_back_exact_records(
    qdrant: Any, collection_name: str, point_ids: list[str], *,
    with_vector: bool = False,
) -> dict[str, dict[str, Any]]:
    ids = list(dict.fromkeys(str(value) for value in point_ids if value))
    if not ids:
        return {}
    records = qdrant.retrieve(collection_name, ids, with_payload=True, with_vector=with_vector)
    return {str(record.get("id")): record for record in records or []
            if isinstance(record, dict) and str(record.get("id")) in ids}


def _record_matches(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    payload = actual.get("payload") if isinstance(actual, dict) else None
    keys = ("memory_kind", "entity_id", "edge_id", "lineage_identity_digest",
            "lineage_scope_key", "lineage_source_key", "lineage_role",
            "relation_type", "source_point_id", "target_point_id",
            "file_version_id", "file_sha256", "current_version_id", "source_deleted",
            "lineage_history_complete", "file_mtime_ns")
    return isinstance(payload, dict) and all(payload.get(key) == expected.get(key)
                                             for key in keys if key in expected)


def apply_capture_plan(
    *, qdrant: Any, collection_name: str,
    plan: dict[str, Any], chunk_points: list[dict[str, Any]],
    read_owned_chunks: Any, lock_timeout: float = 5.0, lock_dir: str = "",
) -> dict[str, Any]:
    """Gate, persist metadata before chunks, and verify exact W1 records."""
    if not plan.get("capturable"):
        return {"acknowledged_ids": [], "read_back_ids": [], "repair_ids": []}
    from qdrant_memory.write_gate import evaluate_mechanical_lineage_write
    with collection_write_lock(collection_name=collection_name,
                               timeout=lock_timeout, lock_dir=lock_dir):
        old_ids = list(plan.get("old_ids") or [])
        reread = read_back_exact_records(qdrant, collection_name, old_ids)
        if set(reread) != set(old_ids) or any(
            _capture_fingerprint(reread.get(point_id, {})) != fingerprint
            for point_id, fingerprint in (plan.get("old_fingerprints") or {}).items()
        ):
            raise RuntimeError("lineage baseline changed before apply")
        new_chunk_ids = [str(point["id"]) for point in chunk_points if str(point["id"]) not in old_ids]
        chunk_collisions = read_back_exact_records(qdrant, collection_name, new_chunk_ids)
        if chunk_collisions:
            raise RuntimeError(f"lineage chunk identity collision: {sorted(chunk_collisions)}")
        source_id = str(plan["source_point_id"])
        source_now = read_back_exact_records(qdrant, collection_name, [source_id])
        if plan.get("source_expected"):
            if source_id not in source_now:
                raise RuntimeError("lineage source disappeared before apply")
            if _capture_fingerprint(source_now[source_id]) != plan["source_fingerprint"]:
                raise RuntimeError("lineage source changed before apply")
        elif source_id in source_now:
            head = (source_now[source_id].get("payload") or {}).get("current_version_id")
            if head and head != plan.get("version_point_id"):
                raise RuntimeError("lineage source appeared with conflicting head before apply")
            raise RuntimeError("lineage source appeared before apply")
        owned_now = read_owned_chunks()
        owned_ids = {str(point.get("id")) for point in owned_now if point.get("id") is not None}
        if owned_ids != set(old_ids):
            raise RuntimeError("lineage chunk inventory changed before apply")
        records = list(plan.get("structural_records") or [])
        record_ids = [str(record["id"]) for record in records]
        existing = read_back_exact_records(qdrant, collection_name, record_ids)
        writes, updates, repairs = [], [], []
        for record in records:
            point_id, payload, prior = str(record["id"]), dict(record["payload"]), existing.get(str(record["id"]))
            unchanged = False
            if prior:
                prior_payload = prior.get("payload") or {}
                if prior_payload.get("lineage_identity_digest") != payload.get("lineage_identity_digest"):
                    raise RuntimeError(f"lineage identity collision at {point_id}")
                if prior_payload.get("head_event_id") or prior_payload.get("pending_event_id"):
                    raise RuntimeError("W1 refuses event-managed sources")
                if ("current_version_id" in payload
                        and prior_payload.get("current_version_id")
                        and prior_payload.get("current_version_id") != payload["current_version_id"]):
                    raise RuntimeError("lineage head conflict before apply")
                payload = {**prior_payload, **payload}
                for key in ("created_at", "updated_at", "canonical", "requires_review",
                            "fact_status", "stale", "truth_confidence", "usefulness_weight"):
                    if key in prior_payload:
                        payload[key] = prior_payload[key]
                unchanged = _record_matches(prior, payload)
                if not unchanged:
                    repairs.append(point_id)
            elif not plan.get("is_new"):
                repairs.append(point_id)
            decision = evaluate_mechanical_lineage_write(
                payload, operation="index_capture", evidence=record["evidence"],
                collection_name=collection_name)
            if decision.decision != "store":
                raise RuntimeError(f"mechanical lineage gate rejected {point_id}: {decision.reasons}")
            if unchanged:
                continue
            if prior:
                updates.append((point_id, payload))
            else:
                writes.append({"id": point_id, "vector": {}, "payload": payload})
        if writes:
            qdrant.upsert(collection_name, writes)
        for point_id, payload in updates:
            qdrant.update_payload(collection_name, point_id, payload)
        for chunk in plan.get("chunk_patches") or []:
            matches = [point for point in chunk_points if str(point.get("id")) == str(chunk.id)]
            if not matches:
                raise RuntimeError(f"missing prepared chunk point {chunk.id}")
            if chunk.id in old_ids:
                payload = matches[0]["payload"]
                keys = ("lineage_schema_version", "lineage_entity_id", "file_version_id",
                        "file_version_entity_id", "chunker_version", "file_mtime_ns",
                        "lineage_pending", "derived_from", "manifest_version")
                patch = {key: payload[key] for key in keys if key in payload}
                old_payload = reread[chunk.id].get("payload") or {}
                if "memory_kind" not in old_payload:
                    patch["memory_kind"] = "source_chunk"
                if any(old_payload.get(key) != value for key, value in patch.items()):
                    qdrant.update_payload(collection_name, chunk.id, patch)
                    repairs.append(str(chunk.id))
            else:
                qdrant.upsert(collection_name, matches)
        expected_ids = record_ids + [str(point["id"]) for point in chunk_points]
        read_back = read_back_exact_records(qdrant, collection_name, expected_ids)
        if set(read_back) != set(expected_ids):
            raise RuntimeError(f"lineage read-back missing exact IDs: {sorted(set(expected_ids) - set(read_back))}")
        for record in records:
            if not _record_matches(read_back[str(record["id"])], record["payload"]):
                raise RuntimeError(f"lineage read-back invariant mismatch: {record['id']}")
        for point in chunk_points:
            actual, expected = read_back[str(point["id"])].get("payload", {}), point["payload"]
            for key in ("file_version_id", "lineage_entity_id", "file_sha256", "chunk_hash", "chunk_index"):
                if actual.get(key) != expected.get(key):
                    raise RuntimeError(f"chunk read-back invariant mismatch: {point['id']}:{key}")
        return {"acknowledged_ids": [str(item["id"]) for item in writes] + [point_id for point_id, _ in updates] + [str(point["id"]) for point in chunk_points],
                "read_back_ids": sorted(read_back),
                "repair_ids": sorted(set(repairs))}


DEPENDENCY_RELATIONS = frozenset({"DERIVED_FROM", "EXTRACTED_FROM", "SUMMARIZES"})
REVIEW_CAUSE_CAP = 32
DEFAULT_INVALIDATION_DEPTH = 8
HARD_INVALIDATION_DEPTH = 16
DEFAULT_INVALIDATION_POINTS = 4096
DEFAULT_INVALIDATION_EDGES = 8192
EVENT_STATES = (
    "prepared",
    "dependents_marked",
    "chunks_staged",
    "roots_retired",
    "committed",
)
EVENT_KINDS = frozenset({
    "observed", "modified", "rechunked", "deleted", "restored",
    "source_stale", "source_verified",
})


def _point_snapshot(point: dict[str, Any]) -> str:
    return lineage_digest([
        "lineage-transition-point-v1",
        str(point.get("id") or ""),
        point.get("payload") if isinstance(point.get("payload"), dict) else {},
        point.get("vector") if "vector" in point else None,
    ])


def _scope_matches(
    payload: dict[str, Any], *, profile_id: str,
    user_id_hash: str, chat_id_hash: str,
) -> bool:
    return (
        payload.get("profile_id") == profile_id
        and payload.get("user_id_hash", "") == user_id_hash
        and payload.get("chat_id_hash", "") == chat_id_hash
    )


def _scroll_bounded(
    qdrant: Any,
    collection_name: str,
    filter_value: dict[str, Any],
    *,
    limit: int,
    with_vector: bool = True,
) -> tuple[list[dict[str, Any]], bool]:
    """Read at most ``limit`` points plus one sentinel and report completeness."""
    wanted = max(0, int(limit))
    if wanted == 0:
        return [], False
    points: list[dict[str, Any]] = []
    offset: Any = None
    if callable(getattr(qdrant, "scroll_page", None)):
        while len(points) <= wanted:
            batch, offset = qdrant.scroll_page(
                collection_name,
                filter_value,
                limit=min(256, wanted + 1 - len(points)),
                offset=offset,
                with_payload=True,
                with_vector=with_vector,
            )
            points.extend(batch)
            if offset is None or not batch:
                break
        return points[:wanted], len(points) <= wanted and offset is None
    try:
        points = qdrant.scroll_by_filter(
            collection_name,
            filter_value,
            limit=min(256, wanted + 1),
            with_payload=True,
            with_vector=with_vector,
            max_total=wanted + 1,
        )
    except TypeError:
        points = qdrant.scroll_by_filter(
            collection_name,
            filter_value,
            limit=wanted + 1,
            with_payload=True,
            with_vector=with_vector,
        )
    return list(points[:wanted]), len(points) <= wanted


def _exact_filter(**values: Any) -> dict[str, Any]:
    return {
        "must": [
            {"key": key, "match": {"value": value}}
            for key, value in values.items()
        ]
    }


def _inline_target(entry: dict[str, Any]) -> tuple[str, str, str]:
    relation_raw = entry.get("relation_type")
    relation = "DERIVED_FROM" if relation_raw in (None, "") else str(relation_raw)
    point_id = str(entry.get("point_id") or entry.get("child_node_id") or "")
    source_uri = str(entry.get("source_uri") or "")
    if not point_id and source_uri.startswith("memory://point/"):
        point_id = source_uri.removeprefix("memory://point/")
    collection = str(entry.get("collection") or entry.get("collection_name") or "")
    return point_id, relation, collection


def find_direct_dependents(
    *,
    qdrant: Any,
    collection_name: str,
    target_point_id: str | list[str],
    profile_id: str,
    user_id_hash: str = "",
    chat_id_hash: str = "",
    max_results: int = DEFAULT_INVALIDATION_POINTS,
) -> dict[str, Any]:
    """Find exact incoming dependency citations without mutating Qdrant."""
    targets = sorted({
        str(value) for value in (
            target_point_id if isinstance(target_point_id, list) else [target_point_id]
        ) if str(value)
    })
    if not targets:
        return {"complete": False, "points": [], "edge_keys": [], "errors": ["target_point_id is required"]}
    target_match = {"value": targets[0]} if len(targets) == 1 else {"any": targets}
    remaining = max(0, int(max_results))
    found: dict[str, dict[str, Any]] = {}
    edge_keys: set[tuple[str, str, str]] = set()
    errors: list[str] = []

    graph_filter = {
        "must": [
            {"key": "target_point_id", "match": target_match},
            {"key": "relation_type", "match": {"any": sorted(DEPENDENCY_RELATIONS)}},
            {"key": "profile_id", "match": {"value": profile_id}},
        ],
        "must_not": [
            {"key": "lineage_retired", "match": {"value": True}},
        ],
    }
    graph_edges, complete = _scroll_bounded(
        qdrant, collection_name, graph_filter, limit=remaining + 1, with_vector=False,
    )
    if not complete or len(graph_edges) > remaining:
        errors.append("direct dependency graph lookup exceeded bound")
    graph_edges = graph_edges[:remaining]
    source_ids: set[str] = set()
    for edge in graph_edges:
        payload = edge.get("payload") if isinstance(edge, dict) else None
        if not isinstance(payload, dict):
            errors.append("dependency edge payload is missing")
            continue
        if not _scope_matches(payload, profile_id=profile_id, user_id_hash=user_id_hash, chat_id_hash=chat_id_hash):
            # Category only: an out-of-scope edge id must never be echoed into a
            # persisted impact report or a reconcile refusal string.
            errors.append("dependency edge scope mismatch")
            continue
        relation = payload.get("relation_type")
        actual_target = str(payload.get("target_point_id") or "")
        if actual_target not in targets or relation not in DEPENDENCY_RELATIONS:
            errors.append(f"dependency edge post-parse mismatch: {edge.get('id')}")
            continue
        source_id = str(payload.get("source_point_id") or "")
        if not source_id:
            errors.append(f"dependency edge source point is missing: {edge.get('id')}")
            continue
        if (
            relation == "DERIVED_FROM"
            and payload.get("lineage_operation") in {"index_capture", "index_reconcile"}
            and payload.get("target_entity_type") == "source"
            and is_uuid_string(payload.get("lineage_retired_by_event_id"))
        ):
            continue
        source_ids.add(source_id)
        edge_keys.add((source_id, actual_target, str(relation)))

    if source_ids:
        source_points = qdrant.retrieve(
            collection_name, sorted(source_ids), with_payload=True, with_vector=True,
        )
        by_id = {str(point.get("id")): point for point in source_points}
        missing = source_ids - set(by_id)
        if missing:
            errors.append(f"dependency source points are missing: {sorted(missing)}")
        for point_id, point in by_id.items():
            payload = point.get("payload") if isinstance(point, dict) else None
            if not isinstance(payload, dict) or not _scope_matches(
                payload, profile_id=profile_id,
                user_id_hash=user_id_hash, chat_id_hash=chat_id_hash,
            ):
                errors.append("dependency point scope mismatch")
                continue
            found[point_id] = point

    nested_filters = [
        {"nested": {"key": "derived_from", "filter": {"must": [
            {"key": key, "match": ({"value": values[0]} if len(values) == 1 else {"any": values})}
        ]}}}
        for key, values in (
            ("point_id", targets),
            ("child_node_id", targets),
            ("source_uri", [f"memory://point/{target}" for target in targets]),
        )
    ]
    for nested in nested_filters:
        inline_filter = {"must": [
            {"key": "profile_id", "match": {"value": profile_id}}, nested,
        ]}
        offset: Any = None
        while True:
            try:
                if hasattr(qdrant, "scroll_page"):
                    points, offset = qdrant.scroll_page(
                        collection_name, inline_filter, limit=256, offset=offset,
                        with_payload=True, with_vector=True,
                    )
                else:
                    points = qdrant.scroll_by_filter(
                        collection_name, inline_filter, limit=256,
                        with_payload=True, with_vector=True,
                    )
                    offset = None
            except Exception as exc:
                errors.append(f"direct dependency inline lookup failed: {exc}")
                break
            for point in points:
                point_id = str(point.get("id") or "")
                payload = point.get("payload") if isinstance(point, dict) else None
                if not point_id or not isinstance(payload, dict):
                    errors.append("inline dependency payload is missing")
                    continue
                if not _scope_matches(payload, profile_id=profile_id, user_id_hash=user_id_hash, chat_id_hash=chat_id_hash):
                    errors.append("inline dependency scope mismatch")
                    continue
                matched = False
                derivations = payload.get("derived_from")
                if not isinstance(derivations, list):
                    errors.append(f"inline dependency list is malformed: {point_id}")
                    continue
                for entry in derivations:
                    if not isinstance(entry, dict):
                        errors.append(f"inline dependency entry is malformed: {point_id}")
                        continue
                    cited_id, relation, cited_collection = _inline_target(entry)
                    if cited_id not in targets:
                        continue
                    if cited_collection and cited_collection != collection_name:
                        errors.append(f"cross-collection dependency is unsupported: {point_id}")
                        continue
                    if relation not in DEPENDENCY_RELATIONS:
                        if relation not in NON_DEPENDENCY_RELATIONS and entry.get("relation_type") not in (None, ""):
                            errors.append(f"unsupported dependency relation: {point_id}:{relation}")
                        continue
                    matched = True
                    edge_keys.add((point_id, cited_id, relation))
                if matched:
                    found[point_id] = point
                    if len(found) > int(max_results):
                        errors.append("direct dependency inline lookup exceeded bound")
                        break
            if len(found) > int(max_results) or offset is None or not points:
                break
        if len(found) > int(max_results):
            break

    if len(found) > int(max_results):
        found = {key: found[key] for key in sorted(found)[:int(max_results)]}
    return {
        "complete": not errors,
        "points": [found[key] for key in sorted(found)],
        "edge_keys": [list(item) for item in sorted(edge_keys)],
        "errors": errors,
    }


def plan_invalidation(
    *,
    qdrant: Any,
    collection_name: str,
    root_point_ids: list[str],
    event_id: str | None,
    profile_id: str,
    user_id_hash: str = "",
    chat_id_hash: str = "",
    max_depth: int = DEFAULT_INVALIDATION_DEPTH,
    max_points: int = DEFAULT_INVALIDATION_POINTS,
    max_edges: int = DEFAULT_INVALIDATION_EDGES,
) -> dict[str, Any]:
    """Plan a bounded exact-ID invalidation closure. This function never writes."""
    errors: list[str] = []
    ordinary_refusal = not event_id
    if event_id and not is_uuid_string(event_id):
        errors.append("event_id must be a UUID")
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or not 0 <= max_depth <= HARD_INVALIDATION_DEPTH:
        errors.append(f"max_depth must be between 0 and {HARD_INVALIDATION_DEPTH}")
    if isinstance(max_points, bool) or not isinstance(max_points, int) or not 1 <= max_points <= DEFAULT_INVALIDATION_POINTS:
        errors.append(f"max_points must be between 1 and {DEFAULT_INVALIDATION_POINTS}")
    if isinstance(max_edges, bool) or not isinstance(max_edges, int) or not 1 <= max_edges <= DEFAULT_INVALIDATION_EDGES:
        errors.append(f"max_edges must be between 1 and {DEFAULT_INVALIDATION_EDGES}")
    roots = sorted({str(point_id) for point_id in root_point_ids if str(point_id)})
    if errors:
        return {"complete": False, "root_ids": roots, "dependent_ids": [], "changes": [], "errors": errors}

    visited = {(collection_name, profile_id, point_id) for point_id in roots}
    frontier = roots
    depth = 0
    dependent_roots: dict[str, set[str]] = {}
    points: dict[str, dict[str, Any]] = {}
    edges: set[tuple[str, str, str]] = set()
    while frontier and not errors:
        direct = find_direct_dependents(
            qdrant=qdrant, collection_name=collection_name,
            target_point_id=frontier, profile_id=profile_id,
            user_id_hash=user_id_hash, chat_id_hash=chat_id_hash,
            max_results=max_points - len(points),
        )
        if not direct["complete"]:
            errors.extend(direct["errors"])
            break
        direct_edges = {tuple(item) for item in direct["edge_keys"]}
        if len(edges | direct_edges) > max_edges:
            errors.append("dependency edge cap exceeded")
            break
        edges.update(direct_edges)
        unseen = [point for point in direct["points"] if (
            collection_name, profile_id, str(point.get("id") or "")
        ) not in visited]
        if depth >= max_depth and unseen:
            errors.append("dependency depth bound reached with unseen dependents")
            break
        next_frontier: list[str] = []
        for point in sorted(unseen, key=lambda item: str(item.get("id") or "")):
            point_id = str(point.get("id") or "")
            if len(points) >= max_points:
                errors.append("dependency point cap exceeded")
                break
            parents = {target for source, target, _relation in direct_edges if source == point_id}
            inherited: set[str] = set()
            for parent_id in parents:
                inherited.update(dependent_roots.get(parent_id, {parent_id} if parent_id in roots else set()))
            visited.add((collection_name, profile_id, point_id))
            points[point_id] = point
            dependent_roots.setdefault(point_id, set()).update(inherited)
            next_frontier.append(point_id)
        frontier = sorted(next_frontier)
        depth += 1

    if ordinary_refusal:
        errors.append("ordinary_root_transition_cause_unratified")
    changes: list[dict[str, Any]] = []
    if not errors:
        for point_id in sorted(points):
            point = points[point_id]
            payload = point.get("payload") or {}
            causes = payload.get("lineage_review_event_ids", [])
            if causes in (None, ""):
                causes = []
            if not isinstance(causes, list) or any(not is_uuid_string(value) for value in causes):
                errors.append(f"malformed lineage review causes: {point_id}")
                break
            merged = list(dict.fromkeys(str(value) for value in causes))
            patch: dict[str, Any] = {"requires_review": True}
            if event_id not in merged:
                if len(merged) < REVIEW_CAUSE_CAP:
                    merged.append(str(event_id))
                else:
                    patch["lineage_review_causes_truncated"] = True
            patch["lineage_review_event_ids"] = merged
            status = payload.get("fact_status")
            if status in (None, "", "active"):
                patch["fact_status"] = "review_required"
            changes.append({
                "point_id": point_id,
                "snapshot_digest": _point_snapshot(point),
                "before_payload": payload,
                "before_vector": point.get("vector") if "vector" in point else None,
                "patch": patch,
                "root_ids": sorted(dependent_roots.get(point_id, set())),
            })
    return {
        "complete": not errors,
        "root_ids": roots,
        "dependent_ids": sorted(points),
        "edge_count": len(edges),
        "changes": changes if not errors else [],
        "bounds": {"max_depth": max_depth, "max_points": max_points, "max_edges": max_edges},
        "errors": errors,
    }


def lineage_impact_proposal_digest(proposal: dict[str, Any]) -> str:
    """Bind persisted lineage impact to the proposal that contains it."""
    excluded = {
        "lineage_impact_proposal_sha256",
        "guarded_auto_proposal_sha256",
        "guarded_auto_snapshot",
    }
    return lineage_digest({key: value for key, value in proposal.items() if key not in excluded})


def build_lineage_impact_snapshot(
    *,
    qdrant: Any,
    collection_name: str,
    root_point_ids: list[str],
    profile_id: str,
    user_id_hash: str = "",
    chat_id_hash: str = "",
    event_id: str | None = None,
) -> dict[str, Any]:
    """Build a read-only exact snapshot for a proposed root action."""
    try:
        plan = plan_invalidation(
            qdrant=qdrant, collection_name=collection_name,
            root_point_ids=root_point_ids, event_id=event_id, profile_id=profile_id,
            user_id_hash=user_id_hash, chat_id_hash=chat_id_hash,
        )
        ids = sorted(set(plan.get("root_ids") or []) | set(plan.get("dependent_ids") or []))
        points = {
            str(point.get("id")): point
            for point in qdrant.retrieve(
                collection_name, ids, with_payload=True, with_vector=True,
            )
        }
    except Exception as exc:
        return {
            "schema_version": 1,
            "root_ids": sorted(set(str(value) for value in root_point_ids)),
            "dependent_ids": [],
            "snapshot_digests": {},
            "bounds": {},
            "complete": False,
            "errors": [f"lineage impact lookup failed: {exc}"],
            "proposed_review_changes": [],
        }
    missing = sorted(set(ids) - set(points))
    errors = list(plan.get("errors") or [])
    if missing:
        errors.append(f"lineage impact points are missing: {missing}")
    return {
        "schema_version": 1,
        "root_ids": sorted(set(plan.get("root_ids") or [])),
        "dependent_ids": sorted(set(plan.get("dependent_ids") or [])),
        "snapshot_digests": {
            point_id: _point_snapshot(points[point_id])
            for point_id in sorted(points)
        },
        "bounds": dict(plan.get("bounds") or {}),
        "complete": plan.get("complete") is True and not missing,
        "errors": errors,
        "proposed_review_changes": [
            {"point_id": change["point_id"], "patch": dict(change["patch"])}
            for change in plan.get("changes") or []
        ],
    }


def validate_lineage_impact_snapshot(
    *,
    qdrant: Any,
    collection_name: str,
    impact: Any,
    expected_root_ids: list[str],
    proposal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a persisted impact snapshot against current exact points."""
    problems: list[str] = []
    if not isinstance(impact, dict) or impact.get("schema_version") != 1:
        return {"valid": False, "problems": ["lineage impact snapshot is missing"]}
    if impact.get("complete") is not True or impact.get("errors"):
        problems.append("lineage impact is incomplete")
    if proposal is not None:
        expected_digest = str(proposal.get("lineage_impact_proposal_sha256") or "")
        if not expected_digest or expected_digest != lineage_impact_proposal_digest(proposal):
            problems.append("lineage impact proposal digest changed")
    roots = sorted(set(str(value) for value in impact.get("root_ids") or []))
    if roots != sorted(set(expected_root_ids)):
        problems.append("lineage impact roots changed")
    digests = impact.get("snapshot_digests")
    if not isinstance(digests, dict):
        problems.append("lineage impact digests are missing")
        digests = {}
    ids = sorted(digests)
    current = {
        str(point.get("id")): point
        for point in qdrant.retrieve(
            collection_name, ids, with_payload=True, with_vector=True,
        )
    }
    if set(current) != set(ids):
        problems.append("lineage impact point set changed")
    elif any(_point_snapshot(current[point_id]) != digests[point_id] for point_id in ids):
        problems.append("lineage impact point state changed")
    return {"valid": not problems, "problems": problems}


def validate_transition_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Validate a transition plan before any persistence boundary."""
    problems: list[str] = []
    if not isinstance(plan, dict):
        return {"valid": False, "problems": ["transition plan must be an object"]}
    if plan.get("complete") is not True:
        problems.append("transition plan is incomplete")
    event_id = plan.get("event_id")
    if not is_uuid_string(event_id):
        problems.append("transition event_id must be a UUID")
    invalidation = plan.get("invalidation")
    if not isinstance(invalidation, dict) or invalidation.get("complete") is not True:
        problems.append("invalidation plan is incomplete")
    for key in ("retired_ids", "new_chunk_ids"):
        values = plan.get(key)
        if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
            problems.append(f"{key} must be a list of exact point IDs")
    overlap = set(plan.get("retired_ids") or []) & set(plan.get("new_chunk_ids") or [])
    if overlap:
        problems.append(f"retired and new inventories overlap: {sorted(overlap)}")
    return {"valid": not problems, "problems": problems}


def apply_invalidation(
    *, qdrant: Any, collection_name: str, plan: dict[str, Any],
) -> dict[str, Any]:
    """Apply and verify exact payload-only review demotions."""
    if plan.get("complete") is not True:
        raise RuntimeError(f"invalidation plan is incomplete: {plan.get('errors') or []}")
    changes = list(plan.get("changes") or [])
    ids = [str(change["point_id"]) for change in changes]
    current = {
        str(point.get("id")): point
        for point in qdrant.retrieve(collection_name, ids, with_payload=True, with_vector=True)
    }
    if set(current) != set(ids):
        raise RuntimeError(f"dependent lookup drift: {sorted(set(ids) - set(current))}")
    changed: list[str] = []
    for change in changes:
        point_id = str(change["point_id"])
        point = current[point_id]
        before_payload = change.get("before_payload")
        before_vector = change.get("before_vector")
        current_payload = point.get("payload") or {}
        if before_payload is not None:
            expected_payload = {**before_payload, **change["patch"]}
            if point.get("vector") != before_vector:
                raise RuntimeError(f"dependent vector drift: {point_id}")
            if current_payload == expected_payload:
                continue
        elif all(current_payload.get(key) == value for key, value in change["patch"].items()):
            continue
        if _point_snapshot(point) != change["snapshot_digest"]:
            raise RuntimeError(f"dependent payload drift: {point_id}")
        qdrant.update_payload(collection_name, point_id, dict(change["patch"]))
        changed.append(point_id)
    read_back = {
        str(point.get("id")): point
        for point in qdrant.retrieve(collection_name, ids, with_payload=True, with_vector=True)
    }
    if set(read_back) != set(ids):
        raise RuntimeError("dependent read-back is incomplete")
    for change in changes:
        point_id = str(change["point_id"])
        point = read_back[point_id]
        before_payload = change.get("before_payload")
        if before_payload is not None:
            expected_payload = {**before_payload, **change["patch"]}
            valid = point.get("payload") == expected_payload and point.get("vector") == change.get("before_vector")
        else:
            payload = point.get("payload") or {}
            valid = all(payload.get(key) == value for key, value in change["patch"].items())
        if not valid:
            raise RuntimeError(f"dependent review update verification failed: {point_id}")
    return {"changed_ids": changed, "verified_ids": sorted(ids)}


def build_change_event(
    *,
    source_key: str,
    scope_key: str,
    profile_id: str,
    file_path: str,
    source_uri: str,
    event_kind: str,
    previous_event_id: str | None,
    from_version_id: str | None,
    to_version_id: str | None,
    desired_inventory_digest: str,
    created_point_ids: list[str],
    retired_point_ids: list[str],
    new_chunk_ids: list[str] | None = None,
    patched_point_ids: list[str],
    lineage_baseline_basis: str,
    lineage_missing_fields: list[str],
    review_changes: list[dict[str, Any]] | None = None,
    lineage_observation: str = "read_bytes",
    lineage_history_complete: bool = False,
    from_file_sha256: str | None = None,
    to_file_sha256: str | None = None,
    event_state: str = "prepared",
    observed_at: str | None = None,
    user_id_hash: str = "",
    chat_id_hash: str = "",
) -> tuple[str, dict[str, Any]]:
    """Build the strict W2-owned change-event record without widening W0."""
    if event_kind not in EVENT_KINDS:
        raise ValueError(f"unsupported event_kind: {event_kind}")
    if event_state not in EVENT_STATES:
        raise ValueError(f"unsupported event_state: {event_state}")
    if previous_event_id not in (None, "") and not is_uuid_string(previous_event_id):
        raise ValueError("previous_event_id must be a UUID or null")
    for key, value in (("from_version_id", from_version_id), ("to_version_id", to_version_id)):
        if value not in (None, "") and not is_uuid_string(value):
            raise ValueError(f"{key} must be a UUID or null")
    for key, value in (("from_file_sha256", from_file_sha256), ("to_file_sha256", to_file_sha256)):
        if value is not None and not is_sha256_hex(value):
            raise ValueError(f"{key} must be a lowercase SHA-256 or absent")
    if not is_sha256_hex(desired_inventory_digest):
        raise ValueError("desired_inventory_digest must be a lowercase SHA-256")
    identity_digest = lineage_digest([
        "lineage-event-v1", source_key, previous_event_id or "", event_kind,
        from_version_id or "", to_version_id or "", desired_inventory_digest,
    ])
    logical_id = make_entity_id("event", identity_digest, profile_id=profile_id)
    point_id = storage_point_id(logical_id)
    payload = build_entity_payload(
        entity_type="event",
        label=f"lineage-event-{identity_digest}",
        logical_entity_id=logical_id,
        profile_id=profile_id,
        source_uri=source_uri,
        created_at=observed_at,
        file_path=file_path,
        lineage_record=True,
        lineage_schema_version=LINEAGE_SCHEMA_VERSION,
        lineage_operation="index_reconcile",
        lineage_identity_digest=entity_identity_digest("event", identity_digest, profile_id=profile_id),
        lineage_source_key=source_key,
        lineage_scope_key=scope_key,
        lineage_observation=lineage_observation,
        lineage_role="change_event",
    )
    payload.update({
        "user_id_hash": user_id_hash,
        "chat_id_hash": chat_id_hash,
        "event_kind": event_kind,
        "previous_event_id": previous_event_id,
        "from_version_id": from_version_id,
        "to_version_id": to_version_id,
        "event_state": event_state,
        "desired_inventory_digest": desired_inventory_digest,
        "created_point_ids": sorted(set(created_point_ids)),
        "retired_point_ids": sorted(set(retired_point_ids)),
        "new_chunk_ids": sorted(set(new_chunk_ids or [])),
        "patched_point_ids": sorted(set(patched_point_ids)),
        "lineage_baseline_basis": lineage_baseline_basis,
        "lineage_missing_fields": sorted(set(lineage_missing_fields)),
        "lineage_history_complete": bool(lineage_history_complete),
        "review_changes": list(review_changes or []),
        "observed_at": payload.get("created_at"),
    })
    if from_file_sha256 is not None:
        payload["from_file_sha256"] = from_file_sha256
    if to_file_sha256 is not None:
        payload["to_file_sha256"] = to_file_sha256
    return point_id, payload


def plan_reconciliation(
    *,
    qdrant: Any,
    collection_name: str,
    profile_id: str,
    user_id_hash: str,
    chat_id_hash: str,
    file_path: str,
    manifest: dict[str, Any] | None,
    chunks: list[Any],
    existing_points: list[dict[str, Any]],
    existing_source: dict[str, Any] | None,
    ownership_errors: list[str] | None = None,
    max_depth: int = DEFAULT_INVALIDATION_DEPTH,
    max_points: int = DEFAULT_INVALIDATION_POINTS,
    max_edges: int = DEFAULT_INVALIDATION_EDGES,
) -> dict[str, Any]:
    """Build one deterministic W2 file transition without writing."""
    scope_key = make_scope_key(
        collection_name=collection_name, profile_id=profile_id,
        user_id_hash=user_id_hash, chat_id_hash=chat_id_hash,
    )
    source_key = make_source_key(scope_key=scope_key, resolved_file_path=file_path)
    source_entity = source_logical_id(source_key=source_key, profile_id=profile_id)
    source_id = storage_point_id(source_entity)
    errors = list(ownership_errors or [])
    source_payload = (existing_source or {}).get("payload") or {}
    pending_event_id = source_payload.get("pending_event_id")
    if pending_event_id in (None, "") and source_payload.get("head_event_id") not in (None, ""):
        head_event_id = str(source_payload["head_event_id"])
        head_event = read_back_exact_records(qdrant, collection_name, [head_event_id]).get(head_event_id)
        if ((head_event or {}).get("payload") or {}).get("event_state") != "committed":
            pending_event_id = head_event_id
    pending_event_payload: dict[str, Any] = {}
    pending_new_chunk_ids: set[str] = set()
    if pending_event_id not in (None, ""):
        if not is_uuid_string(pending_event_id):
            errors.append("source pending_event_id is malformed")
        else:
            pending_event = read_back_exact_records(
                qdrant, collection_name, [str(pending_event_id)]
            ).get(str(pending_event_id))
            pending_event_payload = (pending_event or {}).get("payload") or {}
            if (
                not pending_event
                or pending_event_payload.get("lineage_role") != "change_event"
                or pending_event_payload.get("lineage_source_key") != source_key
                or pending_event_payload.get("lineage_scope_key") != scope_key
            ):
                errors.append("pending event binding mismatch")
            else:
                pending_new_chunk_ids = {
                    str(value) for value in pending_event_payload.get("new_chunk_ids", [])
                    if isinstance(value, str) and value
                }
    if existing_source and (
        str(existing_source.get("id") or "") != source_id
        or source_payload.get("lineage_source_key") != source_key
        or source_payload.get("lineage_scope_key") != scope_key
        or not _scope_matches(source_payload, profile_id=profile_id, user_id_hash=user_id_hash, chat_id_hash=chat_id_hash)
    ):
        errors.append("source head binding mismatch")

    old_hashes: set[str] = set()
    missing_fields: set[str] = set()
    for point in existing_points:
        if str(point.get("id") or "") in pending_new_chunk_ids:
            continue
        payload = point.get("payload") if isinstance(point, dict) else None
        if not isinstance(payload, dict) or not _scope_matches(
            payload, profile_id=profile_id,
            user_id_hash=user_id_hash, chat_id_hash=chat_id_hash,
        ):
            errors.append("owned inventory scope mismatch")
            continue
        value = payload.get("file_sha256")
        binding_gaps = _w1_chunk_binding_problems(payload)
        if value in (None, "") or binding_gaps:
            missing_fields.update(binding_gaps or ["file_sha256"])
        elif not is_sha256_hex(value):
            errors.append(f"malformed old file_sha256: {point.get('id')}")
        else:
            old_hashes.add(str(value))
    if len(old_hashes) > 1 or (old_hashes and missing_fields):
        errors.append("ambiguous old hash basis")

    old_version_id = source_payload.get("current_version_id")
    old_version: dict[str, Any] | None = None
    old_hash = next(iter(old_hashes), None)
    if old_version_id not in (None, ""):
        if not is_uuid_string(old_version_id):
            errors.append("source current_version_id is malformed")
        else:
            records = read_back_exact_records(qdrant, collection_name, [str(old_version_id)])
            old_version = records.get(str(old_version_id))
            if old_version:
                payload = old_version.get("payload") or {}
                head_hash = payload.get("file_sha256")
                if (
                    payload.get("lineage_role") != "file_version"
                    or payload.get("lineage_source_key") != source_key
                    or payload.get("lineage_scope_key") != scope_key
                    or not is_sha256_hex(head_hash)
                ):
                    errors.append("source head version binding mismatch")
                elif old_hash and old_hash != head_hash:
                    errors.append("source head disagrees with old chunk hashes")
                else:
                    old_hash = str(head_hash)
            elif not old_hash:
                errors.append("source head version is missing without an old hash basis")
    elif old_hash:
        old_version_digest = make_version_identity_digest(source_key=source_key, file_sha256=old_hash)
        old_version_id = storage_point_id(version_logical_id(
            version_identity_digest=old_version_digest, profile_id=profile_id,
        ))

    if existing_source is None and any(
        isinstance(point, dict)
        and isinstance(point.get("payload"), dict)
        and (
            point["payload"].get("file_version_id") not in (None, "")
            or not _w1_chunk_binding_problems(point["payload"])
        )
        for point in existing_points
    ):
        errors.append("event-free W1 source baseline is incomplete")
    bootstrap_required = bool(
        existing_source
        and old_version_id not in (None, "")
        and source_payload.get("head_event_id") in (None, "")
        and source_payload.get("pending_event_id") in (None, "")
    )
    if bootstrap_required:
        if old_version is None or not old_hash:
            errors.append("event-free W1 source baseline is incomplete")
        chunk_counts = [
            (point.get("payload") or {}).get("chunk_count")
            for point in existing_points if isinstance(point, dict)
        ]
        chunk_indices = [
            (point.get("payload") or {}).get("chunk_index")
            for point in existing_points if isinstance(point, dict)
        ]
        invalid_inventory = (
            bool(existing_points)
            and (
                any(isinstance(value, bool) or not isinstance(value, int) for value in chunk_counts)
                or len(set(chunk_counts)) != 1
                or chunk_counts[0] != len(existing_points)
                or any(isinstance(value, bool) or not isinstance(value, int) for value in chunk_indices)
                or sorted(chunk_indices) != list(range(len(existing_points)))
            )
        ) or (
            not existing_points
            and (
                int(((old_version or {}).get("payload") or {}).get("file_size") or 0) > 0
                or old_hash != hashlib.sha256(b"").hexdigest()
            )
        )
        if invalid_inventory:
            errors.append("event-free W1 chunk inventory is incomplete or ambiguous")
        for point in existing_points:
            payload = point.get("payload") if isinstance(point, dict) else None
            expected_chunk_id = None
            if isinstance(payload, dict) and all(
                payload.get(key) not in (None, "")
                for key in ("chunker_version", "chunk_index", "chunk_hash")
            ):
                try:
                    expected_chunk_id = make_versioned_file_chunk_id(
                        scope_key=scope_key,
                        resolved_file_path=file_path,
                        file_sha256=str(old_hash),
                        chunker_version=str(payload["chunker_version"]),
                        chunk_index=payload["chunk_index"],
                        chunk_hash=str(payload["chunk_hash"]),
                    )
                except (TypeError, ValueError):
                    expected_chunk_id = None
            if not isinstance(payload, dict) or (
                payload.get("file_version_id") != old_version_id
                or payload.get("file_sha256") != old_hash
                or _w1_chunk_binding_problems(payload)
                or expected_chunk_id != str(point.get("id") or "")
            ):
                errors.append(f"event-free W1 chunk binding mismatch: {point.get('id')}")

    if pending_event_payload:
        pending_from_hash = pending_event_payload.get("from_file_sha256")
        pending_from_version = pending_event_payload.get("from_version_id")
        if pending_from_hash is not None and not is_sha256_hex(pending_from_hash):
            errors.append("pending event from_file_sha256 is malformed")
        else:
            old_hash = pending_from_hash
        if pending_from_version not in (None, ""):
            if not is_uuid_string(pending_from_version):
                errors.append("pending event from_version_id is malformed")
            else:
                old_version_id = str(pending_from_version)
                old_version = read_back_exact_records(
                    qdrant, collection_name, [old_version_id]
                ).get(old_version_id)
    new_hash = None if manifest is None else manifest.get("file_sha256")
    if new_hash is not None and not is_sha256_hex(new_hash):
        errors.append("new file_sha256 is malformed")
    if pending_event_payload and pending_event_payload.get("to_file_sha256") != new_hash:
        errors.append("pending event snapshot no longer matches the prepared file")
    source_uri = (
        str(manifest.get("source_uri") or "") if manifest is not None
        else str(source_payload.get("source_uri") or expected_file_uri(file_path) or "")
    )
    if not source_uri:
        errors.append("source_uri is required")

    desired_ids: list[str] = []
    new_version_id: str | None = None
    new_version_entity: str | None = None
    if new_hash is not None:
        new_version_digest = make_version_identity_digest(source_key=source_key, file_sha256=str(new_hash))
        new_version_entity = version_logical_id(
            version_identity_digest=new_version_digest, profile_id=profile_id,
        )
        new_version_id = storage_point_id(new_version_entity)
        for chunk in chunks:
            chunk.id = make_versioned_file_chunk_id(
                scope_key=scope_key, resolved_file_path=file_path,
                file_sha256=str(new_hash), chunker_version=chunk.chunker_version,
                chunk_index=int(chunk.chunk_index), chunk_hash=chunk.chunk_hash,
            )
            chunk.manifest_version = 2
            chunk.lineage_schema_version = LINEAGE_SCHEMA_VERSION
            chunk.lineage_entity_id = memory_point_endpoint_logical_id(
                scope_key=scope_key, point_id=chunk.id, profile_id=profile_id,
            )
            chunk.file_version_id = new_version_id
            chunk.file_version_entity_id = new_version_entity
            chunk.lineage_pending = True
            chunk.derived_from = [{
                "point_id": new_version_id,
                "source_uri": f"memory://point/{new_version_id}",
                "relation_type": "DERIVED_FROM",
                "derivation_type": "indexed_chunk",
                "content_hash": f"sha256:{new_hash}",
            }]
            desired_ids.append(str(chunk.id))
    inventory_ids = sorted(str(point.get("id")) for point in existing_points if point.get("id") is not None)
    if pending_event_payload:
        retired_ids = sorted(str(value) for value in pending_event_payload.get("retired_point_ids", []) if isinstance(value, str))
        new_chunk_ids = sorted(pending_new_chunk_ids)
        old_ids = retired_ids
        if set(new_chunk_ids) - set(desired_ids):
            errors.append("pending event desired chunk inventory drift")
    else:
        old_ids = inventory_ids
        retired_ids = sorted(set(old_ids) - set(desired_ids))
        new_chunk_ids = sorted(set(desired_ids) - set(old_ids))
    for chunk in chunks:
        if str(chunk.id) not in new_chunk_ids:
            chunk.lineage_pending = False
    bootstrap_only = bool(
        bootstrap_required and manifest is not None and old_hash == new_hash
        and set(old_ids) == set(desired_ids)
    )
    if (not bootstrap_required and not pending_event_payload and manifest is not None
            and old_hash == new_hash and set(old_ids) == set(desired_ids)):
        return {
            "complete": not errors,
            "noop": True,
            "errors": errors,
            "file_path": file_path,
            "source_point_id": source_id,
        }

    if errors:
        return {"complete": False, "noop": False, "errors": sorted(set(errors)), "file_path": file_path}
    if manifest is None and not old_ids and old_version_id in (None, ""):
        return {"complete": True, "noop": True, "errors": [], "file_path": file_path, "source_point_id": source_id}

    baseline_basis = (
        "indexed_file_sha256" if old_hash or not old_ids
        else "missing_indexed_file_sha256"
    )
    if pending_event_payload:
        baseline_basis = str(pending_event_payload.get("lineage_baseline_basis") or baseline_basis)
        missing_fields = set(pending_event_payload.get("lineage_missing_fields") or [])
    previous_event_id = (
        pending_event_payload.get("previous_event_id")
        if pending_event_payload else source_payload.get("head_event_id")
    )
    bootstrap_event_point: dict[str, Any] | None = None
    if bootstrap_required and not (
        manifest is not None and old_hash == new_hash and set(old_ids) == set(desired_ids)
    ):
        bootstrap_digest = lineage_digest([
            "lineage-inventory-v1", source_key, str(old_hash), sorted(old_ids), [],
        ])
        bootstrap_id, bootstrap_payload = build_change_event(
            source_key=source_key, scope_key=scope_key, profile_id=profile_id,
            file_path=file_path, source_uri=str(source_payload.get("source_uri") or source_uri),
            event_kind="observed", previous_event_id=None, from_version_id=None,
            to_version_id=str(old_version_id), desired_inventory_digest=bootstrap_digest,
            created_point_ids=[], retired_point_ids=[], patched_point_ids=[],
            lineage_baseline_basis="indexed_file_sha256", lineage_missing_fields=[],
            to_file_sha256=str(old_hash), lineage_observation="indexed_payload",
            lineage_history_complete=False, event_state="committed",
            user_id_hash=user_id_hash,
            chat_id_hash=chat_id_hash,
        )
        existing_bootstrap = read_back_exact_records(
            qdrant, collection_name, [bootstrap_id]
        ).get(bootstrap_id)
        if existing_bootstrap:
            bootstrap_payload = dict(existing_bootstrap.get("payload") or {})
        bootstrap_event_point = {"id": bootstrap_id, "vector": {}, "payload": bootstrap_payload}
        previous_event_id = bootstrap_id
    if previous_event_id not in (None, "") and not is_uuid_string(previous_event_id):
        return {"complete": False, "errors": ["source head_event_id is malformed"], "file_path": file_path}
    desired_digest = lineage_digest([
        "lineage-inventory-v1", source_key, str(new_hash or ""), sorted(desired_ids), retired_ids,
    ])
    if pending_event_payload:
        event_kind = str(pending_event_payload.get("event_kind") or "")
        if event_kind not in EVENT_KINDS:
            errors.append("pending event_kind is malformed")
    elif bootstrap_only:
        event_kind = "observed"
    elif manifest is None:
        event_kind = "deleted"
    elif old_hash is None:
        event_kind = "observed"
    elif old_hash == new_hash:
        event_kind = "rechunked"
    else:
        prior_new = read_back_exact_records(qdrant, collection_name, [str(new_version_id)]).get(str(new_version_id))
        event_kind = "restored" if prior_new else "modified"

    event_from_version_id = None if event_kind == "observed" else (str(old_version_id) if old_hash else None)
    event_observation = "indexed_payload" if bootstrap_required and event_kind == "observed" else "read_bytes"

    # First build obtains the deterministic event ID needed by invalidation and edges.
    event_id, _ = build_change_event(
        source_key=source_key, scope_key=scope_key, profile_id=profile_id,
        file_path=file_path, source_uri=source_uri, event_kind=event_kind,
        previous_event_id=previous_event_id, from_version_id=event_from_version_id,
        to_version_id=new_version_id, desired_inventory_digest=desired_digest,
        created_point_ids=new_chunk_ids, retired_point_ids=retired_ids, patched_point_ids=[],
        lineage_baseline_basis=baseline_basis,
        lineage_missing_fields=sorted(missing_fields),
        from_file_sha256=None if event_kind == "observed" else old_hash,
        to_file_sha256=str(new_hash) if new_hash is not None else None,
        lineage_observation=event_observation,
        lineage_history_complete=False,
        user_id_hash=user_id_hash, chat_id_hash=chat_id_hash,
    )
    if pending_event_id not in (None, "") and str(pending_event_id) != event_id:
        return {"complete": False, "errors": ["pending event identity drift"], "file_path": file_path}
    existing_event = read_back_exact_records(qdrant, collection_name, [event_id]).get(event_id)
    existing_event_payload = (existing_event or {}).get("payload") or {}
    observed_at = existing_event_payload.get("created_at")
    event_state = str(existing_event_payload.get("event_state") or "prepared")
    if event_state not in EVENT_STATES:
        return {"complete": False, "errors": ["pending event state is malformed"], "file_path": file_path}
    roots = list(retired_ids)
    if old_hash and old_version_id and (new_hash is None or old_hash != new_hash):
        roots.append(str(old_version_id))
    invalidation = plan_invalidation(
        qdrant=qdrant, collection_name=collection_name,
        root_point_ids=roots, event_id=event_id, profile_id=profile_id,
        user_id_hash=user_id_hash, chat_id_hash=chat_id_hash,
        max_depth=max_depth, max_points=max_points, max_edges=max_edges,
    )
    if invalidation.get("complete") is not True:
        return {
            "complete": False, "errors": list(invalidation.get("errors") or []),
            "file_path": file_path, "invalidation": invalidation,
        }

    metadata_points: list[dict[str, Any]] = []
    root_state_updates: list[dict[str, Any]] = []
    if old_hash and old_version_id:
        if old_version is None:
            sample = (existing_points[0].get("payload") or {}) if existing_points else {}
            old_id, old_payload, _ = build_file_version(
                collection_name=collection_name, profile_id=profile_id,
                user_id_hash=user_id_hash, chat_id_hash=chat_id_hash,
                scope_key=scope_key, source_key=source_key, file_path=file_path,
                source_uri=source_uri, file_sha256=old_hash,
                file_size=int(sample.get("file_size") or 0),
                file_mtime=str(sample.get("file_mtime") or ""),
                file_mtime_ns=int(sample.get("file_mtime_ns") or 0),
            )
            old_payload["lineage_operation"] = "index_reconcile"
            metadata_points.append({"id": old_id, "vector": {}, "payload": old_payload})
        if old_version_id != new_version_id:
            root_state_updates.append({
                "point_id": str(old_version_id),
                "patch": {"fact_status": "stale" if manifest is None else "superseded", "stale": manifest is None},
            })

    created_ids = list(new_chunk_ids)
    if (not bootstrap_only and new_hash is not None and manifest is not None
            and new_version_id and new_version_entity):
        version_id, version_payload, _ = build_file_version(
            collection_name=collection_name, profile_id=profile_id,
            user_id_hash=user_id_hash, chat_id_hash=chat_id_hash,
            scope_key=scope_key, source_key=source_key, file_path=file_path,
            source_uri=source_uri, file_sha256=str(new_hash),
            file_size=int(manifest["file_size"]), file_mtime=str(manifest["file_mtime_iso"]),
            file_mtime_ns=int(manifest["file_mtime_ns"]),
        )
        version_payload.update(lineage_operation="index_reconcile", fact_status="active", stale=False)
        metadata_points.append({"id": version_id, "vector": {}, "payload": version_payload})
        created_ids.append(version_id)

        part_payload = build_mechanical_edge_payload(
            relation_type="PART_OF", source_entity_id=new_version_entity,
            target_entity_id=source_entity, profile_id=profile_id,
            source_point_id=version_id, target_point_id=source_id,
            source_entity_type="source", target_entity_type="source",
            lineage_operation="index_reconcile", lineage_source_key=source_key,
            lineage_scope_key=scope_key, observation="read_bytes",
            source_content_hash=f"sha256:{new_hash}", file_version_id=version_id,
            file_path=file_path, file_sha256=str(new_hash), lineage_event_id=event_id,
        )
        part_payload.update(user_id_hash=user_id_hash, chat_id_hash=chat_id_hash)
        part_id = storage_point_id(str(part_payload["edge_id"]))
        metadata_points.append({"id": part_id, "vector": {}, "payload": part_payload})
        created_ids.append(part_id)
        for chunk in chunks:
            edge_payload = build_mechanical_edge_payload(
                relation_type="DERIVED_FROM",
                source_entity_id=chunk.lineage_entity_id,
                target_entity_id=new_version_entity,
                profile_id=profile_id, source_point_id=chunk.id,
                target_point_id=version_id,
                source_entity_type="memory_point", target_entity_type="source",
                lineage_operation="index_reconcile", lineage_source_key=source_key,
                lineage_scope_key=scope_key, observation="indexed_payload",
                source_content_hash=f"sha256:{chunk.chunk_hash}",
                target_content_hash=f"sha256:{new_hash}",
                file_version_id=version_id, file_path=file_path,
                file_sha256=str(new_hash), locator=chunk.locator(),
                lineage_event_id=event_id,
            )
            edge_payload.update(user_id_hash=user_id_hash, chat_id_hash=chat_id_hash)
            edge_id = storage_point_id(str(edge_payload["edge_id"]))
            metadata_points.append({"id": edge_id, "vector": {}, "payload": edge_payload})
            created_ids.append(edge_id)
        if old_hash and old_hash != new_hash and old_version_id:
            old_version_entity = version_logical_id(
                version_identity_digest=make_version_identity_digest(
                    source_key=source_key, file_sha256=old_hash,
                ),
                profile_id=profile_id,
            )
            supersedes = build_mechanical_edge_payload(
                relation_type="SUPERSEDES", source_entity_id=new_version_entity,
                target_entity_id=old_version_entity, profile_id=profile_id,
                source_point_id=version_id, target_point_id=str(old_version_id),
                source_entity_type="source", target_entity_type="source",
                lineage_operation="index_reconcile", lineage_source_key=source_key,
                lineage_scope_key=scope_key, observation="read_bytes",
                source_content_hash=f"sha256:{new_hash}",
                target_content_hash=f"sha256:{old_hash}",
                file_path=file_path, file_sha256=str(new_hash),
                lineage_event_id=event_id,
            )
            supersedes.update(user_id_hash=user_id_hash, chat_id_hash=chat_id_hash)
            edge_id = storage_point_id(str(supersedes["edge_id"]))
            metadata_points.append({"id": edge_id, "vector": {}, "payload": supersedes})
            created_ids.append(edge_id)

    review_changes = [
        {
            "point_id": change["point_id"],
            "snapshot_digest": change["snapshot_digest"],
            "patch": change["patch"],
            "root_ids": change["root_ids"],
        }
        for change in invalidation["changes"]
    ]
    event_id, event_payload = build_change_event(
        source_key=source_key, scope_key=scope_key, profile_id=profile_id,
        file_path=file_path, source_uri=source_uri, event_kind=event_kind,
        previous_event_id=previous_event_id, from_version_id=event_from_version_id,
        to_version_id=new_version_id, desired_inventory_digest=desired_digest,
        created_point_ids=created_ids, retired_point_ids=retired_ids,
        new_chunk_ids=new_chunk_ids,
        patched_point_ids=invalidation["dependent_ids"],
        lineage_baseline_basis=baseline_basis,
        lineage_missing_fields=sorted(missing_fields), review_changes=review_changes,
        from_file_sha256=None if event_kind == "observed" else old_hash,
        to_file_sha256=str(new_hash) if new_hash is not None else None,
        lineage_observation=event_observation,
        lineage_history_complete=False,
        event_state=event_state, observed_at=observed_at,
        user_id_hash=user_id_hash, chat_id_hash=chat_id_hash,
    )
    if existing_event:
        event_payload = {**existing_event_payload, **event_payload}
    metadata_points.append({"id": event_id, "vector": {}, "payload": event_payload})

    source_created = source_payload.get("created_at")
    pending_source = build_source_node_payload(
        source_key=source_key, scope_key=scope_key, profile_id=profile_id,
        file_path=file_path, source_uri=source_uri,
        lineage_operation="index_reconcile", created_at=source_created,
    )
    pending_source.update({
        "user_id_hash": user_id_hash,
        "chat_id_hash": chat_id_hash,
        "current_version_id": str(old_version_id) if old_hash else None,
        "head_event_id": previous_event_id,
        "pending_event_id": event_id,
        "source_deleted": False,
    })
    metadata_points.append({"id": source_id, "vector": {}, "payload": pending_source})
    baseline_records = list(existing_points) + ([existing_source] if existing_source else [])
    if old_version:
        baseline_records.append(old_version)
    baseline_ids = sorted({
        str(point["id"]) for point in baseline_records
        if point and point.get("id") is not None
    })
    full_baseline = {
        str(point.get("id")): point
        for point in qdrant.retrieve(
            collection_name, baseline_ids, with_payload=True, with_vector=True,
        )
    }
    if set(full_baseline) != set(baseline_ids):
        return {"complete": False, "errors": ["reconciliation baseline read is incomplete"], "file_path": file_path}
    baseline_snapshots = {
        point_id: _point_snapshot(full_baseline[point_id])
        for point_id in baseline_ids
    }
    return {
        "complete": True,
        "noop": False,
        "errors": [],
        "file_path": file_path,
        "event_id": event_id,
        "event_kind": event_kind,
        "source_point_id": source_id,
        "new_version_id": new_version_id,
        "old_version_id": str(old_version_id) if old_hash else None,
        "retired_ids": retired_ids,
        "new_chunk_ids": new_chunk_ids,
        "desired_chunk_ids": desired_ids,
        "metadata_points": metadata_points,
        "bootstrap_event_point": bootstrap_event_point,
        "baseline_snapshots": baseline_snapshots,
        "invalidation": invalidation,
        "invalidation_context": {
            "profile_id": profile_id,
            "user_id_hash": user_id_hash,
            "chat_id_hash": chat_id_hash,
        },
        "root_state_updates": root_state_updates,
        "source_commit_patch": {
            "current_version_id": new_version_id,
            "head_event_id": event_id,
            "pending_event_id": None,
            "source_deleted": manifest is None,
        },
        "lineage_baseline_basis": baseline_basis,
        "lineage_missing_fields": sorted(missing_fields),
    }


def _set_event_state(
    qdrant: Any, collection_name: str, event_id: str, state: str,
) -> None:
    current = read_back_exact_records(qdrant, collection_name, [event_id])
    payload = (current.get(event_id) or {}).get("payload") or {}
    prior = str(payload.get("event_state") or "prepared")
    if prior in EVENT_STATES and EVENT_STATES.index(prior) >= EVENT_STATES.index(state):
        return
    qdrant.update_payload(collection_name, event_id, {"event_state": state})


def _retire_chunk_derivation_edges(
    *, qdrant: Any, collection_name: str, retired_ids: list[str],
    old_version_id: str | None, profile_id: str, event_id: str,
) -> None:
    if not retired_ids or not old_version_id:
        return
    edges, complete = _scroll_bounded(
        qdrant,
        collection_name,
        {"must": [
            {"key": "source_point_id", "match": {"any": retired_ids}},
            {"key": "relation_type", "match": {"value": "DERIVED_FROM"}},
            {"key": "profile_id", "match": {"value": profile_id}},
        ]},
        limit=DEFAULT_INVALIDATION_EDGES,
        with_vector=False,
    )
    if not complete:
        raise RuntimeError("retired chunk derivation lookup exceeded bound")
    marked: list[str] = []
    for edge in edges:
        payload = edge.get("payload") if isinstance(edge, dict) else None
        if not isinstance(payload, dict):
            raise RuntimeError("retired chunk derivation payload is missing")
        if (
            str(payload.get("source_point_id") or "") in retired_ids
            and str(payload.get("target_point_id") or "") == old_version_id
            and payload.get("lineage_operation") in {"index_capture", "index_reconcile"}
            and payload.get("target_entity_type") == "source"
        ):
            edge_id = str(edge.get("id") or "")
            if not edge_id:
                raise RuntimeError("retired chunk derivation ID is missing")
            qdrant.update_payload(
                collection_name, edge_id, {
                    "lineage_retired": True,
                    "lineage_retired_by_event_id": event_id,
                },
            )
            marked.append(edge_id)
    if marked:
        read_back = read_back_exact_records(qdrant, collection_name, marked)
        if any(
            ((read_back.get(edge_id) or {}).get("payload") or {}).get("lineage_retired") is not True
            or ((read_back.get(edge_id) or {}).get("payload") or {}).get(
                "lineage_retired_by_event_id"
            ) != event_id
            for edge_id in marked
        ):
            raise RuntimeError("retired chunk derivation marking failed")


def apply_reconciliation_plan(
    *,
    qdrant: Any,
    collection_name: str,
    plan: dict[str, Any],
    chunk_points: list[dict[str, Any]],
    lock_timeout: float = 5.0,
    lock_dir: str = "",
) -> dict[str, Any]:
    """Persist one resumable W2 file transition in the mandated order."""
    validation = validate_transition_plan(plan)
    if not validation["valid"]:
        raise RuntimeError(f"invalid transition plan: {validation['problems']}")
    expected_chunk_ids = list(plan.get("new_chunk_ids") or [])
    provided_chunk_ids = [str(point.get("id") or "") for point in chunk_points]
    if sorted(provided_chunk_ids) != sorted(expected_chunk_ids):
        raise RuntimeError("chunk_points do not exactly cover new_chunk_ids")
    with collection_write_lock(
        collection_name=collection_name, timeout=lock_timeout, lock_dir=lock_dir,
    ):
        baseline_ids = sorted((plan.get("baseline_snapshots") or {}).keys())
        baseline = {
            str(point.get("id")): point
            for point in qdrant.retrieve(
                collection_name, baseline_ids, with_payload=True, with_vector=True,
            )
        }
        if set(baseline) != set(baseline_ids):
            raise RuntimeError("reconciliation baseline disappeared before apply")
        for point_id, digest in (plan.get("baseline_snapshots") or {}).items():
            if _point_snapshot(baseline[point_id]) != digest:
                raise RuntimeError(f"reconciliation baseline drift: {point_id}")

        planned_invalidation = plan["invalidation"]
        context = plan.get("invalidation_context") or {}
        bounds = planned_invalidation.get("bounds") or {}
        locked_invalidation = plan_invalidation(
            qdrant=qdrant,
            collection_name=collection_name,
            root_point_ids=list(planned_invalidation.get("root_ids") or []),
            event_id=str(plan["event_id"]),
            profile_id=str(context.get("profile_id") or ""),
            user_id_hash=str(context.get("user_id_hash") or ""),
            chat_id_hash=str(context.get("chat_id_hash") or ""),
            max_depth=int(bounds.get("max_depth", DEFAULT_INVALIDATION_DEPTH)),
            max_points=int(bounds.get("max_points", DEFAULT_INVALIDATION_POINTS)),
            max_edges=int(bounds.get("max_edges", DEFAULT_INVALIDATION_EDGES)),
        )
        if locked_invalidation.get("complete") is not True:
            raise RuntimeError(
                "locked invalidation closure is incomplete: "
                + "; ".join(locked_invalidation.get("errors") or [])
            )
        planned_ids = set(planned_invalidation.get("dependent_ids") or [])
        locked_ids = set(locked_invalidation.get("dependent_ids") or [])
        if not locked_ids.issubset(planned_ids):
            raise RuntimeError(
                f"locked invalidation closure grew: {sorted(locked_ids - planned_ids)}"
            )
        if (
            locked_invalidation.get("dependent_ids") != planned_invalidation.get("dependent_ids")
            or locked_invalidation.get("changes") != planned_invalidation.get("changes")
        ):
            raise RuntimeError("locked invalidation closure changed; re-plan required")

        bootstrap_event = plan.get("bootstrap_event_point")
        if bootstrap_event:
            bootstrap_id = str(bootstrap_event["id"])
            qdrant.upsert(collection_name, [bootstrap_event])
            bootstrap_read = read_back_exact_records(qdrant, collection_name, [bootstrap_id])
            if (bootstrap_read.get(bootstrap_id) or {}).get("payload") != bootstrap_event["payload"]:
                raise RuntimeError("W1 baseline event read-back failed")
            qdrant.update_payload(collection_name, plan["source_point_id"], {
                "head_event_id": bootstrap_id,
                "pending_event_id": None,
                "current_version_id": plan.get("old_version_id"),
                "source_deleted": False,
            })
            source_read = read_back_exact_records(
                qdrant, collection_name, [plan["source_point_id"]]
            )
            if ((source_read.get(plan["source_point_id"]) or {}).get("payload") or {}).get("head_event_id") != bootstrap_id:
                raise RuntimeError("W1 baseline event head commit failed")

        metadata_points = list(plan.get("metadata_points") or [])
        source_points = [
            point for point in metadata_points
            if str(point["id"]) == str(plan["source_point_id"])
        ]
        metadata_records = [
            point for point in metadata_points
            if str(point["id"]) != str(plan["source_point_id"])
        ]
        if metadata_records:
            qdrant.upsert(collection_name, metadata_records)
        metadata_ids = [str(point["id"]) for point in metadata_records]
        read_back = read_back_exact_records(qdrant, collection_name, metadata_ids)
        if set(read_back) != set(metadata_ids):
            raise RuntimeError("reconciliation metadata read-back is incomplete")
        if source_points:
            qdrant.upsert(collection_name, source_points)
            source_read = read_back_exact_records(
                qdrant, collection_name, [plan["source_point_id"]]
            )
            if plan["source_point_id"] not in source_read:
                raise RuntimeError("reconciliation source read-back is incomplete")

        invalidation_result = apply_invalidation(
            qdrant=qdrant, collection_name=collection_name,
            plan=locked_invalidation,
        )
        _set_event_state(qdrant, collection_name, plan["event_id"], "dependents_marked")

        if chunk_points:
            qdrant.upsert(collection_name, chunk_points)
            chunks_read = {
                str(point.get("id")): point
                for point in qdrant.retrieve(
                    collection_name, plan["new_chunk_ids"],
                    with_payload=True, with_vector=False,
                )
            }
            if set(chunks_read) != set(plan["new_chunk_ids"]):
                raise RuntimeError("staged chunk read-back is incomplete")
            for point in chunk_points:
                actual = chunks_read[str(point["id"])].get("payload") or {}
                expected = point.get("payload") or {}
                if actual.get("content_hash") != expected.get("content_hash") or actual.get("lineage_pending") is not True:
                    raise RuntimeError(f"staged chunk verification failed: {point['id']}")
        _set_event_state(qdrant, collection_name, plan["event_id"], "chunks_staged")

        for update in plan.get("root_state_updates") or []:
            qdrant.update_payload(collection_name, update["point_id"], update["patch"])
        retired_ids = list(plan.get("retired_ids") or [])
        if retired_ids:
            _retire_chunk_derivation_edges(
                qdrant=qdrant,
                collection_name=collection_name,
                retired_ids=retired_ids,
                old_version_id=plan.get("old_version_id"),
                profile_id=str((plan.get("invalidation_context") or {}).get("profile_id") or ""),
                event_id=str(plan["event_id"]),
            )
            qdrant.delete_ids(collection_name, retired_ids)
            if qdrant.retrieve(collection_name, retired_ids, with_payload=True, with_vector=False):
                raise RuntimeError("retired chunk IDs remain after delete")
        _set_event_state(qdrant, collection_name, plan["event_id"], "roots_retired")

        for point_id in plan.get("new_chunk_ids") or []:
            qdrant.update_payload(collection_name, point_id, {"lineage_pending": False})
        qdrant.update_payload(collection_name, plan["source_point_id"], dict(plan["source_commit_patch"]))
        _set_event_state(qdrant, collection_name, plan["event_id"], "committed")
        committed = read_back_exact_records(
            qdrant, collection_name,
            [plan["event_id"], plan["source_point_id"]] + list(plan.get("new_chunk_ids") or []),
        )
        event_payload = (committed.get(plan["event_id"]) or {}).get("payload") or {}
        source_payload = (committed.get(plan["source_point_id"]) or {}).get("payload") or {}
        if event_payload.get("event_state") != "committed":
            raise RuntimeError("event commit read-back failed")
        if source_payload.get("pending_event_id") is not None or source_payload.get("head_event_id") != plan["event_id"]:
            raise RuntimeError("source commit read-back failed")
        return {
            "event_id": plan["event_id"],
            "retired_ids": retired_ids,
            "new_chunk_ids": list(plan.get("new_chunk_ids") or []),
            "dependent_ids": invalidation_result["verified_ids"],
            "committed": True,
        }


def resume_reconciliation(
    *,
    qdrant: Any,
    collection_name: str,
    plan: dict[str, Any],
    chunk_points: list[dict[str, Any]],
    lock_timeout: float = 5.0,
    lock_dir: str = "",
) -> dict[str, Any]:
    """Resume the exact deterministic plan; apply is idempotent by event and ID."""
    return apply_reconciliation_plan(
        qdrant=qdrant, collection_name=collection_name, plan=plan,
        chunk_points=chunk_points, lock_timeout=lock_timeout, lock_dir=lock_dir,
    )
