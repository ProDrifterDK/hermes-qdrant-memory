"""Pure lineage identity, evidence, and structural payload helpers (W0).

W0 scope only: canonical identity material, exact write-ownership checks, the
typed :class:`LineageEvidence` internal evidence record, structural graph
payload builders, and strict payload validation. There is deliberately no
Qdrant orchestration, indexing, reconciliation, locking, or propagation here —
those belong to later waves.

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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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
