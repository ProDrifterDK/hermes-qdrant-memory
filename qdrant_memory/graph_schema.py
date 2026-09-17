"""Graph entity and edge schema primitives for the Qdrant memory plugin.

This module provides typed helpers for building, validating, and serializing
graph-aware memory payloads (entities and edges) that are stored as Qdrant
points alongside regular memory chunks. All payloads are backward-compatible
with existing search/ranking, and all entity/edge creation is non-canonical
and review-gated by default.

Safety invariants enforced here:
- Entity/edge IDs are deterministic, safe (no secrets), and validated.
- No entity or edge is ever auto-promoted to canonical truth.
- Secret-bearing fields are redacted/rejected everywhere.
- Provenance is required for both entities and edges (at least one safe
  provenance handle: non-empty sanitized source_point_ids, a safe source_uri,
  or a valid content_hash).
- ``usefulness_weight`` and ``truth_confidence`` are kept separate.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .lesson_extractor import contains_secret
from .schema import (
    FactStatus,
    MemoryKind,
    RELATION_TYPES,
    RelationType,
    _metadata_is_empty,
    _sanitize_source_metadata,
    make_point_id,
    now_iso,
    valid_fact_status,
    valid_point_id_link,
    valid_relation_type,
)

# ---------------------------------------------------------------------------
# Constants and patterns
# ---------------------------------------------------------------------------

# Entity type vocabulary — open set but we track known types for validation.
KNOWN_ENTITY_TYPES = frozenset({
    "concept",
    "person",
    "project",
    "tool",
    "technology",
    "organization",
    "location",
    "event",
    "decision",
    "artifact",
    "task",
    "metric",
    "hypothesis",
    "mechanism",
    "experiment",
    "failure_mode",
    "source",
    "session",
    "feedback_event",
    "memory_point",
    "agent",
    "worktree",
    "review",
    "blocker",
    "dependency",
    "seed",
})

# Characters allowed in entity label slugs (the textual key used for ID generation).
_ENTITY_SLUG_RE = re.compile(r"[^a-zA-Z0-9_-]+")
_ENTITY_SLUG_CLEAN_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")

# Max number of aliases per entity.
_MAX_ALIASES = 32
# Max alias length.
_MAX_ALIAS_LEN = 256
# Max number of source_point_ids per edge.
_MAX_SOURCE_POINT_IDS = 64

# UUID shape (lowercase, canonical 8-4-4-4-12). Real Qdrant only accepts
# unsigned integers or UUIDs as point IDs; metadata graph records use the
# UUID mapping so their storage IDs are service-valid while payloads keep
# the logical entity-*/edge-* handles.
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")

# Lineage (W0) controlled vocabularies. Single source of truth: the
# mechanical lineage module imports these from here.
LINEAGE_OPERATIONS = (
    "index_capture",
    "index_reconcile",
    "approved_citation",
    "source_revalidation",
    "restore",
)
LINEAGE_OBSERVATIONS = (
    "read_bytes",
    "indexed_payload",
    "approved_manifest",
    "legacy_unpinned",
)
EDGE_CLASSES = ("mechanical", "semantic")
LINEAGE_ROLES = ("file_source", "file_version", "change_event")


# ---------------------------------------------------------------------------
# W0 structural shape domains — the single shared validation surface
# ---------------------------------------------------------------------------
# One definition per documented W0 shape domain, called by BOTH the
# mechanical write gate (``qdrant_memory.lineage.validate_lineage_payload`` /
# ``evidence_shape_problems``) and every W0-capable builder path
# (``GraphEntity.to_payload`` / ``GraphEdge.to_payload`` and the
# ``_validated_lineage_*_fields`` helpers). Builder and gate cannot disagree
# about a documented shape because they call these same functions.
#
# Omission semantics: an optional W0 field is omitted ONLY by its sentinel
# (``None`` or ``""``; the evidence locator uses an empty mapping). Any other
# value — including ``0``, ``False`` or ``{}`` — is a PRESENT value and must
# be exactly the documented token. On final payload dicts, key presence is
# presence: a present key with a sentinel value is a failed validation, not
# an omission. Every check is exact-token: stripping, case-folding, or
# newline tolerance would normalize a different token into a storable one.

# Documented omission sentinels for optional W0 builder/evidence fields.
W0_OMISSION_SENTINELS = (None, "")

# Raw 64-char lowercase hex SHA-256 digest.
_W0_SHA256_RE = re.compile(r"^[a-f0-9]{64}\Z")
# Documented content-hash token: "sha256:<64 lowercase hex>".
_W0_CONTENT_HASH_RE = re.compile(r"^sha256:[a-f0-9]{64}\Z")
# Logical graph handles: ``entity-<16 lowercase hex>`` / ``edge-<16 hex>``.
_W0_ENTITY_HANDLE_RE = re.compile(r"^entity-[a-f0-9]{16}\Z")
_W0_EDGE_HANDLE_RE = re.compile(r"^edge-[a-f0-9]{16}\Z")


def w0_is_omitted(value: Any) -> bool:
    """True iff *value* is the documented W0 omission sentinel (``None``/``""``).

    Never truthiness: ``0``, ``False`` and ``{}`` are NOT omitted — they are
    present, wrong-typed values.
    """
    return value is None or value == ""


def w0_sha256_hex_problems(value: Any, field: str = "value") -> list[str]:
    """Problems for the raw 64-lowercase-hex SHA-256 digest domain."""
    if not isinstance(value, str) or not _W0_SHA256_RE.match(value):
        return [f"{field} must be 64 lowercase hex chars"]
    return []


def w0_content_hash_problems(value: Any, field: str = "value") -> list[str]:
    """Problems for the documented ``sha256:<64 lowercase hex>`` domain."""
    if not isinstance(value, str) or not _W0_CONTENT_HASH_RE.match(value):
        return [f"{field} must be a documented sha256:<64 lowercase hex> content hash"]
    return []


def w0_uuid_problems(value: Any, field: str = "value") -> list[str]:
    """Problems for the exact canonical UUID reference-token domain."""
    if not is_uuid_string(value):
        return [f"{field} must be an exact canonical UUID"]
    return []


def w0_logical_handle_problems(value: Any, field: str = "value", handle_kind: str = "entity") -> list[str]:
    """Problems for the exact ``entity-<16 hex>`` / ``edge-<16 hex>`` domain.

    A padded or newline-suffixed handle is a different token and is refused,
    never stripped into the clean spelling.
    """
    pattern = _W0_ENTITY_HANDLE_RE if handle_kind == "entity" else _W0_EDGE_HANDLE_RE
    if not isinstance(value, str) or not pattern.match(value):
        return [f"{field} must be an exact {handle_kind}-<16 lowercase hex> logical handle"]
    return []


def w0_vocabulary_problems(value: Any, field: str, allowed: Any) -> list[str]:
    """Problems for an exact-token controlled-vocabulary domain.

    The raw value must already be a member of *allowed*; a padded,
    newline-suffixed, or case-shifted token is a different token and is
    refused, never normalized into the documented spelling.
    """
    if not isinstance(value, str) or value not in allowed:
        return [f"{field} must be exactly one of {tuple(allowed)}"]
    return []


def w0_uri_problems(value: Any, field: str = "source_uri") -> list[str]:
    """Problems for the URI token domain: a non-empty string containing no
    whitespace and no NUL. URIs are bound by exact equality downstream, so a
    whitespace-bearing raw token must refuse, never strip into the clean URI.
    """
    if not isinstance(value, str) or not value:
        return [f"{field} must be a non-empty string"]
    if "\x00" in value or any(character.isspace() for character in value):
        return [f"{field} must not contain whitespace or NUL characters"]
    return []


def w0_file_path_problems(value: Any, field: str = "file_path") -> list[str]:
    """Problems for the exact resolved-path token domain.

    Case and internal spelling (spaces included) are identity-relevant and
    preserved; only structural malformation (non-string, empty, NUL) refuses.
    """
    if not isinstance(value, str) or not value:
        return [f"{field} must be a non-empty string"]
    if "\x00" in value:
        return [f"{field} must not contain NUL characters"]
    return []


# Real indexer file-chunk locator shape (``FileChunk.payload``): exactly
# these keys, positive non-bool integer lines, bounded heading. A locator is
# evidence only in this shape.
_W0_LOCATOR_KEYS = frozenset({"line_start", "line_end", "heading"})
# Heading bound already enforced project-wide (sources.py inspect/stat).
_W0_LOCATOR_MAX_HEADING = 200
# "Bounded locator" (plan section 4): cap the serialized form.
_W0_LOCATOR_MAX_JSON = 512


def w0_locator_problems(locator: Any) -> list[str]:
    """Validate one file-chunk locator against the real indexer shape.

    The indexed chunk -> file-version ``DERIVED_FROM`` direction is automatic
    only when "locator and chunk hash match the prepared snapshot", so a
    locator is provenance only when it is exactly the shape the indexer
    emits: required positive ``line_start``, optional ``line_end`` that is
    not reversed, optional bounded ``heading``, no unknown keys, bounded
    serialized size. Malformed non-empty metadata is a problem, never
    silently accepted as provenance. Shared by the gate and the W0 builder
    paths (builders refuse instead of sanitizing an invalid raw locator into
    the exact valid evidence shape).
    """
    if not isinstance(locator, dict) or not locator:
        return ["locator must be a non-empty mapping"]
    unknown = sorted(str(key) for key in locator if key not in _W0_LOCATOR_KEYS)
    if unknown:
        return [f"locator has unknown keys: {unknown}"]
    problems: list[str] = []
    line_start = locator.get("line_start")
    if isinstance(line_start, bool) or not isinstance(line_start, int) or line_start < 1:
        problems.append("locator line_start must be a positive integer (>=1, bools are invalid)")
    # Optional members are validated by KEY PRESENCE: an explicit None is a
    # present, wrong-typed value and refuses, while an absent key is a valid
    # omission.
    if "line_end" in locator:
        line_end = locator["line_end"]
        if isinstance(line_end, bool) or not isinstance(line_end, int) or line_end < 1:
            problems.append("locator line_end must be a positive integer (>=1, bools are invalid)")
        elif (
            isinstance(line_start, int)
            and not isinstance(line_start, bool)
            and line_start >= 1
            and line_end < line_start
        ):
            problems.append("locator line_end must be >= line_start")
    if "heading" in locator:
        heading = locator["heading"]
        if not isinstance(heading, str):
            problems.append("locator heading must be a string")
        elif len(heading) > _W0_LOCATOR_MAX_HEADING:
            problems.append("locator heading exceeds the bounded heading length")
    try:
        serialized = canonical_json(locator)
    except (TypeError, ValueError):
        return [*problems, "locator must be canonical-JSON serializable"]
    if len(serialized) > _W0_LOCATOR_MAX_JSON:
        problems.append("locator exceeds the bounded serialized size")
    return problems


def canonical_json(value: Any) -> str:
    """Canonical JSON: sorted keys, compact separators, UTF-8, no NaN.

    Single shared definition: used by the lineage identity helpers and the
    bounded-locator serialization check above.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def is_uuid_string(value: Any) -> bool:
    """Return True iff *value* is a canonical lowercase UUID string."""
    return isinstance(value, str) and bool(_UUID_RE.match(value))


def make_graph_point_id(logical_id: str) -> str:
    """Map a logical graph handle (entity-*/edge-*) to its Qdrant storage ID.

    Real Qdrant rejects raw ``entity-<hex>`` / ``edge-<hex>`` handles as point
    IDs (HTTP 400). Storage IDs are deterministic UUIDs derived via
    ``make_point_id("graph-record-v1", logical_id)``; payloads keep the
    logical handles unchanged.
    """
    text = (logical_id or "").strip()
    if not text:
        raise ValueError("logical_id is required for graph point ID mapping")
    if contains_secret(text):
        raise ValueError("logical_id must not contain secrets")
    return make_point_id("graph-record-v1", text)


# ---------------------------------------------------------------------------
# ID generation and validation
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    """Normalize text into a safe slug component for ID generation."""
    slug = _ENTITY_SLUG_RE.sub("-", (text or "").strip().lower()).strip("-")
    return slug


def entity_identity_digest(entity_type: str, label: str, *, profile_id: str = "default") -> str:
    """Full SHA-256 identity digest whose truncation yields make_entity_id.

    Structural lineage records store this full digest so a logical-handle
    collision (two identities sharing the truncated 16-hex handle) is
    detectable as a hard error instead of a silent overwrite.
    """
    etype = (entity_type or "").strip().lower()
    label_text = (label or "").strip()
    if not etype:
        raise ValueError("entity_type is required for entity ID generation")
    if not label_text:
        raise ValueError("label is required for entity ID generation")
    raw = f"entity|{profile_id}|{etype}|{_slugify(label_text)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def make_entity_id(entity_type: str, label: str, *, profile_id: str = "default") -> str:
    """Generate a deterministic, safe entity ID.

    The ID is derived from entity_type, label, and profile_id via SHA-256.
    It never contains raw user text or secrets, only a hex digest.

    Raises ValueError if entity_type or label are empty.
    """
    digest = entity_identity_digest(entity_type, label, profile_id=profile_id)
    return f"entity-{digest[:16]}"


def make_edge_id(
    source_entity_id: str,
    target_entity_id: str,
    relation_type: str,
    *,
    profile_id: str = "default",
) -> str:
    """Generate a deterministic, safe edge ID.

    The ID is derived from the ordered tuple (source, target, relation_type, profile_id).
    It is stable and idempotent: repeated calls with the same arguments return the same ID.

    Raises ValueError if any required argument is empty.
    """
    src = (source_entity_id or "").strip()
    tgt = (target_entity_id or "").strip()
    rel = (relation_type or "").strip()
    if not src:
        raise ValueError("source_entity_id is required for edge ID generation")
    if not tgt:
        raise ValueError("target_entity_id is required for edge ID generation")
    if not rel:
        raise ValueError("relation_type is required for edge ID generation")
    raw = f"edge|{profile_id}|{src}|{rel}|{tgt}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"edge-{digest[:16]}"


def validate_entity_id(entity_id: Any) -> str:
    """Validate that a value is a well-formed entity ID.

    Returns the validated ID string, or raises ValueError.
    """
    if not isinstance(entity_id, str):
        raise ValueError("entity_id must be a string")
    text = entity_id.strip()
    if not text:
        raise ValueError("entity_id must not be empty")
    if contains_secret(text):
        raise ValueError("entity_id must not contain secrets")
    if not text.startswith("entity-"):
        raise ValueError("entity_id must start with 'entity-'")
    if not re.match(r"^entity-[a-f0-9]{16}$", text):
        raise ValueError("entity_id must match format 'entity-<16 hex chars>'")
    return text


def validate_edge_id(edge_id: Any) -> str:
    """Validate that a value is a well-formed edge ID.

    Returns the validated ID string, or raises ValueError.
    """
    if not isinstance(edge_id, str):
        raise ValueError("edge_id must be a string")
    text = edge_id.strip()
    if not text:
        raise ValueError("edge_id must not be empty")
    if contains_secret(text):
        raise ValueError("edge_id must not contain secrets")
    if not text.startswith("edge-"):
        raise ValueError("edge_id must start with 'edge-'")
    if not re.match(r"^edge-[a-f0-9]{16}$", text):
        raise ValueError("edge_id must match format 'edge-<16 hex chars>'")
    return text


def valid_entity_id(entity_id: Any) -> str | None:
    """Return validated entity_id or None (does not raise)."""
    try:
        return validate_entity_id(entity_id)
    except ValueError:
        return None


def valid_edge_id(edge_id: Any) -> str | None:
    """Return validated edge_id or None (does not raise)."""
    try:
        return validate_edge_id(edge_id)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Alias sanitization
# ---------------------------------------------------------------------------

def sanitize_aliases(aliases: Any) -> list[str]:
    """Sanitize and deduplicate a list of entity aliases.

    - Strips whitespace.
    - Rejects empty strings.
    - Rejects aliases containing secrets.
    - Deduplicates while preserving order.
    - Caps at _MAX_ALIASES entries.
    """
    if not isinstance(aliases, (list, tuple, set)):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in aliases:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text or len(text) > _MAX_ALIAS_LEN:
            continue
        if contains_secret(text):
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= _MAX_ALIASES:
            break
    return result


# ---------------------------------------------------------------------------
# Source point ID sanitization
# ---------------------------------------------------------------------------

def sanitize_source_point_ids(value: Any) -> list[str]:
    """Sanitize a list of source point IDs for an edge or entity.

    Reuses the existing ``valid_point_id_link`` helper for safety.
    """
    if not isinstance(value, (list, tuple)):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value[:_MAX_SOURCE_POINT_IDS]:
        point_id = valid_point_id_link(item)
        if point_id and point_id not in seen:
            result.append(point_id)
            seen.add(point_id)
    return result


# ---------------------------------------------------------------------------
# Direct-field sanitization (tags, content_hash, profile_id)
# ---------------------------------------------------------------------------

# Allowed characters for tag slugs.
_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-/]{0,127}$")
# content_hash must look like a hash: prefix:hex or pure hex.
_CONTENT_HASH_RE = re.compile(
    r"^(?:[a-zA-Z0-9_-]{1,32}:)?[a-fA-F0-9]{6,256}$"
)
# profile_id is a simple slug, similar to point IDs but allows no separators
# that could be path-traversal vectors.
_PROFILE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
# Max number of tags per entity/edge.
_MAX_TAGS = 64

# Reserved keys that ``extra`` must never inject into a graph payload.
# These are the provenance, schema, and structural fields that have dedicated
# sanitization paths in ``to_payload()``.  If ``extra`` contains any of these
# keys, they are silently dropped — callers must use the proper keyword args.
_RESERVED_PAYLOAD_KEYS = frozenset({
    "entity_id",
    "edge_id",
    "entity_type",
    "label",
    "source_entity_id",
    "target_entity_id",
    "relation_type",
    "text",
    "source",
    "source_type",
    "chunk_type",
    "memory_kind",
    "confidence",
    "canonical",
    "requires_review",
    "usefulness_weight",
    "truth_confidence",
    "created_at",
    "updated_at",
    "profile_id",
    "tags",
    "fact_status",
    "aliases",
    "description",
    "source_uri",
    "content_hash",
    "source_point_ids",
    "observed_at",
    "valid_from",
    "valid_until",
    # Top-level schema-looking keys. ``extra`` must never be allowed to inject
    # fields that look like schema metadata (e.g. ``schema``, ``schema_version``,
    # ``version``) — those are owned by the memory subsystem itself and any
    # caller-supplied value would be misleading at best and a forgery vector at
    # worst.  Callers wanting to record schema annotations must use the proper
    # keyword args (none exist today; if added, they will be promoted out of the
    # reserved set).
    "schema",
    "schema_version",
    "version",
    "extra",
    # Ownership keys are part of the exact write-scope contract; ``extra``
    # must never be able to forge them.
    "user_id_hash",
    "chat_id_hash",
    # Structural lineage (W0) fields. These have dedicated validated builder
    # arguments; ``extra`` must never be able to forge them.
    "lineage_record",
    "lineage_pending",
    "lineage_schema_version",
    "lineage_operation",
    "lineage_identity_digest",
    "lineage_source_key",
    "lineage_scope_key",
    "lineage_observation",
    "lineage_role",
    "lineage_entity_id",
    "lineage_event_id",
    "edge_class",
    "source_point_id",
    "target_point_id",
    "source_entity_type",
    "target_entity_type",
    "source_content_hash",
    "target_content_hash",
    "file_version_id",
    "file_version_entity_id",
    "file_path",
    "file_sha256",
    "file_size",
    "file_mtime",
    "file_mtime_ns",
    "chunker_version",
    "manifest_version",
    "locator",
    "derived_from",
})

# ---------------------------------------------------------------------------
# Lineage metadata validation (explicit builder arguments, never ``extra``)
# ---------------------------------------------------------------------------

def _validate_lineage_enum(value: Any, name: str, allowed: tuple[str, ...], *, exact: bool = False) -> str:
    """Validate an optional lineage enum field. Empty passes; bad values raise.

    Legacy mode (``exact=False``, non-lineage callers) strips before matching,
    unchanged. W0 structural mode (``exact=True``) is an exact-token check:
    the raw value must already be the documented token — stripping or
    case-folding a padded token would normalize input the shared gate refuses.
    """
    if exact:
        if w0_is_omitted(value):
            return ""
        problems = w0_vocabulary_problems(value, name, allowed)
        if problems:
            raise ValueError(problems[0])
        return value
    # Legacy (non-lineage) branch, behavior restored: falsy values omit,
    # truthy values are stripped before matching.
    text = str(value or "").strip()
    if not text:
        return ""
    if text not in allowed:
        raise ValueError(f"invalid {name}: {text!r}")
    return text


def _validate_lineage_digest64(value: Any, name: str) -> str:
    """Validate an optional full lowercase SHA-256 identity digest.

    Exact-token, shared ``w0_sha256_hex_problems`` domain: ``None`` and the
    empty string are the omission sentinels, but any other value -
    whitespace-padded, newline-suffixed, non-string - is refused. No
    stripping: a padded digest must not be normalized into a valid token.
    """
    if w0_is_omitted(value):
        return ""
    problems = w0_sha256_hex_problems(value, name)
    if problems:
        raise ValueError(problems[0])
    return value


def _validated_lineage_entity_fields(
    *,
    lineage_record: bool,
    lineage_schema_version: Any,
    lineage_operation: Any,
    lineage_identity_digest: Any,
    lineage_source_key: Any,
    lineage_scope_key: Any,
    lineage_observation: Any,
    lineage_role: Any,
    file_path: Any = None,
) -> dict[str, Any]:
    """Validate explicit entity lineage metadata and return payload fields."""
    fields: dict[str, Any] = {}
    if not isinstance(lineage_record, bool):
        raise ValueError("lineage_record must be a boolean")
    if lineage_schema_version is not None:
        if isinstance(lineage_schema_version, bool) or not isinstance(lineage_schema_version, int):
            raise ValueError("lineage_schema_version must be an integer (booleans are invalid)")
        if lineage_schema_version < 1:
            raise ValueError("lineage_schema_version must be >= 1")
    if lineage_record:
        if lineage_schema_version != 1:
            raise ValueError("lineage_record=True requires lineage_schema_version=1")
        operation = _validate_lineage_enum(lineage_operation, "lineage_operation", LINEAGE_OPERATIONS, exact=True)
        identity_digest = _validate_lineage_digest64(lineage_identity_digest, "lineage_identity_digest")
        if not operation or not identity_digest:
            raise ValueError("lineage_record=True requires lineage_operation and lineage_identity_digest")
        fields["lineage_record"] = True
        fields["lineage_schema_version"] = 1
        fields["lineage_operation"] = operation
        fields["lineage_identity_digest"] = identity_digest
        source_key = _validate_lineage_digest64(lineage_source_key, "lineage_source_key")
        scope_key = _validate_lineage_digest64(lineage_scope_key, "lineage_scope_key")
        if source_key:
            fields["lineage_source_key"] = source_key
        if scope_key:
            fields["lineage_scope_key"] = scope_key
        observation = _validate_lineage_enum(lineage_observation, "lineage_observation", LINEAGE_OBSERVATIONS, exact=True)
        if observation:
            fields["lineage_observation"] = observation
        role = _validate_lineage_enum(lineage_role, "lineage_role", LINEAGE_ROLES, exact=True)
        if role:
            fields["lineage_role"] = role
        if not w0_is_omitted(file_path) and not isinstance(file_path, str):
            # W0 exact-token: a non-string path must refuse, never coerce
            # through str() into a look-alike path token. The empty string
            # and None stay the documented omission sentinels.
            raise ValueError("file_path must be a string")
        path = "" if w0_is_omitted(file_path) else file_path
        if path:
            if "\x00" in path or contains_secret(path):
                raise ValueError("file_path must be a safe path string")
            # Case and exact spelling are identity-relevant; no normalization.
            fields["file_path"] = path
    elif any(
        value not in (None, "", False)
        for value in (
            lineage_schema_version,
            lineage_operation,
            lineage_identity_digest,
            lineage_source_key,
            lineage_scope_key,
            lineage_observation,
            lineage_role,
            file_path,
        )
    ):
        # Partial lineage annotations are exactly the forgery vector the
        # reserved-key wall exists to prevent; fail closed instead.
        raise ValueError("lineage metadata fields require lineage_record=True")
    return fields


def _validated_lineage_edge_fields(
    *,
    lineage_record: bool,
    lineage_schema_version: Any,
    lineage_operation: Any,
    lineage_identity_digest: Any,
    lineage_source_key: Any,
    lineage_scope_key: Any,
    lineage_observation: Any,
    lineage_event_id: Any,
    edge_class: Any,
    source_point_id: Any,
    target_point_id: Any,
    source_entity_type: Any,
    target_entity_type: Any,
    source_content_hash: Any,
    target_content_hash: Any,
    file_version_id: Any,
    file_path: Any,
    file_sha256: Any,
    locator: Any,
) -> dict[str, Any]:
    """Validate explicit edge lineage/mechanical metadata; return payload fields."""
    fields = _validated_lineage_entity_fields(
        lineage_record=lineage_record,
        lineage_schema_version=lineage_schema_version,
        lineage_operation=lineage_operation,
        lineage_identity_digest=lineage_identity_digest,
        lineage_source_key=lineage_source_key,
        lineage_scope_key=lineage_scope_key,
        lineage_observation=lineage_observation,
        lineage_role="",
    )
    edge_class_text = _validate_lineage_enum(edge_class, "edge_class", EDGE_CLASSES, exact=lineage_record)
    if edge_class_text:
        fields["edge_class"] = edge_class_text

    for arg_name, value in (("source_point_id", source_point_id), ("target_point_id", target_point_id)):
        if w0_is_omitted(value):
            continue
        # Mechanical-edge point IDs are documented as exact canonical UUIDs
        # (the write gate validates the same shared domain). No stripping: a
        # padded or newline-suffixed ID is a different token and is refused,
        # not normalized into a storable one.
        problems = w0_uuid_problems(value, arg_name)
        if problems:
            raise ValueError(problems[0])
        fields[arg_name] = value

    for arg_name, value in (("source_entity_type", source_entity_type), ("target_entity_type", target_entity_type)):
        if w0_is_omitted(value):
            continue
        if lineage_record:
            # W0 exact-token endpoint type: the raw value must already be a
            # known entity type; stripping/lowercasing ' MEMORY_POINT ' into
            # 'memory_point' would normalize input the shared gate refuses.
            problems = w0_vocabulary_problems(value, arg_name, KNOWN_ENTITY_TYPES)
            if problems:
                raise ValueError(problems[0])
            fields[arg_name] = value
            continue
        etype_text = str(value or "").strip().lower()
        if not etype_text:
            continue
        if etype_text not in KNOWN_ENTITY_TYPES:
            raise ValueError(f"{arg_name} must be a known entity type: {etype_text!r}")
        fields[arg_name] = etype_text

    for arg_name, value in (("source_content_hash", source_content_hash), ("target_content_hash", target_content_hash)):
        if w0_is_omitted(value):
            continue
        # Documented W0 endpoint hash domain (shared validator), exact token:
        # no stripping, and the legacy lax sanitizer must not normalize
        # whitespace-padded or newline-suffixed input into a storable digest.
        problems = w0_content_hash_problems(value, arg_name)
        if problems:
            raise ValueError(problems[0])
        fields[arg_name] = value

    if not w0_is_omitted(file_version_id):
        # file_version_id is the exact storage point ID (UUID) of the file
        # version record; the logical handle lives in file_version_entity_id.
        # Exact canonical token (shared domain): no stripping.
        problems = w0_uuid_problems(file_version_id, "file_version_id")
        if problems:
            raise ValueError(problems[0])
        fields["file_version_id"] = file_version_id

    if w0_is_omitted(file_path):
        pass
    elif lineage_record and not isinstance(file_path, str):
        # W0 exact raw typing only for structural records: a non-string path
        # must refuse, never coerce through str() into a look-alike token.
        raise ValueError("file_path must be a string")
    else:
        # Legacy non-lineage coercion restored exactly: str(value or "") —
        # and the coerced path is emitted ONLY when truthy, so 123 still
        # emits '123' while 0/False/[]/{} omit the key entirely (pre-pass
        # behavior).
        path = str(file_path or "")
        if path:
            if "\x00" in path or contains_secret(path):
                raise ValueError("file_path must be a safe path string")
            # Case and exact spelling are identity-relevant; no normalization.
            fields["file_path"] = path

    if not w0_is_omitted(file_sha256):
        # Exact raw-hex token (shared domain): no stripping; a padded or
        # newline-suffixed digest is refused, not normalized.
        problems = w0_sha256_hex_problems(file_sha256, "file_sha256")
        if problems:
            raise ValueError(problems[0])
        fields["file_sha256"] = file_sha256

    if not w0_is_omitted(lineage_event_id):
        # lineage_event_id is an exact UUID reference token supplied by the
        # caller; W0 does not look up the referenced record or commit state.
        # Exact canonical token (shared domain): no stripping.
        problems = w0_uuid_problems(lineage_event_id, "lineage_event_id")
        if problems:
            raise ValueError(problems[0])
        fields["lineage_event_id"] = lineage_event_id

    if locator is not None:
        if not isinstance(locator, dict):
            raise ValueError("locator must be a dict when provided")
        if lineage_record:
            # W0: the raw locator must already be the exact real indexer shape
            # (shared w0_locator_problems domain). Sanitizing an invalid raw
            # locator would erase unknown keys and truncate overlong headings,
            # fabricating the very evidence shape the gate demands — refuse
            # instead. Valid locators serialize verbatim.
            problems = w0_locator_problems(locator)
            if problems:
                raise ValueError("; ".join(problems))
            fields["locator"] = dict(locator)
        else:
            sanitized_locator = _sanitize_source_metadata(locator)
            if sanitized_locator is not None and not isinstance(sanitized_locator, dict):
                raise ValueError("locator must sanitize to a dict")
            if isinstance(sanitized_locator, dict) and sanitized_locator:
                fields["locator"] = sanitized_locator
    return fields



def sanitize_tags(value: Any) -> list[str]:
    """Sanitize and deduplicate a list of tags.

    - Strips whitespace.
    - Rejects empty strings.
    - Rejects tags containing secrets.
    - Rejects tags that don't match the safe slug pattern.
    - Deduplicates while preserving order (case-insensitive).
    - Caps at _MAX_TAGS entries.
    """
    if not isinstance(value, (list, tuple, set)):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text:
            continue
        if contains_secret(text):
            continue
        if not _TAG_RE.match(text):
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= _MAX_TAGS:
            break
    return result


def sanitize_content_hash(value: Any) -> str:
    """Sanitize a content_hash value.

    Returns the validated hash string or empty string if invalid/unsafe.
    """
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if not text:
        return ""
    if contains_secret(text):
        return ""
    if not _CONTENT_HASH_RE.match(text):
        return ""
    return text


def sanitize_profile_id(value: Any) -> str:
    """Sanitize a profile_id value.

    Returns the validated profile_id or "default" if invalid/unsafe.
    """
    if not isinstance(value, str):
        return "default"
    text = value.strip()
    if not text:
        return "default"
    if contains_secret(text):
        return "default"
    if not _PROFILE_ID_RE.match(text):
        return "default"
    return text


# -----------------------------------------------------------------------
# Controlled-timestamp sanitization (created_at / updated_at)
# -----------------------------------------------------------------------

# Allowed pattern for ISO-8601-ish timestamps: dates, times, optional
# fractional seconds, optional 'Z' or +HH:MM offset.  Deliberately
# restrictive so secret-bearing strings (which tend to contain '=', '/',
# spaces, etc.) are always rejected.
_TIMESTAMP_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}[T ][0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:?[0-9]{2})?$"
)


def sanitize_timestamp(value: Any) -> str:
    """Sanitize a controlled timestamp value (created_at / updated_at).

    These fields are caller-supplied and must not persist secret-bearing
    values.  Returns the stripped, validated timestamp string if it matches
    the ISO-8601-ish pattern and does not contain secrets; otherwise returns
    the current UTC timestamp from :func:`now_iso`.
    """
    if not isinstance(value, str):
        return now_iso()
    text = value.strip()
    if not text:
        return now_iso()
    if contains_secret(text):
        return now_iso()
    if not _TIMESTAMP_RE.match(text):
        return now_iso()
    return text


def _filter_reserved_from_extra(extra: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *extra* with all reserved payload keys removed.

    This must be applied *before* ``_sanitize_source_metadata`` so that
    reserved keys (e.g. ``relation_type``) cannot trigger schema-level
    validation errors or bypass direct-field sanitization paths.
    """
    return {
        key: val
        for key, val in extra.items()
        if key not in _RESERVED_PAYLOAD_KEYS
    }


# ---------------------------------------------------------------------------
# Provenance validation
# ---------------------------------------------------------------------------

def _has_safe_provenance(
    *,
    source_point_ids: list[str] | None = None,
    source_uri: str = "",
    content_hash: str = "",
) -> bool:
    """Return True if at least one safe provenance handle is present."""
    if source_point_ids and len(source_point_ids) > 0:
        return True
    if source_uri and not contains_secret(source_uri) and source_uri.strip():
        return True
    ch = sanitize_content_hash(content_hash)
    if ch:
        return True
    return False


# ---------------------------------------------------------------------------
# Weight validation
# ---------------------------------------------------------------------------

def _validate_weight(value: Any, name: str, *, default: float = 0.0) -> float:
    """Validate a float weight in [0.0, 1.0]."""
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite float in [0.0, 1.0]")
    import math
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite float in [0.0, 1.0]")
    return max(0.0, min(1.0, result))


# ---------------------------------------------------------------------------
# Entity record
# ---------------------------------------------------------------------------

@dataclass
class GraphEntity:
    """A typed graph entity record for Qdrant storage.

    Entities are never canonical by default. They represent extracted or
    declared concepts that can participate in graph edges. All entities
    start as ``requires_review=True`` and ``canonical=False``.

    The ``to_payload()`` method produces a JSON-serializable dict compatible
    with existing Qdrant payload conventions and legacy search.
    """

    entity_type: str
    label: str
    aliases: list[str] = field(default_factory=list)
    description: str = ""
    profile_id: str = "default"
    confidence: float = 0.5
    canonical: bool = False
    requires_review: bool = True
    fact_status: str = "active"
    usefulness_weight: float = 0.0
    truth_confidence: float = 0.0
    source_uri: str = ""
    content_hash: str = ""
    source_point_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    # Explicit logical-handle override for structural lineage records whose
    # identity is a bare digest rather than the display label. When set, it is
    # strictly validated and used verbatim as the payload entity_id.
    logical_entity_id: str = ""
    lineage_record: bool = False
    lineage_schema_version: int | None = None
    lineage_operation: str = ""
    lineage_identity_digest: str = ""
    lineage_source_key: str = ""
    lineage_scope_key: str = ""
    lineage_observation: str = ""
    lineage_role: str = ""
    file_path: str = ""

    @property
    def entity_id(self) -> str:
        if self.logical_entity_id:
            return validate_entity_id(self.logical_entity_id)
        return make_entity_id(self.entity_type, self.label, profile_id=self.profile_id)

    def to_payload(self) -> dict[str, Any]:
        """Serialize to a safe, JSON-serializable Qdrant payload dict.

        Raises ValueError if required fields are missing/invalid, if any
        direct field contains secrets, or if provenance is absent.
        """
        etype = (self.entity_type or "").strip().lower()
        if self.lineage_record:
            if not isinstance(self.label, str):
                raise ValueError("label must be a string")
            # Structural identity is exact-token. Preserve malformed-but-safe
            # spellings for the final gate to reject; never strip them into a
            # different, storable label/text pair.
            label_text = self.label
        else:
            label_text = (self.label or "").strip()
        if not etype:
            raise ValueError("entity_type is required")
        if not label_text:
            raise ValueError("label is required")
        if contains_secret(label_text) or contains_secret(etype):
            raise ValueError("entity label/type must not contain secrets")
        if self.lineage_record:
            # W0 exact-token: the raw entity_type must already be the
            # documented token; strip/lower normalization is legacy-only.
            problems = w0_vocabulary_problems(self.entity_type, "entity_type", KNOWN_ENTITY_TYPES)
            if problems:
                raise ValueError(problems[0])
            etype = self.entity_type
        if self.lineage_record and not w0_is_omitted(self.source_uri):
            # Adjudicated W0 serializer contract, hoisted BEFORE the legacy
            # provenance checks touch the value: classify with the shared URI
            # domain. Unsafe values (non-string, secret-bearing) raise;
            # malformed-but-serializable raw strings keep their exact spelling
            # (emitted verbatim below) and fail closed at the final gate.
            if not isinstance(self.source_uri, str) or contains_secret(self.source_uri):
                raise ValueError("source_uri must be a string without secrets")
            w0_uri_problems(self.source_uri, field="source_uri")

        # Sanitize direct user-controlled fields. Structural ownership is an
        # exact gate binding: preserve safe raw profile spellings so this
        # serializer cannot turn a rejected owner into a different owner.
        if self.lineage_record:
            if not isinstance(self.profile_id, str) or contains_secret(self.profile_id):
                raise ValueError("profile_id must be a string without secrets")
            profile_id = self.profile_id
        else:
            profile_id = sanitize_profile_id(self.profile_id)
        tags = sanitize_tags(self.tags)
        content_hash = sanitize_content_hash(self.content_hash)
        if self.lineage_record and not w0_is_omitted(self.content_hash):
            # W0 shared hash domain, exact token: refuse before the legacy lax
            # sanitizer can strip a padded or newline-suffixed digest into a
            # storable one.
            problems = w0_content_hash_problems(self.content_hash, "content_hash")
            if problems:
                raise ValueError(problems[0])

        # Provenance requirement: must have at least one safe provenance handle
        source_points = sanitize_source_point_ids(
            getattr(self, "source_point_ids", None)
        )
        has_uri = bool(
            self.source_uri
            and not contains_secret(self.source_uri)
            and self.source_uri.strip()
        )
        if not _has_safe_provenance(
            source_point_ids=source_points,
            source_uri=self.source_uri,
            content_hash=content_hash,
        ):
            raise ValueError(
                "entity requires provenance via source_point_ids, "
                "source_uri, or content_hash"
            )

        eid = make_entity_id(self.entity_type, self.label, profile_id=profile_id)
        if self.logical_entity_id:
            if self.lineage_record:
                # Adjudicated W0 serializer contract: the shared handle domain
                # is classified here, BEFORE any legacy normalizer (the legacy
                # strip validator is bypassed entirely). Unsafe values
                # (non-string, secret-bearing) raise; a malformed-but-
                # serializable raw string keeps its exact spelling so the
                # final gate is the enforcing write boundary that refuses it.
                if not isinstance(self.logical_entity_id, str) or contains_secret(self.logical_entity_id):
                    raise ValueError("logical_entity_id must be a string without secrets")
                w0_logical_handle_problems(self.logical_entity_id, field="logical_entity_id")
                eid = self.logical_entity_id
            else:
                eid = validate_entity_id(self.logical_entity_id)
        payload: dict[str, Any] = {
            "text": label_text,  # legacy search compatibility
            "source": "graph_entity",
            "source_type": "graph",
            "chunk_type": "entity",
            "memory_kind": MemoryKind.GRAPH_ENTITY.value,
            "entity_id": eid,
            "entity_type": etype,
            "label": label_text,
            "confidence": _validate_weight(self.confidence, "confidence", default=0.5),
            "canonical": False,  # NEVER auto-promote
            "requires_review": True,  # ALWAYS review-gated by default
            "usefulness_weight": _validate_weight(self.usefulness_weight, "usefulness_weight"),
            "truth_confidence": _validate_weight(self.truth_confidence, "truth_confidence"),
            "created_at": sanitize_timestamp(self.created_at),
            "updated_at": sanitize_timestamp(self.updated_at),
            "profile_id": profile_id,
            "tags": tags,
        }

        # Validate fact_status
        fs = valid_fact_status(self.fact_status)
        if fs:
            payload["fact_status"] = fs
        else:
            payload["fact_status"] = FactStatus.ACTIVE.value

        # Aliases
        aliases = sanitize_aliases(self.aliases)
        if aliases:
            payload["aliases"] = aliases

        # Description (sanitized)
        if self.description and not contains_secret(self.description):
            desc = self.description.strip()
            if desc:
                payload["description"] = desc

        # Provenance
        if self.lineage_record:
            # Already classified above (shared URI domain); the value, when
            # present, serializes verbatim — never stripped — so the final
            # gate refuses malformed raw spellings.
            if not w0_is_omitted(self.source_uri):
                payload["source_uri"] = self.source_uri
        elif self.source_uri and not contains_secret(self.source_uri):
            uri = self.source_uri.strip()
            if uri:
                payload["source_uri"] = uri
        if content_hash:
            payload["content_hash"] = content_hash
        if source_points:
            payload["source_point_ids"] = source_points

        # Explicit validated lineage metadata (never injectable via extra).
        payload.update(
            _validated_lineage_entity_fields(
                lineage_record=self.lineage_record,
                lineage_schema_version=self.lineage_schema_version,
                lineage_operation=self.lineage_operation,
                lineage_identity_digest=self.lineage_identity_digest,
                lineage_source_key=self.lineage_source_key,
                lineage_scope_key=self.lineage_scope_key,
                lineage_observation=self.lineage_observation,
                lineage_role=self.lineage_role,
                file_path=self.file_path,
            )
        )

        # Extra metadata (sanitized, reserved keys always excluded)
        if self.extra:
            safe_extra = _filter_reserved_from_extra(self.extra)
            if safe_extra:
                sanitized_extra = _sanitize_source_metadata(safe_extra)
                if isinstance(sanitized_extra, dict):
                    for key, val in sanitized_extra.items():
                        if key in _RESERVED_PAYLOAD_KEYS:
                            continue
                        if key not in payload and not _metadata_is_empty(val):
                            payload[key] = val

        return payload


# ---------------------------------------------------------------------------
# Edge record
# ---------------------------------------------------------------------------

@dataclass
class GraphEdge:
    """A typed graph edge record for Qdrant storage.

    Edges represent typed relationships between entities. Legacy non-lineage
    edges require their own generic provenance handle. Structural edges may
    instead carry endpoint provenance for the mechanical gate. Edges are never
    canonical by default.

    ``usefulness_weight`` tracks answer/session utility signals.
    ``truth_confidence`` tracks evidence/provenance confidence.
    These two are deliberately separate and must not be conflated.
    """

    source_entity_id: str
    target_entity_id: str
    relation_type: str
    profile_id: str = "default"
    source_point_ids: list[str] = field(default_factory=list)
    confidence: float = 0.5
    usefulness_weight: float = 0.0
    truth_confidence: float = 0.0
    canonical: bool = False
    requires_review: bool = True
    fact_status: str = "active"
    source_uri: str = ""
    content_hash: str = ""
    observed_at: str = ""
    valid_from: str = ""
    valid_until: str = ""
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    # Explicit validated lineage/mechanical fields (never injectable via extra).
    lineage_record: bool = False
    lineage_schema_version: int | None = None
    lineage_operation: str = ""
    lineage_identity_digest: str = ""
    lineage_source_key: str = ""
    lineage_scope_key: str = ""
    lineage_observation: str = ""
    lineage_event_id: str = ""
    edge_class: str = ""
    source_point_id: str = ""
    target_point_id: str = ""
    source_entity_type: str = ""
    target_entity_type: str = ""
    source_content_hash: str = ""
    target_content_hash: str = ""
    file_version_id: str = ""
    file_path: str = ""
    file_sha256: str = ""
    locator: dict[str, Any] | None = None

    @property
    def edge_id(self) -> str:
        return make_edge_id(
            self.source_entity_id,
            self.target_entity_id,
            self.relation_type,
            profile_id=self.profile_id,
        )

    def to_payload(self) -> dict[str, Any]:
        """Serialize to a safe, JSON-serializable Qdrant payload dict.

        Raises ValueError if required fields are missing/invalid, or if a
        non-lineage edge has no generic provenance handle.
        """
        if self.lineage_record:
            # Adjudicated W0 serializer contract: the shared handle domain is
            # classified here, BEFORE any legacy normalizer (the legacy strip
            # validator is bypassed entirely). Unsafe values (non-string,
            # secret-bearing) raise; a malformed-but-serializable raw string
            # keeps its exact spelling so the final gate is the enforcing
            # write boundary that refuses it.
            for handle_value, handle_field in (
                (self.source_entity_id, "source_entity_id"),
                (self.target_entity_id, "target_entity_id"),
            ):
                if not isinstance(handle_value, str) or contains_secret(handle_value):
                    raise ValueError(f"{handle_field} must be a string without secrets")
                w0_logical_handle_problems(handle_value, field=handle_field)
            src = self.source_entity_id
            tgt = self.target_entity_id
            # W0 exact-token relation vocabulary: the raw relation must
            # already be the documented token (legacy strip normalization
            # would turn ' summarizes ' into the storable 'SUMMARIZES').
            problems = w0_vocabulary_problems(self.relation_type, "relation_type", RELATION_TYPES)
            if problems:
                raise ValueError(problems[0])
            rel = self.relation_type
            if not w0_is_omitted(self.source_uri):
                # Adjudicated W0 serializer contract, hoisted BEFORE the
                # legacy provenance checks touch the value: classify with the
                # shared URI domain. Unsafe values raise; malformed-but-
                # serializable raw strings keep their exact spelling (emitted
                # verbatim below) and fail closed at the final gate.
                if not isinstance(self.source_uri, str) or contains_secret(self.source_uri):
                    raise ValueError("source_uri must be a string without secrets")
                w0_uri_problems(self.source_uri, field="source_uri")
        else:
            src = validate_entity_id(self.source_entity_id)
            tgt = validate_entity_id(self.target_entity_id)
            rel = valid_relation_type(self.relation_type)
            if rel is None:
                raise ValueError(f"unknown relation_type: {self.relation_type or '<empty>'}")

        # Sanitize direct user-controlled fields. Structural ownership is an
        # exact gate binding: preserve safe raw profile spellings so this
        # serializer cannot turn a rejected owner into a different owner.
        if self.lineage_record:
            if not isinstance(self.profile_id, str) or contains_secret(self.profile_id):
                raise ValueError("profile_id must be a string without secrets")
            profile_id = self.profile_id
        else:
            profile_id = sanitize_profile_id(self.profile_id)
        tags = sanitize_tags(self.tags)
        content_hash = sanitize_content_hash(self.content_hash)
        if self.lineage_record and not w0_is_omitted(self.content_hash):
            # W0 shared hash domain, exact token: refuse before the legacy lax
            # sanitizer can strip a padded or newline-suffixed digest into a
            # storable one.
            problems = w0_content_hash_problems(self.content_hash, "content_hash")
            if problems:
                raise ValueError(problems[0])

        # Provenance requirement: must have at least one source
        source_points = sanitize_source_point_ids(self.source_point_ids)
        has_uri = bool(self.source_uri and not contains_secret(self.source_uri) and self.source_uri.strip())
        if not self.lineage_record and not _has_safe_provenance(
            source_point_ids=source_points,
            source_uri=self.source_uri,
            content_hash=content_hash,
        ):
            raise ValueError(
                "edge requires provenance via source_point_ids, source_uri, or content_hash"
            )

        eid = make_edge_id(
            self.source_entity_id,
            self.target_entity_id,
            self.relation_type,
            profile_id=profile_id,
        )
        payload: dict[str, Any] = {
            "text": f"{src} {rel} {tgt}",  # legacy search compatibility
            "source": "graph_edge",
            "source_type": "graph",
            "chunk_type": "edge",
            "memory_kind": MemoryKind.GRAPH_EDGE.value,
            "edge_id": eid,
            "source_entity_id": src,
            "target_entity_id": tgt,
            "relation_type": rel,
            "confidence": _validate_weight(self.confidence, "confidence", default=0.5),
            "canonical": False,  # NEVER auto-promote
            "requires_review": True,  # ALWAYS review-gated by default
            "usefulness_weight": _validate_weight(self.usefulness_weight, "usefulness_weight"),
            "truth_confidence": _validate_weight(self.truth_confidence, "truth_confidence"),
            "created_at": sanitize_timestamp(self.created_at),
            "updated_at": sanitize_timestamp(self.updated_at),
            "profile_id": profile_id,
            "tags": tags,
        }

        # Validate fact_status
        fs = valid_fact_status(self.fact_status)
        if fs:
            payload["fact_status"] = fs
        else:
            payload["fact_status"] = FactStatus.ACTIVE.value

        # Source point IDs
        if source_points:
            payload["source_point_ids"] = source_points

        # Provenance URI
        if self.lineage_record:
            # Already classified above (shared URI domain); the value, when
            # present, serializes verbatim — never stripped — so the final
            # gate refuses malformed raw spellings.
            if not w0_is_omitted(self.source_uri):
                payload["source_uri"] = self.source_uri
        elif has_uri:
            payload["source_uri"] = self.source_uri.strip()
        if content_hash:
            payload["content_hash"] = content_hash

        # Temporal fields
        for key in ("observed_at", "valid_from", "valid_until"):
            val = getattr(self, key, "")
            if val and isinstance(val, str):
                sanitized = _sanitize_source_metadata(val)
                if isinstance(sanitized, str) and sanitized:
                    payload[key] = sanitized

        # Explicit validated lineage/mechanical metadata (never via extra).
        payload.update(
            _validated_lineage_edge_fields(
                lineage_record=self.lineage_record,
                lineage_schema_version=self.lineage_schema_version,
                lineage_operation=self.lineage_operation,
                lineage_identity_digest=self.lineage_identity_digest,
                lineage_source_key=self.lineage_source_key,
                lineage_scope_key=self.lineage_scope_key,
                lineage_observation=self.lineage_observation,
                lineage_event_id=self.lineage_event_id,
                edge_class=self.edge_class,
                source_point_id=self.source_point_id,
                target_point_id=self.target_point_id,
                source_entity_type=self.source_entity_type,
                target_entity_type=self.target_entity_type,
                source_content_hash=self.source_content_hash,
                target_content_hash=self.target_content_hash,
                file_version_id=self.file_version_id,
                file_path=self.file_path,
                file_sha256=self.file_sha256,
                locator=self.locator,
            )
        )

        # Extra metadata (sanitized, reserved keys always excluded)
        if self.extra:
            safe_extra = _filter_reserved_from_extra(self.extra)
            if safe_extra:
                sanitized_extra = _sanitize_source_metadata(safe_extra)
                if isinstance(sanitized_extra, dict):
                    for key, val in sanitized_extra.items():
                        if key in _RESERVED_PAYLOAD_KEYS:
                            continue
                        if key not in payload and not _metadata_is_empty(val):
                            payload[key] = val

        return payload


# ---------------------------------------------------------------------------
# Convenience builder functions
# ---------------------------------------------------------------------------

def build_entity_payload(
    *,
    entity_type: str,
    label: str,
    aliases: list[str] | None = None,
    description: str = "",
    profile_id: str = "default",
    confidence: float = 0.5,
    fact_status: str = "active",
    usefulness_weight: float = 0.0,
    truth_confidence: float = 0.0,
    source_uri: str = "",
    content_hash: str = "",
    source_point_ids: list[str] | None = None,
    tags: list[str] | None = None,
    extra: dict[str, Any] | None = None,
    created_at: str | None = None,
    logical_entity_id: str = "",
    lineage_record: bool = False,
    lineage_schema_version: int | None = None,
    lineage_operation: str = "",
    lineage_identity_digest: str = "",
    lineage_source_key: str = "",
    lineage_scope_key: str = "",
    lineage_observation: str = "",
    lineage_role: str = "",
    file_path: str = "",
) -> dict[str, Any]:
    """Build a graph entity payload dict without writing it anywhere.

    Entities are always ``canonical=False`` and ``requires_review=True``.
    Provenance is required: at least one of ``source_uri``, ``content_hash``,
    or ``source_point_ids`` must be provided and pass sanitization.
    """
    entity = GraphEntity(
        entity_type=entity_type,
        label=label,
        aliases=aliases or [],
        description=description,
        profile_id=profile_id,
        confidence=confidence,
        fact_status=fact_status,
        usefulness_weight=usefulness_weight,
        truth_confidence=truth_confidence,
        source_uri=source_uri,
        content_hash=content_hash,
        source_point_ids=source_point_ids or [],
        tags=tags or [],
        extra=extra or {},
        created_at=created_at or now_iso(),
        logical_entity_id=logical_entity_id,
        lineage_record=lineage_record,
        lineage_schema_version=lineage_schema_version,
        lineage_operation=lineage_operation,
        lineage_identity_digest=lineage_identity_digest,
        lineage_source_key=lineage_source_key,
        lineage_scope_key=lineage_scope_key,
        lineage_observation=lineage_observation,
        lineage_role=lineage_role,
        file_path=file_path,
    )
    return entity.to_payload()


def build_edge_payload(
    *,
    source_entity_id: str,
    target_entity_id: str,
    relation_type: str,
    profile_id: str = "default",
    source_point_ids: list[str] | None = None,
    confidence: float = 0.5,
    usefulness_weight: float = 0.0,
    truth_confidence: float = 0.0,
    fact_status: str = "active",
    source_uri: str = "",
    content_hash: str = "",
    observed_at: str = "",
    valid_from: str = "",
    valid_until: str = "",
    tags: list[str] | None = None,
    extra: dict[str, Any] | None = None,
    created_at: str | None = None,
    lineage_record: bool = False,
    lineage_schema_version: int | None = None,
    lineage_operation: str = "",
    lineage_identity_digest: str = "",
    lineage_source_key: str = "",
    lineage_scope_key: str = "",
    lineage_observation: str = "",
    lineage_event_id: str = "",
    edge_class: str = "",
    source_point_id: str = "",
    target_point_id: str = "",
    source_entity_type: str = "",
    target_entity_type: str = "",
    source_content_hash: str = "",
    target_content_hash: str = "",
    file_version_id: str = "",
    file_path: str = "",
    file_sha256: str = "",
    locator: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a graph edge payload dict without writing it anywhere.

    Edges are always ``canonical=False`` and ``requires_review=True``.
    Non-lineage edges require at least one of ``source_point_ids``,
    ``source_uri``, or ``content_hash``. Structural edges may omit those
    generic handles because the final mechanical gate validates their endpoint
    provenance.
    """
    edge = GraphEdge(
        source_entity_id=source_entity_id,
        target_entity_id=target_entity_id,
        relation_type=relation_type,
        profile_id=profile_id,
        source_point_ids=source_point_ids or [],
        confidence=confidence,
        usefulness_weight=usefulness_weight,
        truth_confidence=truth_confidence,
        fact_status=fact_status,
        source_uri=source_uri,
        content_hash=content_hash,
        observed_at=observed_at,
        valid_from=valid_from,
        valid_until=valid_until,
        tags=tags or [],
        extra=extra or {},
        created_at=created_at or now_iso(),
        lineage_record=lineage_record,
        lineage_schema_version=lineage_schema_version,
        lineage_operation=lineage_operation,
        lineage_identity_digest=lineage_identity_digest,
        lineage_source_key=lineage_source_key,
        lineage_scope_key=lineage_scope_key,
        lineage_observation=lineage_observation,
        lineage_event_id=lineage_event_id,
        edge_class=edge_class,
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
    return edge.to_payload()
