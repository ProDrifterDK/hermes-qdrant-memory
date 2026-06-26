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
import re
from dataclasses import dataclass, field
from typing import Any

from .lesson_extractor import contains_secret
from .schema import (
    FactStatus,
    MemoryKind,
    RelationType,
    _metadata_is_empty,
    _sanitize_source_metadata,
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


# ---------------------------------------------------------------------------
# ID generation and validation
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    """Normalize text into a safe slug component for ID generation."""
    slug = _ENTITY_SLUG_RE.sub("-", (text or "").strip().lower()).strip("-")
    return slug


def make_entity_id(entity_type: str, label: str, *, profile_id: str = "default") -> str:
    """Generate a deterministic, safe entity ID.

    The ID is derived from entity_type, label, and profile_id via SHA-256.
    It never contains raw user text or secrets — only a hex digest.

    Raises ValueError if entity_type or label are empty.
    """
    etype = (entity_type or "").strip().lower()
    label_text = (label or "").strip()
    if not etype:
        raise ValueError("entity_type is required for entity ID generation")
    if not label_text:
        raise ValueError("label is required for entity ID generation")
    raw = f"entity|{profile_id}|{etype}|{_slugify(label_text)}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
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
})


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

    @property
    def entity_id(self) -> str:
        return make_entity_id(self.entity_type, self.label, profile_id=self.profile_id)

    def to_payload(self) -> dict[str, Any]:
        """Serialize to a safe, JSON-serializable Qdrant payload dict.

        Raises ValueError if required fields are missing/invalid, if any
        direct field contains secrets, or if provenance is absent.
        """
        etype = (self.entity_type or "").strip().lower()
        label_text = (self.label or "").strip()
        if not etype:
            raise ValueError("entity_type is required")
        if not label_text:
            raise ValueError("label is required")
        if contains_secret(label_text) or contains_secret(etype):
            raise ValueError("entity label/type must not contain secrets")

        # Sanitize direct user-controlled fields
        profile_id = sanitize_profile_id(self.profile_id)
        tags = sanitize_tags(self.tags)
        content_hash = sanitize_content_hash(self.content_hash)

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
        if self.source_uri and not contains_secret(self.source_uri):
            uri = self.source_uri.strip()
            if uri:
                payload["source_uri"] = uri
        if content_hash:
            payload["content_hash"] = content_hash
        if source_points:
            payload["source_point_ids"] = source_points

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

    Edges represent typed relationships between entities. They always
    carry provenance (source_point_ids or source_uri), and are never
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

        Raises ValueError if required fields are missing/invalid or if
        provenance is absent.
        """
        src = validate_entity_id(self.source_entity_id)
        tgt = validate_entity_id(self.target_entity_id)
        rel = valid_relation_type(self.relation_type)
        if rel is None:
            raise ValueError(f"unknown relation_type: {self.relation_type or '<empty>'}")

        # Sanitize direct user-controlled fields
        profile_id = sanitize_profile_id(self.profile_id)
        tags = sanitize_tags(self.tags)
        content_hash = sanitize_content_hash(self.content_hash)

        # Provenance requirement: must have at least one source
        source_points = sanitize_source_point_ids(self.source_point_ids)
        has_uri = bool(self.source_uri and not contains_secret(self.source_uri) and self.source_uri.strip())
        if not _has_safe_provenance(
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
        if has_uri:
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
) -> dict[str, Any]:
    """Build a graph edge payload dict without writing it anywhere.

    Edges are always ``canonical=False`` and ``requires_review=True``.
    Provenance is required: at least one of ``source_point_ids``,
    ``source_uri``, or ``content_hash`` must be provided and pass sanitization.
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
    )
    return edge.to_payload()
