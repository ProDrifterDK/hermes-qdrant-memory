"""Graph entity and edge schema primitives for the Qdrant memory plugin.

This module provides typed helpers for building, validating, and serializing
graph-aware memory payloads (entities and edges) that are stored as Qdrant
points alongside regular memory chunks. All payloads are backward-compatible
with existing search/ranking, and all entity/edge creation is non-canonical
and review-gated by default.

Safety invariants enforced here:
- Entity/edge IDs are deterministic, safe (no secrets), and validated.
- No entity or edge is ever auto-promoted to canonical truth.
- Secret-bearing fields are redacted/rejected.
- Provenance is required for edges.
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
    """Sanitize a list of source point IDs for an edge.

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
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def entity_id(self) -> str:
        return make_entity_id(self.entity_type, self.label, profile_id=self.profile_id)

    def to_payload(self) -> dict[str, Any]:
        """Serialize to a safe, JSON-serializable Qdrant payload dict."""
        etype = (self.entity_type or "").strip().lower()
        label_text = (self.label or "").strip()
        if not etype:
            raise ValueError("entity_type is required")
        if not label_text:
            raise ValueError("label is required")
        if contains_secret(label_text) or contains_secret(etype):
            raise ValueError("entity label/type must not contain secrets")

        eid = self.entity_id
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
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "profile_id": self.profile_id,
            "tags": list(self.tags),
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
        if self.content_hash:
            payload["content_hash"] = self.content_hash

        # Extra metadata (sanitized)
        if self.extra:
            sanitized_extra = _sanitize_source_metadata(self.extra)
            if isinstance(sanitized_extra, dict):
                for key, val in sanitized_extra.items():
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

        # Provenance requirement: must have at least one source
        source_points = sanitize_source_point_ids(self.source_point_ids)
        has_uri = bool(self.source_uri and not contains_secret(self.source_uri) and self.source_uri.strip())
        if not source_points and not has_uri:
            raise ValueError(
                "edge requires provenance via source_point_ids or source_uri"
            )

        payload: dict[str, Any] = {
            "text": f"{src} {rel} {tgt}",  # legacy search compatibility
            "source": "graph_edge",
            "source_type": "graph",
            "chunk_type": "edge",
            "memory_kind": MemoryKind.GRAPH_EDGE.value,
            "edge_id": self.edge_id,
            "source_entity_id": src,
            "target_entity_id": tgt,
            "relation_type": rel,
            "confidence": _validate_weight(self.confidence, "confidence", default=0.5),
            "canonical": False,  # NEVER auto-promote
            "requires_review": True,  # ALWAYS review-gated by default
            "usefulness_weight": _validate_weight(self.usefulness_weight, "usefulness_weight"),
            "truth_confidence": _validate_weight(self.truth_confidence, "truth_confidence"),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "profile_id": self.profile_id,
            "tags": list(self.tags),
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
        if self.content_hash:
            payload["content_hash"] = self.content_hash

        # Temporal fields
        for key in ("observed_at", "valid_from", "valid_until"):
            val = getattr(self, key, "")
            if val and isinstance(val, str):
                sanitized = _sanitize_source_metadata(val)
                if isinstance(sanitized, str) and sanitized:
                    payload[key] = sanitized

        # Extra metadata (sanitized)
        if self.extra:
            sanitized_extra = _sanitize_source_metadata(self.extra)
            if isinstance(sanitized_extra, dict):
                for key, val in sanitized_extra.items():
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
    tags: list[str] | None = None,
    extra: dict[str, Any] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a graph entity payload dict without writing it anywhere.

    Entities are always ``canonical=False`` and ``requires_review=True``.
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
    Provenance is required (source_point_ids or source_uri).
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
