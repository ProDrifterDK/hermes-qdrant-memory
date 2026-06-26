from __future__ import annotations

import hashlib
import math
import re
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .lesson_extractor import contains_secret


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_point_id(source: str, text: str) -> str:
    digest = hashlib.sha256(f"{source}\n{text}".encode("utf-8")).hexdigest()
    return str(uuid.UUID(digest[:32]))


class MemoryKind(str, Enum):
    CONVERSATION_TURN = "conversation_turn"
    MANUAL_FACT = "manual_fact"
    SOURCE_CHUNK = "source_chunk"
    LEARNING = "learning"
    ASSERTION = "assertion"
    DECISION = "decision"
    USER_PREFERENCE = "user_preference"
    PROJECT_INVARIANT = "project_invariant"
    TOOL_QUIRK = "tool_quirk"
    WORKFLOW_LESSON = "workflow_lesson"
    RISK = "risk"
    PROPOSAL = "proposal"
    SUMMARY = "summary"
    GRAPH_ENTITY = "graph_entity"
    GRAPH_EDGE = "graph_edge"


class RelationType(str, Enum):
    DERIVED_FROM = "DERIVED_FROM"
    EXTRACTED_FROM = "EXTRACTED_FROM"
    SUMMARIZES = "SUMMARIZES"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    SUPERSEDES = "SUPERSEDES"
    REFERENCES = "REFERENCES"
    APPLIES_TO = "APPLIES_TO"
    USES_TOOL = "USES_TOOL"
    PREFERS = "PREFERS"
    BLOCKS = "BLOCKS"
    IS_A = "IS_A"
    PART_OF = "PART_OF"
    RELATED_TO = "RELATED_TO"
    CREATED_BY = "CREATED_BY"
    DEPENDS_ON = "DEPENDS_ON"
    LOCATED_IN = "LOCATED_IN"


class FactStatus(str, Enum):
    ACTIVE = "active"
    STALE = "stale"
    DEPRECATED = "deprecated"
    DISPUTED = "disputed"
    SUPERSEDED = "superseded"
    REVIEW_REQUIRED = "review_required"


MEMORY_KINDS = tuple(kind.value for kind in MemoryKind)
RELATION_TYPES = tuple(relation.value for relation in RelationType)
FACT_STATUSES = tuple(status.value for status in FactStatus)
_MEMORY_KIND_SET = set(MEMORY_KINDS)
_RELATION_TYPE_SET = set(RELATION_TYPES)
_FACT_STATUS_SET = set(FACT_STATUSES)
_POINT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")


def _grammar_value(value: Any) -> str:
    if isinstance(value, Enum):
        return str(value.value).strip()
    return str(value or "").strip()


def valid_memory_kind(value: Any) -> str | None:
    text = _grammar_value(value)
    return text if text in _MEMORY_KIND_SET else None


def validate_memory_kind(value: Any) -> str:
    text = _grammar_value(value)
    if text not in _MEMORY_KIND_SET:
        raise ValueError(f"unknown memory_kind: {text or '<empty>'}")
    return text


def valid_relation_type(value: Any) -> str | None:
    text = _grammar_value(value)
    return text if text in _RELATION_TYPE_SET else None


def validate_relation_type(value: Any) -> str:
    text = _grammar_value(value)
    if text not in _RELATION_TYPE_SET:
        raise ValueError(f"unknown relation_type: {text or '<empty>'}")
    return text


def valid_fact_status(value: Any) -> str | None:
    text = _grammar_value(value)
    return text if text in _FACT_STATUS_SET else None


def validate_fact_status(value: Any) -> str:
    text = _grammar_value(value)
    if text not in _FACT_STATUS_SET:
        raise ValueError(f"unknown fact_status: {text or '<empty>'}")
    return text


def valid_point_id_link(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or "://" in text or "/" in text or "\\" in text or not _POINT_ID_RE.match(text):
        return None
    if contains_secret(text):
        return None
    return text


def sanitize_point_id_links(value: Any, *, max_links: int = 32) -> list[str]:
    if not isinstance(value, list):
        return []
    links: list[str] = []
    seen: set[str] = set()
    for item in value[:max_links]:
        point_id = valid_point_id_link(item)
        if point_id and point_id not in seen:
            links.append(point_id)
            seen.add(point_id)
    return links


def validate_point_id_links(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of explicit point IDs")
    links: list[str] = []
    seen: set[str] = set()
    for item in value:
        point_id = valid_point_id_link(item)
        if not point_id:
            raise ValueError(f"{field_name} entries must be explicit point IDs")
        if point_id not in seen:
            links.append(point_id)
            seen.add(point_id)
    return links


@dataclass
class SourceLocator:
    line_start: int | None = None
    line_end: int | None = None
    heading: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.line_start is not None:
            data["line_start"] = int(self.line_start)
        if self.line_end is not None:
            data["line_end"] = int(self.line_end)
        if self.heading:
            data["heading"] = self.heading
        for key, value in self.extra.items():
            if key not in data:
                data[key] = value
        sanitized = _sanitize_source_metadata(data)
        return sanitized if isinstance(sanitized, dict) else {}


@dataclass
class DerivationEdge:
    source_uri: str = ""
    locator: SourceLocator | dict[str, Any] | None = None
    derivation_type: str = ""
    relation_type: str | RelationType = ""
    source_type: str = ""
    content_hash: str = ""
    source_modified_at: str = ""

    def to_payload(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "source_uri": self.source_uri,
            "source_type": self.source_type,
            "locator": self.locator,
            "content_hash": self.content_hash,
            "source_modified_at": self.source_modified_at,
            "derivation_type": self.derivation_type,
            "relation_type": self.relation_type,
        }
        sanitized = _sanitize_source_metadata(data)
        return sanitized if isinstance(sanitized, dict) else {}


@dataclass
class SourceReference:
    source_uri: str = ""
    source_type: str = ""
    locator: SourceLocator | dict[str, Any] | None = None
    content_hash: str = ""
    source_modified_at: str = ""
    derivation_type: str = ""
    relation_type: str | RelationType = ""
    derived_from: list[DerivationEdge | SourceReference | dict[str, Any] | str] = field(default_factory=list)
    canonical: bool | None = None
    stale: bool | None = None
    requires_review: bool | None = None

    def to_payload(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "source_uri": self.source_uri,
            "source_type": self.source_type,
            "locator": self.locator,
            "content_hash": self.content_hash,
            "source_modified_at": self.source_modified_at,
            "derivation_type": self.derivation_type,
            "relation_type": self.relation_type,
            "derived_from": self.derived_from,
            "canonical": self.canonical,
            "stale": self.stale,
            "requires_review": self.requires_review,
        }
        sanitized = _sanitize_source_metadata(data)
        return sanitized if isinstance(sanitized, dict) else {}


@dataclass
class MemoryChunk:
    text: str
    source: str
    source_type: str = "manual"
    chunk_type: str = "fact"
    importance: int = 5
    confidence: float = 1.0
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=now_iso)
    last_accessed: str = field(default_factory=now_iso)
    profile_id: str = "default"
    platform: str = "cli"
    session_id: str = ""
    point_id: str = ""

    def id(self) -> str:
        return self.point_id or make_point_id(self.source, self.text)


_HEADING_RE = re.compile(r"^#\s+(Relevant Long-Term Memory|Past Learnings)\b", re.IGNORECASE | re.MULTILINE)
_TRIVIAL_RE = re.compile(r"^(ok|okay|thanks|thank you|got it|sure|yes|no|yep|nope|k|ty|thx|np)\.?$", re.I)


def _strip_marked_blocks(text: str) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    out: list[str] = []
    skipping = False
    for line in lines:
        if re.match(r"^#\s+(Relevant Long-Term Memory|Past Learnings)\b", line.strip(), re.I):
            skipping = True
            continue
        if skipping:
            # Stop when a new top-level heading that is not provider context appears.
            if line.startswith("# ") and not re.match(r"^#\s+(Relevant Long-Term Memory|Past Learnings)\b", line.strip(), re.I):
                skipping = False
                out.append(line)
            continue
        out.append(line)
    return "\n".join(out)


def clean_text_for_memory(text: str) -> str:
    text = _strip_marked_blocks(text or "")
    text = re.sub(r"```qdrant-memory[\s\S]*?```", "", text, flags=re.I)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def score_importance(text: str, source_type: str = "conversation") -> int:
    lowered = (text or "").lower()
    if _TRIVIAL_RE.match(lowered.strip()):
        return 1
    score = 5
    if source_type in {"manual", "builtin_memory", "user_profile"}:
        score += 2
    if any(word in lowered for word in ("remember", "important", "decision", "prefer", "correction", "never", "always")):
        score += 2
    if len(text or "") < 40 and score < 8:
        score -= 2
    return max(1, min(10, score))


def _metadata_is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


_SOURCE_REF_PAYLOAD_KEYS = {
    "source_uri",
    "locator",
    "content_hash",
    "source_modified_at",
    "derivation_type",
    "relation_type",
    "derived_from",
    "canonical",
    "stale",
    "requires_review",
    "observed_at",
    "valid_from",
    "valid_until",
    "fact_status",
    "supersedes",
    "superseded_by",
    "invalidated_by",
}


def _sanitize_source_metadata(value: Any) -> Any:
    """Normalize provenance metadata into JSON-serializable, secret-free values."""
    if hasattr(value, "to_payload") and callable(getattr(value, "to_payload")):
        value = value.to_payload()
    elif is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)

    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or contains_secret(stripped):
            return None
        return stripped
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, dict):
        sanitized_dict: dict[str, Any] = {}
        for key, item in value.items():
            key_s = str(key).strip()
            if not key_s or contains_secret(key_s):
                continue
            if key_s == "relation_type":
                if _metadata_is_empty(item):
                    continue
                sanitized_item = validate_relation_type(item)
            else:
                sanitized_item = _sanitize_source_metadata(item)
            if not _metadata_is_empty(sanitized_item):
                sanitized_dict[key_s] = sanitized_item
        return sanitized_dict or None
    if isinstance(value, (list, tuple, set)):
        sanitized_list = []
        for item in value:
            sanitized_item = _sanitize_source_metadata(item)
            if not _metadata_is_empty(sanitized_item):
                sanitized_list.append(sanitized_item)
        return sanitized_list or None

    as_string = str(value).strip()
    if not as_string or contains_secret(as_string):
        return None
    return as_string


def _add_source_metadata(payload: dict[str, Any], key: str, value: Any) -> None:
    sanitized = _sanitize_source_metadata(value)
    if not _metadata_is_empty(sanitized):
        payload[key] = sanitized


def _add_temporal_string(payload: dict[str, Any], key: str, value: Any) -> None:
    sanitized = _sanitize_source_metadata(value)
    if isinstance(sanitized, str) and sanitized:
        payload[key] = sanitized


def build_payload(
    *,
    text: str,
    source: str,
    source_type: str = "manual",
    chunk_type: str = "fact",
    importance: int | None = None,
    confidence: float = 1.0,
    tags: list[str] | None = None,
    profile_id: str = "default",
    platform: str = "cli",
    user_id_hash: str = "",
    chat_id_hash: str = "",
    session_id: str = "",
    project_path: str = "",
    model: str = "",
    provider: str = "qdrant",
    memory_kind: str | MemoryKind | None = None,
    created_at: str | None = None,
    fact_metadata: dict[str, Any] | None = None,
    source_ref: SourceReference | dict[str, Any] | None = None,
    source_uri: str | None = None,
    locator: SourceLocator | dict[str, Any] | None = None,
    content_hash: str | None = None,
    source_modified_at: str | None = None,
    derivation_type: str | None = None,
    derived_from: list[DerivationEdge | SourceReference | dict[str, Any] | str] | None = None,
    canonical: bool | None = None,
    stale: bool | None = None,
    requires_review: bool | None = None,
    observed_at: str | None = None,
    valid_from: str | None = None,
    valid_until: str | None = None,
    fact_status: str | FactStatus | None = None,
    supersedes: list[Any] | None = None,
    superseded_by: list[Any] | None = None,
    invalidated_by: list[Any] | None = None,
) -> dict[str, Any]:
    created = created_at or now_iso()
    imp = importance if importance is not None else score_importance(text, source_type)
    payload = {
        "text": text,
        "source": source,
        "source_type": source_type,
        "chunk_type": chunk_type,
        "importance": max(1, min(10, int(imp))),
        "confidence": float(confidence),
        "access_count": 0,
        "created_at": created,
        "last_accessed": created,
        "decay_score": 1.0,
        "tags": list(tags or []),
        "profile_id": profile_id,
        "platform": platform,
        "user_id_hash": user_id_hash,
        "chat_id_hash": chat_id_hash,
        "session_id": session_id,
        "project_path": project_path,
        "model": model,
        "provider": provider,
    }
    for key, value in (fact_metadata or {}).items():
        if key in {"fact_key", "subject", "topic", "entity", "reconsolidation_key"} and value and not contains_secret(str(value)):
            payload[key] = value

    if not _metadata_is_empty(memory_kind):
        payload["memory_kind"] = validate_memory_kind(memory_kind)

    if source_ref:
        sanitized_ref = _sanitize_source_metadata(source_ref)
        if isinstance(sanitized_ref, dict):
            for key in _SOURCE_REF_PAYLOAD_KEYS:
                if key not in sanitized_ref:
                    continue
                if key == "fact_status":
                    payload[key] = validate_fact_status(sanitized_ref[key])
                elif key in {"supersedes", "superseded_by", "invalidated_by"}:
                    links = validate_point_id_links(sanitized_ref[key], key)
                    if links:
                        payload[key] = links
                elif key in {"observed_at", "valid_from", "valid_until"}:
                    _add_temporal_string(payload, key, sanitized_ref[key])
                else:
                    payload[key] = sanitized_ref[key]
    _add_source_metadata(payload, "source_uri", source_uri)
    _add_source_metadata(payload, "locator", locator)
    _add_source_metadata(payload, "content_hash", content_hash)
    _add_source_metadata(payload, "source_modified_at", source_modified_at)
    _add_source_metadata(payload, "derivation_type", derivation_type)
    _add_source_metadata(payload, "derived_from", derived_from)
    _add_source_metadata(payload, "canonical", canonical)
    _add_source_metadata(payload, "stale", stale)
    _add_source_metadata(payload, "requires_review", requires_review)
    _add_temporal_string(payload, "observed_at", observed_at)
    _add_temporal_string(payload, "valid_from", valid_from)
    _add_temporal_string(payload, "valid_until", valid_until)
    if not _metadata_is_empty(fact_status):
        payload["fact_status"] = validate_fact_status(fact_status)
    for key, value in (
        ("supersedes", supersedes),
        ("superseded_by", superseded_by),
        ("invalidated_by", invalidated_by),
    ):
        links = validate_point_id_links(value, key)
        if links:
            payload[key] = links
    return payload


def _required_assertion_string(name: str, value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(f"{name} is required for assertion payloads")
    return text


def build_assertion_payload(
    *,
    claim_text: str,
    subject: Any,
    predicate: Any,
    object: Any,
    confidence: float = 0.5,
    source_uri: str | None = None,
    locator: SourceLocator | dict[str, Any] | None = None,
    derived_from: list[DerivationEdge | SourceReference | dict[str, Any] | str] | None = None,
    evidence: Any = None,
    tags: list[str] | None = None,
    profile_id: str = "default",
    platform: str = "cli",
    user_id_hash: str = "",
    chat_id_hash: str = "",
    session_id: str = "",
    project_path: str = "",
    model: str = "",
    provider: str = "qdrant",
    created_at: str | None = None,
    observed_at: str | None = None,
    valid_from: str | None = None,
    valid_until: str | None = None,
    fact_status: str | FactStatus | None = None,
    supersedes: list[Any] | None = None,
    superseded_by: list[Any] | None = None,
    invalidated_by: list[Any] | None = None,
) -> dict[str, Any]:
    """Build a review-gated assertion payload without writing it anywhere.

    Assertions are source-backed factual claim candidates. This helper only
    returns payload data; it does not upsert points or provide a promotion path.
    Extraction confidence is stored as confidence, while truth/canonical status
    remains explicitly non-canonical and review-required.
    """
    claim = _required_assertion_string("claim_text", claim_text)
    payload = build_payload(
        text=claim,
        source="hermes_assertion",
        source_type="assertion",
        chunk_type="assertion",
        confidence=confidence,
        tags=tags,
        profile_id=profile_id,
        platform=platform,
        user_id_hash=user_id_hash,
        chat_id_hash=chat_id_hash,
        session_id=session_id,
        project_path=project_path,
        model=model,
        provider=provider,
        memory_kind=MemoryKind.ASSERTION,
        created_at=created_at,
        source_uri=source_uri,
        locator=locator,
        derived_from=derived_from,
        canonical=False,
        requires_review=True,
        observed_at=observed_at,
        valid_from=valid_from,
        valid_until=valid_until,
        fact_status=fact_status,
        supersedes=supersedes,
        superseded_by=superseded_by,
        invalidated_by=invalidated_by,
    )
    payload.update(
        {
            "claim_text": claim,
            "subject": _required_assertion_string("subject", subject),
            "predicate": _required_assertion_string("predicate", predicate),
            "object": _required_assertion_string("object", object),
            "canonical": False,
            "requires_review": True,
        }
    )

    sanitized_evidence = _sanitize_source_metadata(evidence)
    if not _metadata_is_empty(sanitized_evidence):
        payload["evidence"] = sanitized_evidence

    if not any(key in payload for key in ("source_uri", "locator", "derived_from", "evidence")):
        raise ValueError("assertion payload requires provenance via source_uri, locator, derived_from, or evidence")
    return payload
