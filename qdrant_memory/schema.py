from __future__ import annotations

import hashlib
import math
import re
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any

from .lesson_extractor import contains_secret


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_point_id(source: str, text: str) -> str:
    digest = hashlib.sha256(f"{source}\n{text}".encode("utf-8")).hexdigest()
    return str(uuid.UUID(digest[:32]))


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
    derived_from: list[DerivationEdge | SourceReference | dict[str, Any]] = field(default_factory=list)
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
    "derived_from",
    "canonical",
    "stale",
    "requires_review",
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
    created_at: str | None = None,
    fact_metadata: dict[str, Any] | None = None,
    source_ref: SourceReference | dict[str, Any] | None = None,
    source_uri: str | None = None,
    locator: SourceLocator | dict[str, Any] | None = None,
    content_hash: str | None = None,
    source_modified_at: str | None = None,
    derivation_type: str | None = None,
    derived_from: list[DerivationEdge | SourceReference | dict[str, Any]] | None = None,
    canonical: bool | None = None,
    stale: bool | None = None,
    requires_review: bool | None = None,
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

    if source_ref:
        sanitized_ref = _sanitize_source_metadata(source_ref)
        if isinstance(sanitized_ref, dict):
            for key in _SOURCE_REF_PAYLOAD_KEYS:
                if key in sanitized_ref:
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
    return payload
