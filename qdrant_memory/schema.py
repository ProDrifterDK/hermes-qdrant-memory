from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .lesson_extractor import contains_secret


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_point_id(source: str, text: str) -> str:
    digest = hashlib.sha256(f"{source}\n{text}".encode("utf-8")).hexdigest()
    return str(uuid.UUID(digest[:32]))


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
    return payload
