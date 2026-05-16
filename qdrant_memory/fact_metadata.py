from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

from .lesson_extractor import contains_secret

_METADATA_KEYS = {"fact_key", "subject", "topic", "entity", "reconsolidation_key"}
_EXPLICIT_TAG_PREFIXES = {
    "fact": "fact_key",
    "fact_key": "fact_key",
    "reconsolidation": "reconsolidation_key",
    "reconsolidation_key": "reconsolidation_key",
    "subject": "subject",
    "topic": "topic",
    "entity": "entity",
}
_GENERIC_HEADINGS = {"intro", "introduction", "notes", "note", "todo", "todos", "plan", "misc", "general"}
_FACT_PATTERNS = [
    re.compile(r"^(?:the\s+)?(?P<subject>[A-Z][A-Za-z0-9 _./-]{3,100}?)\s+(?:is|are|=|:)\s+\S.+$"),
    re.compile(r"^(?:use|prefer)\s+(?P<subject>[A-Za-z0-9 _./-]{3,100}?)\s+(?:for|when|in)\s+\S.+$", re.I),
]


def normalize_key_part(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", ".", text)
    text = re.sub(r"\.+", ".", text).strip(".")
    return text[:160].strip(".")


def _human_value(value: str, *, max_len: int = 160) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())[:max_len].strip()


def _safe_set(out: dict[str, str], key: str, value: str, *, slug: bool = False) -> None:
    if key not in _METADATA_KEYS:
        return
    clean = normalize_key_part(value) if slug else _human_value(value)
    if clean and not contains_secret(clean):
        out[key] = clean


def extract_explicit_metadata_from_tags(tags: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in tags or []:
        tag = str(raw or "").strip().lstrip("#")
        if not tag or contains_secret(tag):
            continue
        match = re.match(r"^(?P<prefix>fact_key|fact|reconsolidation_key|reconsolidation|subject|topic|entity)[:=](?P<value>.+)$", tag, re.I)
        if not match:
            continue
        key = _EXPLICIT_TAG_PREFIXES[match.group("prefix").lower()]
        _safe_set(out, key, match.group("value"), slug=key in {"fact_key", "reconsolidation_key"})
    if out.get("fact_key") and not out.get("reconsolidation_key"):
        out["reconsolidation_key"] = out["fact_key"]
    return out


def _subject_from_text(text: str) -> str:
    lines = str(text or "").strip().splitlines()
    first_line = re.sub(r"\s+", " ", lines[0].strip())[:240] if lines else ""
    if not first_line or first_line.startswith(("User:", "Assistant:")):
        return ""
    for pattern in _FACT_PATTERNS:
        match = pattern.match(first_line)
        if match:
            subject = _human_value(match.group("subject"))
            if subject.lower() in _GENERIC_HEADINGS:
                return ""
            return subject
    return ""


def _entity_from_command(command: str) -> str:
    first = str(command or "").strip().split()
    if not first:
        return ""
    executable = Path(first[0]).name
    return _human_value(executable, max_len=80)


def _derive_learning_metadata(*, learning_type: str = "", tool_name: str = "", command: str = "", project_path: str = "") -> dict[str, str]:
    topic = _human_value(learning_type)
    entity = _human_value(tool_name) or _entity_from_command(command)
    if not entity and project_path:
        entity = _human_value(Path(project_path).name)
    out: dict[str, str] = {}
    if topic:
        out["topic"] = topic
    if entity:
        out["entity"] = entity
    if topic and entity:
        key = f"learning.{normalize_key_part(topic)}.{normalize_key_part(entity)}"
        if key and not contains_secret(key):
            out["fact_key"] = key
            out["reconsolidation_key"] = key
    return out


def derive_fact_metadata(
    *,
    text: str,
    tags: list[str] | None = None,
    source_type: str = "manual",
    chunk_type: str = "fact",
    heading: str = "",
    file_path: str = "",
    learning_type: str = "",
    tool_name: str = "",
    command: str = "",
    project_path: str = "",
    explicit: dict[str, Any] | None = None,
) -> dict[str, str]:
    combined = "\n".join(str(x or "") for x in [text, " ".join(tags or []), command, str(explicit or {})])
    if contains_secret(combined):
        return {}

    out = extract_explicit_metadata_from_tags(tags)
    for key, value in (explicit or {}).items():
        if key in _METADATA_KEYS:
            _safe_set(out, key, str(value), slug=key in {"fact_key", "reconsolidation_key"})

    if source_type == "learning":
        for key, value in _derive_learning_metadata(learning_type=learning_type or chunk_type, tool_name=tool_name, command=command, project_path=project_path).items():
            out.setdefault(key, value)

    if heading:
        clean_heading = _human_value(heading)
        if clean_heading:
            out.setdefault("topic", clean_heading)

    if not out.get("subject"):
        subject = _subject_from_text(text)
        if subject:
            out["subject"] = subject

    if out.get("subject") and not out.get("fact_key"):
        key = normalize_key_part(out["subject"])
        if key:
            out["fact_key"] = key
            out.setdefault("reconsolidation_key", key)

    if out.get("fact_key"):
        out.setdefault("reconsolidation_key", out["fact_key"])

    # topic/entity are useful filters; only fact_key/reconsolidation_key should drive M10 grouping.
    return {key: value for key, value in out.items() if key in _METADATA_KEYS and value}
