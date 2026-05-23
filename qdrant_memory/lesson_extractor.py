from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

VALID_LEARNING_TYPES = {"tool_failure_lesson", "user_correction", "workflow_lesson", "environment_quirk"}

_SECRET_PATTERNS = [
    re.compile(r"(?<![A-Za-z0-9_])sk-[A-Za-z0-9_-]{10,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)authorization\s*:\s*bearer\s+\S+"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{6,}"),
    re.compile(r"(?i)(api[_-]?key|password|secret)\s*[:=]\s*\S+"),
    re.compile(r"(?i)token\s*[:=]\s*\S+"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"https?://[^\s:@]+:[^\s:@]+@"),
]
_TOKEN_CONTEXT_PATTERN = re.compile(r"(?i)\b(?:access\s+)?token\s+(?!budget\b|cache\b|count(?:ing)?\b|counter\b|estimate\b|estimates\b|estimation\b|hard\b|limit\b|overhead\b|window\b|context\b|izer\b)(\S{6,})")
_CREDENTIAL_CONTEXT_PATTERN = re.compile(r"(?i)\b(api[_-]?key|password|secret)\s+(?!detection\b|patterns?\b|redaction\b|bearing\b|review\b|heuristic\b)(\S{6,})")

_EXPLICIT_CORRECTION_PATTERNS = [
    re.compile(r"(?i)\bactually[, ]+(?P<body>.+)"),
    re.compile(r"(?i)\bno[, ]+(?P<body>.+)"),
    re.compile(r"(?i)\bcorrection[: ]+(?P<body>.+)"),
    re.compile(r"(?i)\bremember this[: ]+(?P<body>.+)"),
    re.compile(r"(?i)\bi prefer (?P<body>.+)"),
]

_NOT_PATTERN = re.compile(r"(?i)(?P<correct>[^.\n,;:]{2,80}?)\s*,?\s+not\s+(?P<mistake>[^.\n,;:]{2,80})")
_TOOL_FAILURE_PATTERN = re.compile(r"(?i)(failed|error|exception|traceback|exit code|command not found|not found|http\s*\d{3})")
_RESOLUTION_PATTERN = re.compile(r"(?i)(correction|fix|solution|resolved|use|instead|rerun|run)[: ]+(?P<body>.+)")


def contains_secret(text: str) -> bool:
    text = text or ""
    return any(pattern.search(text) for pattern in _SECRET_PATTERNS) or bool(_TOKEN_CONTEXT_PATTERN.search(text)) or bool(_CREDENTIAL_CONTEXT_PATTERN.search(text))


def _stable_id(parts: Iterable[str]) -> str:
    raw = "|".join((part or "").strip() for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


@dataclass
class LearningCandidate:
    lesson: str
    learning_type: str
    trigger: str = ""
    mistake: str = ""
    correction: str = ""
    evidence: str = ""
    tool_name: str = ""
    command: str = ""
    confidence: float = 0.85
    importance: int = 7
    tags: list[str] = field(default_factory=lambda: ["auto_candidate", "gated"])
    source_hook: str = "unknown"
    gated: bool = True
    reason: str = "heuristic candidate"

    @property
    def candidate_id(self) -> str:
        return _stable_id([self.learning_type, self.trigger, self.mistake, self.correction, self.lesson])

    def to_dict(self, *, include_metadata: bool = False) -> dict[str, Any]:
        item: dict[str, Any] = {
            "candidate_id": self.candidate_id,
            "lesson": self.lesson,
            "learning_type": self.learning_type,
            "trigger": self.trigger,
            "mistake": self.mistake,
            "correction": self.correction,
            "evidence": self.evidence,
            "tool_name": self.tool_name,
            "command": self.command,
            "confidence": round(float(self.confidence), 4),
            "importance": int(self.importance),
            "tags": list(self.tags),
            "gated": self.gated,
            "reason": self.reason,
        }
        if include_metadata:
            item["metadata"] = {"source_hook": self.source_hook}
        return item


def candidate_to_learning_args(candidate: LearningCandidate) -> dict[str, Any]:
    return {
        "lesson": candidate.lesson,
        "learning_type": candidate.learning_type,
        "trigger": candidate.trigger,
        "mistake": candidate.mistake,
        "correction": candidate.correction,
        "evidence": candidate.evidence,
        "tool_name": candidate.tool_name,
        "command": candidate.command,
        "importance": candidate.importance,
        "confidence": candidate.confidence,
        "tags": list(dict.fromkeys([*candidate.tags, f"source:{candidate.source_hook}"])),
        "promote_to_skill_candidate": candidate.learning_type in {"workflow_lesson", "tool_failure_lesson"},
    }


def _message_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    if not isinstance(message, dict):
        return str(message or "")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if content is None:
        return json.dumps(message, ensure_ascii=False)
    return str(content)


def _message_role(message: Any) -> str:
    return str(message.get("role") or "") if isinstance(message, dict) else ""


def _message_tool_name(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    return str(message.get("name") or message.get("tool_name") or "")


def _dedupe_and_gate(candidates: list[LearningCandidate], *, min_confidence: float, max_candidates: int) -> list[LearningCandidate]:
    seen: set[str] = set()
    accepted: list[LearningCandidate] = []
    for candidate in candidates:
        if candidate.learning_type not in VALID_LEARNING_TYPES:
            continue
        combined = "\n".join([candidate.lesson, candidate.trigger, candidate.mistake, candidate.correction, candidate.evidence, candidate.command])
        if contains_secret(combined):
            continue
        if candidate.confidence < min_confidence:
            continue
        if candidate.candidate_id in seen:
            continue
        seen.add(candidate.candidate_id)
        accepted.append(candidate)
        if len(accepted) >= max_candidates:
            break
    return accepted


def _extract_user_correction(user_content: str, assistant_content: str, *, source_hook: str) -> LearningCandidate | None:
    text = (user_content or "").strip()
    if not text or contains_secret(text):
        return None
    if not any(pattern.search(text) for pattern in _EXPLICIT_CORRECTION_PATTERNS):
        return None
    mistake = ""
    correction = text
    not_match = _NOT_PATTERN.search(text)
    if not_match:
        correction = not_match.group("correct").strip(" .,:;\n\t")
        mistake = not_match.group("mistake").strip(" .,:;\n\t")
    lesson = f"User correction: use {correction}."
    if mistake:
        lesson = f"User correction: use {correction}, not {mistake}."
    return LearningCandidate(
        lesson=lesson,
        learning_type="user_correction",
        trigger=text[:240],
        mistake=mistake,
        correction=correction,
        evidence="Explicit user correction followed by assistant acknowledgement." if assistant_content else "Explicit user correction.",
        confidence=0.9,
        importance=8,
        tags=["auto_candidate", "gated", "user_correction"],
        source_hook=source_hook,
        reason="explicit user correction signal",
    )


def extract_learning_candidates_from_turn(
    user_content: str,
    assistant_content: str,
    *,
    source_hook: str = "sync_turn",
    min_confidence: float = 0.85,
    max_candidates: int = 3,
) -> list[LearningCandidate]:
    candidates: list[LearningCandidate] = []
    correction = _extract_user_correction(user_content, assistant_content, source_hook=source_hook)
    if correction:
        candidates.append(correction)
    return _dedupe_and_gate(candidates, min_confidence=min_confidence, max_candidates=max_candidates)


def _extract_tool_failure_candidates(messages: list[Any], *, source_hook: str) -> list[LearningCandidate]:
    candidates: list[LearningCandidate] = []
    for idx, message in enumerate(messages):
        role = _message_role(message)
        text = _message_text(message).strip()
        if not text or contains_secret(text) or not _TOOL_FAILURE_PATTERN.search(text):
            continue
        is_toolish = role == "tool" or bool(_message_tool_name(message)) or "failed" in text.lower() or "error" in text.lower()
        if not is_toolish:
            continue
        next_text = ""
        for later in messages[idx + 1 : idx + 4]:
            if _message_role(later) == "assistant":
                next_text = _message_text(later).strip()
                break
        resolution = _RESOLUTION_PATTERN.search(next_text or "")
        if not resolution or contains_secret(next_text):
            continue
        correction = resolution.group("body").strip()
        lesson = f"When {text[:120]}, {correction}"
        candidates.append(
            LearningCandidate(
                lesson=lesson,
                learning_type="tool_failure_lesson",
                trigger=text[:240],
                mistake=text[:240],
                correction=correction,
                evidence=f"Tool failure followed by assistant correction: {next_text[:240]}",
                tool_name=_message_tool_name(message),
                command=_extract_command(text),
                confidence=0.86,
                importance=8,
                tags=["auto_candidate", "gated", "tool_failure"],
                source_hook=source_hook,
                reason="tool failure with explicit follow-up correction",
            )
        )
    return candidates


def _extract_command(text: str) -> str:
    if "command" in text.lower():
        return text[:160]
    return ""


def extract_learning_candidates_from_messages(
    messages: list[Any],
    *,
    source_hook: str = "on_session_end",
    min_confidence: float = 0.85,
    max_candidates: int = 3,
) -> list[LearningCandidate]:
    candidates: list[LearningCandidate] = []
    for idx, message in enumerate(messages):
        if _message_role(message) != "user":
            continue
        assistant_text = ""
        for later in messages[idx + 1 : idx + 3]:
            if _message_role(later) == "assistant":
                assistant_text = _message_text(later)
                break
        candidate = _extract_user_correction(_message_text(message), assistant_text, source_hook=source_hook)
        if candidate:
            candidates.append(candidate)
    candidates.extend(_extract_tool_failure_candidates(messages, source_hook=source_hook))
    return _dedupe_and_gate(candidates, min_confidence=min_confidence, max_candidates=max_candidates)
