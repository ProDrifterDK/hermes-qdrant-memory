"""Secret detection and the learning candidate type.

``contains_secret`` and ``redact_secret_spans`` are used across the plugin. ``LearningCandidate``
and ``candidate_to_learning_args`` feed the learning approval backend
(``QdrantMemoryProvider.approve_learning_candidate``). The heuristic learning extractor this
module was named for (user-correction and tool-failure regexes run by the session hooks) was
removed in capture CP1: it fired on the Spanish word "no" and on host warnings, and its
candidates lived only in provider memory.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

_SECRET_PATTERNS = [
    re.compile(r"(?<![A-Za-z0-9_])sk-[A-Za-z0-9_-]{10,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)authorization\s*:\s*bearer\s+\S+"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{6,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"https?://[^\s:@]+:[^\s:@]+@"),
]

# Explicit credential-key assignments are a fail-closed boundary. Separator
# whitespace is horizontal-only so an apparent assignment cannot borrow its
# value from a later line or code fence. The value capture keeps quoted or
# bracketed same-line values together so exact placeholders with spaces can be
# recognized without weakening real credential assignments.
_CREDENTIAL_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)(?<![A-Za-z0-9])"
    r"(?:api[_-]?key|password|passwd|secret|token|authorization|bearer|"
    r"credential(?:s)?|private[_-]?key)"
    r"[ \t]*[:=][ \t]*"
    r"(?P<value>\"[^\"\r\n]*\"|'[^'\r\n]*'|<[^>\r\n]*>|"
    r"\[[^\]\r\n]*\]|\([^\)\r\n]*\)|`[^`\r\n]*`|[^\s,;]+)"
)

# Word-followed-by-apparent-value patterns. Each captures the RHS so
# `contains_secret` can apply `_looks_like_secret_value` to it; without that
# post-filter we flagged ordinary prose like "context token discussion",
# "Token Bucket algorithm", or "prompts for password before continuing".
_TOKEN_CONTEXT_PATTERN = re.compile(
    r"(?i)\b(?:access\s+)?token\s+"
    r"(?!budget\b|cache\b|count(?:ing)?\b|counter\b|estimate\b|estimates\b|estimation\b|"
    r"hard\b|limit\b|overhead\b|window\b|context\b|izer\b|management\b|usage\b|rate\b|"
    r"rotation\b|expiry\b|ttl\b|lifetime\b|session\b|storage\b|size\b|bucket\b|ring\b|"
    r"network\b|cost\b|discussion\b|metric\b|metrics\b|counts?\b|number\b|amount\b|"
    r"table\b|tables\b|chart\b|charts\b|report\b|reports\b|comparisons?\b|comparison\b|"
    r"analys[ei]s\b|taxonomy\b|classification\b|category\b|categories\b|family\b|"
    r"throughput\b|latency\b|efficiency\b|economics\b|pricing\b|price\b|policy\b|"
    r"bucket\b)"
    r"([^\s,;\"'`][^\s,;\"']{5,40})"
)
_CREDENTIAL_CONTEXT_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|password|secret)\s+"
    r"(?!detection\b|patterns?\b|redaction\b|bearing\b|review\b|heuristic\b|"
    r"management\b|rotation\b|field\b|input\b|value\b|prompt\b|policy\b|"
    r"before\b|after\b|during\b|strength\b|complexity\b|expiry\b|reset\b|change\b|"
    r"update\b|changeover\b|hashing\b|hash\b|encryption\b|salting\b|salt\b|"
    r"storage\b|stores?\b|store\b|leak(?:s|ed|age)?\b|safety\b|safekeeping\b|"
    r"validation\b|checking\b|checks?\b|mistakes?\b|tips?\b|advice\b|guide\b|"
    r"explanation\b|context\b|discussion\b|comparison\b|comparisons?\b|"
    r"length\b|size\b|format\b|scheme\b|re(?:use|using)\b|generator\b|"
    r"vault\b|manager\b|generator\b|requirements?\b|requirement\b|"
    r"best\b|practices?\b|practice\b|recommendations?\b|recommendation\b)"
    r"([^\s,;\"'`][^\s,;\"']{5,40})"
)

# Narrow exact sentinels for placeholder / redaction values. Matching is
# case-insensitive after quote and trailing sentence-punctuation normalization.
_SECRET_PLACEHOLDER_VALUES = frozenset(
    {
        "***",
        "****",
        "<redacted>",
        "[redacted]",
        "<placeholder>",
        "[placeholder]",
        "<input required>",
        "(empty)",
        "<empty>",
        "[empty]",
        "[redacted: possible secret-bearing value]",
        "[redacted: possible secret-bearing memory]",
    }
)
_PLACEHOLDER_TRAILING_PUNCTUATION = ".,;:!?"
_PLACEHOLDER_QUOTES = {'"', "'", "`"}


def _normalize_placeholder_value(value: str) -> str:
    """Normalize only what is needed for exact placeholder recognition."""
    normalized = str(value or "").strip().rstrip(_PLACEHOLDER_TRAILING_PUNCTUATION).strip()
    if len(normalized) >= 2 and normalized[0] in _PLACEHOLDER_QUOTES and normalized[-1] == normalized[0]:
        normalized = normalized[1:-1].strip().rstrip(_PLACEHOLDER_TRAILING_PUNCTUATION).strip()
    return normalized.casefold()


def _is_secret_placeholder(value: str) -> bool:
    return _normalize_placeholder_value(value) in _SECRET_PLACEHOLDER_VALUES


def _looks_like_secret_value(value: str) -> bool:
    """Return True iff a loose context value looks credential-like.

    Explicit key assignments do not use this heuristic: they fail closed unless
    ``value`` is an exact placeholder. Loose prose contexts retain a conservative
    length/shape filter so token-budget and password-detection discussion stays
    usable. Punctuation alone is never evidence of a secret.
    """
    if not value or _is_secret_placeholder(value):
        return False
    if len(value) < 8:
        return False
    has_digit = any(ch.isdigit() for ch in value)
    alphanumeric_count = sum(ch.isalnum() for ch in value)
    return has_digit or alphanumeric_count >= 16

def contains_secret(text: str) -> bool:
    """Return True when ``text`` contains a secret-shaped value.

    Provider-specific patterns remain high-confidence defense in depth. Explicit
    credential-key assignments fail closed unless their complete RHS is an exact
    placeholder. Loose word-followed-by-value contexts retain shape filtering,
    and every match is checked so an earlier benign value cannot hide a later
    credential.
    """
    text = text or ""
    if any(pattern.search(text) for pattern in _SECRET_PATTERNS):
        return True

    for match in _CREDENTIAL_ASSIGNMENT_PATTERN.finditer(text):
        if not _is_secret_placeholder(match.group("value")):
            return True

    for match in _TOKEN_CONTEXT_PATTERN.finditer(text):
        if _looks_like_secret_value(match.group(1)):
            return True

    for match in _CREDENTIAL_CONTEXT_PATTERN.finditer(text):
        if _looks_like_secret_value(match.group(2)):
            return True

    return False


SECRET_SPAN_REPLACEMENT = "[redacted: possible secret-bearing value]"

# `_SECRET_PATTERNS` flags a PEM key by its header line only. Redacting just the
# header would leave the key body in place, so the span runs to the END marker (or
# to the end of the text when the block is cut off).
_PRIVATE_KEY_BLOCK_PATTERN = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)"
)


def redact_secret_spans(text: str) -> str:
    """Return ``text`` with every span ``contains_secret`` flags replaced by a marker.

    Uses the same patterns and the same placeholder and shape filters as
    ``contains_secret``, so it redacts exactly what that detector would refuse and
    nothing it would accept. The result is checked with ``contains_secret`` again; if
    anything is still flagged, the whole text is replaced (fail closed).
    """
    text = text or ""
    if not contains_secret(text):
        return text
    spans: list[tuple[int, int]] = []
    for pattern in _SECRET_PATTERNS:
        spans.extend(match.span() for match in pattern.finditer(text))
    spans.extend(match.span() for match in _PRIVATE_KEY_BLOCK_PATTERN.finditer(text))
    for match in _CREDENTIAL_ASSIGNMENT_PATTERN.finditer(text):
        if not _is_secret_placeholder(match.group("value")):
            spans.append(match.span("value"))
    for match in _TOKEN_CONTEXT_PATTERN.finditer(text):
        if _looks_like_secret_value(match.group(1)):
            spans.append(match.span(1))
    for match in _CREDENTIAL_CONTEXT_PATTERN.finditer(text):
        if _looks_like_secret_value(match.group(2)):
            spans.append(match.span(2))
    pieces: list[str] = []
    cursor = 0
    for start, end in sorted(spans):
        if end <= cursor:
            continue
        if start > cursor:
            pieces.append(text[cursor:start])
        pieces.append(SECRET_SPAN_REPLACEMENT)
        cursor = max(cursor, end)
    pieces.append(text[cursor:])
    redacted = "".join(pieces)
    if contains_secret(redacted):
        return SECRET_SPAN_REPLACEMENT
    return redacted


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
    # The message text the lesson was read from, verbatim from the hook transcript, and its
    # role. Session origin (C'3) cites the state.db row holding it; never part of to_dict.
    source_text: str = field(default="", compare=False, repr=False)
    source_role: str = field(default="", compare=False, repr=False)
    # Frozen by the provider when it accepts the candidate (session, time, hook).
    origin_capture: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

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
