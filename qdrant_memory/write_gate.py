from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from qdrant_memory.lesson_extractor import contains_secret
from qdrant_memory.schema import clean_text_for_memory, score_importance

WRITE_DECISIONS = {"store", "skip", "draft_review", "learning_candidate", "skill_candidate", "reject"}
_DERIVED_TYPES = {"summary", "consolidation_summary", "proposal", "draft", "reconsolidation", "derived_memory"}
_LEARNING_TARGETS = {"learning", "learnings", "procedural_learning"}


@dataclass(frozen=True)
class WriteDecision:
    decision: str
    reasons: list[str]
    confidence: float
    requires_review: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clamp_confidence(value: float | None, default: float = 0.8) -> float:
    try:
        parsed = float(default if value is None else value)
    except Exception:
        parsed = default
    return max(0.0, min(1.0, parsed))


def _metadata_contains_secret(value: Any) -> bool:
    if isinstance(value, str):
        return contains_secret(value)
    if isinstance(value, dict):
        return any(_metadata_contains_secret(key) or _metadata_contains_secret(item) for key, item in value.items())
    if isinstance(value, (list, tuple, set)):
        return any(_metadata_contains_secret(item) for item in value)
    return False


def _semantic_text_for_quality(text: str) -> str:
    parts: list[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if ":" in stripped:
            label, value = stripped.split(":", 1)
            if label.strip().lower() in {"lesson", "trigger", "mistake", "correction", "evidence", "tool", "command"}:
                stripped = value.strip()
        if stripped:
            parts.append(stripped)
    return "\n".join(parts) or text


def _decision(decision: str, reasons: list[str], *, confidence: float, requires_review: bool, metadata: dict[str, Any] | None = None) -> WriteDecision:
    if decision not in WRITE_DECISIONS:
        raise ValueError(f"unsupported write decision: {decision}")
    return WriteDecision(decision=decision, reasons=reasons, confidence=_clamp_confidence(confidence), requires_review=requires_review, metadata=metadata or {})


def evaluate_write_candidate(
    *,
    text: str,
    target: str = "memory",
    source_type: str = "",
    derivation_type: str = "",
    source_uri: str = "",
    derived_from: list[dict[str, Any]] | None = None,
    confidence: float | None = None,
    duplicate: dict[str, Any] | None = None,
    promote_to_skill_candidate: bool = False,
    metadata: dict[str, Any] | None = None,
) -> WriteDecision:
    """Classify a candidate write before it becomes durable memory.

    The gate is intentionally conservative: it does not mutate anything. Callers use
    the structured decision to choose store/skip/review behavior while preserving the
    existing explicit approval and dry-run gates.
    """

    cleaned = clean_text_for_memory(text or "")
    normalized_target = str(target or "memory").strip().lower()
    normalized_source_type = str(source_type or "").strip().lower()
    normalized_derivation_type = str(derivation_type or "").strip().lower()
    source_uri = str(source_uri or "").strip()
    derived_edges = [edge for edge in (derived_from or []) if isinstance(edge, dict)]
    candidate_confidence = _clamp_confidence(confidence)
    metadata = metadata or {}

    if not cleaned:
        return _decision("skip", ["empty_text"], confidence=1.0, requires_review=False)
    if contains_secret(cleaned) or _metadata_contains_secret(metadata) or _metadata_contains_secret(derived_edges) or contains_secret(source_uri):
        return _decision("reject", ["possible_secret"], confidence=1.0, requires_review=True)
    if duplicate:
        return _decision(
            "skip",
            ["duplicate_candidate"],
            confidence=_clamp_confidence(duplicate.get("score") if isinstance(duplicate, dict) else None, 0.95),
            requires_review=False,
            metadata={"duplicate": duplicate},
        )

    quality_text = _semantic_text_for_quality(cleaned)
    scored_importance = score_importance(quality_text, normalized_source_type or normalized_target or "conversation")
    if scored_importance <= 1:
        return _decision("skip", ["low_information"], confidence=0.9, requires_review=False, metadata={"importance": scored_importance})
    importance = scored_importance
    try:
        metadata_importance = int(metadata.get("importance") or 0)
    except Exception:
        metadata_importance = 0
    if 1 <= metadata_importance <= 10:
        importance = max(importance, metadata_importance)
    if importance <= 2:
        return _decision("skip", ["low_information"], confidence=0.9, requires_review=False, metadata={"importance": importance})

    is_derived = normalized_derivation_type in _DERIVED_TYPES or normalized_source_type in {"summary", "consolidation_report", "proposal"}
    has_provenance = bool(source_uri or derived_edges)
    if is_derived and not has_provenance:
        return _decision("draft_review", ["missing_provenance"], confidence=candidate_confidence, requires_review=True, metadata={"importance": importance})
    if is_derived and candidate_confidence < 0.65:
        return _decision("draft_review", ["low_confidence_derived_write"], confidence=candidate_confidence, requires_review=True, metadata={"importance": importance})

    if promote_to_skill_candidate:
        if candidate_confidence >= 0.85 and importance >= 8:
            return _decision("skill_candidate", ["skill_candidate"], confidence=candidate_confidence, requires_review=True, metadata={"importance": importance})
        return _decision("draft_review", ["weak_skill_candidate"], confidence=candidate_confidence, requires_review=True, metadata={"importance": importance})

    if normalized_target in _LEARNING_TARGETS or normalized_source_type == "learning":
        return _decision("learning_candidate", ["learning_candidate"], confidence=candidate_confidence, requires_review=True, metadata={"importance": importance})

    return _decision("store", ["storeable"], confidence=candidate_confidence, requires_review=False, metadata={"importance": importance})


def decision_to_json(decision: WriteDecision) -> str:
    return json.dumps(decision.to_dict(), sort_keys=True)
