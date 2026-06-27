from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from qdrant_memory.lesson_extractor import contains_secret
from qdrant_memory.schema import clean_text_for_memory, score_importance

WRITE_DECISIONS = {"store", "skip", "draft_review", "learning_candidate", "skill_candidate", "reject"}
_DERIVED_TYPES = {"summary", "consolidation_summary", "proposal", "draft", "reconsolidation", "derived_memory"}
_LEARNING_TARGETS = {"learning", "learnings", "procedural_learning"}
_MEMORY_TARGETS = {"memory", "memories"}
_CANDIDATE_KIND_TO_MEMORY_KIND = {
    "preference_candidate": {"user_preference"},
    "invariant_candidate": {"project_invariant"},
    "risk_candidate": {"risk"},
    "status_update_candidate": {"assertion"},
    "assertion_candidate": {"assertion"},
    "memory_candidate": {"decision", "tool_quirk", "workflow_lesson", "manual_fact", "source_chunk", "summary"},
    "graph_entity_candidate": {"graph_entity"},
    "graph_edge_candidate": {"graph_edge"},
}
_DRAFT_RISKS = {"high"}
_REJECT_RISKS = {"critical", "forbidden"}


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


def _candidate_payload(candidate: Any) -> dict[str, Any]:
    payload = getattr(candidate, "proposed_payload", None)
    return payload if isinstance(payload, dict) else {}


def _candidate_to_dict(candidate: Any) -> dict[str, Any]:
    to_dict = getattr(candidate, "to_dict", None)
    if callable(to_dict):
        data = to_dict()
        if isinstance(data, dict):
            return data
    return {
        "candidate_id": getattr(candidate, "candidate_id", ""),
        "candidate_type": getattr(candidate, "candidate_type", ""),
        "source_uri": getattr(candidate, "source_uri", ""),
        "locator": getattr(candidate, "locator", {}),
        "derived_from": getattr(candidate, "derived_from", []),
        "proposed_payload": _candidate_payload(candidate),
        "reason": getattr(candidate, "reason", ""),
        "confidence": getattr(candidate, "confidence", 0.0),
        "risk": getattr(candidate, "risk", "unknown"),
        "requires_review": getattr(candidate, "requires_review", True),
        "created_at": getattr(candidate, "created_at", ""),
    }


def _candidate_payload_text(candidate: Any, payload: dict[str, Any] | None = None) -> str:
    payload = payload if payload is not None else _candidate_payload(candidate)
    return str(payload.get("text") or payload.get("claim_text") or "")


def _source_uris_from_edges(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    uris: list[str] = []
    for edge in value:
        if isinstance(edge, dict):
            source_uri = str(edge.get("source_uri") or "").strip()
            if source_uri:
                uris.append(source_uri)
    return uris


def _candidate_has_source_provenance(candidate: Any, payload: dict[str, Any]) -> bool:
    source_uris = [
        str(getattr(candidate, "source_uri", "") or "").strip(),
        str(payload.get("source_uri") or "").strip(),
    ]
    candidate_edges = getattr(candidate, "derived_from", []) or []
    payload_edges = payload.get("derived_from") or []
    evidence = payload.get("evidence") or []
    source_uris.extend(_source_uris_from_edges(candidate_edges))
    source_uris.extend(_source_uris_from_edges(payload_edges))
    source_uris.extend(_source_uris_from_edges(evidence))
    has_source_uri = any(source_uris)
    has_source_detail = bool(candidate_edges or payload_edges or evidence or getattr(candidate, "locator", None) or payload.get("locator"))
    return has_source_uri and has_source_detail


def _candidate_destination_valid(candidate_type: str, payload: dict[str, Any], target: str) -> bool:
    normalized_target = str(target or "memory").strip().lower()
    if normalized_target not in _MEMORY_TARGETS:
        return False
    for key in ("target", "destination", "collection", "collection_name"):
        destination = str(payload.get(key) or "").strip().lower()
        if destination and destination not in _MEMORY_TARGETS:
            return False
    if candidate_type == "ontology_suggestion":
        return True
    return candidate_type in _CANDIDATE_KIND_TO_MEMORY_KIND


def _candidate_kind_valid(candidate_type: str, payload: dict[str, Any]) -> bool:
    expected_kinds = _CANDIDATE_KIND_TO_MEMORY_KIND.get(candidate_type)
    if expected_kinds is None:
        return candidate_type == "ontology_suggestion"
    memory_kind = str(payload.get("memory_kind") or "")
    return memory_kind in expected_kinds


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


def evaluate_extraction_candidate_write(
    candidate: Any,
    *,
    target: str = "memory",
    persisted_payload: dict[str, Any] | None = None,
    low_confidence_threshold: float = 0.65,
) -> WriteDecision:
    """Validate a source extraction candidate through the shared write gate.

    This intentionally accepts ``Any`` so approval paths cannot bypass the gate by
    constructing candidate-like objects outside ``extraction_candidates.py``.
    """

    payload = dict(persisted_payload or _candidate_payload(candidate))
    candidate_type = str(getattr(candidate, "candidate_type", "") or "").strip()
    candidate_risk = str(getattr(candidate, "risk", "unknown") or "unknown").strip().lower()
    candidate_confidence = _clamp_confidence(getattr(candidate, "confidence", None), 0.0)
    metadata = {
        **payload,
        "candidate_id": getattr(candidate, "candidate_id", ""),
        "candidate_type": candidate_type,
        "candidate_risk": candidate_risk,
    }
    candidate_payload = _candidate_to_dict(candidate)
    if persisted_payload is not None:
        candidate_payload = {**candidate_payload, "persisted_payload": payload}

    if contains_secret(json.dumps(candidate_payload, sort_keys=True, default=str)) or _metadata_contains_secret(payload):
        return _decision(
            "reject",
            ["possible_secret"],
            confidence=1.0,
            requires_review=True,
            metadata={"candidate_type": candidate_type},
        )
    # Reject identity-bearing / non-allowlisted graph candidates at the write gate.
    # Phase 3 uses an allowlist: only known non-personal entity types are safe.
    # Any entity type NOT in the allowlist is rejected (no raw label in payload).
    if candidate_type in ("graph_entity_candidate", "graph_edge_candidate"):
        try:
            from qdrant_memory.improve import (
                is_identity_bearing_value,
                is_identity_bearing_entity_type,
            )
            label_val = str(payload.get("label") or payload.get("text") or "")
            source_uri_val = str(payload.get("source_uri") or "")
            if is_identity_bearing_value(label_val) or is_identity_bearing_value(source_uri_val):
                return _decision(
                    "reject",
                    ["identity_bearing_graph_candidate"],
                    confidence=1.0,
                    requires_review=True,
                    metadata={"candidate_type": candidate_type},
                )
            if candidate_type == "graph_edge_candidate":
                # For graph edges, validate both endpoint entity types against
                # the allowlist using source_entity_type / target_entity_type
                # metadata.  If the metadata is absent or malformed, fail
                # closed (reject) — never allow an edge without proven endpoint
                # type safety.
                src_etype = str(payload.get("source_entity_type") or "").strip().lower()
                tgt_etype = str(payload.get("target_entity_type") or "").strip().lower()
                if not src_etype or not tgt_etype:
                    return _decision(
                        "reject",
                        ["identity_bearing_graph_candidate"],
                        confidence=1.0,
                        requires_review=True,
                        metadata={"candidate_type": candidate_type},
                    )
                if is_identity_bearing_entity_type(src_etype) or is_identity_bearing_entity_type(tgt_etype):
                    return _decision(
                        "reject",
                        ["identity_bearing_graph_candidate"],
                        confidence=1.0,
                        requires_review=True,
                        metadata={"candidate_type": candidate_type},
                    )
            else:
                # graph_entity_candidate: validate the entity_type field.
                entity_type_val = str(payload.get("entity_type") or "")
                if is_identity_bearing_entity_type(entity_type_val):
                    return _decision(
                        "reject",
                        ["identity_bearing_graph_candidate"],
                        confidence=1.0,
                        requires_review=True,
                        metadata={"candidate_type": candidate_type},
                    )
        except ImportError:
            pass  # improve module not available; skip extra check
    if not _candidate_destination_valid(candidate_type, payload, target):
        return _decision(
            "reject",
            ["unsupported_destination"],
            confidence=candidate_confidence,
            requires_review=True,
            metadata={"candidate_type": candidate_type},
        )
    if not _candidate_has_source_provenance(candidate, payload):
        return _decision(
            "reject",
            ["missing_source_provenance"],
            confidence=candidate_confidence,
            requires_review=True,
            metadata={"candidate_type": candidate_type},
        )
    if candidate_risk in _REJECT_RISKS:
        return _decision(
            "reject",
            ["candidate_risk_too_high"],
            confidence=candidate_confidence,
            requires_review=True,
            metadata={"candidate_type": candidate_type, "risk": candidate_risk},
        )
    if candidate_type == "ontology_suggestion":
        return _decision(
            "draft_review",
            ["ontology_suggestion_review_only"],
            confidence=candidate_confidence,
            requires_review=True,
            metadata={"candidate_type": candidate_type},
        )
    if not _candidate_kind_valid(candidate_type, payload):
        return _decision(
            "draft_review",
            ["candidate_kind_mismatch"],
            confidence=candidate_confidence,
            requires_review=True,
            metadata={"candidate_type": candidate_type},
        )
    if candidate_risk in _DRAFT_RISKS:
        return _decision(
            "draft_review",
            ["candidate_risk_requires_review"],
            confidence=candidate_confidence,
            requires_review=True,
            metadata={"candidate_type": candidate_type, "risk": candidate_risk},
        )
    if candidate_confidence < low_confidence_threshold:
        return _decision(
            "draft_review",
            ["low_confidence_source_extraction"],
            confidence=candidate_confidence,
            requires_review=True,
            metadata={"candidate_type": candidate_type},
        )

    return evaluate_write_candidate(
        text=_candidate_payload_text(candidate, payload),
        target=target,
        source_type=str(payload.get("source_type") or "source_extraction"),
        derivation_type=str(payload.get("derivation_type") or "source_extraction"),
        source_uri=str(getattr(candidate, "source_uri", "") or payload.get("source_uri") or ""),
        derived_from=list(getattr(candidate, "derived_from", None) or payload.get("derived_from") or []),
        confidence=candidate_confidence,
        metadata=metadata,
    )


def decision_to_json(decision: WriteDecision) -> str:
    return json.dumps(decision.to_dict(), sort_keys=True)
