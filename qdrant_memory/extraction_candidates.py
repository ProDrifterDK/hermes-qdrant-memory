from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Mapping

from qdrant_memory.schema import now_iso, validate_fact_status, validate_memory_kind, validate_relation_type


class ExtractionCandidateType(str, Enum):
    MEMORY = "memory_candidate"
    ASSERTION = "assertion_candidate"
    PREFERENCE = "preference_candidate"
    INVARIANT = "invariant_candidate"
    RISK = "risk_candidate"
    STATUS_UPDATE = "status_update_candidate"
    ONTOLOGY_SUGGESTION = "ontology_suggestion"


EXTRACTION_CANDIDATE_TYPES = tuple(candidate_type.value for candidate_type in ExtractionCandidateType)
_EXTRACTION_CANDIDATE_TYPE_SET = set(EXTRACTION_CANDIDATE_TYPES)


@dataclass(frozen=True)
class ExtractionCandidate:
    candidate_id: str
    candidate_type: str
    source_uri: str
    locator: dict[str, Any] = field(default_factory=dict)
    derived_from: list[Any] = field(default_factory=list)
    proposed_payload: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    confidence: float = 0.0
    risk: str = "unknown"
    requires_review: bool = True
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_type": self.candidate_type,
            "source_uri": self.source_uri,
            "locator": self.locator,
            "derived_from": self.derived_from,
            "proposed_payload": self.proposed_payload,
            "reason": self.reason,
            "confidence": self.confidence,
            "risk": self.risk,
            "requires_review": self.requires_review,
            "created_at": self.created_at,
        }


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _normalize_candidate_type(candidate_type: str | ExtractionCandidateType) -> str:
    value = candidate_type.value if isinstance(candidate_type, ExtractionCandidateType) else str(candidate_type or "").strip()
    if value not in _EXTRACTION_CANDIDATE_TYPE_SET:
        raise ValueError(f"unknown candidate_type: {value or '<empty>'}")
    return value


def _normalize_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except Exception as exc:
        raise ValueError("confidence must be numeric") from exc
    if not math.isfinite(confidence):
        raise ValueError("confidence must be finite")
    return max(0.0, min(1.0, confidence))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if hasattr(value, "to_payload") and callable(getattr(value, "to_payload")):
        value = value.to_payload()
    elif is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)

    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("candidate payload must not contain non-finite floats")
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        normalized = [_jsonable(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    raise TypeError(f"candidate payload contains non-serializable value: {type(value).__name__}")


def _validate_payload_grammar(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_s = str(key)
            if key_s == "memory_kind" and not _is_empty(item):
                validate_memory_kind(item)
            elif key_s == "relation_type" and not _is_empty(item):
                validate_relation_type(item)
            elif key_s == "fact_status" and not _is_empty(item):
                validate_fact_status(item)
            _validate_payload_grammar(item)
    elif isinstance(value, list):
        for item in value:
            _validate_payload_grammar(item)


def validate_proposed_payload(proposed_payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(proposed_payload, Mapping):
        raise TypeError("proposed_payload must be a mapping")
    normalized = _jsonable(proposed_payload)
    if not isinstance(normalized, dict):  # defensive; _jsonable preserves mappings as dicts.
        raise TypeError("proposed_payload must normalize to a JSON object")
    _validate_payload_grammar(normalized)
    json.dumps(normalized, sort_keys=True, allow_nan=False)
    return normalized


def _stable_json(value: Any) -> str:
    normalized = _jsonable(value)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def make_extraction_candidate_id(
    *,
    candidate_type: str | ExtractionCandidateType,
    source_uri: str,
    locator: Mapping[str, Any] | None = None,
    derived_from: list[Any] | None = None,
    proposed_payload: Mapping[str, Any],
    reason: str = "",
    confidence: float = 0.0,
    risk: str = "unknown",
    lifecycle_id: str = "",
) -> str:
    fingerprint = {
        "candidate_type": _normalize_candidate_type(candidate_type),
        "source_uri": str(source_uri or "").strip(),
        "locator": _jsonable(locator or {}),
        "derived_from": _jsonable(derived_from or []),
        "proposed_payload": validate_proposed_payload(proposed_payload),
        "reason": str(reason or "").strip(),
        "confidence": _normalize_confidence(confidence),
        "risk": str(risk or "unknown").strip() or "unknown",
        "lifecycle_id": str(lifecycle_id or "").strip(),
    }
    digest = hashlib.sha256(_stable_json(fingerprint).encode("utf-8")).hexdigest()
    return str(uuid.UUID(digest[:32]))


def build_extraction_candidate(
    *,
    candidate_type: str | ExtractionCandidateType,
    source_uri: str,
    proposed_payload: Mapping[str, Any],
    locator: Mapping[str, Any] | None = None,
    derived_from: list[Any] | None = None,
    reason: str = "",
    confidence: float = 0.0,
    risk: str = "unknown",
    requires_review: bool = True,
    created_at: str | None = None,
    lifecycle_id: str = "",
) -> ExtractionCandidate:
    normalized_type = _normalize_candidate_type(candidate_type)
    normalized_source_uri = str(source_uri or "").strip()
    if not normalized_source_uri:
        raise ValueError("source_uri is required")
    normalized_reason = str(reason or "").strip()
    if not normalized_reason:
        raise ValueError("reason is required")
    normalized_confidence = _normalize_confidence(confidence)
    normalized_locator = _jsonable(locator or {})
    if not isinstance(normalized_locator, dict):
        raise TypeError("locator must normalize to a JSON object")
    normalized_derived_from = _jsonable(derived_from or [])
    if not isinstance(normalized_derived_from, list):
        raise TypeError("derived_from must normalize to a JSON array")
    _validate_payload_grammar(normalized_locator)
    _validate_payload_grammar(normalized_derived_from)
    normalized_payload = validate_proposed_payload(proposed_payload)
    candidate_id = make_extraction_candidate_id(
        candidate_type=normalized_type,
        source_uri=normalized_source_uri,
        locator=normalized_locator,
        derived_from=normalized_derived_from,
        proposed_payload=normalized_payload,
        reason=normalized_reason,
        confidence=normalized_confidence,
        risk=risk,
        lifecycle_id=lifecycle_id,
    )
    return ExtractionCandidate(
        candidate_id=candidate_id,
        candidate_type=normalized_type,
        source_uri=normalized_source_uri,
        locator=normalized_locator,
        derived_from=normalized_derived_from,
        proposed_payload=normalized_payload,
        reason=normalized_reason,
        confidence=normalized_confidence,
        risk=str(risk or "unknown").strip() or "unknown",
        requires_review=bool(requires_review),
        created_at=created_at or now_iso(),
    )


def preview_extraction_candidates(candidates: list[ExtractionCandidate]) -> dict[str, Any]:
    items = [candidate.to_dict() for candidate in candidates]
    json.dumps(items, sort_keys=True, allow_nan=False)
    return {"candidates": items, "count": len(items), "dry_run": True}
