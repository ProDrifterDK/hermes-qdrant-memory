"""Source-first extraction: deterministic line signals, the extraction write gate, proposals.

``extract_source_candidates_from_text`` stays because ``improve.py`` builds its candidates
with it. ``evaluate_source_extraction_candidate`` and ``build_source_extraction_proposal`` are
the gate and draft path of the extraction approval backend
(``QdrantMemoryProvider.approve_extraction_candidate``). The session-hook producer that ran
these rules over every compressed or ended transcript (``extract_source_candidates_from_messages``)
and its preview were removed in capture CP1: it matched no message on the labelled sample.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from qdrant_memory.extraction_candidates import ExtractionCandidate, build_extraction_candidate
from qdrant_memory.lesson_extractor import contains_secret
from qdrant_memory.schema import build_assertion_payload, build_payload, clean_text_for_memory
from qdrant_memory.write_gate import WriteDecision, evaluate_extraction_candidate_write

SOURCE_EXTRACTION_DERIVATION_TYPE = "source_extraction"
LOW_CONFIDENCE_DRAFT_THRESHOLD = 0.65

_DECISION_RE = re.compile(r"(?i)^\s*(?:decision|decided)\s*[:\-]\s*(?P<body>.+)")
_TOOL_QUIRK_RE = re.compile(r"(?i)^\s*(?:tool\s+quirk|environment\s+quirk|quirk)\s*[:\-]\s*(?P<body>.+)")
_INVARIANT_RE = re.compile(r"(?i)^\s*(?:project\s+invariant|invariant)\s*[:\-]\s*(?P<body>.+)")
_RESOLVED_CONFLICT_RE = re.compile(r"(?i)^\s*(?:resolved\s+conflict|conflict\s+resolved)\s*[:\-]\s*(?P<body>.+)")
_POSSIBLE_RE = re.compile(
    r"(?i)^\s*(?:possible|maybe)\s+(?P<label>decision|tool\s+quirk|project\s+invariant|invariant|preference|correction)\s*[:\-]\s*(?P<body>.+)"
)
_EXPLICIT_CORRECTION_RE = re.compile(r"(?i)^\s*(?:actually|correction|remember this|i prefer|prefer)\b[:, ]*\s*(?P<body>.+)")
_PRIVATE_IDENTITY_RE = re.compile(
    r"(?i)\b(?:my|user(?:'s)?|customer(?:'s)?|person(?:'s)?)\s+"
    r"(?:name|surname|email|phone|address|birth(?:day|date)?|age|ssn|passport|identity|id)\b"
)

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


def _clamp_confidence(value: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = LOW_CONFIDENCE_DRAFT_THRESHOLD
    return max(0.0, min(1.0, parsed))


def _is_unsafe_source_line(text: str) -> bool:
    clean = clean_text_for_memory(text or "")
    if not clean:
        return True
    return contains_secret(clean) or bool(_PRIVATE_IDENTITY_RE.search(clean))


def _edge(
    *,
    source_uri: str,
    source_type: str,
    locator: Mapping[str, Any] | None,
    derivation_type: str,
    point_id: str = "",
) -> dict[str, Any]:
    edge: dict[str, Any] = {
        "source_uri": source_uri,
        "source_type": source_type,
        "derivation_type": derivation_type,
        "relation_type": "EXTRACTED_FROM",
    }
    if locator:
        edge["locator"] = dict(locator)
    if point_id:
        edge["point_id"] = point_id
    return edge


def _clean_body(body: str) -> str:
    return clean_text_for_memory(body).strip(" .\n\t")


def _line_signal(line: str) -> tuple[str, str, str, float, str] | None:
    """Return candidate_type, memory_kind, body, confidence, reason for a source line."""

    text = clean_text_for_memory(line)
    if not text:
        return None
    possible = _POSSIBLE_RE.match(text)
    if possible:
        label = possible.group("label").lower().replace(" ", "_")
        body = _clean_body(possible.group("body"))
        if label in {"preference", "correction"}:
            return "preference_candidate", "user_preference", body, 0.5, "low-confidence explicit preference signal"
        if label in {"project_invariant", "invariant"}:
            return "invariant_candidate", "project_invariant", body, 0.5, "low-confidence invariant signal"
        if label == "tool_quirk":
            return "memory_candidate", "tool_quirk", body, 0.5, "low-confidence tool quirk signal"
        return "memory_candidate", "decision", body, 0.5, "low-confidence decision signal"

    checks = [
        (_DECISION_RE, "memory_candidate", "decision", 0.88, "explicit decision signal"),
        (_TOOL_QUIRK_RE, "memory_candidate", "tool_quirk", 0.86, "explicit tool quirk signal"),
        (_INVARIANT_RE, "invariant_candidate", "project_invariant", 0.9, "explicit project invariant signal"),
        (_RESOLVED_CONFLICT_RE, "status_update_candidate", "assertion", 0.82, "resolved conflict signal"),
    ]
    for pattern, candidate_type, memory_kind, confidence, reason in checks:
        match = pattern.match(text)
        if match:
            body = _clean_body(match.group("body"))
            return candidate_type, memory_kind, body, confidence, reason

    correction = _EXPLICIT_CORRECTION_RE.match(text)
    if correction:
        body = _clean_body(correction.group("body"))
        lowered = body.lower()
        if any(word in lowered for word in ("prefer", "use", "always", "never", "instead", "not ")):
            return "preference_candidate", "user_preference", body, 0.9, "explicit user correction or preference signal"
    return None


def _payload_for_signal(
    *,
    candidate_type: str,
    memory_kind: str,
    text: str,
    source_uri: str,
    source_type: str,
    locator: Mapping[str, Any],
    derived_from: list[dict[str, Any]],
    confidence: float,
) -> dict[str, Any]:
    if candidate_type in {"assertion_candidate", "status_update_candidate"}:
        payload = build_assertion_payload(
            claim_text=text,
            subject="source_extraction.resolved_conflict",
            predicate="resolved_conflict",
            object=text,
            confidence=confidence,
            source_uri=source_uri,
            locator=dict(locator),
            derived_from=derived_from,
            evidence=derived_from,
            tags=["source_extraction", "resolved_conflict"],
            fact_status="review_required",
        )
        payload["derivation_type"] = SOURCE_EXTRACTION_DERIVATION_TYPE
        return payload
    return build_payload(
        text=text,
        source="source_extraction",
        source_type="source_extraction",
        chunk_type=memory_kind,
        confidence=confidence,
        tags=["source_extraction", memory_kind],
        memory_kind=memory_kind,
        source_uri=source_uri,
        locator=dict(locator),
        derived_from=derived_from,
        derivation_type=SOURCE_EXTRACTION_DERIVATION_TYPE,
        canonical=False,
        requires_review=True,
    )


def _build_candidate(
    *,
    candidate_type: str,
    memory_kind: str,
    body: str,
    source_uri: str,
    source_type: str,
    locator: Mapping[str, Any],
    derivation_type: str,
    confidence: float,
    reason: str,
    lifecycle_id: str = "",
    point_id: str = "",
) -> ExtractionCandidate | None:
    clean_body = _clean_body(body)
    if not clean_body or _is_unsafe_source_line(clean_body):
        return None
    normalized_confidence = _clamp_confidence(confidence)
    derived_from = [
        _edge(
            source_uri=source_uri,
            source_type=source_type,
            locator=locator,
            derivation_type=derivation_type,
            point_id=point_id,
        )
    ]
    if memory_kind == "decision":
        text = f"Decision: {clean_body}."
    elif memory_kind == "tool_quirk":
        text = f"Tool quirk: {clean_body}."
    elif memory_kind == "project_invariant":
        text = f"Project invariant: {clean_body}."
    elif memory_kind == "user_preference":
        text = f"User preference/correction: {clean_body}."
    elif memory_kind == "assertion":
        text = f"Resolved conflict: {clean_body}."
    else:
        text = clean_body
    payload = _payload_for_signal(
        candidate_type=candidate_type,
        memory_kind=memory_kind,
        text=text,
        source_uri=source_uri,
        source_type=source_type,
        locator=locator,
        derived_from=derived_from,
        confidence=normalized_confidence,
    )
    return build_extraction_candidate(
        candidate_type=candidate_type,
        source_uri=source_uri,
        locator=dict(locator),
        derived_from=derived_from,
        proposed_payload=payload,
        reason=reason,
        confidence=normalized_confidence,
        risk="low" if normalized_confidence >= LOW_CONFIDENCE_DRAFT_THRESHOLD else "medium",
        requires_review=True,
        lifecycle_id=lifecycle_id,
    )


def _dedupe_candidates(candidates: Iterable[ExtractionCandidate], *, max_candidates: int) -> list[ExtractionCandidate]:
    seen: set[str] = set()
    accepted: list[ExtractionCandidate] = []
    for candidate in candidates:
        if candidate.candidate_id in seen:
            continue
        seen.add(candidate.candidate_id)
        accepted.append(candidate)
        if len(accepted) >= max(1, int(max_candidates or 1)):
            break
    return accepted


def extract_source_candidates_from_text(
    text: str,
    *,
    source_uri: str,
    source_type: str = "source_text",
    locator: Mapping[str, Any] | None = None,
    derivation_type: str = "source_text",
    lifecycle_id: str = "",
    min_confidence: float = LOW_CONFIDENCE_DRAFT_THRESHOLD,
    max_candidates: int = 8,
    point_id: str = "",
) -> list[ExtractionCandidate]:
    """Extract source-backed candidates from explicit signals using deterministic rules only."""

    source_uri = str(source_uri or "").strip()
    if not source_uri or _is_unsafe_source_line(source_uri):
        return []
    try:
        floor = float(min_confidence)
    except Exception:
        floor = LOW_CONFIDENCE_DRAFT_THRESHOLD
    base_locator = dict(locator or {})
    candidates: list[ExtractionCandidate] = []
    for index, line in enumerate((text or "").splitlines(), start=1):
        if _is_unsafe_source_line(line):
            continue
        signal = _line_signal(line)
        if not signal:
            continue
        candidate_type, memory_kind, body, confidence, reason = signal
        if confidence < floor:
            continue
        line_locator = dict(base_locator)
        line_locator.setdefault("line_start", index)
        line_locator.setdefault("line_end", index)
        candidate = _build_candidate(
            candidate_type=candidate_type,
            memory_kind=memory_kind,
            body=body,
            source_uri=source_uri,
            source_type=source_type,
            locator=line_locator,
            derivation_type=derivation_type,
            confidence=confidence,
            reason=reason,
            lifecycle_id=lifecycle_id,
            point_id=point_id,
        )
        if candidate:
            candidates.append(candidate)
    return _dedupe_candidates(candidates, max_candidates=max_candidates)


def _candidate_payload_text(candidate: ExtractionCandidate) -> str:
    payload = candidate.proposed_payload or {}
    return str(payload.get("text") or payload.get("claim_text") or "")


def _candidate_has_provenance(candidate: ExtractionCandidate) -> bool:
    payload = candidate.proposed_payload or {}
    return bool(
        str(candidate.source_uri or "").strip()
        or candidate.derived_from
        or payload.get("source_uri")
        or payload.get("derived_from")
        or payload.get("evidence")
    )


def _candidate_kind_valid(candidate: ExtractionCandidate) -> bool:
    expected_kinds = _CANDIDATE_KIND_TO_MEMORY_KIND.get(candidate.candidate_type)
    if expected_kinds is None:
        return candidate.candidate_type == "ontology_suggestion"
    memory_kind = str((candidate.proposed_payload or {}).get("memory_kind") or "")
    return memory_kind in expected_kinds


def evaluate_source_extraction_candidate(
    candidate: ExtractionCandidate,
    *,
    persisted_payload: dict[str, Any] | None = None,
) -> WriteDecision:
    """Validate an extraction candidate through the shared conservative write gate."""

    return evaluate_extraction_candidate_write(
        candidate,
        target="memory",
        persisted_payload=persisted_payload,
        low_confidence_threshold=LOW_CONFIDENCE_DRAFT_THRESHOLD,
    )


def build_source_extraction_proposal(candidate: ExtractionCandidate, write_decision: WriteDecision | dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    decision_payload = write_decision.to_dict() if isinstance(write_decision, WriteDecision) else write_decision
    report = {
        "report_id": f"source-extraction-{candidate.candidate_id[:8]}",
        "report_type": "source_extraction",
        "source_uri": candidate.source_uri,
        "dry_run_first": True,
    }
    proposal = {
        "proposal_id": candidate.candidate_id,
        "proposal_type": "source_extraction_candidate",
        "candidate_type": candidate.candidate_type,
        "suggested_action": "manual_review",
        "affected_ids": [candidate.candidate_id],
        "confidence": candidate.confidence,
        "write_decision": decision_payload,
    }
    point = {
        "id": candidate.candidate_id,
        "text": _candidate_payload_text(candidate),
        "payload": {**candidate.proposed_payload, "source_extraction_candidate_id": candidate.candidate_id},
    }
    return report, proposal, [point]
