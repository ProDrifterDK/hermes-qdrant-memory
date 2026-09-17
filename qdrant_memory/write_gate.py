from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from qdrant_memory.lesson_extractor import contains_secret
from qdrant_memory.schema import clean_text_for_memory, score_importance

# Regex for a 64-char lowercase hex SHA-256 hash (matches the RAPTOR
# builder's ``hashlib.sha256(...).hexdigest()`` output). Defined here
# (rather than imported from ``qdrant_memory.raptor``) to keep
# ``write_gate`` free of circular imports and to guarantee the same
# shape is enforced by both the pre- and post-enrichment gates.
# Exact-token anchor ('\Z', not '$'): a trailing newline must not pass.
_SHA256_HEX_RE = re.compile(r"^[a-f0-9]{64}\Z")
# RAPTOR derivation/relation types for provenance edges. Mirrored from
# ``qdrant_memory.raptor.schema`` to keep the post-enrichment gate
# self-contained.
_RAPTOR_DERIVATION_TYPE = "raptor_summary"
_RAPTOR_RELATION_TYPE = "SUMMARIZES"


def _is_sha256_hex(value: Any) -> bool:
    """Return True iff *value* is a 64-char lowercase hex SHA-256 string."""
    return isinstance(value, str) and bool(_SHA256_HEX_RE.match(value))


def _is_raptor_provenance_edge(edge: Any) -> bool:
    """Return True iff *edge* is a structurally valid RAPTOR provenance edge.

    A RAPTOR provenance edge is a mapping with:
    - non-empty string ``source_uri``
    - non-empty string ``child_node_id``
    - ``derivation_type`` exactly equal to ``"raptor_summary"``
    - ``relation_type`` exactly equal to ``"SUMMARIZES"``
    """
    if not isinstance(edge, Mapping):
        return False
    source_uri = edge.get("source_uri")
    child_node_id = edge.get("child_node_id")
    derivation_type = edge.get("derivation_type")
    relation_type = edge.get("relation_type")
    return (
        isinstance(source_uri, str)
        and bool(source_uri.strip())
        and isinstance(child_node_id, str)
        and bool(child_node_id.strip())
        and derivation_type == _RAPTOR_DERIVATION_TYPE
        and relation_type == _RAPTOR_RELATION_TYPE
    )

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


_MECHANICAL_EDGE_CLASS = "mechanical"


def structural_lineage_marker_reasons(payload: Any) -> list[str]:
    """Value-semantic detection of structural write-class markers.

    Retrieval and consolidation exclude structural records by truthiness, so
    ANY truthy ``lineage_record`` / ``lineage_pending`` value — not just the
    literal boolean ``True`` — creates a record those readers would hide.
    ``edge_class="mechanical"`` is likewise a structural write-class claim a
    semantic write path must never certify, whatever the relation type.
    Absent and falsy markers (``False``, ``0``, ``""``) remain allowed shared
    metadata, and ordinary semantic fields (endpoint types, source paths) are
    never touched.
    """
    if not isinstance(payload, Mapping):
        return []
    reasons: list[str] = []
    if payload.get("lineage_record"):
        reasons.append("lineage_record")
    if payload.get("lineage_pending"):
        reasons.append("lineage_pending")
    edge_class = payload.get("edge_class")
    if isinstance(edge_class, str) and edge_class.strip().lower() == _MECHANICAL_EDGE_CLASS:
        reasons.append("edge_class=mechanical")
    return reasons


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

    # The mechanical lineage candidate class is internal to the lineage
    # writer path. It must never be submittable through the ordinary
    # extraction gate, whatever payload flags it carries. Any truthy
    # ``lineage_record`` / ``lineage_pending`` marker of ANY type is equally
    # structural (truthiness is what retrieval and consolidation exclude on,
    # so ``1`` or ``"true"`` would create a record those readers hide), and a
    # payload claiming the mechanical edge class is a structural write-class
    # claim the semantic path must never certify — a claim edge labeled
    # mechanical is still rejected.
    structural_reasons = structural_lineage_marker_reasons(payload)
    if candidate_type == "mechanical_lineage_candidate" or structural_reasons:
        return _decision(
            "reject",
            ["mechanical_lineage_candidate_forbidden", *structural_reasons],
            confidence=1.0,
            requires_review=True,
            metadata={"candidate_type": candidate_type},
        )
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


def evaluate_mechanical_lineage_write(
    payload: Any,
    *,
    operation: str,
    evidence: Any,
    collection_name: str,
) -> WriteDecision:
    """Gate one structural lineage write using the independent evidence record.

    ``collection_name`` is the exact collection the caller will upsert into;
    it must match the evidence, and the payload's scope digest is recomputed
    from this collection plus the evidence ownership tuple, so one evidence
    object can never authorize writes into a different collection.

    This is a separate function and internal evidence type from the reviewed
    semantic path — never a user-controlled boolean. It returns ``store`` only
    for the approved operation/relation matrix and permitted source/version
    records; everything else (claim edges, forged endpoint types, missing or
    altered provenance, ownership mismatch, unsupported operations, and any
    validation error) fails closed with ``reject``. Unlike the semantic path,
    import or validation errors here propagate — they are never swallowed.

    A ``store`` decision authorizes structural persistence only: stored graph
    payloads keep ``requires_review=True`` and ``canonical=False``.
    """
    gate_metadata: dict[str, Any] = {"candidate_type": "mechanical_lineage_candidate"}
    from qdrant_memory import lineage as _lineage
    from qdrant_memory.graph_schema import is_uuid_string as _is_uuid

    # Original tokens are validated before any normalization: an operation or
    # collection name that only matches after stripping is a different token
    # and fails closed, per the identity conventions in section 4.
    op = operation if isinstance(operation, str) else ""
    if op not in _lineage.LINEAGE_OPERATIONS:
        return _decision("reject", ["unsupported_lineage_operation"], confidence=1.0, requires_review=True, metadata=gate_metadata)
    if not isinstance(payload, dict):
        return _decision("reject", ["lineage_payload_invalid"], confidence=1.0, requires_review=True, metadata=gate_metadata)
    if not isinstance(evidence, _lineage.LineageEvidence):
        return _decision("reject", ["lineage_evidence_invalid"], confidence=1.0, requires_review=True, metadata=gate_metadata)
    if str(evidence.operation or "") != op:
        return _decision("reject", ["operation_mismatch"], confidence=1.0, requires_review=True, metadata=gate_metadata)
    if not str(evidence.collection_name or "").strip() or not str(evidence.profile_id or "").strip():
        return _decision("reject", ["lineage_evidence_invalid"], confidence=1.0, requires_review=True, metadata=gate_metadata)

    def _collection_token_invalid(token: Any) -> bool:
        # Whitespace-bearing collection tokens are malformed, not equivalent
        # to their stripped form: writes bind to the exact configured name.
        return not isinstance(token, str) or not token or any(ch.isspace() for ch in token)

    target_collection = collection_name
    if _collection_token_invalid(target_collection) or _collection_token_invalid(evidence.collection_name):
        return _decision("reject", ["lineage_collection_invalid"], confidence=1.0, requires_review=True, metadata=gate_metadata)
    if target_collection != evidence.collection_name:
        return _decision("reject", ["lineage_collection_mismatch"], confidence=1.0, requires_review=True, metadata=gate_metadata)

    if contains_secret(json.dumps(payload, sort_keys=True, default=str)) or _metadata_contains_secret(evidence.to_dict()):
        return _decision("reject", ["possible_secret"], confidence=1.0, requires_review=True, metadata=gate_metadata)

    if not _lineage.ownership_matches(payload, evidence):
        return _decision("reject", ["lineage_ownership_mismatch"], confidence=1.0, requires_review=True, metadata=gate_metadata)

    internal = _lineage.evidence_internal_mismatches(evidence)
    if internal:
        return _decision(
            "reject",
            ["lineage_evidence_invalid", *internal[:8]],
            confidence=1.0,
            requires_review=True,
            metadata=gate_metadata,
        )

    problems = _lineage.validate_lineage_payload(payload)
    if problems:
        return _decision(
            "reject",
            ["lineage_payload_invalid", *problems[:8]],
            confidence=1.0,
            requires_review=True,
            metadata=gate_metadata,
        )

    missing = _lineage.evidence_missing_requirements(payload, evidence)
    if missing:
        return _decision(
            "reject",
            ["lineage_evidence_missing", *missing[:8]],
            confidence=1.0,
            requires_review=True,
            metadata=gate_metadata,
        )

    # The evidence record itself is held to the same shared W0 shape domains
    # as the payload: a present evidence value outside its documented domain
    # (malformed digest, non-UUID token, whitespace-bearing URI, invalid
    # locator, off-vocabulary token) is refused before any comparison.
    evidence_shape = _lineage.evidence_shape_problems(evidence)
    if evidence_shape:
        return _decision(
            "reject",
            ["lineage_evidence_invalid", *evidence_shape[:8]],
            confidence=1.0,
            requires_review=True,
            metadata=gate_metadata,
        )

    mismatches = _lineage.evidence_payload_mismatches(payload, evidence)
    if mismatches:
        return _decision(
            "reject",
            ["lineage_evidence_mismatch", *mismatches[:8]],
            confidence=1.0,
            requires_review=True,
            metadata=gate_metadata,
        )

    identity = _lineage.structural_identity_mismatches(payload, evidence)
    if identity:
        return _decision(
            "reject",
            ["lineage_identity_mismatch", *identity[:8]],
            confidence=1.0,
            requires_review=True,
            metadata=gate_metadata,
        )

    if payload.get("memory_kind") == "graph_edge":
        relation = str(payload.get("relation_type") or "")
        allowed_relations = _lineage.OPERATION_RELATION_MATRIX.get(op, ())
        if relation not in allowed_relations:
            return _decision(
                "reject",
                ["lineage_operation_relation_mismatch"],
                confidence=1.0,
                requires_review=True,
                metadata=gate_metadata,
            )
        if relation == "SUPERSEDES":
            if op == "approved_citation":
                return _decision(
                    "reject",
                    ["lineage_operation_relation_mismatch"],
                    confidence=1.0,
                    requires_review=True,
                    metadata=gate_metadata,
                )
            # The current event reference must come from the evidence and be
            # distinct from its predecessor. W0 binds UUID tokens here; it
            # does not look up event persistence or transition commit state.
            predecessor = str(evidence.predecessor_event_id or "")
            event = str(evidence.event_id or "")
            if not _is_uuid(predecessor):
                return _decision(
                    "reject",
                    ["supersedes_predecessor_required"],
                    confidence=1.0,
                    requires_review=True,
                    metadata=gate_metadata,
                )
            if not _is_uuid(event):
                return _decision(
                    "reject",
                    ["supersedes_event_required"],
                    confidence=1.0,
                    requires_review=True,
                    metadata=gate_metadata,
                )
            if predecessor == event:
                return _decision(
                    "reject",
                    ["supersedes_event_not_distinct"],
                    confidence=1.0,
                    requires_review=True,
                    metadata=gate_metadata,
                )
            if str(payload.get("lineage_event_id") or "") != event:
                return _decision(
                    "reject",
                    ["supersedes_event_mismatch"],
                    confidence=1.0,
                    requires_review=True,
                    metadata=gate_metadata,
                )
        # Approval is required for every approved-citation relation: the
        # operation itself is not approval, and citation-class relations
        # require a recorded approval reference under any operation.
        if relation in ("SUMMARIZES", "EXTRACTED_FROM") or op == "approved_citation":
            if not str(evidence.approval_ref or "").strip():
                return _decision(
                    "reject",
                    ["approved_citation_required"],
                    confidence=1.0,
                    requires_review=True,
                    metadata=gate_metadata,
                )

    return _decision(
        "store",
        ["mechanical_lineage_store"],
        confidence=1.0,
        requires_review=True,
        metadata=gate_metadata,
    )


def decision_to_json(decision: WriteDecision) -> str:
    return json.dumps(decision.to_dict(), sort_keys=True)


# ---------------------------------------------------------------------------
# RAPTOR summary write-gate helper
# ---------------------------------------------------------------------------

_RAPTOR_REQUIRED_FIELDS = ("raptor_node_id", "raptor_child_ids", "source_hashes")
_RAPTOR_CITATION_KEYS = ("derived_from", "evidence", "source_uri", "citations")


def _has_raptor_citations(metadata: dict[str, Any]) -> bool:
    """Check that metadata contains at least one form of citation/provenance."""
    for key in _RAPTOR_CITATION_KEYS:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return True
        if isinstance(value, (list, dict)) and value:
            return True
    return False


def evaluate_raptor_summary_write(
    *,
    text: str,
    metadata: dict[str, Any] | None = None,
    confidence: float | None = None,
) -> WriteDecision:
    """Evaluate a model-authored RAPTOR summary for write acceptance.

    RAPTOR summaries are always derived content — they must route to
    review/reject unless they carry full provenance:

    - Required RAPTOR node fields: ``raptor_node_id``, ``raptor_child_ids``,
      ``source_hashes``.
    - ``canonical`` MUST be exactly the boolean ``False`` (any other value
      is rejected — type-loose checks let tampered digest-consistent
      manifests bypass this invariant).
    - ``requires_review`` MUST be exactly the boolean ``True`` (any other
      value is rejected).
    - ``source_hashes`` MUST be a non-empty list of 64-char lowercase hex
      SHA-256 strings (matches the builder's ``hashlib.sha256(...).hexdigest()``).
      Lists like ``[None]``, ``[{}]``, ``[""]``, or short/non-hex strings
      are rejected.
    - ``derived_from`` MUST be a non-empty list of RAPTOR provenance
      edges, each with non-empty string ``source_uri`` and
      ``child_node_id``, ``derivation_type == "raptor_summary"``, and
      ``relation_type == "SUMMARIZES"``.
    - At least one citation/provenance key: ``derived_from``, ``evidence``,
      ``source_uri``, or ``citations``.
    - Must not contain secrets or recursive contamination markers.

    Returns a :class:`WriteDecision` — never mutates anything. Trust-flag
    and source/provenance violations are returned as ``decision="reject"``
    so the post-enrichment gate cannot diverge from the pre-enrichment
    gate in :mod:`qdrant_memory.raptor.apply`.
    """
    metadata = metadata or {}
    candidate_confidence = _clamp_confidence(confidence)
    cleaned = clean_text_for_memory(text or "")

    # Empty text
    if not cleaned:
        return _decision("skip", ["empty_text"], confidence=1.0, requires_review=False)

    # Secret check — on text and metadata
    if contains_secret(cleaned) or _metadata_contains_secret(metadata):
        return _decision("reject", ["possible_secret"], confidence=1.0, requires_review=True,
                         metadata={"raptor": True})

    # Structural lineage markers are forbidden on RAPTOR summary payloads,
    # whatever their type: a summary marked as structural or pending would be
    # written into the ordinary pool while every structural reader excludes
    # it. This is the same shared gate the post-enrichment path runs, so
    # enrichment-time drift cannot reintroduce the markers either.
    structural_reasons = structural_lineage_marker_reasons(metadata)
    if structural_reasons:
        return _decision(
            "reject",
            ["structural_lineage_marker_forbidden", *structural_reasons],
            confidence=1.0,
            requires_review=True,
            metadata={"raptor": True},
        )

    # Canonical must be exactly the boolean False for RAPTOR summaries.
    # Reject every other value (strings, ints, None, True) — type-loose
    # checks let tampered digest-consistent manifests bypass the trust
    # flag invariant during the post-enrichment gate.
    if metadata.get("canonical") is not False:
        return _decision("reject", ["raptor_summary_must_not_be_canonical"],
                         confidence=1.0, requires_review=True,
                         metadata={"raptor": True,
                                   "bad_canonical": repr(metadata.get("canonical"))})

    # requires_review must be exactly the boolean True. Reject every other
    # value so RAPTOR summaries cannot be silently marked auto-store.
    if metadata.get("requires_review") is not True:
        return _decision("reject", ["raptor_summary_must_not_skip_review"],
                         confidence=1.0, requires_review=True,
                         metadata={"raptor": True,
                                   "bad_requires_review": repr(metadata.get("requires_review"))})

    # Check required RAPTOR fields
    missing_fields = [f for f in _RAPTOR_REQUIRED_FIELDS if not metadata.get(f)]
    if missing_fields:
        return _decision("draft_review", ["missing_raptor_provenance"],
                         confidence=candidate_confidence, requires_review=True,
                         metadata={"raptor": True, "missing_fields": missing_fields})

    # Strict source-hash shape: a non-empty list of 64-char lowercase hex
    # SHA-256 strings. Type-loose checks let ``[None]``, ``[{}]``, ``[""]``,
    # or short/non-hex values pass.
    source_hashes = metadata.get("source_hashes")
    if not isinstance(source_hashes, list) or not source_hashes:
        return _decision("reject", ["raptor_source_hashes_malformed"],
                         confidence=1.0, requires_review=True,
                         metadata={"raptor": True})
    if not all(_is_sha256_hex(h) for h in source_hashes):
        return _decision("reject", ["raptor_source_hashes_malformed"],
                         confidence=1.0, requires_review=True,
                         metadata={"raptor": True})

    # Strict provenance-edge shape: a non-empty list of mappings with the
    # exact RAPTOR provenance schema. The pre-enrichment gate enforces this
    # in ``qdrant_memory/raptor/apply.py``; the post-enrichment gate must
    # not allow the same payloads through after provider metadata is
    # added.
    derived_from = metadata.get("derived_from")
    if not isinstance(derived_from, list) or not derived_from:
        return _decision("reject", ["raptor_derived_from_malformed"],
                         confidence=1.0, requires_review=True,
                         metadata={"raptor": True})
    if not all(_is_raptor_provenance_edge(edge) for edge in derived_from):
        return _decision("reject", ["raptor_derived_from_malformed"],
                         confidence=1.0, requires_review=True,
                         metadata={"raptor": True})

    # Check for at least one citation/provenance (legacy fallback for
    # other citation shapes, retained from Phase 1).
    if not _has_raptor_citations(metadata):
        return _decision("draft_review", ["missing_raptor_citations"],
                         confidence=candidate_confidence, requires_review=True,
                         metadata={"raptor": True})

    # All checks passed — route to review for human/automation approval.
    # RAPTOR summaries are never auto-stored as canonical facts.
    return _decision("draft_review", ["raptor_summary_review_required"],
                     confidence=candidate_confidence, requires_review=True,
                     metadata={"raptor": True})
