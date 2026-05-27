from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from qdrant_memory.consolidation import SECRET_KEYWORDS
from qdrant_memory.extraction_candidates import ExtractionCandidate, ExtractionCandidateType, build_extraction_candidate
from qdrant_memory.lesson_extractor import contains_secret
from qdrant_memory.proposals import proposal_draft_metadata
from qdrant_memory.schema import MEMORY_KINDS, RELATION_TYPES, now_iso


class OntologySuggestionType(str, Enum):
    NEW_MEMORY_KIND = "new_memory_kind"
    NEW_RELATION_TYPE = "new_relation_type"
    MERGE_RENAME_TAGS = "merge_rename_tags"
    NORMALIZE_SUBJECT_ALIASES = "normalize_subject_aliases"
    PROMOTE_FACT_KEY_PATTERN = "promote_fact_key_pattern"


ONTOLOGY_SUGGESTION_TYPES = tuple(suggestion_type.value for suggestion_type in OntologySuggestionType)
_ONTOLOGY_SUGGESTION_TYPE_SET = set(ONTOLOGY_SUGGESTION_TYPES)
_IDENTITY_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d .()\-]{7,}\d)(?!\d)")
_IDENTITY_KEYWORDS = (
    "email",
    "e-mail",
    "phone",
    "address",
    "user_id",
    "userid",
    "chat_id",
    "chatid",
    "profile_id",
    "profileid",
    "account_id",
    "identity",
)


@dataclass(frozen=True)
class OntologySuggestion:
    proposal_id: str
    suggestion_type: str
    title: str
    source_uri: str
    proposed_payload: dict[str, Any]
    evidence: list[Any] = field(default_factory=list)
    locator: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    confidence: float = 0.0
    risk: str = "medium"
    requires_review: bool = True
    manual_review_required: bool = True
    redaction_applied: bool = False
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "proposal_type": "ontology_suggestion",
            "suggestion_type": self.suggestion_type,
            "title": self.title,
            "source_uri": self.source_uri,
            "locator": self.locator,
            "proposed_payload": self.proposed_payload,
            "evidence": self.evidence,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "risk": self.risk,
            "requires_review": self.requires_review,
            "manual_review_required": self.manual_review_required,
            "redaction_applied": self.redaction_applied,
            "suggested_action": "draft_review_only",
            "affected_ids": [],
            "auto_apply_eligible": False,
            "schema_mutation_allowed": False,
            "accepted_change_path": "normal_code_docs_and_tests",
            "created_at": self.created_at,
        }


def _normalize_suggestion_type(suggestion_type: str | OntologySuggestionType) -> str:
    value = suggestion_type.value if isinstance(suggestion_type, OntologySuggestionType) else str(suggestion_type or "")
    value = value.strip()
    if value not in _ONTOLOGY_SUGGESTION_TYPE_SET:
        raise ValueError(f"unknown ontology suggestion type: {value or '<empty>'}")
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
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        normalized = [_jsonable(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("ontology suggestion payload must not contain non-finite floats")
        return value
    return str(value)


def _stable_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _slug(value: str, *, fallback: str = "candidate") -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or fallback


def make_ontology_suggestion_id(
    suggestion_type: str | OntologySuggestionType,
    target_value: str,
    *,
    sensitive: bool = False,
) -> str:
    normalized_type = _normalize_suggestion_type(suggestion_type)
    type_slug = _slug(normalized_type)
    if sensitive:
        digest = hashlib.sha256(_stable_json([normalized_type, target_value]).encode("utf-8")).hexdigest()[:12]
        target_slug = f"manual-review-{digest}"
    else:
        target_slug = _slug(target_value)
    return f"ontology-{type_slug}-{target_slug}"


def _contains_identity(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if any(keyword in key_text for keyword in _IDENTITY_KEYWORDS) and item not in (None, "", [], {}):
                return True
            if _contains_identity(item):
                return True
        return False
    if isinstance(value, list | tuple | set):
        return any(_contains_identity(item) for item in value)
    if isinstance(value, str):
        return bool(_IDENTITY_RE.search(value) or _PHONE_RE.search(value))
    return False


def _contains_secret_like(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if any(keyword in key_text for keyword in SECRET_KEYWORDS) and item not in (None, "", [], {}):
                return True
            if _contains_secret_like(item):
                return True
        return False
    if isinstance(value, list | tuple | set):
        return any(_contains_secret_like(item) for item in value)
    if isinstance(value, str):
        return contains_secret(value)
    return False


def _should_redact_keyed_value(item: Any) -> bool:
    return item not in (None, "", [], {}) and not isinstance(item, bool)


def _redact_secret_like(value: Any) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key).lower()
            if any(keyword in key_text for keyword in SECRET_KEYWORDS) and _should_redact_keyed_value(item):
                redacted[str(key)] = "[redacted: possible secret-bearing value]"
            else:
                redacted[str(key)] = _redact_secret_like(item)
        return redacted
    if isinstance(value, list):
        return [_redact_secret_like(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_secret_like(item) for item in value]
    if isinstance(value, set):
        return sorted((_redact_secret_like(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, str) and (contains_secret(value) or "bearer" in value.lower()):
        return "[redacted: possible secret-bearing value]"
    return value


def _redact_identity(value: Any) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key).lower()
            if any(keyword in key_text for keyword in _IDENTITY_KEYWORDS) and _should_redact_keyed_value(item):
                redacted[str(key)] = "[redacted: identity-bearing value]"
            else:
                redacted[str(key)] = _redact_identity(item)
        return redacted
    if isinstance(value, list):
        return [_redact_identity(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_identity(item) for item in value]
    if isinstance(value, set):
        return sorted((_redact_identity(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, str):
        if _IDENTITY_RE.search(value) or _PHONE_RE.search(value):
            return "[redacted: identity-bearing value]"
    return value


def _redact_sensitive(value: Any) -> Any:
    return _redact_identity(_redact_secret_like(value))


def _normalize_evidence(evidence: list[Any] | tuple[Any, ...] | None) -> list[Any]:
    normalized = _jsonable(list(evidence or []))
    return normalized if isinstance(normalized, list) else []


def _build_suggestion(
    *,
    suggestion_type: str | OntologySuggestionType,
    target_value: str,
    title: str,
    source_uri: str,
    ontology_field: str,
    proposed_payload: Mapping[str, Any],
    evidence: list[Any] | tuple[Any, ...] | None,
    locator: Mapping[str, Any] | None = None,
    rationale: str = "",
    confidence: float = 0.0,
    risk: str = "medium",
) -> OntologySuggestion:
    normalized_type = _normalize_suggestion_type(suggestion_type)
    normalized_source_uri = str(source_uri or "").strip()
    if not normalized_source_uri:
        raise ValueError("source_uri is required")
    normalized_confidence = _normalize_confidence(confidence)
    raw_evidence = _normalize_evidence(evidence)
    raw_locator = _jsonable(locator or {})
    raw_payload = _jsonable(
        {
            **dict(proposed_payload),
            "ontology_field": ontology_field,
            "suggestion_type": normalized_type,
            "target_value": target_value,
            "candidate_value": dict(proposed_payload).get("candidate_value", target_value),
            "schema_mutation_allowed": False,
            "auto_apply_eligible": False,
            "accepted_change_path": "normal_code_docs_and_tests",
            "safety": {
                "proposal_draft_only": True,
                "no_cron_watcher_auto_apply": True,
                "accepted_changes_require_code_docs_tests": True,
            },
        }
    )
    raw_bundle = {"payload": raw_payload, "evidence": raw_evidence, "locator": raw_locator, "rationale": rationale}
    secret_bearing = _contains_secret_like(raw_bundle)
    identity_bearing = _contains_identity(raw_bundle)
    sensitive = secret_bearing or identity_bearing
    proposal_id = make_ontology_suggestion_id(normalized_type, target_value, sensitive=sensitive)

    payload_with_safety = {
        **raw_payload,
        "safety": {
            **raw_payload.get("safety", {}),
            "secret_bearing": secret_bearing,
            "identity_bearing": identity_bearing,
            "manual_review_required": True,
        },
    }
    safe_payload = _redact_sensitive(payload_with_safety)
    safe_evidence = _redact_sensitive(raw_evidence)
    safe_locator = _redact_sensitive(raw_locator)
    safe_rationale = str(_redact_sensitive(rationale))
    redaction_applied = _stable_json({"payload": payload_with_safety, "evidence": raw_evidence}) != _stable_json(
        {"payload": safe_payload, "evidence": safe_evidence}
    )
    normalized_risk = str(risk or "medium").strip().lower() or "medium"
    if sensitive:
        normalized_risk = "high"
    return OntologySuggestion(
        proposal_id=proposal_id,
        suggestion_type=normalized_type,
        title=title,
        source_uri=normalized_source_uri,
        locator=safe_locator if isinstance(safe_locator, dict) else {},
        proposed_payload=safe_payload if isinstance(safe_payload, dict) else {},
        evidence=safe_evidence if isinstance(safe_evidence, list) else [],
        rationale=safe_rationale,
        confidence=normalized_confidence,
        risk=normalized_risk,
        redaction_applied=redaction_applied,
    )


def suggest_new_memory_kind(
    candidate_kind: str,
    *,
    evidence: list[Any] | tuple[Any, ...] | None,
    source_uri: str,
    locator: Mapping[str, Any] | None = None,
    rationale: str = "Repeated memories do not fit the current memory_kind grammar cleanly.",
    confidence: float = 0.0,
) -> OntologySuggestion:
    value = str(candidate_kind or "").strip()
    if not value:
        raise ValueError("candidate_kind is required")
    return _build_suggestion(
        suggestion_type=OntologySuggestionType.NEW_MEMORY_KIND,
        target_value=value,
        title=f"Consider new memory_kind candidate: {value}",
        source_uri=source_uri,
        locator=locator,
        ontology_field="memory_kind",
        proposed_payload={"candidate_value": value, "existing_values": list(MEMORY_KINDS)},
        evidence=evidence,
        rationale=rationale,
        confidence=confidence,
    )


def suggest_new_relation_type(
    candidate_relation: str,
    *,
    evidence: list[Any] | tuple[Any, ...] | None,
    source_uri: str,
    locator: Mapping[str, Any] | None = None,
    rationale: str = "Repeated derivation edges appear to need a more precise relation_type.",
    confidence: float = 0.0,
) -> OntologySuggestion:
    value = str(candidate_relation or "").strip()
    if not value:
        raise ValueError("candidate_relation is required")
    return _build_suggestion(
        suggestion_type=OntologySuggestionType.NEW_RELATION_TYPE,
        target_value=value,
        title=f"Consider new relation_type candidate: {value}",
        source_uri=source_uri,
        locator=locator,
        ontology_field="relation_type",
        proposed_payload={"candidate_value": value, "existing_values": list(RELATION_TYPES)},
        evidence=evidence,
        rationale=rationale,
        confidence=confidence,
    )


def suggest_tag_merge_or_rename(
    *,
    source_tags: list[str] | tuple[str, ...],
    canonical_tag: str,
    evidence: list[Any] | tuple[Any, ...] | None,
    source_uri: str,
    locator: Mapping[str, Any] | None = None,
    rationale: str = "Repeated tag variants appear to describe the same concept.",
    confidence: float = 0.0,
) -> OntologySuggestion:
    tags = [str(tag or "").strip().lstrip("#") for tag in source_tags if str(tag or "").strip()]
    canonical = str(canonical_tag or "").strip().lstrip("#")
    if not tags or not canonical:
        raise ValueError("source_tags and canonical_tag are required")
    return _build_suggestion(
        suggestion_type=OntologySuggestionType.MERGE_RENAME_TAGS,
        target_value=canonical,
        title=f"Consider tag merge/rename to: {canonical}",
        source_uri=source_uri,
        locator=locator,
        ontology_field="tag",
        proposed_payload={"canonical_tag": canonical, "source_tags": tags, "candidate_value": canonical},
        evidence=evidence,
        rationale=rationale,
        confidence=confidence,
    )


def suggest_subject_alias_normalization(
    *,
    canonical_subject: str,
    aliases: list[str] | tuple[str, ...],
    evidence: list[Any] | tuple[Any, ...] | None,
    source_uri: str,
    locator: Mapping[str, Any] | None = None,
    rationale: str = "Repeated subject aliases appear to refer to the same subject.",
    confidence: float = 0.0,
) -> OntologySuggestion:
    canonical = str(canonical_subject or "").strip()
    normalized_aliases = [str(alias or "").strip() for alias in aliases if str(alias or "").strip()]
    if not canonical or not normalized_aliases:
        raise ValueError("canonical_subject and aliases are required")
    return _build_suggestion(
        suggestion_type=OntologySuggestionType.NORMALIZE_SUBJECT_ALIASES,
        target_value=canonical,
        title=f"Consider subject alias normalization: {canonical}",
        source_uri=source_uri,
        locator=locator,
        ontology_field="subject_alias",
        proposed_payload={
            "canonical_subject": canonical,
            "aliases": normalized_aliases,
            "candidate_value": canonical,
        },
        evidence=evidence,
        rationale=rationale,
        confidence=confidence,
    )


def suggest_fact_key_pattern_promotion(
    *,
    pattern: str,
    examples: list[str] | tuple[str, ...],
    evidence: list[Any] | tuple[Any, ...] | None,
    source_uri: str,
    locator: Mapping[str, Any] | None = None,
    rationale: str = "Repeated fact_key shapes may deserve a documented canonical pattern.",
    confidence: float = 0.0,
) -> OntologySuggestion:
    normalized_pattern = str(pattern or "").strip()
    normalized_examples = [str(example or "").strip() for example in examples if str(example or "").strip()]
    if not normalized_pattern or not normalized_examples:
        raise ValueError("pattern and examples are required")
    return _build_suggestion(
        suggestion_type=OntologySuggestionType.PROMOTE_FACT_KEY_PATTERN,
        target_value=normalized_pattern,
        title=f"Consider promoting repeated fact_key pattern: {normalized_pattern}",
        source_uri=source_uri,
        locator=locator,
        ontology_field="fact_key_pattern",
        proposed_payload={
            "pattern": normalized_pattern,
            "examples": normalized_examples,
            "candidate_value": normalized_pattern,
        },
        evidence=evidence,
        rationale=rationale,
        confidence=confidence,
    )


def _suggestion_dict(suggestion: OntologySuggestion | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(suggestion, OntologySuggestion):
        return suggestion.to_dict()
    normalized = _jsonable(suggestion)
    if not isinstance(normalized, dict):
        raise TypeError("ontology suggestion must be a mapping")
    return normalized


def preview_ontology_suggestions(
    suggestions: list[OntologySuggestion | Mapping[str, Any]],
    *,
    qdrant: Any | None = None,
    embeddings: Any | None = None,
) -> dict[str, Any]:
    del qdrant, embeddings  # Explicitly unused: previews must not contact embedding or Qdrant backends.
    proposals = [_redact_sensitive(_suggestion_dict(suggestion)) for suggestion in suggestions]
    return {
        "dry_run": True,
        "proposal_type": "ontology_suggestion",
        "count": len(proposals),
        "proposal_ids": [str(proposal.get("proposal_id") or "") for proposal in proposals],
        "proposals": proposals,
        "auto_apply_allowed": False,
        "schema_mutation_allowed": False,
        "accepted_change_path": "normal_code_docs_and_tests",
    }


def render_ontology_suggestion_markdown(suggestion: OntologySuggestion | Mapping[str, Any], *, report_id: str) -> str:
    proposal = _redact_sensitive(_suggestion_dict(suggestion))
    lines = [
        "# Ontology suggestion draft",
        "",
        "Proposal/draft artifact only. This draft must not mutate schema, update live ontology, or write Qdrant.",
        "manual review is required before any accepted ontology change is implemented.",
        "No cron/watcher auto-apply is permitted for ontology suggestions.",
        "Accepted ontology changes require normal code/docs changes and tests.",
        "",
        f"- report_id: {report_id}",
        f"- proposal_id: {proposal.get('proposal_id', 'N/A')}",
        f"- proposal_type: {proposal.get('proposal_type', 'ontology_suggestion')}",
        f"- suggestion_type: {proposal.get('suggestion_type', 'N/A')}",
        f"- suggested_action: {proposal.get('suggested_action', 'draft_review_only')}",
        f"- manual_review_required: {proposal.get('manual_review_required', True)}",
        f"- auto_apply_eligible: {proposal.get('auto_apply_eligible', False)}",
        f"- schema_mutation_allowed: {proposal.get('schema_mutation_allowed', False)}",
        f"- risk: {proposal.get('risk', 'unknown')}",
        "",
        "## Proposed ontology change",
        "",
        "```json",
        json.dumps(proposal.get("proposed_payload", {}), indent=2, sort_keys=True, ensure_ascii=False),
        "```",
        "",
        "## Evidence",
        "",
        "```json",
        json.dumps(proposal.get("evidence", []), indent=2, sort_keys=True, ensure_ascii=False),
        "```",
        "",
        "## Reviewer checklist",
        "",
        "- Treat this as a review-only grammar improvement proposal.",
        "- Do not apply it through cron, watcher, consolidation apply, or live memory mutation paths.",
        "- If accepted, implement through explicit source edits, documentation updates, and tests.",
        "- Re-check redaction before copying sensitive or identity-bearing evidence elsewhere.",
        "",
    ]
    return "\n".join(lines)


def write_ontology_suggestion_draft(
    suggestion: OntologySuggestion | Mapping[str, Any],
    *,
    report_id: str,
    hermes_home: str,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    proposal = _redact_sensitive(_suggestion_dict(suggestion))
    metadata = proposal_draft_metadata(
        report={"report_id": report_id},
        proposal=proposal,
        points=[],
        hermes_home=hermes_home,
        config=config,
        write_decision={"decision": "draft_review", "requires_review": True},
    )
    metadata.update(
        {
            "kind": "ontology_suggestion_draft",
            "auto_apply_eligible": False,
            "schema_mutation_allowed": False,
            "manual_review_required": True,
            "accepted_change_path": "normal_code_docs_and_tests",
        }
    )
    markdown_path = Path(str(metadata["path"]))
    metadata_path = Path(str(metadata["metadata_path"]))
    markdown_path.write_text(render_ontology_suggestion_markdown(proposal, report_id=str(metadata["report_id"])), encoding="utf-8")
    metadata_path.write_text(json.dumps(_redact_sensitive(metadata), indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def build_ontology_extraction_candidate(
    suggestion: OntologySuggestion | Mapping[str, Any],
    *,
    created_at: str | None = None,
) -> ExtractionCandidate:
    proposal = _redact_sensitive(_suggestion_dict(suggestion))
    payload = proposal.get("proposed_payload") if isinstance(proposal.get("proposed_payload"), dict) else {}
    proposed_payload = {
        "proposal_id": proposal.get("proposal_id"),
        "proposal_type": proposal.get("proposal_type", "ontology_suggestion"),
        "suggestion_type": proposal.get("suggestion_type"),
        "ontology_field": payload.get("ontology_field"),
        "candidate_value": payload.get("candidate_value"),
        "proposal_payload": payload,
        "schema_mutation_allowed": False,
        "auto_apply_eligible": False,
        "accepted_change_path": "normal_code_docs_and_tests",
    }
    source_uri = str(proposal.get("source_uri") or "").strip()
    return build_extraction_candidate(
        candidate_type=ExtractionCandidateType.ONTOLOGY_SUGGESTION,
        source_uri=source_uri,
        locator=proposal.get("locator") if isinstance(proposal.get("locator"), dict) else {},
        derived_from=[{"source_uri": source_uri, "derivation_type": "ontology_suggestion", "relation_type": "DERIVED_FROM"}],
        proposed_payload=proposed_payload,
        reason=str(proposal.get("rationale") or proposal.get("title") or "ontology suggestion"),
        confidence=proposal.get("confidence", 0.0),
        risk=str(proposal.get("risk") or "medium"),
        requires_review=True,
        created_at=created_at,
        lifecycle_id=str(proposal.get("proposal_id") or ""),
    )
