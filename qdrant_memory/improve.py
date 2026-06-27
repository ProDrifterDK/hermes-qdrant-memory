"""Post-session graph improve preview/apply pipeline.

Phase 3 of the graph memory roadmap.  This module provides pure, deterministic
functions for:

1. Building graph entity/edge candidates from explicit source text or
   deterministic graph-grammar markers.
2. Computing stable report IDs and candidate digests.
3. Previewing improve reports (no Qdrant writes, no embeddings).
4. Persisting/loading report artifacts for audit.

The actual *apply* (live Qdrant upsert) is performed by the provider in
``__init__.py`` which routes the persisted payload through the shared write
gate before any mutation.  This module intentionally has zero side effects on
Qdrant.

Safety invariants:
- No LLM extraction — only deterministic rules and explicit graph markers.
- No new dependencies (no Cognee/Neo4j/networkx/etc.).
- IDs never contain raw source text, secrets, or timestamps.
- Every candidate carries source provenance.
- Graph payloads are built through ``graph_schema.build_entity_payload`` /
  ``build_edge_payload`` only.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

from qdrant_memory.consolidation import redact_secrets
from qdrant_memory.extraction_candidates import (
    ExtractionCandidate,
    _stable_json,
    build_extraction_candidate,
)
from qdrant_memory.graph_schema import (
    build_edge_payload,
    build_entity_payload,
    make_edge_id,
    make_entity_id,
    sanitize_profile_id,
)
from qdrant_memory.lesson_extractor import contains_secret
from qdrant_memory.schema import clean_text_for_memory
from qdrant_memory.source_extraction import (
    LOW_CONFIDENCE_DRAFT_THRESHOLD,
    evaluate_source_extraction_candidate,
    extract_source_candidates_from_text,
)
from qdrant_memory.write_gate import WriteDecision

IMPROVE_EXTRACTOR_VERSION = "graph_improve_v1"
IMPROVE_MAX_CANDIDATES_DEFAULT = 20
IMPROVE_MAX_CANDIDATES_HARD_CAP = 50

# Strict canonical report ID format: improve-<12 hex chars>
REPORT_ID_RE = re.compile(r"^improve-[a-f0-9]{12}$")

# ---------------------------------------------------------------------------
# Safe graph entity-type allowlist (Phase 3)
# ---------------------------------------------------------------------------

# Phase 3 uses an ALLOWLIST model: only known non-personal graph entity types
# may be store-eligible via automatic improve extraction.  Any entity type not
# in this set is treated as identity-bearing / unsafe and rejected at
# construction so that raw labels never enter preview JSON, persisted report
# JSON, draft artifacts, live errors, or Qdrant upsert payloads.
#
# To add a new safe type, append it here.  Unknown/unlisted types are rejected
# by default — they are NOT assumed safe.
_SAFE_GRAPH_ENTITY_TYPES = frozenset({
    "project", "tool", "concept", "service", "repo", "package",
})

# Regex patterns for identity values in labels/source URIs
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# Phone-like: sequences of 7+ digits with optional +, separators, spaces
_PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{6,}\d)")
# Identity-key labels (PII-ish field names in labels or entity types)
_IDENTITY_VALUE_KEYWORDS = frozenset({
    "email", "phone", "address", "ssn", "password", "credential",
    "token", "api_key", "passport", "national_id", "tax_id",
})


def is_identity_bearing_value(text: str) -> bool:
    """Check if a label/source value contains identity-bearing content.

    Covers emails, phone-like values, and identity-value keywords.
    """
    s = str(text or "").strip()
    if not s:
        return False
    if _EMAIL_RE.search(s):
        return True
    if _PHONE_RE.search(s):
        return True
    tokens = set(t.lower() for t in re.split(r"[\W_]+", s) if t)
    if tokens & _IDENTITY_VALUE_KEYWORDS:
        return True
    return False


def is_safe_graph_entity_type(entity_type: str) -> bool:
    """Check if an entity type is in the Phase 3 safe allowlist.

    Only allowlisted non-personal types (project, tool, concept, service, repo,
    package) are store-eligible.  All other types — including unlisted
    person/role/customer synonyms — are NOT safe.
    """
    return entity_type.strip().lower() in _SAFE_GRAPH_ENTITY_TYPES


def is_identity_bearing_entity_type(entity_type: str) -> bool:
    """Check if an entity type is identity-bearing / NOT safe for Phase 3.

    Under the allowlist model this returns True for any entity type that is
    NOT in the safe allowlist.  This keeps the existing call-sites working
    while switching from denylist to allowlist semantics.
    """
    return not is_safe_graph_entity_type(entity_type)


def is_identity_bearing_graph_candidate(
    *,
    entity_type: str = "",
    label: str = "",
    source_uri: str = "",
    src_label: str = "",
    tgt_label: str = "",
) -> bool:
    """Check if a graph entity/edge candidate is identity-bearing.

    Returns True if any label/value/source is identity-bearing, or if
    the entity type itself is NOT in the safe allowlist.
    """
    if is_identity_bearing_entity_type(entity_type):
        return True
    for val in [label, source_uri, src_label, tgt_label]:
        if is_identity_bearing_value(val):
            return True
    return False


# ---------------------------------------------------------------------------
# Deterministic graph-grammar markers
# ---------------------------------------------------------------------------

# "Graph entity: <entity_type>: <label>"
_ENTITY_RE = re.compile(
    r"(?i)^\s*graph\s+entity\s*:\s*"
    r"(?P<entity_type>[a-zA-Z][a-zA-Z0-9_]*)\s*:\s*"
    r"(?P<label>.+)$"
)

# "Graph edge: <src_type>:<src_label> -[RELATION_TYPE]-> <tgt_type>:<tgt_label>"
_EDGE_RE = re.compile(
    r"(?i)^\s*graph\s+edge\s*:\s*"
    r"(?P<src_entity_type>[a-zA-Z][a-zA-Z0-9_]*)\s*:\s*(?P<src_label>[^-]+?)"
    r"\s*-\[(?P<relation_type>[A-Z_]+)\]->\s*"
    r"(?P<tgt_entity_type>[a-zA-Z][a-zA-Z0-9_]*)\s*:\s*(?P<tgt_label>.+)$"
)


def _is_unsafe_text(text: str) -> bool:
    clean = clean_text_for_memory(text or "")
    if not clean:
        return True
    return contains_secret(clean)


def _safe_source_uri(source_uri: str, fallback: str = "") -> str:
    uri = str(source_uri or "").strip()
    if not uri or contains_secret(uri):
        return str(fallback or "").strip()
    return uri


# ---------------------------------------------------------------------------
# Stable ID / digest helpers
# ---------------------------------------------------------------------------

def _make_report_id(
    *,
    profile_id: str,
    session_id: str,
    source_scope: str,
    source_handles: list[str],
    candidate_ids: list[str],
) -> str:
    fingerprint = {
        "extractor_version": IMPROVE_EXTRACTOR_VERSION,
        "profile_id": profile_id,
        "session_id": session_id,
        "source_scope": source_scope,
        "source_handles": sorted(source_handles),
        "candidate_ids": sorted(candidate_ids),
    }
    digest = hashlib.sha256(_stable_json(fingerprint).encode("utf-8")).hexdigest()
    return f"improve-{digest[:12]}"


def make_candidate_digest(candidate_item: Mapping[str, Any]) -> str:
    """Compute a stable digest for a report candidate item."""
    write_decision = candidate_item.get("write_decision") or {}
    if isinstance(write_decision, dict):
        wd_digest = [write_decision.get("decision", ""), *write_decision.get("reasons", [])]
    else:
        wd_digest = [str(write_decision)]
    fingerprint = {
        "candidate_id": candidate_item.get("candidate_id", ""),
        "candidate_type": candidate_item.get("candidate_type", ""),
        "target_point_id": candidate_item.get("target_point_id", ""),
        "proposed_payload": candidate_item.get("proposed_payload", {}),
        "derived_from": candidate_item.get("derived_from", []),
        "source_uri": candidate_item.get("source_uri", ""),
        "locator": candidate_item.get("locator", {}),
        "write_decision": wd_digest,
    }
    digest = hashlib.sha256(_stable_json(fingerprint).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


# ---------------------------------------------------------------------------
# Graph candidate builders
# ---------------------------------------------------------------------------

def _build_entity_candidate(
    *,
    entity_type: str,
    label: str,
    source_uri: str,
    source_type: str,
    locator: Mapping[str, Any],
    derivation_type: str,
    confidence: float,
    profile_id: str,
    lifecycle_id: str,
    point_id: str = "",
) -> ExtractionCandidate | None:
    label_clean = clean_text_for_memory(label).strip()
    entity_type_clean = entity_type.strip().lower()
    if not label_clean or not entity_type_clean:
        return None
    if _is_unsafe_text(label_clean) or _is_unsafe_text(entity_type_clean):
        return None
    if is_identity_bearing_graph_candidate(
        entity_type=entity_type_clean, label=label_clean, source_uri=source_uri
    ):
        return None
    safe_uri = _safe_source_uri(source_uri)
    if not safe_uri:
        return None
    safe_profile = sanitize_profile_id(profile_id)
    try:
        payload = build_entity_payload(
            entity_type=entity_type_clean,
            label=label_clean,
            profile_id=safe_profile,
            confidence=confidence,
            source_uri=safe_uri,
            tags=["improve", "graph_entity"],
        )
    except (ValueError, TypeError):
        return None
    # Normalize volatile timestamps so candidate IDs are deterministic.
    # The store path will set real timestamps via provider metadata injection.
    payload["created_at"] = "2000-01-01T00:00:00Z"
    payload["updated_at"] = "2000-01-01T00:00:00Z"
    derived_from = [
        {
            "source_uri": safe_uri,
            "source_type": source_type,
            "derivation_type": derivation_type,
            "relation_type": "EXTRACTED_FROM",
        }
    ]
    if point_id:
        derived_from[0]["point_id"] = point_id
    try:
        return build_extraction_candidate(
            candidate_type="graph_entity_candidate",
            source_uri=safe_uri,
            locator=dict(locator),
            derived_from=derived_from,
            proposed_payload=payload,
            reason=f"graph entity from {derivation_type}",
            confidence=confidence,
            risk="low" if confidence >= LOW_CONFIDENCE_DRAFT_THRESHOLD else "medium",
            requires_review=True,
            lifecycle_id=lifecycle_id,
        )
    except (ValueError, TypeError):
        return None


def _build_edge_candidate(
    *,
    src_entity_type: str,
    src_label: str,
    tgt_entity_type: str,
    tgt_label: str,
    relation_type: str,
    source_uri: str,
    source_type: str,
    locator: Mapping[str, Any],
    derivation_type: str,
    confidence: float,
    profile_id: str,
    lifecycle_id: str,
    point_id: str = "",
) -> ExtractionCandidate | None:
    src_label_clean = clean_text_for_memory(src_label).strip()
    tgt_label_clean = clean_text_for_memory(tgt_label).strip()
    src_type_clean = src_entity_type.strip().lower()
    tgt_type_clean = tgt_entity_type.strip().lower()
    rel_clean = relation_type.strip().upper()
    if not all([src_label_clean, tgt_label_clean, src_type_clean, tgt_type_clean, rel_clean]):
        return None
    if any(_is_unsafe_text(t) for t in [src_label_clean, tgt_label_clean, rel_clean]):
        return None
    if is_identity_bearing_graph_candidate(
        entity_type=src_type_clean, src_label=src_label_clean,
        tgt_label=tgt_label_clean, source_uri=source_uri,
    ) or is_identity_bearing_graph_candidate(
        entity_type=tgt_type_clean, label=tgt_label_clean, source_uri=source_uri,
    ):
        return None
    safe_uri = _safe_source_uri(source_uri)
    if not safe_uri:
        return None
    safe_profile = sanitize_profile_id(profile_id)
    src_entity_id = make_entity_id(src_type_clean, src_label_clean, profile_id=safe_profile)
    tgt_entity_id = make_entity_id(tgt_type_clean, tgt_label_clean, profile_id=safe_profile)
    try:
        payload = build_edge_payload(
            source_entity_id=src_entity_id,
            target_entity_id=tgt_entity_id,
            relation_type=rel_clean,
            profile_id=safe_profile,
            confidence=confidence,
            source_uri=safe_uri,
            tags=["improve", "graph_edge"],
        )
    except (ValueError, TypeError):
        return None
    # Normalize volatile timestamps so candidate IDs are deterministic.
    payload["created_at"] = "2000-01-01T00:00:00Z"
    payload["updated_at"] = "2000-01-01T00:00:00Z"
    derived_from = [
        {
            "source_uri": safe_uri,
            "source_type": source_type,
            "derivation_type": derivation_type,
            "relation_type": "EXTRACTED_FROM",
        }
    ]
    if point_id:
        derived_from[0]["point_id"] = point_id
    try:
        return build_extraction_candidate(
            candidate_type="graph_edge_candidate",
            source_uri=safe_uri,
            locator=dict(locator),
            derived_from=derived_from,
            proposed_payload=payload,
            reason=f"graph edge from {derivation_type}",
            confidence=confidence,
            risk="low" if confidence >= LOW_CONFIDENCE_DRAFT_THRESHOLD else "medium",
            requires_review=True,
            lifecycle_id=lifecycle_id,
        )
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Source material -> graph candidates
# ---------------------------------------------------------------------------

def _extract_graph_markers_from_text(
    text: str,
    *,
    source_uri: str,
    source_type: str,
    locator: Mapping[str, Any],
    derivation_type: str,
    confidence: float,
    profile_id: str,
    lifecycle_id: str,
    point_id: str = "",
) -> list[ExtractionCandidate]:
    candidates: list[ExtractionCandidate] = []
    for index, line in enumerate((text or "").splitlines(), start=1):
        if _is_unsafe_text(line):
            continue
        if is_identity_bearing_value(line):
            continue
        line_locator = dict(locator)
        line_locator.setdefault("line_start", index)
        line_locator.setdefault("line_end", index)
        entity_match = _ENTITY_RE.match(line)
        if entity_match:
            c = _build_entity_candidate(
                entity_type=entity_match.group("entity_type"),
                label=entity_match.group("label"),
                source_uri=source_uri,
                source_type=source_type,
                locator=line_locator,
                derivation_type=derivation_type,
                confidence=confidence,
                profile_id=profile_id,
                lifecycle_id=lifecycle_id,
                point_id=point_id,
            )
            if c:
                candidates.append(c)
            continue
        edge_match = _EDGE_RE.match(line)
        if edge_match:
            c = _build_edge_candidate(
                src_entity_type=edge_match.group("src_entity_type"),
                src_label=edge_match.group("src_label"),
                tgt_entity_type=edge_match.group("tgt_entity_type"),
                tgt_label=edge_match.group("tgt_label"),
                relation_type=edge_match.group("relation_type"),
                source_uri=source_uri,
                source_type=source_type,
                locator=line_locator,
                derivation_type=derivation_type,
                confidence=confidence,
                profile_id=profile_id,
                lifecycle_id=lifecycle_id,
                point_id=point_id,
            )
            if c:
                candidates.append(c)
    return candidates


def _dedupe(candidates: list[ExtractionCandidate], *, max_candidates: int) -> list[ExtractionCandidate]:
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


def extract_improve_candidates_from_text(
    text: str,
    *,
    source_uri: str,
    source_type: str = "source_text",
    locator: Mapping[str, Any] | None = None,
    derivation_type: str = "source_text",
    confidence: float = 0.8,
    profile_id: str = "default",
    lifecycle_id: str = "",
    max_candidates: int = IMPROVE_MAX_CANDIDATES_DEFAULT,
    point_id: str = "",
) -> list[ExtractionCandidate]:
    """Extract graph candidates from explicit source text using deterministic rules.

    Graph markers:
      - "Graph entity: <type>: <label>"
      - "Graph edge: <src_type>:<src_label> -[RELATION_TYPE]-> <tgt_type>:<tgt_label>"

    Also reuses existing source_extraction signal extraction for non-graph
    source-backed candidates (decisions, preferences, invariants, etc.) so that
    improve can surface both graph and regular source-backed candidates.
    """
    cap = min(max_candidates, IMPROVE_MAX_CANDIDATES_HARD_CAP)
    safe_uri = _safe_source_uri(source_uri)
    if not safe_uri:
        return []
    base_locator = dict(locator or {})

    graph_candidates = _extract_graph_markers_from_text(
        text,
        source_uri=safe_uri,
        source_type=source_type,
        locator=base_locator,
        derivation_type=derivation_type,
        confidence=confidence,
        profile_id=profile_id,
        lifecycle_id=lifecycle_id,
        point_id=point_id,
    )

    # Also extract regular source-backed candidates from the same text.
    regular_candidates = extract_source_candidates_from_text(
        text,
        source_uri=safe_uri,
        source_type=source_type,
        locator=base_locator,
        derivation_type=derivation_type,
        lifecycle_id=lifecycle_id,
        max_candidates=cap,
        point_id=point_id,
    )

    return _dedupe(graph_candidates + regular_candidates, max_candidates=cap)


def extract_improve_candidates_from_point(
    point: Any,
    *,
    confidence: float = 0.75,
    profile_id: str = "default",
    lifecycle_id: str = "",
    max_candidates: int = IMPROVE_MAX_CANDIDATES_DEFAULT,
) -> list[ExtractionCandidate]:
    """Extract graph + regular source candidates from a Qdrant point (read-only)."""
    if isinstance(point, Mapping):
        point_id = str(point.get("id") or "")
        raw_payload = point.get("payload")
        payload = raw_payload if isinstance(raw_payload, Mapping) else {}
        text = str(point.get("text") or payload.get("text") or "")
    else:
        point_id = str(getattr(point, "id", "") or "")
        raw_payload = getattr(point, "payload", None)
        payload = raw_payload if isinstance(raw_payload, Mapping) else {}
        text = str(getattr(point, "text", "") or payload.get("text") or "")
    source_uri = str(payload.get("source_uri") or (f"qdrant://memory/{point_id}" if point_id else ""))
    locator = payload.get("locator") if isinstance(payload.get("locator"), Mapping) else {}
    return extract_improve_candidates_from_text(
        text,
        source_uri=source_uri,
        source_type="recalled_point",
        locator=locator,
        derivation_type="recalled_point",
        confidence=confidence,
        profile_id=profile_id,
        lifecycle_id=lifecycle_id,
        max_candidates=max_candidates,
        point_id=point_id,
    )


# ---------------------------------------------------------------------------
# Preview report builder
# ---------------------------------------------------------------------------

def _candidate_target_point_id(candidate: ExtractionCandidate) -> str:
    """Get the Qdrant target point ID for a candidate (entity_id or edge_id)."""
    payload = candidate.proposed_payload or {}
    if candidate.candidate_type == "graph_entity_candidate":
        return str(payload.get("entity_id") or "")
    if candidate.candidate_type == "graph_edge_candidate":
        return str(payload.get("edge_id") or "")
    # Regular candidates use candidate_id as point ID (matching extraction flow)
    return candidate.candidate_id


def _candidate_target_memory_kind(candidate: ExtractionCandidate) -> str:
    payload = candidate.proposed_payload or {}
    return str(payload.get("memory_kind") or "")


def build_improve_report(
    candidates: list[ExtractionCandidate],
    *,
    profile_id: str,
    session_id: str,
    source_scope: str,
    source_handles: list[str],
    persist: bool = True,
    include_metadata: bool = False,
) -> dict[str, Any]:
    """Build a complete improve preview report from candidates.

    This function is pure: it does not embed, upsert, delete, or update Qdrant.
    """
    items: list[dict[str, Any]] = []
    candidate_ids: list[str] = []

    for candidate in candidates:
        decision = evaluate_source_extraction_candidate(candidate)
        item = redact_secrets(candidate.to_dict())
        if not isinstance(item, dict):
            item = candidate.to_dict()
        target_pid = _candidate_target_point_id(candidate)
        item["target_point_id"] = target_pid
        item["target_memory_kind"] = _candidate_target_memory_kind(candidate)
        item["write_decision"] = decision.to_dict()
        item["candidate_digest"] = ""
        item["apply_eligible"] = decision.decision == "store"
        item["would_store"] = decision.decision == "store"
        item["would_create_proposal"] = decision.decision == "draft_review"
        item["candidate_digest"] = make_candidate_digest(item)
        if not include_metadata:
            # Remove verbose metadata to keep reports compact
            item.pop("created_at", None)
        items.append(item)
        candidate_ids.append(candidate.candidate_id)

    report_id = _make_report_id(
        profile_id=profile_id,
        session_id=session_id,
        source_scope=source_scope,
        source_handles=source_handles,
        candidate_ids=candidate_ids,
    )

    report_digest_payload = {
        "report_id": report_id,
        "candidate_digests": [item["candidate_digest"] for item in items],
    }
    report_digest = "sha256:" + hashlib.sha256(
        _stable_json(report_digest_payload).encode("utf-8")
    ).hexdigest()

    counts = {
        "total": len(items),
        "store_eligible": sum(1 for item in items if item["would_store"]),
        "draft_review": sum(1 for item in items if item["would_create_proposal"]),
        "rejected": sum(1 for item in items if item["write_decision"]["decision"] == "reject"),
    }

    report: dict[str, Any] = {
        "report_id": report_id,
        "report_type": "graph_improve_preview",
        "extractor_version": IMPROVE_EXTRACTOR_VERSION,
        "report_digest": report_digest,
        "dry_run": True,
        "persisted": persist,
        "profile_id": profile_id,
        "session_id": session_id,
        "source_scope": source_scope,
        "source_handles": source_handles,
        "candidates": items,
        "counts": counts,
    }
    return report


def persist_improve_report(
    report: dict[str, Any],
    *,
    hermes_home: str,
    artifact_dir: str = "",
) -> dict[str, Any]:
    """Persist an improve report as a local JSON artifact.

    Returns dict with path and report_id.
    """
    from pathlib import Path

    if not artifact_dir:
        artifact_dir = str(Path(hermes_home) / "qdrant_memory" / "improve_reports")
    dir_path = Path(artifact_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    report_id = report.get("report_id", "unknown")
    if not REPORT_ID_RE.match(report_id):
        raise ValueError(f"Refusing to persist report with non-canonical report_id: {report_id!r}")
    file_path = dir_path / f"{report_id}.json"
    file_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report["artifact_path"] = str(file_path)
    return {"path": str(file_path), "report_id": report_id}


def load_improve_report(
    report_id: str,
    *,
    hermes_home: str,
    artifact_dir: str = "",
) -> dict[str, Any] | None:
    """Load a persisted improve report by report_id.

    The report_id must match the strict canonical format ``improve-<12 hex>``.
    Returns None if the ID is non-canonical or the report is not found.
    """
    from pathlib import Path

    if not REPORT_ID_RE.match(report_id):
        return None
    if not artifact_dir:
        artifact_dir = str(Path(hermes_home) / "qdrant_memory" / "improve_reports")
    file_path = Path(artifact_dir) / f"{report_id}.json"
    if not file_path.exists():
        return None
    try:
        loaded = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    # Require exact report_id match after loading
    if not isinstance(loaded, dict):
        return None
    if loaded.get("report_id") != report_id:
        return None
    return loaded


# ---------------------------------------------------------------------------
# Application record (idempotency)
# ---------------------------------------------------------------------------

def _application_record_path(report_id: str, candidate_id: str, *, hermes_home: str) -> Any:
    """Return the path to an application record for a report+candidate."""
    from pathlib import Path
    return (
        Path(hermes_home)
        / "qdrant_memory"
        / "improve_applied"
        / f"{report_id}_{candidate_id}.json"
    )


def is_candidate_applied(report_id: str, candidate_id: str, *, hermes_home: str) -> dict[str, Any] | None:
    """Check if a candidate has been applied for this report.

    Returns the application record dict if applied, or None.
    """
    path = _application_record_path(report_id, candidate_id, hermes_home=hermes_home)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def record_candidate_applied(
    report_id: str,
    candidate_id: str,
    *,
    hermes_home: str,
    target_point_id: str = "",
    candidate_digest: str = "",
    payload_digest: str = "",
) -> dict[str, Any]:
    """Persist an application record after successful live apply.

    This enables idempotent repeat: a second apply for the same
    report+candidate returns ``already_applied`` without touching Qdrant.
    """
    from datetime import datetime
    record = {
        "report_id": report_id,
        "candidate_id": candidate_id,
        "target_point_id": target_point_id,
        "candidate_digest": candidate_digest,
        "payload_digest": payload_digest,
        "applied_at": datetime.utcnow().isoformat() + "Z",
    }
    path = _application_record_path(report_id, candidate_id, hermes_home=hermes_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return record


# ---------------------------------------------------------------------------
# Dependency guard
# ---------------------------------------------------------------------------

def _check_no_external_graph_deps() -> None:
    """Assert that importing this module did not pull in forbidden dependencies."""
    import sys

    forbidden = {
        "cognee",
        "graphiti",
        "neo4j",
        "networkx",
        "langchain",
        "chromadb",
    }
    loaded = set(sys.modules.keys())
    found = forbidden & loaded
    if found:
        raise ImportError(
            f"qdrant_memory.improve must not depend on: {', '.join(sorted(found))}"
        )
