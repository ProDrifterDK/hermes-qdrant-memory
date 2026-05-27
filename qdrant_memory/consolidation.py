from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qdrant_memory.lesson_extractor import contains_secret

SECRET_KEYWORDS = ("api_key", "apikey", "token", "password", "passwd", "secret", "authorization", "bearer", "credential", "private_key")


@dataclass
class ConsolidationPoint:
    id: str
    collection_name: str
    text: str
    payload: dict[str, Any]


def _point_id(point: dict[str, Any]) -> str:
    return str(point.get("id") or "")


def _point_payload(point: dict[str, Any]) -> dict[str, Any]:
    payload = point.get("payload") or {}
    return payload if isinstance(payload, dict) else {}


def _point_text(point: dict[str, Any]) -> str:
    payload = _point_payload(point)
    text = payload.get("text") or payload.get("lesson") or ""
    return str(text)


def normalize_text_fingerprint(text: str) -> str:
    words = [word.strip(".,:;!?()[]{}'\"").lower() for word in text.split()]
    words = [word for word in words if word]
    return " ".join(words)


_PROFILE_MEMORY_SOURCE_TYPES = {
    "profile",
    "profile_memory",
    "user",
    "user_memory",
    "user_profile",
}
_PROFILE_TARGETS = {"profile", "user"}
_FACT_METADATA_KEYS = ("entity", "fact_key", "reconsolidation_key", "subject")
_TERMINAL_FACT_STATUSES = {"deprecated", "superseded"}


def _point_requires_manual_review(point: ConsolidationPoint) -> bool:
    payload = point.payload or {}
    source_type = str(payload.get("source_type") or "").strip().lower()
    target = str(payload.get("target") or payload.get("memory_target") or "").strip().lower()
    if source_type in _PROFILE_MEMORY_SOURCE_TYPES or target in _PROFILE_TARGETS:
        return True
    return any(str(payload.get(key) or "").strip() for key in _FACT_METADATA_KEYS)


def _point_is_identity_bearing(point: ConsolidationPoint) -> bool:
    payload = point.payload or {}
    source_type = str(payload.get("source_type") or "").strip().lower()
    target = str(payload.get("target") or payload.get("memory_target") or "").strip().lower()
    return source_type in _PROFILE_MEMORY_SOURCE_TYPES or target in _PROFILE_TARGETS


def _point_contains_secret(point: ConsolidationPoint) -> bool:
    payload_text = json.dumps(point.payload or {}, sort_keys=True, default=str)
    return contains_secret(point.text) or contains_secret(payload_text)


def _fact_status_for_point(point: ConsolidationPoint) -> str:
    payload = point.payload or {}
    status = str(payload.get("fact_status") or "").strip().lower()
    if status in {"current", "verified", "unknown"}:
        return "active"
    if status:
        return status
    if payload.get("stale") is True:
        return "stale"
    return "active"


def _status_change(point: ConsolidationPoint, to_status: str, *, reason: str, **extra: Any) -> dict[str, Any]:
    change: dict[str, Any] = {"id": point.id, "from": _fact_status_for_point(point), "to": to_status, "reason": reason}
    for key, value in extra.items():
        if value not in (None, "", [], {}):
            change[key] = value
    return change


def _point_observed_at(point: ConsolidationPoint) -> datetime | None:
    payload = point.payload or {}
    for key in ("observed_at", "valid_from", "created_at", "source_modified_at"):
        parsed = _parse_datetime(payload.get(key))
        if parsed:
            return parsed
    return None


def _source_snippet(point: ConsolidationPoint) -> dict[str, Any]:
    payload = point.payload or {}
    snippet: dict[str, Any] = {
        "id": point.id,
        "snippet": _snippet(point.text),
        "source_type": payload.get("source_type", "unknown"),
        "fact_status": _fact_status_for_point(point),
    }
    for key in ("source_uri", "created_at", "observed_at", "valid_from", "valid_until"):
        if payload.get(key):
            snippet[key] = payload.get(key)
    return snippet


def _fact_review_risk(points: list[ConsolidationPoint]) -> str:
    if any(_point_contains_secret(point) or _point_is_identity_bearing(point) for point in points):
        return "high"
    return "medium"


def _manual_review_reason(points: list[ConsolidationPoint], base_reason: str) -> str:
    reasons = [base_reason]
    if any(_point_contains_secret(point) for point in points):
        reasons.append("secret-bearing assertion requires manual review")
    if any(_point_is_identity_bearing(point) for point in points):
        reasons.append("identity-bearing assertion requires manual review")
    return "; ".join(reasons)


_HEADING_NOISE_FINGERPRINTS = {
    "tareas",
    "notas",
    "reflexion",
    "reflexión",
    "contribution",
    "implementation",
    "files modified this turn",
    "plan de implementacion",
    "plan de implementación",
    "risks doubts",
    "next handoff",
}


def _heading_noise_details(text: str) -> tuple[bool, bool, str, str]:
    """Return (is_noise, guarded_safe, normalized, reason) for heading-only chunks."""
    stripped = " ".join(str(text or "").strip().split())
    if not stripped or len(stripped) > 80:
        return False, False, "", "not a short standalone heading"
    had_markdown_hash = stripped.startswith("#")
    while stripped.startswith("#"):
        stripped = stripped[1:].strip()
    normalized = normalize_text_fingerprint(stripped)
    if normalized in _HEADING_NOISE_FINGERPRINTS:
        return True, True, normalized, "known heading/indexer noise fingerprint"
    # Generic all-heading chunks are useful report candidates, but not safe
    # enough for unattended deletion. A legitimate memory such as
    # "# Project Phoenix" looks exactly like a short markdown title.
    if had_markdown_hash and len(normalized.split()) <= 4 and not any(char in stripped for char in ".!?;:"):
        return True, False, normalized, "generic short markdown heading; manual review required"
    return False, False, normalized, "not heading noise"


def is_heading_noise_text(text: str) -> bool:
    """Return True for standalone markdown headings that carry little memory value."""
    return _heading_noise_details(text)[0]


def is_guarded_auto_heading_noise_text(text: str) -> bool:
    """Return True only for known heading-noise fingerprints safe for auto-delete."""
    return _heading_noise_details(text)[1]


def _snippet(text: str, *, max_chars: int = 160) -> str:
    if contains_secret(text):
        return "[redacted: possible secret-bearing memory]"
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _proposal_id(proposal_type: str, affected_ids: list[str]) -> str:
    digest = hashlib.sha256((proposal_type + ":" + ":".join(sorted(affected_ids))).encode("utf-8")).hexdigest()[:16]
    return f"{proposal_type}-{digest}"


def parse_bool_arg(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in {"contains_secret_text", "secret_bearing"} and isinstance(item, bool):
                redacted[key] = item
            elif any(keyword in key_text for keyword in SECRET_KEYWORDS):
                redacted[key] = "[redacted: possible secret-bearing value]"
            else:
                redacted[key] = redact_secrets(item)
        return redacted
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, str) and (contains_secret(value) or "bearer" in value.lower()):
        return "[redacted: possible secret-bearing value]"
    return value


def artifact_root(hermes_home: str, configured_dir: str = "") -> Path:
    root = Path(configured_dir).expanduser() if configured_dir else Path(hermes_home or str(Path.home() / ".hermes")) / "qdrant_memory" / "consolidation"
    root.mkdir(parents=True, exist_ok=True)
    try:
        root.chmod(0o700)
    except Exception:
        pass
    return root


def compute_report_id(report: dict[str, Any], created_at: str) -> str:
    proposal_ids = sorted(str(p.get("proposal_id") or "") for p in report.get("proposals", []))
    seed = json.dumps(
        {
            "created_at": created_at,
            "scope": report.get("scope"),
            "profile_id": report.get("profile_id"),
            "session_id": report.get("session_id"),
            "proposal_ids": proposal_ids,
        },
        sort_keys=True,
    )
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def persist_consolidation_report(report: dict[str, Any], *, hermes_home: str, configured_dir: str = "") -> dict[str, Any]:
    created_at = str(report.get("created_at") or _utc_now())
    report = redact_secrets({**report, "created_at": created_at})
    report_id = str(report.get("report_id") or compute_report_id(report, created_at))
    report["report_id"] = report_id
    report["schema_version"] = 1
    root = artifact_root(hermes_home, configured_dir)
    path = root / f"report-{report_id}.json"
    report["persisted"] = True
    report["artifact"] = {"path": str(path), "proposal_count": len(report.get("proposals", []))}
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def load_consolidation_report(report_id: str, *, hermes_home: str, configured_dir: str = "") -> dict[str, Any]:
    safe_id = "".join(ch for ch in str(report_id) if ch.isalnum() or ch in {"-", "_"})
    if not safe_id or safe_id != str(report_id):
        raise ValueError("invalid report_id")
    path = artifact_root(hermes_home, configured_dir) / f"report-{safe_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"consolidation report not found: {safe_id}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("invalid consolidation report")
    return data


def find_proposal(report: dict[str, Any], proposal_id: str) -> dict[str, Any]:
    for proposal in report.get("proposals", []):
        if str(proposal.get("proposal_id")) == str(proposal_id):
            return proposal
    raise KeyError(f"proposal_id not found: {proposal_id}")


def expected_action_for_proposal(proposal_type: str) -> str | None:
    return {
        "duplicate_cluster": "merge",
        "heading_noise": "delete",
        "stale_low_value": "delete",
        "learning_promotion_candidate": "promote_to_skill",
        "reconsolidation_candidate": "draft_review",
        "fact_conflict_candidate": "draft_review",
        "fact_supersession_candidate": "draft_review",
        "fact_status_update_candidate": "draft_review",
    }.get(proposal_type)


def persist_application_record(record: dict[str, Any], *, hermes_home: str, configured_dir: str = "") -> dict[str, Any]:
    root = artifact_root(hermes_home, configured_dir) / "applications"
    root.mkdir(parents=True, exist_ok=True)
    created_at = str(record.get("created_at") or _utc_now())
    record = redact_secrets({**record, "created_at": created_at, "schema_version": 1})
    seed = f"{created_at}:{record.get('report_id')}:{record.get('proposal_id')}:{record.get('action')}"
    application_id = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    record["application_id"] = application_id
    path = root / f"application-{application_id}.json"
    record["artifact_path"] = str(path)
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return record


def build_skill_draft_text(point: ConsolidationPoint, *, proposal_id: str, report_id: str) -> str:
    payload = redact_secrets(point.payload)
    safe_text = redact_secrets(point.text)
    lesson = str(payload.get("lesson") or payload.get("text") or safe_text).strip()
    trigger = str(payload.get("trigger") or "").strip()
    correction = str(payload.get("correction") or "").strip()
    evidence = str(payload.get("evidence") or "").strip()
    learning_type = str(payload.get("learning_type") or "workflow_lesson")
    return "\n".join(
        [
            "---",
            f"name: draft-{proposal_id}",
            f"description: Draft skill promoted from Qdrant learning {point.id}",
            "---",
            "",
            "# Draft skill from Qdrant learning",
            "",
            "This is a draft artifact only. It is not installed as an active Hermes skill.",
            "",
            f"- report_id: {report_id}",
            f"- proposal_id: {proposal_id}",
            f"- learning_id: {point.id}",
            f"- learning_type: {learning_type}",
            "",
            "## Lesson",
            lesson,
            "",
            "## Trigger",
            trigger or "N/A",
            "",
            "## Correct procedure",
            correction or lesson,
            "",
            "## Evidence",
            evidence or "N/A",
            "",
        ]
    )


def build_reconsolidation_draft_text(points: list[ConsolidationPoint], *, proposal: dict[str, Any], report_id: str) -> str:
    safe_proposal = redact_secrets(proposal)
    lines = [
        "# Reconsolidation review draft",
        "",
        "This is a manual review artifact only. It does not mutate Qdrant memory.",
        "",
        f"- report_id: {report_id}",
        f"- proposal_id: {safe_proposal.get('proposal_id')}",
        f"- fact_key: {safe_proposal.get('fact_key') or 'N/A'}",
        f"- collection_name: {safe_proposal.get('collection_name')}",
        f"- affected_ids: {', '.join(str(i) for i in safe_proposal.get('affected_ids', []))}",
        "",
        "## Candidate statement",
        str(safe_proposal.get("candidate_statement") or "Manual review required."),
        "",
        "## Evidence",
    ]
    for point in points:
        payload = redact_secrets(point.payload)
        lines.extend(
            [
                "",
                f"### {point.id}",
                f"- source_type: {payload.get('source_type', 'unknown')}",
                f"- importance: {payload.get('importance', 'unknown')}",
                f"- confidence: {payload.get('confidence', 'unknown')}",
                "",
                _snippet(point.text, max_chars=500),
            ]
        )
    lines.extend(
        [
            "",
            "## Reviewer checklist",
            "",
            "- Decide whether one fact supersedes another, or whether both are context-dependent.",
            "- If a durable procedural lesson emerges, create or update a Hermes skill manually.",
            "- If a memory should be changed, use explicit memory tools with dry-run first; do not bulk edit by query.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        return None


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def make_filter(scope: dict[str, str], *, source_type: str | None = None) -> dict[str, Any]:
    must: list[dict[str, Any]] = []
    for key, value in scope.items():
        if value:
            must.append({"key": key, "match": {"value": value}})
    if source_type:
        must.append({"key": "source_type", "match": {"value": source_type}})
    return {"must": must}


def _similarity(a: str, b: str) -> float:
    a_words = set(a.split())
    b_words = set(b.split())
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / len(a_words | b_words)


def _heading_noise_proposals(points: list[ConsolidationPoint], *, include_examples: bool, max_groups: int) -> list[dict[str, Any]]:
    candidates = [point for point in points if is_heading_noise_text(point.text) and not point.payload.get("consolidation_quarantined")]
    proposals: list[dict[str, Any]] = []
    for point in candidates[:max_groups]:
        _, guarded_safe, _normalized, reason = _heading_noise_details(point.text)
        proposal: dict[str, Any] = {
            "proposal_id": _proposal_id("heading_noise", [point.id]),
            "proposal_type": "heading_noise",
            "collection_name": point.collection_name,
            "affected_ids": [point.id],
            "suggested_action": "delete_guarded_auto_eligible" if guarded_safe else "delete_review_only",
            "confidence": 0.99 if guarded_safe else 0.70,
            "risk": "low" if guarded_safe else "medium",
            "evidence": [{"id": point.id, "reason": reason}],
            "requires_explicit_approval": True,
            "guarded_auto_eligible": guarded_safe,
        }
        if guarded_safe:
            proposal["preauthorized_policy"] = "guarded-auto:heading-noise"
        else:
            proposal["manual_review_required"] = True
        if include_examples:
            proposal["examples"] = [{"id": point.id, "text": _snippet(point.text)}]
        proposals.append(proposal)
    return proposals


def _duplicate_proposals(points: list[ConsolidationPoint], *, max_groups: int, include_examples: bool, threshold: float = 0.92) -> list[dict[str, Any]]:
    fingerprints = [(point, normalize_text_fingerprint(point.text)) for point in points]
    used: set[str] = set()
    proposals: list[dict[str, Any]] = []
    for point, fingerprint in fingerprints:
        if not fingerprint or point.id in used:
            continue
        group = [point]
        for other, other_fingerprint in fingerprints:
            if other.id == point.id or other.id in used or not other_fingerprint:
                continue
            if _similarity(fingerprint, other_fingerprint) >= threshold:
                group.append(other)
        if len(group) < 2:
            continue
        for member in group:
            used.add(member.id)
        affected_ids = [point.id for point in group if point.id]
        group_fingerprints = {fp for member, fp in fingerprints if member.id in set(affected_ids)}
        exact_normalized = len(group_fingerprints) == 1
        contains_secret_text = any(contains_secret(member.text) or contains_secret(json.dumps(member.payload or {}, sort_keys=True, default=str)) for member in group)
        manual_review_point = any(_point_requires_manual_review(member) for member in group)
        guarded_auto_safe = exact_normalized and not contains_secret_text and not manual_review_point
        proposal: dict[str, Any] = {
            "proposal_id": _proposal_id("duplicate_cluster", affected_ids),
            "proposal_type": "duplicate_cluster",
            "collection_name": group[0].collection_name,
            "affected_ids": affected_ids,
            "suggested_action": "merge_review_only",
            "confidence": 0.99 if exact_normalized else 0.95,
            "risk": "low" if guarded_auto_safe else "medium",
            "evidence": [{"id": point.id, "reason": "identical normalized text" if exact_normalized else "high normalized text overlap"} for point in group],
            "requires_explicit_approval": True,
            "match_kind": "exact_normalized" if exact_normalized else "near_duplicate",
            "guarded_auto_eligible": guarded_auto_safe,
        }
        if contains_secret_text:
            proposal["contains_secret_text"] = True
        if manual_review_point:
            proposal["manual_review_required"] = True
            proposal["manual_review_reason"] = "profile or fact-like memory requires manual review"
        if proposal.get("guarded_auto_eligible"):
            proposal["preauthorized_policy"] = "guarded-auto:exact-duplicate-merge"
        if include_examples:
            proposal["examples"] = [{"id": point.id, "text": _snippet(point.text)} for point in group]
        proposals.append(proposal)
        if len(proposals) >= max_groups:
            break
    return proposals


def _stale_low_value_proposals(
    points: list[ConsolidationPoint],
    *,
    stale_days: int,
    min_importance_for_keep: int,
    include_examples: bool,
    max_groups: int,
) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc)
    proposals: list[dict[str, Any]] = []
    for point in points:
        payload = point.payload
        if payload.get("consolidation_quarantined"):
            continue
        created = _parse_datetime(payload.get("created_at") or payload.get("last_accessed"))
        if not created:
            continue
        age_days = (now - created).days
        importance = _as_int(payload.get("importance"), 5)
        access_count = _as_int(payload.get("access_count"), 0)
        confidence = _as_float(payload.get("confidence"), 1.0)
        if age_days < stale_days or importance >= min_importance_for_keep or access_count > 0 or confidence > 0.5:
            continue
        stale_manual_review = _point_requires_manual_review(point)
        contains_secret_text = contains_secret(point.text) or contains_secret(json.dumps(point.payload or {}, sort_keys=True, default=str))
        guarded_auto_safe = not stale_manual_review and not contains_secret_text
        proposal: dict[str, Any] = {
            "proposal_id": _proposal_id("stale_low_value", [point.id]),
            "proposal_type": "stale_low_value",
            "collection_name": point.collection_name,
            "affected_ids": [point.id],
            "suggested_action": "quarantine_guarded_auto_eligible" if guarded_auto_safe else "quarantine_review_only",
            "confidence": 0.75,
            "risk": "low" if guarded_auto_safe else "medium",
            "evidence": [
                {
                    "id": point.id,
                    "age_days": age_days,
                    "importance": importance,
                    "access_count": access_count,
                    "confidence": confidence,
                }
            ],
            "requires_explicit_approval": True,
            "guarded_auto_eligible": guarded_auto_safe,
        }
        if guarded_auto_safe:
            proposal["preauthorized_policy"] = "guarded-auto:stale-low-value-quarantine"
        else:
            proposal["manual_review_required"] = True
            proposal["manual_review_reason"] = "secret-bearing, profile, or fact-like memory requires manual review"
        if contains_secret_text:
            proposal["contains_secret_text"] = True
        if include_examples:
            proposal["examples"] = [{"id": point.id, "text": _snippet(point.text)}]
        proposals.append(proposal)
        if len(proposals) >= max_groups:
            break
    return proposals


def _learning_promotion_proposals(points: list[ConsolidationPoint], *, include_examples: bool, max_groups: int) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    for point in points:
        payload = point.payload
        if not payload.get("promote_to_skill_candidate"):
            continue
        if payload.get("promoted_to_skill_draft"):
            continue
        if contains_secret(point.text) or contains_secret(json.dumps(payload or {}, sort_keys=True, default=str)):
            continue
        if _as_float(payload.get("confidence"), 0.0) < 0.85 or _as_int(payload.get("importance"), 0) < 8:
            continue
        proposal: dict[str, Any] = {
            "proposal_id": _proposal_id("learning_promotion_candidate", [point.id]),
            "proposal_type": "learning_promotion_candidate",
            "collection_name": point.collection_name,
            "affected_ids": [point.id],
            "suggested_action": "promote_to_skill_review_only",
            "confidence": _as_float(payload.get("confidence"), 0.85),
            "risk": "low",
            "evidence": [
                {
                    "id": point.id,
                    "learning_type": payload.get("learning_type", ""),
                    "importance": _as_int(payload.get("importance"), 0),
                    "confidence": _as_float(payload.get("confidence"), 0.0),
                }
            ],
            "requires_explicit_approval": True,
            "guarded_auto_eligible": True,
            "preauthorized_policy": "guarded-auto:learning-skill-draft",
        }
        if include_examples:
            proposal["examples"] = [{"id": point.id, "text": _snippet(point.text)}]
        proposals.append(proposal)
        if len(proposals) >= max_groups:
            break
    return proposals


def _fact_key(point: ConsolidationPoint) -> str:
    payload = point.payload or {}
    for key in ("reconsolidation_key", "fact_key", "subject"):
        value = str(payload.get(key) or "").strip().lower()
        if value:
            return f"{key}:{value}"
    return ""


def _fact_groups(points: list[ConsolidationPoint], *, min_confidence: float) -> dict[str, list[ConsolidationPoint]]:
    groups: dict[str, list[ConsolidationPoint]] = {}
    for point in points:
        key = _fact_key(point)
        if not key:
            continue
        if _as_float(point.payload.get("confidence"), 1.0) < min_confidence:
            continue
        groups.setdefault(key, []).append(point)
    return groups


def _fact_status_update_proposals(
    points: list[ConsolidationPoint],
    *,
    max_candidates: int,
    min_confidence: float,
    collection_name: str,
) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc)
    for point in points:
        if not _fact_key(point):
            continue
        if _as_float(point.payload.get("confidence"), 1.0) < min_confidence:
            continue
        current_status = _fact_status_for_point(point)
        target_status = ""
        reason = ""
        payload = point.payload or {}
        if payload.get("stale") is True and current_status != "stale":
            target_status = "stale"
            reason = "payload stale flag is set"
        elif payload.get("superseded_by") and current_status != "superseded":
            target_status = "superseded"
            reason = "payload superseded_by link is set"
        elif payload.get("invalidated_by") and current_status != "disputed":
            target_status = "disputed"
            reason = "payload invalidated_by link is set"
        else:
            valid_until = _parse_datetime(payload.get("valid_until"))
            if valid_until and valid_until < now and current_status not in {"stale", *_TERMINAL_FACT_STATUSES}:
                target_status = "stale"
                reason = "valid_until is in the past"
        if not target_status:
            continue

        contains_secret_text = _point_contains_secret(point)
        proposal: dict[str, Any] = {
            "proposal_id": _proposal_id("fact_status_update_candidate", [point.id]),
            "proposal_type": "fact_status_update_candidate",
            "collection_name": collection_name,
            "affected_ids": [point.id],
            "fact_key": _fact_key(point),
            "candidate_statement": "Review proposed fact status metadata before changing memory.",
            "suggested_action": "draft_review_only",
            "confidence": min(0.9, max(0.5, _as_float(point.payload.get("confidence"), 0.75))),
            "risk": _fact_review_risk([point]),
            "source_snippets": [_source_snippet(point)],
            "proposed_status_changes": [_status_change(point, target_status, reason=reason)],
            "manual_review_required": True,
            "manual_review_reason": _manual_review_reason([point], "fact status update requires manual review"),
            "requires_explicit_approval": True,
            "guarded_auto_eligible": False,
            "draft_only": True,
        }
        if contains_secret_text:
            proposal["contains_secret_text"] = True
        proposals.append(proposal)
        if len(proposals) >= max_candidates:
            break
    return proposals


def _fact_supersession_proposal(
    key: str,
    group: list[ConsolidationPoint],
    *,
    collection_name: str,
) -> dict[str, Any] | None:
    dated = [(point, observed_at) for point in group if (observed_at := _point_observed_at(point))]
    if len(dated) < 2:
        return None
    newest, newest_at = max(
        dated,
        key=lambda item: (
            item[1],
            _as_int(item[0].payload.get("importance"), 5),
            _as_float(item[0].payload.get("confidence"), 0.0),
            item[0].id,
        ),
    )
    older = [point for point, observed_at in dated if point.id != newest.id and observed_at < newest_at]
    older = [point for point in older if _fact_status_for_point(point) not in _TERMINAL_FACT_STATUSES]
    if not older:
        return None
    affected_points = [*older, newest]
    affected_ids = [point.id for point in affected_points if point.id]
    contains_secret_text = any(_point_contains_secret(point) for point in affected_points)
    proposed_status_changes = [
        _status_change(point, "superseded", reason="newer assertion may supersede this fact", superseded_by=[newest.id])
        for point in older
    ]
    proposed_status_changes.append(_status_change(newest, "active", reason="newer assertion remains current pending review"))
    proposal: dict[str, Any] = {
        "proposal_id": _proposal_id("fact_supersession_candidate", affected_ids),
        "proposal_type": "fact_supersession_candidate",
        "collection_name": collection_name,
        "affected_ids": affected_ids,
        "fact_key": key,
        "current_or_newer_id": newest.id,
        "superseded_candidate_ids": [point.id for point in older],
        "supersession_reason": f"newer assertion {newest.id} may supersede older assertion(s) for {key}; review only.",
        "candidate_statement": f"Newer fact evidence may supersede older assertion(s) for {key}.",
        "suggested_action": "draft_review_only",
        "confidence": min(0.95, max(_as_float(point.payload.get("confidence"), 0.0) for point in affected_points)),
        "risk": _fact_review_risk(affected_points),
        "source_snippets": [_source_snippet(point) for point in affected_points],
        "proposed_status_changes": proposed_status_changes,
        "manual_review_required": True,
        "manual_review_reason": _manual_review_reason(affected_points, "fact supersession requires manual review"),
        "requires_explicit_approval": True,
        "guarded_auto_eligible": False,
        "draft_only": True,
    }
    if contains_secret_text:
        proposal["contains_secret_text"] = True
    return proposal


def _fact_conflict_and_supersession_proposals(
    points: list[ConsolidationPoint],
    *,
    max_candidates: int,
    min_confidence: float,
    collection_name: str,
) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    for key, raw_group in sorted(_fact_groups(points, min_confidence=min_confidence).items()):
        group = [point for point in raw_group if _fact_status_for_point(point) not in _TERMINAL_FACT_STATUSES]
        if len(group) < 2:
            continue
        unique_texts = {normalize_text_fingerprint(point.text) for point in group if normalize_text_fingerprint(point.text)}
        if len(unique_texts) < 2:
            continue
        affected_ids = [point.id for point in group if point.id]
        contains_secret_text = any(_point_contains_secret(point) for point in group)
        proposal: dict[str, Any] = {
            "proposal_id": _proposal_id("fact_conflict_candidate", affected_ids),
            "proposal_type": "fact_conflict_candidate",
            "collection_name": collection_name,
            "affected_ids": affected_ids,
            "fact_key": key,
            "candidate_statement": f"Conflicting fact assertions share {key}; review before changing status.",
            "suggested_action": "draft_review_only",
            "confidence": min(0.9, max(_as_float(point.payload.get("confidence"), 0.0) for point in group)),
            "risk": _fact_review_risk(group),
            "source_snippets": [_source_snippet(point) for point in group],
            "proposed_status_changes": [
                _status_change(point, "disputed", reason=f"conflicting assertions share {key}") for point in group
            ],
            "manual_review_required": True,
            "manual_review_reason": _manual_review_reason(group, "fact conflict requires manual review"),
            "requires_explicit_approval": True,
            "guarded_auto_eligible": False,
            "draft_only": True,
        }
        if contains_secret_text:
            proposal["contains_secret_text"] = True
        proposals.append(proposal)
        if len(proposals) >= max_candidates:
            break
        supersession = _fact_supersession_proposal(key, group, collection_name=collection_name)
        if supersession:
            proposals.append(supersession)
        if len(proposals) >= max_candidates:
            break
    return proposals


def _reconsolidation_proposals(
    points: list[ConsolidationPoint],
    *,
    include_examples: bool,
    max_candidates: int,
    min_confidence: float,
    collection_name: str,
) -> list[dict[str, Any]]:
    groups: dict[str, list[ConsolidationPoint]] = {}
    for point in points:
        key = _fact_key(point)
        if not key:
            continue
        confidence = _as_float(point.payload.get("confidence"), 1.0)
        if confidence < min_confidence:
            continue
        groups.setdefault(key, []).append(point)

    proposals: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        if len(group) < 2:
            continue
        unique_texts = {normalize_text_fingerprint(point.text) for point in group if normalize_text_fingerprint(point.text)}
        if len(unique_texts) < 2:
            continue
        affected_ids = [point.id for point in group if point.id]
        contains_secret_text = any(contains_secret(point.text) for point in group)
        important_fact = any(_as_int(point.payload.get("importance"), 5) >= 7 or str(point.payload.get("source_type") or "") in {"manual", "user_memory", "profile_memory"} for point in group)
        canonical = max(
            group,
            key=lambda point: (
                _as_int(point.payload.get("importance"), 5),
                _as_float(point.payload.get("confidence"), 0.0),
                str(point.payload.get("created_at") or ""),
                point.id,
            ),
        )
        evidence = []
        for point in group:
            evidence.append(
                {
                    "id": point.id,
                    "source_type": point.payload.get("source_type", ""),
                    "importance": _as_int(point.payload.get("importance"), 5),
                    "confidence": _as_float(point.payload.get("confidence"), 1.0),
                    "snippet": _snippet(point.text) if include_examples else "redacted/manual review",
                }
            )
        proposal: dict[str, Any] = {
            "proposal_id": _proposal_id("reconsolidation_candidate", affected_ids),
            "proposal_type": "reconsolidation_candidate",
            "collection_name": collection_name,
            "affected_ids": affected_ids,
            "canonical_or_current_id": canonical.id,
            "conflicting_ids": [point.id for point in group if point.id != canonical.id],
            "fact_key": key,
            "candidate_statement": "Secret-bearing conflicting memories require manual review." if contains_secret_text else f"Conflicting memories share {key}; review before changing any fact.",
            "suggested_action": "reconsolidate_review_only",
            "confidence": min(0.9, max(_as_float(point.payload.get("confidence"), 0.0) for point in group)),
            "risk": "high",
            "important_fact": important_fact,
            "manual_review_required": True,
            "evidence": evidence,
            "requires_explicit_approval": True,
        }
        if include_examples:
            proposal["examples"] = [{"id": point.id, "text": _snippet(point.text)} for point in group]
        proposals.append(proposal)
        if len(proposals) >= max_candidates:
            break
    return proposals


def _quality_warning_proposals(points: list[ConsolidationPoint], *, max_groups: int) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    for point in points:
        if not contains_secret(point.text):
            continue
        proposal = {
            "proposal_id": _proposal_id("quality_warning", [point.id]),
            "proposal_type": "quality_warning",
            "collection_name": point.collection_name,
            "affected_ids": [point.id],
            "suggested_action": "manual_secret_review_only",
            "confidence": 0.9,
            "risk": "high",
            "evidence": [{"id": point.id, "reason": "possible secret-bearing memory; text redacted"}],
            "requires_explicit_approval": True,
        }
        proposals.append(proposal)
        if len(proposals) >= max_groups:
            break
    return proposals


def points_from_qdrant(raw_points: list[dict[str, Any]], *, collection_name: str) -> list[ConsolidationPoint]:
    points: list[ConsolidationPoint] = []
    for raw in raw_points:
        point_id = _point_id(raw)
        payload = _point_payload(raw)
        text = _point_text(raw)
        if not point_id:
            continue
        points.append(ConsolidationPoint(id=point_id, collection_name=collection_name, text=text, payload=payload))
    return points


def build_consolidation_report(
    *,
    memory_points: list[ConsolidationPoint],
    learning_points: list[ConsolidationPoint],
    collection_name: str,
    learning_collection_name: str,
    scope: str,
    include_examples: bool = False,
    max_groups: int = 20,
    stale_days: int = 90,
    min_importance_for_keep: int = 4,
    duplicate_threshold: float = 0.92,
    consolidation_enabled: bool = False,
    reconsolidation_enabled: bool = False,
    include_reconsolidation: bool = False,
    reconsolidation_max_candidates: int = 10,
    reconsolidation_min_confidence: float = 0.6,
) -> dict[str, Any]:
    proposals: list[dict[str, Any]] = []
    memory_heading_noise = [point for point in memory_points if is_heading_noise_text(point.text)]
    learning_heading_noise = [point for point in learning_points if is_heading_noise_text(point.text)]
    memory_non_heading = [point for point in memory_points if not is_heading_noise_text(point.text)]
    learning_non_heading = [point for point in learning_points if not is_heading_noise_text(point.text)]
    proposals.extend(_quality_warning_proposals(memory_points + learning_points, max_groups=max_groups))
    proposals.extend(_heading_noise_proposals(memory_heading_noise + learning_heading_noise, include_examples=include_examples, max_groups=max_groups))
    proposals.extend(_duplicate_proposals(memory_non_heading, max_groups=max_groups, include_examples=include_examples, threshold=duplicate_threshold))
    proposals.extend(_duplicate_proposals(learning_non_heading, max_groups=max_groups, include_examples=include_examples, threshold=duplicate_threshold))
    proposals.extend(
        _stale_low_value_proposals(
            memory_non_heading,
            stale_days=stale_days,
            min_importance_for_keep=min_importance_for_keep,
            include_examples=include_examples,
            max_groups=max_groups,
        )
    )
    proposals.extend(_learning_promotion_proposals(learning_points, include_examples=include_examples, max_groups=max_groups))
    if include_reconsolidation:
        proposals.extend(
            _fact_status_update_proposals(
                memory_points,
                max_candidates=reconsolidation_max_candidates,
                min_confidence=reconsolidation_min_confidence,
                collection_name=collection_name,
            )
        )
        proposals.extend(
            _fact_conflict_and_supersession_proposals(
                memory_points,
                max_candidates=reconsolidation_max_candidates,
                min_confidence=reconsolidation_min_confidence,
                collection_name=collection_name,
            )
        )
        proposals.extend(
            _reconsolidation_proposals(
                memory_points,
                include_examples=include_examples,
                max_candidates=reconsolidation_max_candidates,
                min_confidence=reconsolidation_min_confidence,
                collection_name=collection_name,
            )
        )
    proposals = proposals[:max_groups]
    summary: dict[str, int] = {}
    for proposal in proposals:
        ptype = str(proposal.get("proposal_type") or "unknown")
        summary[ptype] = summary.get(ptype, 0) + 1
    return {
        "dry_run": True,
        "report_only": True,
        "mutations_performed": False,
        "scope": scope,
        "collections": {"memory": collection_name, "learning": learning_collection_name},
        "consolidation_enabled": consolidation_enabled,
        "reconsolidation_enabled": reconsolidation_enabled,
        "include_reconsolidation": include_reconsolidation,
        "reconsolidation_report_only": True,
        "analyzed": {"memory_points": len(memory_points), "learning_points": len(learning_points)},
        "summary": summary,
        "proposals": proposals,
        "warnings": ["M9a persists report artifacts when requested/defaulted; qdrant_memory_consolidation_apply is required for gated live actions."],
        "next_steps": [
            "Review proposals manually.",
            "Preview a proposal with qdrant_memory_consolidation_apply using dry_run=true.",
            "Apply only one explicit proposal_id at a time with dry_run=false and approve=true.",
        ],
    }
