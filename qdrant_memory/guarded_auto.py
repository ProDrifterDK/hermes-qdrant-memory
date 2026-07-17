from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from qdrant_memory.consolidation import (
    _point_requires_manual_review,
    is_guarded_auto_heading_noise_text,
    normalize_text_fingerprint,
)
from qdrant_memory.lesson_extractor import contains_secret


_SAFE_HANDLE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
_GUARDED_AUTO_PROPOSAL_TYPES = {
    "duplicate_cluster",
    "heading_noise",
    "learning_promotion_candidate",
    "stale_low_value",
}


@dataclass(frozen=True)
class GuardedAutoPolicy:
    mode: str = "report-only"
    max_actions: int = 10
    duplicate_min_confidence: float = 0.98
    duplicate_max_cluster_size: int = 20
    learning_min_confidence: float = 0.90
    stale_quarantine_enabled: bool = True
    quarantine_days: int = 30

    @property
    def enabled(self) -> bool:
        return self.mode == "guarded-auto"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _affected_ids(proposal: dict[str, Any]) -> list[str]:
    raw = proposal.get("affected_ids") or []
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if str(item)]


def _secret_or_manual_only(proposal: dict[str, Any]) -> bool:
    if proposal.get("contains_secret_text") or proposal.get("secret_bearing"):
        return True
    if proposal.get("manual_review_required"):
        return True
    text = json.dumps(proposal, sort_keys=True).lower()
    return "secret-bearing" in text or "possible secret" in text or "manual_secret_review" in text


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_handle(value: Any) -> str:
    text = str(value or "").strip()
    if _SAFE_HANDLE_RE.fullmatch(text):
        return text
    return f"sha256:{_sha256_text(text)[:16]}"


def _safe_proposal_type(value: Any) -> str:
    text = str(value or "")
    return text if text in _GUARDED_AUTO_PROPOSAL_TYPES else "review_only"


def _point_content_digest(point: Any) -> str:
    return _sha256_text(str(getattr(point, "text", "") or ""))


def _point_payload_digest(point: Any) -> str:
    return _sha256_text(_canonical_json(getattr(point, "payload", {}) or {}))


def _point_digest_records(points: list[Any]) -> list[dict[str, str]]:
    return sorted(
        [
            {
                "id": str(getattr(point, "id", "") or ""),
                "content_sha256": _point_content_digest(point),
                "payload_sha256": _point_payload_digest(point),
            }
            for point in points
        ],
        key=lambda item: item["id"],
    )


def _proposal_digest(proposal: dict[str, Any]) -> str:
    material = {key: value for key, value in proposal.items() if key != "guarded_auto_proposal_sha256"}
    return _sha256_text(_canonical_json(material))


def seal_guarded_auto_proposals(
    report: dict[str, Any],
    points: list[Any],
    *,
    stale_days: int,
    min_importance_for_keep: int,
    duplicate_min_confidence: float,
    duplicate_max_cluster_size: int,
    learning_min_confidence: float,
) -> dict[str, Any]:
    """Bind guarded-auto proposals to exact point state and policy criteria."""
    by_collection_and_id = {
        (str(getattr(point, "collection_name", "") or ""), str(getattr(point, "id", "") or "")): point
        for point in points
    }
    for proposal in report.get("proposals", []):
        if not isinstance(proposal, dict):
            continue
        policy = str(proposal.get("preauthorized_policy") or "")
        if proposal.get("guarded_auto_eligible") is not True or not policy.startswith("guarded-auto:"):
            continue
        collection_name = str(proposal.get("collection_name") or "")
        affected_ids = _affected_ids(proposal)
        proposal_points = [
            by_collection_and_id[(collection_name, point_id)]
            for point_id in affected_ids
            if (collection_name, point_id) in by_collection_and_id
        ]
        proposal["guarded_auto_snapshot"] = {
            "schema_version": 1,
            "criteria": {
                "stale_days": int(stale_days),
                "min_importance_for_keep": int(min_importance_for_keep),
                "duplicate_min_confidence": float(duplicate_min_confidence),
                "duplicate_max_cluster_size": int(duplicate_max_cluster_size),
                "learning_min_confidence": float(learning_min_confidence),
            },
            "point_digests": _point_digest_records(proposal_points),
        }
        proposal["guarded_auto_proposal_sha256"] = _proposal_digest(proposal)
    return report


def guarded_auto_report_metadata_matches(report: dict[str, Any], requested_report_id: str) -> bool:
    """Verify persisted report identity fields used by the exact-ID apply gate."""
    if str(report.get("report_id") or "") != str(requested_report_id):
        return False
    created_at = str(report.get("created_at") or "")
    if not created_at:
        return False
    proposal_ids = sorted(str(p.get("proposal_id") or "") for p in report.get("proposals", []) if isinstance(p, dict))
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
    return _sha256_text(seed)[:16] == requested_report_id


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
    except Exception:
        return None


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _points_are_sensitive(points: list[Any]) -> bool:
    for point in points:
        payload = getattr(point, "payload", {}) or {}
        if contains_secret(str(getattr(point, "text", "") or "")):
            return True
        if contains_secret(_canonical_json(payload)) or _point_requires_manual_review(point):
            return True
    return False


def validate_guarded_auto_current_points(
    proposal: dict[str, Any],
    points: list[Any],
    *,
    stale_days: int,
    min_importance_for_keep: int,
    duplicate_min_confidence: float,
    duplicate_max_cluster_size: int,
    learning_min_confidence: float,
) -> tuple[bool, str]:
    """Re-derive guarded-auto eligibility from freshly retrieved exact points."""
    expected_proposal_digest = str(proposal.get("guarded_auto_proposal_sha256") or "")
    if not expected_proposal_digest or expected_proposal_digest != _proposal_digest(proposal):
        return False, "report metadata changed; generate a fresh report"
    snapshot = proposal.get("guarded_auto_snapshot")
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != 1:
        return False, "guarded-auto snapshot missing; generate a fresh report"
    expected_criteria = {
        "stale_days": int(stale_days),
        "min_importance_for_keep": int(min_importance_for_keep),
        "duplicate_min_confidence": float(duplicate_min_confidence),
        "duplicate_max_cluster_size": int(duplicate_max_cluster_size),
        "learning_min_confidence": float(learning_min_confidence),
    }
    if snapshot.get("criteria") != expected_criteria:
        return False, "guarded-auto criteria changed; generate a fresh report"
    affected_ids = _affected_ids(proposal)
    current_ids = [str(getattr(point, "id", "") or "") for point in points]
    if len(affected_ids) != len(set(affected_ids)) or set(current_ids) != set(affected_ids):
        return False, "affected points changed; generate a fresh report"
    if snapshot.get("point_digests") != _point_digest_records(points):
        return False, "affected point content or payload changed; generate a fresh report"
    if _secret_or_manual_only(proposal) or _points_are_sensitive(points):
        return False, "current points require manual review; generate a fresh report"

    proposal_type = str(proposal.get("proposal_type") or "")
    confidence = _as_float(proposal.get("confidence"), 0.0)
    risk = str(proposal.get("risk") or "").lower()
    if proposal.get("guarded_auto_eligible") is not True or risk != "low":
        return False, "proposal is no longer guarded-auto eligible; generate a fresh report"

    if proposal_type == "duplicate_cluster":
        fingerprints = {normalize_text_fingerprint(str(getattr(point, "text", "") or "")) for point in points}
        eligible = (
            len(points) >= 2
            and "" not in fingerprints
            and len(fingerprints) == 1
            and proposal.get("match_kind") == "exact_normalized"
            and proposal.get("preauthorized_policy") == "guarded-auto:exact-duplicate-merge"
            and confidence >= duplicate_min_confidence
            and len(points) <= duplicate_max_cluster_size
        )
    elif proposal_type == "heading_noise":
        eligible = (
            len(points) == 1
            and is_guarded_auto_heading_noise_text(str(getattr(points[0], "text", "") or ""))
            and proposal.get("preauthorized_policy") == "guarded-auto:heading-noise"
        )
    elif proposal_type == "stale_low_value":
        payload = getattr(points[0], "payload", {}) or {} if len(points) == 1 else {}
        created = _parse_datetime(payload.get("created_at") or payload.get("last_accessed"))
        age_days = (datetime.now(timezone.utc) - created).days if created else -1
        eligible = (
            len(points) == 1
            and not payload.get("consolidation_quarantined")
            and created is not None
            and age_days >= stale_days
            and _as_int(payload.get("importance"), 5) < min_importance_for_keep
            and _as_int(payload.get("access_count"), 0) == 0
            and _as_float(payload.get("confidence"), 1.0) <= 0.5
            and proposal.get("preauthorized_policy") == "guarded-auto:stale-low-value-quarantine"
        )
    elif proposal_type == "learning_promotion_candidate":
        payload = getattr(points[0], "payload", {}) or {} if len(points) == 1 else {}
        eligible = (
            len(points) == 1
            and payload.get("promote_to_skill_candidate") is True
            and not payload.get("promoted_to_skill_draft")
            and _as_float(payload.get("confidence"), confidence) >= learning_min_confidence
            and _as_int(payload.get("importance"), 0) >= 8
            and proposal.get("preauthorized_policy") == "guarded-auto:learning-skill-draft"
        )
    else:
        eligible = False
    if not eligible:
        return False, "current points are no longer guarded-auto eligible; generate a fresh report"
    return True, "current exact points match the persisted guarded-auto snapshot"


def _public_result_summary(result: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {"status": "applied"}
    for key in ("dry_run", "applied", "would_apply"):
        if isinstance(result.get(key), bool):
            summary[key] = result[key]
    for key in ("application_id", "canonical_id"):
        if result.get(key):
            summary[key] = _safe_handle(result[key])
    for key in ("deleted_ids", "quarantined_ids"):
        values = result.get(key)
        if isinstance(values, list):
            summary[key] = [_safe_handle(value) for value in values]
    return summary


def guarded_auto_action_for_proposal(proposal: dict[str, Any], policy: GuardedAutoPolicy) -> tuple[str | None, str]:
    """Return (action, reason) for preauthorized low-risk watcher actions.

    Actions are intentionally narrow and must still go through the provider's
    exact report_id/proposal_id apply path. Returning None means report/review
    only.
    """
    if not policy.enabled:
        return None, "autonomy mode is report-only"
    if not isinstance(proposal, dict):
        return None, "invalid proposal"
    proposal_type = str(proposal.get("proposal_type") or "")
    affected = _affected_ids(proposal)
    if not affected:
        return None, "proposal has no explicit affected_ids"
    if _secret_or_manual_only(proposal):
        return None, "proposal requires manual review or may contain secrets"

    confidence = _as_float(proposal.get("confidence"), 0.0)
    risk = str(proposal.get("risk") or "").lower()

    if proposal_type == "heading_noise":
        if (
            risk == "low"
            and proposal.get("guarded_auto_eligible") is True
            and proposal.get("preauthorized_policy") == "guarded-auto:heading-noise"
        ):
            return "delete", "known heading/indexer noise is preauthorized for exact-ID cleanup"
        return None, "heading_noise proposal is not marked guarded-auto eligible"

    if proposal_type == "duplicate_cluster":
        if proposal.get("guarded_auto_eligible") is not True:
            return None, "duplicate cluster is not guarded-auto eligible"
        if proposal.get("preauthorized_policy") != "guarded-auto:exact-duplicate-merge":
            return None, "duplicate cluster lacks the exact-duplicate preauthorization policy"
        if proposal.get("match_kind") != "exact_normalized":
            return None, "duplicate cluster is not exact_normalized"
        if confidence < policy.duplicate_min_confidence:
            return None, "duplicate confidence below guarded-auto threshold"
        if len(affected) > policy.duplicate_max_cluster_size:
            return None, "duplicate cluster exceeds guarded-auto max size"
        if risk != "low":
            return None, "duplicate risk is not low"
        return "merge", "exact normalized duplicate cluster is preauthorized for merge"

    if proposal_type == "learning_promotion_candidate":
        if proposal.get("guarded_auto_eligible") is not True:
            return None, "learning promotion is not guarded-auto eligible"
        if proposal.get("preauthorized_policy") != "guarded-auto:learning-skill-draft":
            return None, "learning promotion lacks the draft-only preauthorization policy"
        if risk != "low":
            return None, "learning promotion risk is not low"
        if confidence < policy.learning_min_confidence:
            return None, "learning confidence below guarded-auto threshold"
        if len(affected) != 1:
            return None, "learning promotion only handles one explicit point at a time"
        return "promote_to_skill", "high-confidence learning promotion is preauthorized as draft-only"

    if proposal_type == "stale_low_value":
        if not policy.stale_quarantine_enabled:
            return None, "stale quarantine disabled"
        if proposal.get("guarded_auto_eligible") is not True:
            return None, "stale proposal is not guarded-auto eligible"
        if proposal.get("preauthorized_policy") != "guarded-auto:stale-low-value-quarantine":
            return None, "stale proposal lacks the quarantine preauthorization policy"
        if risk != "low":
            return None, "stale quarantine risk is not low"
        if len(affected) != 1:
            return None, "stale quarantine only handles one explicit point at a time"
        return "quarantine", "stale low-value point is preauthorized for reversible quarantine"

    return None, "proposal type is review-only"


def apply_guarded_auto(provider: Any, report: dict[str, Any], policy: GuardedAutoPolicy) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "mode": policy.mode,
        "enabled": policy.enabled,
        "attempted": 0,
        "applied": [],
        "skipped": [],
        "errors": [],
    }
    if not policy.enabled:
        return summary

    report_id = str(report.get("report_id") or "").strip()
    raw_proposals = report.get("proposals")
    proposals = raw_proposals if isinstance(raw_proposals, list) else []
    if not report_id:
        summary["errors"].append({"code": "missing_report_id"})
        return summary

    for proposal in proposals:
        if not isinstance(proposal, dict):
            continue
        proposal_id = str(proposal.get("proposal_id") or "").strip()
        proposal_handle = _safe_handle(proposal_id)
        proposal_type = _safe_proposal_type(proposal.get("proposal_type"))
        action, reason = guarded_auto_action_for_proposal(proposal, policy)
        if not action:
            summary["skipped"].append({"proposal_id": proposal_handle, "proposal_type": proposal_type, "reason": reason})
            continue
        if summary["attempted"] >= policy.max_actions:
            summary["skipped"].append({"proposal_id": proposal_handle, "proposal_type": proposal_type, "reason": "guarded-auto max_actions reached"})
            continue
        args = {
            "report_id": report_id,
            "proposal_id": proposal_id,
            "action": action,
            "dry_run": False,
            "approve": True,
            "_guarded_auto": True,
        }
        if action == "quarantine":
            args["quarantine_days"] = policy.quarantine_days
        summary["attempted"] += 1
        try:
            raw = provider.handle_tool_call("qdrant_memory_consolidation_apply", args)
        except Exception:
            summary["errors"].append({"code": "provider_exception", "proposal_handle": proposal_handle, "action": action})
            continue
        try:
            result = json.loads(raw)
        except Exception:
            summary["errors"].append({"code": "invalid_provider_response", "proposal_handle": proposal_handle, "action": action})
            continue
        if not isinstance(result, dict):
            summary["errors"].append({"code": "invalid_provider_response", "proposal_handle": proposal_handle, "action": action})
            continue
        if isinstance(result, dict) and result.get("error"):
            summary["errors"].append({"code": "provider_rejected", "proposal_handle": proposal_handle, "action": action})
            continue
        summary["applied"].append(
            {
                "proposal_id": proposal_handle,
                "proposal_type": proposal_type,
                "action": action,
                "reason": reason,
                "result": _public_result_summary(result),
            }
        )
    return summary
