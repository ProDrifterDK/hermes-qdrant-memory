from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from qdrant_memory.backup import create_point_snapshot_backup, read_backup_manifest
from qdrant_memory.consolidation import (
    _point_requires_manual_review,
    is_guarded_auto_heading_noise_text,
    normalize_text_fingerprint,
    points_from_qdrant,
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


def _lineage_impact_allows_guarded_auto(proposal: dict[str, Any]) -> bool:
    if str(proposal.get("proposal_type") or "") not in {
        "heading_noise", "duplicate_cluster", "stale_low_value",
    }:
        return True
    impact = proposal.get("lineage_impact")
    if impact is None:
        return True
    return (
        isinstance(impact, dict)
        and impact.get("schema_version") == 1
        and impact.get("complete") is True
        and not impact.get("errors")
        and not impact.get("dependent_ids")
    )


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
    if not _lineage_impact_allows_guarded_auto(proposal):
        return False, "lineage impact requires manual review; generate a fresh report"
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
            and not (getattr(points[0], "payload", {}) or {}).get("consolidation_quarantined")
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
    for key in ("application_id", "canonical_id", "pre_apply_backup_id"):
        if result.get(key):
            summary[key] = _safe_handle(result[key])
    snapshot = result.get("pre_mutation_snapshot")
    if isinstance(snapshot, dict) and snapshot.get("backup_id"):
        summary["snapshot_backup_id"] = _safe_handle(snapshot["backup_id"])
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
    if not _lineage_impact_allows_guarded_auto(proposal):
        return None, "lineage impact requires manual review"

    confidence = _as_float(proposal.get("confidence"), 0.0)
    risk = str(proposal.get("risk") or "").lower()

    if proposal_type == "heading_noise":
        if (
            risk == "low"
            and proposal.get("guarded_auto_eligible") is True
            and proposal.get("preauthorized_policy") == "guarded-auto:heading-noise"
        ):
            return "quarantine", "known heading/indexer noise is preauthorized for reversible exact-ID quarantine"
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


def _public_backup_summary(backup: Any) -> dict[str, Any]:
    if not isinstance(backup, dict):
        return {}
    collections = backup.get("collections") if isinstance(backup.get("collections"), dict) else {}
    return {
        "backup_id": _safe_handle(backup.get("backup_id")),
        "scope": str(backup.get("scope") or ""),
        "collections": {
            str(scope): {
                "collection_name": _safe_handle(details.get("collection_name")),
                "count": _as_int(details.get("count"), 0),
            }
            for scope, details in collections.items()
            if isinstance(details, dict)
        },
    }


def _backup_scope_for(report: dict[str, Any], proposals: list[dict[str, Any]]) -> str:
    collections = report.get("collections") if isinstance(report.get("collections"), dict) else {}
    learning_name = str(collections.get("learning") or "")
    scopes = {
        "learning" if learning_name and str(proposal.get("collection_name") or "") == learning_name else "memory"
        for proposal in proposals
    }
    return scopes.pop() if len(scopes) == 1 else "both"


def guarded_auto_backup_manifest(config: dict[str, Any], hermes_home: str, backup_id: str) -> dict[str, Any]:
    """Return the manifest of a full (non-partial) plugin backup, or raise."""
    manifest = read_backup_manifest(config, hermes_home=hermes_home, backup_id=backup_id)
    if manifest.get("partial") is True:
        raise ValueError("a partial point snapshot is not a full pre-mutation backup")
    return manifest


def write_guarded_auto_snapshot(
    *,
    raw_points: list[Any],
    report_id: str,
    proposal: dict[str, Any],
    action: str,
    collection_name: str,
    config: dict[str, Any],
    hermes_home: str,
) -> dict[str, Any]:
    """Write the digest-bound snapshot of the points one guarded-auto action is about to change.

    The snapshot is a partial plugin backup (payloads and vectors), so
    ``hermes qdrant restore --backup <backup_id> --no-dry-run --approve`` undoes exactly this action. It is
    refused unless the points read now are the ones the report sealed: same IDs, same
    content and payload digests.
    """
    affected_ids = _affected_ids(proposal)
    points = points_from_qdrant(list(raw_points or []), collection_name=collection_name)
    if sorted(str(point.id) for point in points) != sorted(set(affected_ids)):
        raise ValueError("snapshot points do not match the proposal's affected IDs")
    digests = _point_digest_records(points)
    sealed = proposal.get("guarded_auto_snapshot") if isinstance(proposal.get("guarded_auto_snapshot"), dict) else {}
    if sealed.get("point_digests") != digests:
        raise ValueError("snapshot points do not match the sealed proposal digests")
    if collection_name == str(config.get("collection_name") or ""):
        scope = "memory"
    elif collection_name == str(config.get("learning_collection_name") or ""):
        scope = "learning"
    else:
        raise ValueError("snapshot collection is not a configured collection")
    manifest = create_point_snapshot_backup(
        list(raw_points),
        config,
        hermes_home=hermes_home,
        scope=scope,
        collection_name=collection_name,
        snapshot={
            "kind": "guarded_auto_pre_mutation",
            "report_id": report_id,
            "proposal_id": str(proposal.get("proposal_id") or ""),
            "proposal_type": str(proposal.get("proposal_type") or ""),
            "proposal_sha256": str(proposal.get("guarded_auto_proposal_sha256") or ""),
            "action": action,
            "point_digests": digests,
        },
    )
    details = (manifest.get("collections") or {}).get(scope) or {}
    return {
        "backup_id": manifest.get("backup_id"),
        "backup_dir": manifest.get("backup_dir"),
        "collection_name": collection_name,
        "point_ids": sorted(str(point.id) for point in points),
        "point_digests": digests,
        "file_sha256": details.get("sha256"),
    }


def apply_guarded_auto(provider: Any, report: dict[str, Any], policy: GuardedAutoPolicy) -> dict[str, Any]:
    """Apply the preauthorized proposals of one watcher report.

    Order: static policy decision per proposal, capped at ``max_actions``; a dry-run
    preflight through the provider's exact-ID apply path (every current-point check runs,
    nothing is written); one full plugin backup of the touched collections
    (``provider.guarded_auto_backup``); then each live apply carrying that backup ID. The
    apply path refuses a watcher mutation without a verified backup and writes a
    digest-bound point snapshot plus an application record per action. If the backup fails,
    nothing is applied.

    In report-only mode the same static decision is reported as ``would_apply`` and the
    provider is not called.
    """
    summary: dict[str, Any] = {
        "mode": policy.mode,
        "enabled": policy.enabled,
        "attempted": 0,
        "applied": [],
        "skipped": [],
        "errors": [],
        "would_apply": [],
    }
    raw_proposals = report.get("proposals")
    proposals = raw_proposals if isinstance(raw_proposals, list) else []
    preview_policy = policy if policy.enabled else GuardedAutoPolicy(**{**policy.__dict__, "mode": "guarded-auto"})

    candidates: list[tuple[dict[str, Any], str, str]] = []
    for proposal in proposals:
        if not isinstance(proposal, dict):
            continue
        proposal_handle = _safe_handle(str(proposal.get("proposal_id") or "").strip())
        proposal_type = _safe_proposal_type(proposal.get("proposal_type"))
        action, reason = guarded_auto_action_for_proposal(proposal, preview_policy)
        if not action:
            if policy.enabled:
                summary["skipped"].append({"proposal_id": proposal_handle, "proposal_type": proposal_type, "reason": reason})
            continue
        if len(candidates) >= policy.max_actions:
            if policy.enabled:
                summary["skipped"].append({"proposal_id": proposal_handle, "proposal_type": proposal_type, "reason": "guarded-auto max_actions reached"})
            continue
        candidates.append((proposal, action, reason))
    if not policy.enabled:
        summary["would_apply"] = [
            {
                "proposal_id": _safe_handle(str(proposal.get("proposal_id") or "")),
                "proposal_type": _safe_proposal_type(proposal.get("proposal_type")),
                "action": action,
                "affected_ids": [_safe_handle(value) for value in _affected_ids(proposal)],
            }
            for proposal, action, _reason in candidates
        ]
        return summary

    report_id = str(report.get("report_id") or "").strip()
    if not report_id:
        summary["errors"].append({"code": "missing_report_id"})
        return summary

    def _call(args: dict[str, Any], proposal_handle: str, action: str) -> dict[str, Any] | None:
        try:
            raw = provider.handle_tool_call("qdrant_memory_consolidation_apply", args)
        except Exception:
            summary["errors"].append({"code": "provider_exception", "proposal_handle": proposal_handle, "action": action})
            return None
        try:
            result = json.loads(raw)
        except Exception:
            summary["errors"].append({"code": "invalid_provider_response", "proposal_handle": proposal_handle, "action": action})
            return None
        if not isinstance(result, dict):
            summary["errors"].append({"code": "invalid_provider_response", "proposal_handle": proposal_handle, "action": action})
            return None
        return result

    def _base_args(proposal: dict[str, Any], action: str) -> dict[str, Any]:
        args = {
            "report_id": report_id,
            "proposal_id": str(proposal.get("proposal_id") or "").strip(),
            "action": action,
            "_guarded_auto": True,
        }
        if action == "quarantine":
            args["quarantine_days"] = policy.quarantine_days
        return args

    ready: list[tuple[dict[str, Any], str, str]] = []
    for proposal, action, reason in candidates:
        proposal_handle = _safe_handle(str(proposal.get("proposal_id") or "").strip())
        result = _call({**_base_args(proposal, action), "dry_run": True}, proposal_handle, action)
        if result is None:
            continue
        if result.get("error") or result.get("would_apply") is not True:
            # A proposal the report marked preauthorized that the exact-ID apply path now
            # refuses (changed point, lineage-managed target, ...) is surfaced, not hidden.
            summary["errors"].append({"code": "provider_rejected", "proposal_handle": proposal_handle, "action": action})
            continue
        ready.append((proposal, action, reason))
        summary["would_apply"].append(
            {
                "proposal_id": proposal_handle,
                "proposal_type": _safe_proposal_type(proposal.get("proposal_type")),
                "action": action,
                "affected_ids": [_safe_handle(value) for value in _affected_ids(proposal)],
            }
        )
    if not ready:
        return summary

    backup_scope = _backup_scope_for(report, [proposal for proposal, _action, _reason in ready])
    backup_fn = getattr(provider, "guarded_auto_backup", None)
    backup: Any = None
    if callable(backup_fn):
        try:
            backup = backup_fn(scope=backup_scope)
        except Exception:
            backup = None
    backup_id = str(backup.get("backup_id") or "").strip() if isinstance(backup, dict) else ""
    if not backup_id:
        summary["errors"].append({"code": "backup_failed", "scope": backup_scope})
        for proposal, _action, _reason in ready:
            summary["skipped"].append(
                {
                    "proposal_id": _safe_handle(str(proposal.get("proposal_id") or "").strip()),
                    "proposal_type": _safe_proposal_type(proposal.get("proposal_type")),
                    "reason": "pre-mutation backup failed; nothing applied",
                }
            )
        return summary
    summary["backup"] = _public_backup_summary(backup)

    for proposal, action, reason in ready:
        proposal_handle = _safe_handle(str(proposal.get("proposal_id") or "").strip())
        proposal_type = _safe_proposal_type(proposal.get("proposal_type"))
        summary["attempted"] += 1
        result = _call(
            {**_base_args(proposal, action), "dry_run": False, "approve": True, "guarded_auto_backup_id": backup_id},
            proposal_handle,
            action,
        )
        if result is None:
            continue
        if result.get("error"):
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
