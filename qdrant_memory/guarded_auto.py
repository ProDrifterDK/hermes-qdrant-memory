from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


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
        summary["errors"].append({"error": "guarded-auto requires a persisted report_id"})
        return summary

    for proposal in proposals:
        if not isinstance(proposal, dict):
            continue
        proposal_id = str(proposal.get("proposal_id") or "").strip()
        action, reason = guarded_auto_action_for_proposal(proposal, policy)
        if not action:
            summary["skipped"].append({"proposal_id": proposal_id, "proposal_type": proposal.get("proposal_type"), "reason": reason})
            continue
        if summary["attempted"] >= policy.max_actions:
            summary["skipped"].append({"proposal_id": proposal_id, "proposal_type": proposal.get("proposal_type"), "reason": "guarded-auto max_actions reached"})
            continue
        args = {
            "report_id": report_id,
            "proposal_id": proposal_id,
            "action": action,
            "dry_run": False,
            "approve": True,
        }
        if action == "quarantine":
            args["quarantine_days"] = policy.quarantine_days
        summary["attempted"] += 1
        try:
            raw = provider.handle_tool_call("qdrant_memory_consolidation_apply", args)
            result = json.loads(raw)
        except Exception as exc:
            summary["errors"].append({"proposal_id": proposal_id, "action": action, "reason": reason, "error": str(exc)})
            continue
        if isinstance(result, dict) and result.get("error"):
            summary["errors"].append({"proposal_id": proposal_id, "action": action, "reason": reason, "error": result.get("error")})
            continue
        summary["applied"].append(
            {
                "proposal_id": proposal_id,
                "proposal_type": proposal.get("proposal_type"),
                "action": action,
                "reason": reason,
                "result": result,
            }
        )
    return summary
