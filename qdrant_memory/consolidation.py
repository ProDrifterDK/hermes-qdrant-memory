from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from qdrant_memory.lesson_extractor import contains_secret


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
        proposal: dict[str, Any] = {
            "proposal_id": _proposal_id("duplicate_cluster", affected_ids),
            "proposal_type": "duplicate_cluster",
            "collection_name": group[0].collection_name,
            "affected_ids": affected_ids,
            "suggested_action": "merge_review_only",
            "confidence": 0.95,
            "risk": "medium",
            "evidence": [{"id": point.id, "reason": "identical normalized text"} for point in group],
            "requires_explicit_approval": True,
        }
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
        created = _parse_datetime(payload.get("created_at") or payload.get("last_accessed"))
        if not created:
            continue
        age_days = (now - created).days
        importance = _as_int(payload.get("importance"), 5)
        access_count = _as_int(payload.get("access_count"), 0)
        confidence = _as_float(payload.get("confidence"), 1.0)
        if age_days < stale_days or importance >= min_importance_for_keep or access_count > 0 or confidence > 0.5:
            continue
        proposal: dict[str, Any] = {
            "proposal_id": _proposal_id("stale_low_value", [point.id]),
            "proposal_type": "stale_low_value",
            "collection_name": point.collection_name,
            "affected_ids": [point.id],
            "suggested_action": "delete_review_only",
            "confidence": 0.75,
            "risk": "high",
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
        }
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
        }
        if include_examples:
            proposal["examples"] = [{"id": point.id, "text": _snippet(point.text)}]
        proposals.append(proposal)
        if len(proposals) >= max_groups:
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
) -> dict[str, Any]:
    proposals: list[dict[str, Any]] = []
    proposals.extend(_quality_warning_proposals(memory_points + learning_points, max_groups=max_groups))
    proposals.extend(_duplicate_proposals(memory_points, max_groups=max_groups, include_examples=include_examples, threshold=duplicate_threshold))
    proposals.extend(_duplicate_proposals(learning_points, max_groups=max_groups, include_examples=include_examples, threshold=duplicate_threshold))
    proposals.extend(
        _stale_low_value_proposals(
            memory_points,
            stale_days=stale_days,
            min_importance_for_keep=min_importance_for_keep,
            include_examples=include_examples,
            max_groups=max_groups,
        )
    )
    proposals.extend(_learning_promotion_proposals(learning_points, include_examples=include_examples, max_groups=max_groups))
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
        "analyzed": {"memory_points": len(memory_points), "learning_points": len(learning_points)},
        "summary": summary,
        "proposals": proposals,
        "warnings": ["M8 is report-only. No writes, deletes, metadata updates, merges, or approvals were performed."],
        "next_steps": [
            "Review proposals manually.",
            "Do not apply merge/delete/promote suggestions without explicit approval.",
            "Future M9 may add gated apply-by-proposal-id semantics.",
        ],
    }
