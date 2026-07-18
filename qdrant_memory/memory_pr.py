from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from qdrant_memory.consolidation import (
    IDENTITY_REDACTED_SNIPPET,
    SECRET_KEYWORDS,
    expected_action_for_proposal,
    identity_bearing_payload,
    redact_secrets,
)
from qdrant_memory.lesson_extractor import contains_secret

SCHEMA_NAME = "hermes-qdrant-memory.memory-pr"
SCHEMA_VERSION = 1
MAX_ID_CHARS = 128
MAX_AFFECTED_POINTS = 100
MAX_SNIPPET_CHARS = 360
MAX_SUMMARY_CHARS = 1200
MAX_FIELD_CHARS = 1200
MAX_SNAPSHOT_STRING_CHARS = 4096
MAX_CONTAINER_ITEMS = 100
MAX_NESTING_DEPTH = 8

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_REDACTED_VALUE = "[redacted: possible secret-bearing value]"
_TRUNCATED_VALUE = "[truncated: review field limit]"
_PROVENANCE_KEYS = (
    "source_type",
    "source_uri",
    "locator",
    "content_hash",
    "source_modified_at",
    "created_at",
    "updated_at",
    "observed_at",
    "valid_from",
    "valid_until",
    "derivation_type",
    "derived_from",
)
_IDENTITY_PROVENANCE_VALUE_KEYS = {"source_uri", "locator", "content_hash", "derived_from"}


class MemoryPRValidationError(ValueError):
    """Raised when exact Memory PR inputs cannot be safely reconciled."""


def validate_exact_id(value: Any, label: str = "id") -> str:
    """Return a canonical exact ID or fail closed for path-like input."""
    if not isinstance(value, str) or not value or len(value) > MAX_ID_CHARS or not _ID_RE.fullmatch(value):
        raise MemoryPRValidationError(f"invalid {label}; expected an exact alphanumeric, hyphen, or underscore ID")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _bounded_text(value: Any, *, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if contains_secret(text) or "bearer" in text.casefold():
        return _REDACTED_VALUE
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _safe_mapping_key(value: Any) -> str:
    key = str(value)
    if contains_secret(key) or "bearer" in key.casefold():
        return "[redacted: possible secret-bearing key]"
    return _bounded_text(key, max_chars=160)


def _bound_sanitized(value: Any, *, max_string_chars: int, depth: int = 0) -> Any:
    if depth >= MAX_NESTING_DEPTH:
        return _TRUNCATED_VALUE
    if isinstance(value, Mapping):
        bounded: dict[str, Any] = {}
        items = sorted(value.items(), key=lambda item: str(item[0]))
        for index, (key, item) in enumerate(items):
            if index >= MAX_CONTAINER_ITEMS:
                bounded["[truncated]"] = _TRUNCATED_VALUE
                break
            bounded[_safe_mapping_key(key)] = _bound_sanitized(item, max_string_chars=max_string_chars, depth=depth + 1)
        return bounded
    if isinstance(value, (list, tuple)):
        result = [
            _bound_sanitized(item, max_string_chars=max_string_chars, depth=depth + 1)
            for item in list(value)[:MAX_CONTAINER_ITEMS]
        ]
        if len(value) > MAX_CONTAINER_ITEMS:
            result.append(_TRUNCATED_VALUE)
        return result
    if isinstance(value, str):
        return _bounded_text(value, max_chars=max_string_chars)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _bounded_text(value, max_chars=max_string_chars)


def sanitize_for_review(value: Any, *, max_string_chars: int = MAX_FIELD_CHARS) -> Any:
    """Recursively redact first, then apply deterministic size bounds."""
    return _bound_sanitized(redact_secrets(value), max_string_chars=max_string_chars)


def _point_id(point: Any) -> str:
    if isinstance(point, Mapping):
        return str(point.get("id") or "")
    return str(getattr(point, "id", "") or "")


def _point_payload(point: Any) -> dict[str, Any]:
    raw = point.get("payload") if isinstance(point, Mapping) else getattr(point, "payload", None)
    return dict(raw) if isinstance(raw, Mapping) else {}


def _point_text(point: Any) -> str:
    payload = _point_payload(point)
    if isinstance(point, Mapping) and point.get("text") not in (None, ""):
        return str(point.get("text") or "")
    return str(getattr(point, "text", "") or payload.get("text") or payload.get("lesson") or "")


def _point_collection(point: Any) -> str:
    if isinstance(point, Mapping):
        return str(point.get("collection_name") or "")
    return str(getattr(point, "collection_name", "") or "")


def _snapshot_projection(point: Any) -> dict[str, Any]:
    payload = _point_payload(point)
    if identity_bearing_payload(payload):
        safe_text: Any = IDENTITY_REDACTED_SNIPPET
        safe_payload: Any = {
            "identity_bearing": True,
            "source_type": sanitize_for_review(payload.get("source_type", "unknown"), max_string_chars=160),
            "canonical": bool(payload.get("canonical", False)),
            "stale": bool(payload.get("stale", False)),
            "requires_review": True,
            "fact_status": sanitize_for_review(payload.get("fact_status") or "active", max_string_chars=80),
        }
    else:
        safe_text = sanitize_for_review(_point_text(point), max_string_chars=MAX_SNAPSHOT_STRING_CHARS)
        safe_payload = sanitize_for_review(payload, max_string_chars=MAX_SNAPSHOT_STRING_CHARS)
    return {"text": safe_text, "payload": safe_payload}


def stable_point_snapshot_digest(point: Any) -> str:
    """Hash stable sanitized point content without exposing the snapshot."""
    return _sha256(_snapshot_projection(point))


def _point_secret_bearing(point: Any) -> bool:
    def nested_secret(value: Any) -> bool:
        if isinstance(value, Mapping):
            for key, item in value.items():
                key_text = str(key).casefold()
                if (
                    contains_secret(str(key))
                    or "bearer" in key_text
                    or any(keyword in key_text for keyword in SECRET_KEYWORDS)
                ):
                    return True
                if nested_secret(item):
                    return True
            return False
        if isinstance(value, (list, tuple)):
            return any(nested_secret(item) for item in value)
        if isinstance(value, str):
            return contains_secret(value) or "bearer" in value.casefold()
        return False

    payload_text = _canonical_json(_point_payload(point))
    text = _point_text(point)
    return (
        nested_secret(_point_payload(point))
        or contains_secret(text)
        or contains_secret(payload_text)
        or "bearer" in text.casefold()
        or "bearer" in payload_text.casefold()
    )


def review_point_snapshots(points: Sequence[Any]) -> list[dict[str, str]]:
    """Build the digest-only report snapshot records used for drift checks."""
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for point in points:
        point_id = validate_exact_id(_point_id(point), "point_id")
        if point_id in seen:
            raise MemoryPRValidationError("duplicate current affected point ID")
        seen.add(point_id)
        records.append({"id": point_id, "snapshot_digest": stable_point_snapshot_digest(point)})
    return sorted(records, key=lambda item: item["id"])


def attach_review_point_snapshots(report: dict[str, Any], points: Sequence[Any]) -> dict[str, Any]:
    """Attach digest-only exact-point snapshots to every generated proposal."""
    by_collection_and_id: dict[tuple[str, str], Any] = {}
    for point in points:
        point_id = validate_exact_id(_point_id(point), "point_id")
        collection_name = validate_exact_id(_point_collection(point), "collection_name")
        key = (collection_name, point_id)
        if key in by_collection_and_id:
            raise MemoryPRValidationError("duplicate report source point ID in collection")
        by_collection_and_id[key] = point
    proposals = report.get("proposals")
    if not isinstance(proposals, list):
        raise MemoryPRValidationError("invalid consolidation report proposals")
    for proposal in proposals:
        if not isinstance(proposal, dict):
            raise MemoryPRValidationError("invalid consolidation report proposal")
        collection_name = validate_exact_id(str(proposal.get("collection_name") or ""), "collection_name")
        affected_ids = _affected_ids(proposal)
        proposal_points: list[Any] = []
        for point_id in affected_ids:
            point = by_collection_and_id.get((collection_name, point_id))
            if point is None:
                raise MemoryPRValidationError("proposal affected point missing from consolidation report inputs")
            proposal_points.append(point)
        proposal["review_point_snapshots"] = review_point_snapshots(proposal_points)
    return report


def _proposal_from_report(report: Mapping[str, Any], proposal_id: str) -> dict[str, Any]:
    proposals = report.get("proposals")
    if not isinstance(proposals, list):
        raise MemoryPRValidationError("invalid consolidation report proposals")
    matches = [
        item for item in proposals if isinstance(item, Mapping) and str(item.get("proposal_id") or "") == proposal_id
    ]
    if len(matches) != 1:
        raise MemoryPRValidationError(f"proposal_id not found or not unique: {proposal_id}")
    proposal = dict(matches[0])
    validate_exact_id(proposal.get("proposal_id"), "proposal_id")
    return proposal


def _affected_ids(proposal: Mapping[str, Any]) -> list[str]:
    raw_ids = proposal.get("affected_ids")
    if not isinstance(raw_ids, list) or not raw_ids or len(raw_ids) > MAX_AFFECTED_POINTS:
        raise MemoryPRValidationError("proposal must contain a bounded non-empty affected_ids list")
    ids = [validate_exact_id(item, "affected point ID") for item in raw_ids]
    if len(ids) != len(set(ids)):
        raise MemoryPRValidationError("proposal contains duplicate affected point IDs")
    return sorted(ids)


def _report_snapshot_map(proposal: Mapping[str, Any], affected_ids: Sequence[str]) -> dict[str, str]:
    raw = proposal.get("review_point_snapshots")
    if raw in (None, []):
        return {}
    if not isinstance(raw, list):
        raise MemoryPRValidationError("invalid report review point snapshots")
    result: dict[str, str] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise MemoryPRValidationError("invalid report review point snapshot")
        point_id = validate_exact_id(item.get("id"), "snapshot point ID")
        digest = str(item.get("snapshot_digest") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise MemoryPRValidationError("invalid report review point snapshot digest")
        if point_id in result:
            raise MemoryPRValidationError("duplicate report review point snapshot ID")
        result[point_id] = digest
    if set(result) != set(affected_ids):
        raise MemoryPRValidationError("report snapshot affected point IDs do not match proposal affected point IDs")
    return result


def _fact_status(payload: Mapping[str, Any]) -> str:
    status = str(payload.get("fact_status") or "").strip().lower()
    if status:
        return _bounded_text(status, max_chars=80)
    return "stale" if payload.get("stale") is True else "active"


def _point_provenance(payload: Mapping[str, Any], *, identity_bearing: bool) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in _PROVENANCE_KEYS:
        value = payload.get(key)
        if value in (None, "", [], {}):
            continue
        if identity_bearing and key in _IDENTITY_PROVENANCE_VALUE_KEYS:
            result[key] = IDENTITY_REDACTED_SNIPPET
        else:
            result[key] = sanitize_for_review(value, max_string_chars=MAX_FIELD_CHARS)
    if "source_type" not in result:
        result["source_type"] = "unknown"
    return result


def _evidence_for_point(point: Any, report_digest: str | None) -> dict[str, Any]:
    point_id = validate_exact_id(_point_id(point), "point_id")
    payload = _point_payload(point)
    identity_bearing = identity_bearing_payload(payload)
    secret_bearing = _point_secret_bearing(point)
    current_digest = stable_point_snapshot_digest(point)
    drift_status = (
        "unknown"
        if report_digest is None or identity_bearing or secret_bearing
        else ("unchanged" if report_digest == current_digest else "changed")
    )
    snippet = (
        IDENTITY_REDACTED_SNIPPET
        if identity_bearing
        else _bounded_text(
            sanitize_for_review(_point_text(point), max_string_chars=MAX_SNIPPET_CHARS), max_chars=MAX_SNIPPET_CHARS
        )
    )
    return {
        "id": point_id,
        "snippet": snippet or "[empty memory text]",
        "identity_bearing": identity_bearing,
        "secret_bearing": secret_bearing,
        "snapshot_scope": "redacted_sensitive_state" if identity_bearing or secret_bearing else "sanitized_point",
        "provenance": _point_provenance(payload, identity_bearing=identity_bearing),
        "state": {
            "canonical": bool(payload.get("canonical", False)),
            "stale": bool(payload.get("stale", False)),
            "requires_review": bool(payload.get("requires_review", False) or identity_bearing),
            "fact_status": _fact_status(payload),
        },
        "snapshot_digest": current_digest,
        "report_snapshot_digest": report_digest,
        "drift_status": drift_status,
    }


def _summary_text(proposal: Mapping[str, Any]) -> str:
    for key in ("candidate_statement", "summary", "reason", "supersession_reason", "manual_review_reason"):
        value = proposal.get(key)
        if value in (None, "", [], {}):
            continue
        safe = sanitize_for_review(value, max_string_chars=MAX_SUMMARY_CHARS)
        if isinstance(safe, str):
            return _bounded_text(safe, max_chars=MAX_SUMMARY_CHARS)
        return _bounded_text(_canonical_json(safe), max_chars=MAX_SUMMARY_CHARS)
    return "Review the current exact point evidence against the persisted proposal."


def _identity_safe_status_changes(changes: Any, identity_ids: set[str]) -> Any:
    safe = sanitize_for_review(changes or [], max_string_chars=MAX_FIELD_CHARS)
    if not identity_ids or not isinstance(safe, list):
        return safe
    result: list[Any] = []
    for item in safe:
        if not isinstance(item, Mapping) or str(item.get("id") or "") not in identity_ids:
            result.append(item)
            continue
        preserved = {key: value for key, value in item.items() if key in {"id", "from", "to", "superseded_by"}}
        preserved["reason"] = IDENTITY_REDACTED_SNIPPET
        result.append(preserved)
    return result


def _identity_safe_persisted_evidence(value: Any, identity_ids: set[str]) -> Any:
    safe = sanitize_for_review(value or [], max_string_chars=MAX_SNIPPET_CHARS)
    if not identity_ids or not isinstance(safe, list):
        return safe
    result: list[Any] = []
    for item in safe:
        if not isinstance(item, Mapping) or str(item.get("id") or "") not in identity_ids:
            result.append(item)
            continue
        redacted = dict(item)
        for key in ("text", "snippet", "lesson", "source_uri", "locator", "derived_from"):
            if key in redacted:
                redacted[key] = IDENTITY_REDACTED_SNIPPET
        result.append(redacted)
    return result


def _dry_run_next_step(report_id: str, proposal_id: str, expected_action: str | None) -> dict[str, Any]:
    if not expected_action:
        return {
            "tool": None,
            "arguments": None,
            "statement": "No automatic apply path exists for this proposal type; continue with manual review only.",
        }
    return {
        "tool": "qdrant_memory_consolidation_apply",
        "arguments": {
            "report_id": report_id,
            "proposal_id": proposal_id,
            "action": expected_action,
            "dry_run": True,
        },
        "statement": "Preview this exact proposal through the existing dry-run apply gate. This packet does not execute it.",
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_memory_pr(
    *,
    report: Mapping[str, Any],
    report_id: str,
    proposal_id: str,
    current_points: Sequence[Any],
    generated_at: str | None = None,
    generation_mode: str = "live",
    resolved_collection_name: str | None = None,
) -> dict[str, Any]:
    """Build one deterministic, sanitized Memory PR packet without writes."""
    exact_report_id = validate_exact_id(report_id, "report_id")
    exact_proposal_id = validate_exact_id(proposal_id, "proposal_id")
    embedded_report_id = validate_exact_id(report.get("report_id"), "report_id")
    if embedded_report_id != exact_report_id:
        raise MemoryPRValidationError("requested report_id does not match the persisted report")
    if generation_mode not in {"live", "fixture"}:
        raise MemoryPRValidationError("generation_mode must be live or fixture")

    proposal = _proposal_from_report(report, exact_proposal_id)
    affected_ids = _affected_ids(proposal)
    report_snapshots = _report_snapshot_map(proposal, affected_ids)

    points_by_id: dict[str, Any] = {}
    for point in current_points:
        point_id = validate_exact_id(_point_id(point), "current point ID")
        if point_id in points_by_id:
            raise MemoryPRValidationError("duplicate current affected point ID")
        points_by_id[point_id] = point
    if set(points_by_id) != set(affected_ids):
        raise MemoryPRValidationError("current affected point IDs do not exactly match proposal affected point IDs")

    collection_name = validate_exact_id(
        str(resolved_collection_name or proposal.get("collection_name") or "memory"),
        "collection_name",
    )
    for point in points_by_id.values():
        point_collection = _point_collection(point)
        if point_collection and point_collection != collection_name:
            raise MemoryPRValidationError("current point collection does not match proposal collection")

    evidence = [
        _evidence_for_point(points_by_id[point_id], report_snapshots.get(point_id)) for point_id in affected_ids
    ]
    identity_ids = {item["id"] for item in evidence if item.get("identity_bearing")}
    drift_values = {item["drift_status"] for item in evidence}
    drift_status = "changed" if "changed" in drift_values else ("unknown" if "unknown" in drift_values else "unchanged")
    proposal_type = _bounded_text(proposal.get("proposal_type") or "unknown", max_chars=120)
    expected_action = expected_action_for_proposal(str(proposal.get("proposal_type") or ""))

    stable_review_content = {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "generation_mode": generation_mode,
        "report_id": exact_report_id,
        "proposal_id": exact_proposal_id,
        "proposal_type": proposal_type,
        "expected_action": expected_action,
        "suggested_action": sanitize_for_review(
            proposal.get("suggested_action") or proposal.get("action") or expected_action or "manual_review",
            max_string_chars=120,
        ),
        "risk": sanitize_for_review(proposal.get("risk") or "unknown", max_string_chars=80),
        "confidence": sanitize_for_review(proposal.get("confidence"), max_string_chars=80),
        "collection_name": collection_name,
        "affected_point_ids": affected_ids,
        "proposal_summary": (
            "Identity-bearing proposal summary suppressed; review exact status and provenance fields only."
            if identity_ids
            else _summary_text(proposal)
        ),
        "proposed_status_changes": _identity_safe_status_changes(proposal.get("proposed_status_changes"), identity_ids),
        "persisted_evidence": _identity_safe_persisted_evidence(
            proposal.get("source_snippets") or proposal.get("examples"), identity_ids
        ),
        "current_evidence": evidence,
        "drift_status": drift_status,
        "write_boundary": {
            "review_only": True,
            "qdrant_mutation": False,
            "memory_payload_mutation": False,
            "source_mutation": False,
            "user_file_mutation": False,
            "artifact_writes": "Only explicit caller-selected JSON and HTML artifact output is permitted.",
        },
        "reviewer_checklist": [
            "Confirm the exact report, proposal, collection, and affected point IDs.",
            "Review every drift label before relying on persisted proposal evidence.",
            "Compare current provenance, canonical, stale, review, and fact-status fields.",
            "Decide whether the proposed status changes are justified by the current evidence.",
            "Use the exact dry-run next step only after review; do not infer mutation authority from this packet.",
        ],
        "dry_run_next_step": _dry_run_next_step(exact_report_id, exact_proposal_id, expected_action),
    }
    content_digest = _sha256(stable_review_content)
    return {
        **stable_review_content,
        "memory_pr_id": f"mpr-{content_digest[:20]}",
        "content_digest": content_digest,
        "generated_at": generated_at or _utc_now(),
    }


def _esc(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return html.escape(str(value), quote=True)


def _badge(label: Any, kind: str = "neutral") -> str:
    safe_kind = kind if kind in {"neutral", "amber", "ok", "danger", "unknown"} else "neutral"
    return f'<span class="badge badge--{safe_kind}">{_esc(label)}</span>'


def _drift_badge(value: Any) -> str:
    drift = str(value or "unknown")
    return _badge(
        f"drift: {drift}", {"unchanged": "ok", "changed": "danger", "unknown": "unknown"}.get(drift, "unknown")
    )


def _status_changes_html(changes: Any) -> str:
    if not isinstance(changes, list) or not changes:
        return '<p class="muted">No structured status changes were included in the proposal.</p>'
    rows: list[str] = []
    for item in changes[:MAX_CONTAINER_ITEMS]:
        change = item if isinstance(item, Mapping) else {"reason": item}
        rows.append(
            "<tr>"
            f'<th scope="row"><code>{_esc(change.get("id"))}</code></th>'
            f"<td>{_esc(change.get('from'))}</td>"
            f"<td>{_esc(change.get('to'))}</td>"
            f"<td>{_esc(change.get('reason'))}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table><caption>Persisted proposed point-status transitions</caption>'
        '<thead><tr><th scope="col">Point ID</th><th scope="col">Before</th><th scope="col">After</th><th scope="col">Reason</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _provenance_html(provenance: Any) -> str:
    if not isinstance(provenance, Mapping) or not provenance:
        return '<p class="muted">No provenance fields available.</p>'
    items = "".join(f"<dt>{_esc(key)}</dt><dd>{_esc(value)}</dd>" for key, value in sorted(provenance.items()))
    return f'<dl class="provenance">{items}</dl>'


def _evidence_html(evidence: Any) -> str:
    cards: list[str] = []
    for item in evidence if isinstance(evidence, list) else []:
        if not isinstance(item, Mapping):
            continue
        state = item.get("state") if isinstance(item.get("state"), Mapping) else {}
        state_badges = "".join(
            [
                _badge(f"fact: {state.get('fact_status', 'unknown')}", "neutral"),
                _badge(
                    f"canonical: {str(bool(state.get('canonical'))).lower()}",
                    "amber" if state.get("canonical") else "neutral",
                ),
                _badge(f"stale: {str(bool(state.get('stale'))).lower()}", "danger" if state.get("stale") else "ok"),
                _badge(
                    f"review: {str(bool(state.get('requires_review'))).lower()}",
                    "amber" if state.get("requires_review") else "ok",
                ),
                _drift_badge(item.get("drift_status")),
            ]
        )
        identity_note = (
            '<p class="identity-note">Identity-bearing memory: snippet intentionally suppressed.</p>'
            if item.get("identity_bearing")
            else ""
        )
        cards.append(
            '<article class="evidence-card">'
            '<header class="evidence-card__header">'
            f'<div><p class="eyebrow">Current exact point</p><h3><code>{_esc(item.get("id"))}</code></h3></div>'
            f'<div class="badge-row">{state_badges}</div>'
            "</header>"
            f"<blockquote>{_esc(item.get('snippet'))}</blockquote>{identity_note}"
            '<section aria-label="Point provenance"><h4>Provenance</h4>'
            f"{_provenance_html(item.get('provenance'))}</section>"
            '<dl class="digest-pair">'
            f"<div><dt>Report snapshot</dt><dd><code>{_esc(item.get('report_snapshot_digest'))}</code></dd></div>"
            f"<div><dt>Current snapshot</dt><dd><code>{_esc(item.get('snapshot_digest'))}</code></dd></div>"
            "</dl></article>"
        )
    return "".join(cards) or '<p class="muted">No current evidence available.</p>'


def render_memory_pr_html(packet: Mapping[str, Any]) -> str:
    """Render a static, escaped, self-contained Memory PR review artifact."""
    checklist = packet.get("reviewer_checklist") if isinstance(packet.get("reviewer_checklist"), list) else []
    checklist_html = "".join(f"<li>{_esc(item)}</li>" for item in checklist)
    next_step = packet.get("dry_run_next_step") if isinstance(packet.get("dry_run_next_step"), Mapping) else {}
    arguments = next_step.get("arguments")
    next_step_code = _canonical_json(arguments) if arguments is not None else str(next_step.get("statement") or "")
    title = f"Memory PR · {packet.get('proposal_type', 'review')}"
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src 'none'; font-src 'none'; script-src 'none'; connect-src 'none'; media-src 'none'; object-src 'none'; frame-src 'none'; base-uri 'none'; form-action 'none'">
  <meta name="color-scheme" content="dark light">
  <title>{_esc(title)}</title>
  <style>
    :root {{ --void:#090a0b; --panel:#111315; --raised:#17191c; --line:#34373b; --text:#f4f0e8; --muted:#b9b2a5; --amber:#ffbd52; --amber-ink:#211706; --green:#81d8a1; --red:#ff8e82; --unknown:#c7b8e8; --focus:#fff1a8; }}
    * {{ box-sizing:border-box; }}
    html {{ background:var(--void); color:var(--text); font-family:"Avenir Next","Segoe UI",sans-serif; line-height:1.55; }}
    body {{ margin:0; min-width:280px; background:radial-gradient(circle at 85% 5%,#2b2112 0,transparent 24rem),var(--void); }}
    body::before {{ content:""; position:fixed; inset:0; pointer-events:none; opacity:.12; background-image:linear-gradient(#fff 1px,transparent 1px),linear-gradient(90deg,#fff 1px,transparent 1px); background-size:48px 48px; mask-image:linear-gradient(to bottom,#000,transparent 60%); }}
    a {{ color:var(--amber); }} a:focus-visible, summary:focus-visible {{ outline:3px solid var(--focus); outline-offset:4px; }}
    .skip-link {{ position:absolute; left:1rem; top:-5rem; z-index:10; padding:.8rem 1rem; color:#000; background:var(--focus); font-weight:800; }} .skip-link:focus {{ top:1rem; }}
    .shell {{ width:min(1180px,calc(100% - 2rem)); margin-inline:auto; position:relative; }}
    .masthead {{ padding:clamp(3.5rem,10vw,8rem) 0 2.5rem; border-bottom:1px solid var(--line); }}
    .kicker,.eyebrow {{ margin:0 0 .45rem; color:var(--amber); font-size:.75rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; }}
    h1,h2,h3,h4,p {{ overflow-wrap:anywhere; }} h1,h2 {{ font-family:"Iowan Old Style","Palatino Linotype",Palatino,serif; }}
    h1 {{ max-width:16ch; margin:.2rem 0 1rem; font-size:clamp(2.7rem,8vw,6.4rem); line-height:.92; letter-spacing:-.045em; font-weight:600; }}
    h2 {{ margin:0 0 1.2rem; font-size:clamp(1.7rem,4vw,2.7rem); line-height:1.05; }} h3 {{ margin:.15rem 0 0; font-size:1.05rem; }} h4 {{ margin:1.2rem 0 .6rem; }}
    .lede {{ max-width:64ch; color:var(--muted); font-size:clamp(1rem,2vw,1.25rem); }}
    .id-strip {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:1px; margin:2rem 0 0; border:1px solid var(--line); background:var(--line); }}
    .id-strip div {{ padding:1rem; background:var(--panel); }} dt {{ color:var(--muted); font-size:.75rem; letter-spacing:.08em; text-transform:uppercase; }} dd {{ margin:.25rem 0 0; }} code {{ font-family:"Cascadia Mono","IBM Plex Mono",ui-monospace,monospace; font-size:.88em; overflow-wrap:anywhere; }}
    main {{ padding:3rem 0 5rem; }} .section {{ padding:clamp(2rem,6vw,4.5rem) 0; border-bottom:1px solid var(--line); }}
    .summary-grid {{ display:grid; grid-template-columns:minmax(0,1.5fr) minmax(16rem,.7fr); gap:2rem; align-items:start; }}
    .summary-card,.safety-card {{ padding:clamp(1.25rem,4vw,2rem); border:1px solid var(--line); background:linear-gradient(145deg,var(--raised),var(--panel)); box-shadow:0 18px 60px #0006; }}
    .summary-card p {{ font-size:1.15rem; }} .safety-card {{ border-left:4px solid var(--amber); }}
    .badge-row {{ display:flex; flex-wrap:wrap; gap:.45rem; align-items:center; }} .badge {{ display:inline-flex; align-items:center; min-height:1.8rem; padding:.18rem .58rem; border:1px solid currentColor; border-radius:99rem; font-size:.72rem; font-weight:800; letter-spacing:.035em; text-transform:uppercase; }}
    .badge--neutral {{ color:var(--muted); }} .badge--amber {{ color:var(--amber); }} .badge--ok {{ color:var(--green); }} .badge--danger {{ color:var(--red); }} .badge--unknown {{ color:var(--unknown); }}
    .evidence-grid {{ display:grid; gap:1.2rem; }} .evidence-card {{ padding:clamp(1.1rem,3vw,1.7rem); border:1px solid var(--line); background:var(--panel); }}
    .evidence-card__header {{ display:flex; justify-content:space-between; gap:1rem; align-items:flex-start; }} blockquote {{ margin:1.4rem 0; padding:1rem 1.2rem; border-left:3px solid var(--amber); background:#0c0d0e; font-family:"Iowan Old Style",Palatino,serif; font-size:1.1rem; }}
    .identity-note {{ color:var(--amber); font-weight:700; }} .provenance {{ display:grid; grid-template-columns:minmax(8rem,.4fr) minmax(0,1fr); gap:.35rem 1rem; margin:0; }}
    .provenance dt,.provenance dd {{ padding:.35rem 0; border-bottom:1px dotted var(--line); }} .digest-pair {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin:1.5rem 0 0; }} .digest-pair>div {{ min-width:0; padding:.8rem; background:#0c0d0e; }}
    .table-wrap {{ overflow-x:auto; }} table {{ width:100%; border-collapse:collapse; background:var(--panel); }} caption {{ padding:.8rem; color:var(--muted); text-align:left; }} th,td {{ padding:.85rem; border:1px solid var(--line); text-align:left; vertical-align:top; }} thead {{ background:var(--raised); }}
    .review-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:2rem; }} .checklist {{ padding-left:1.35rem; }} .checklist li {{ padding:.45rem 0 .45rem .35rem; }} .next-step {{ padding:1.2rem; border:1px solid var(--amber); background:#171208; }} .next-step pre {{ margin:1rem 0 0; white-space:pre-wrap; overflow-wrap:anywhere; color:var(--focus); }}
    .digest-footer {{ display:grid; grid-template-columns:1fr 2fr; gap:1rem; padding:2rem 0 3rem; color:var(--muted); }} .digest-footer code {{ color:var(--text); }} .muted {{ color:var(--muted); }}
    @media (max-width:760px) {{ .id-strip,.summary-grid,.review-grid,.digest-footer,.digest-pair {{ grid-template-columns:1fr; }} .evidence-card__header {{ display:block; }} .badge-row {{ margin-top:1rem; }} .provenance {{ grid-template-columns:1fr; }} .provenance dd {{ margin-top:-.5rem; }} }}
    @media (prefers-reduced-motion:reduce) {{ *,*::before,*::after {{ scroll-behavior:auto!important; animation:none!important; transition:none!important; }} }}
    @media (forced-colors:active) {{ .badge,.summary-card,.safety-card,.evidence-card,.next-step {{ border:1px solid CanvasText; }} }}
    @media print {{ :root {{ --void:#fff; --panel:#fff; --raised:#f1f1f1; --line:#555; --text:#000; --muted:#333; --amber:#744700; --green:#135c2d; --red:#8b1d15; --unknown:#4c3b70; }} body {{ background:#fff; }} body::before,.skip-link {{ display:none; }} .shell {{ width:100%; }} .masthead {{ padding:0 0 1rem; }} h1 {{ font-size:36pt; }} .section {{ break-inside:avoid; padding:1.4rem 0; }} .evidence-card {{ break-inside:avoid; box-shadow:none; }} a {{ color:#000; text-decoration:none; }} }}
  </style>
</head>
<body>
  <a class="skip-link" href="#main">Skip to review content</a>
  <header class="masthead"><div class="shell">
    <p class="kicker">Memory PR / review-only evidence packet</p>
    <h1>{_esc(packet.get("proposal_type"))}</h1>
    <p class="lede">Git made code changes reviewable. Memory PR makes persistent agent memory reviewable.</p>
    <dl class="id-strip">
      <div><dt>Memory PR</dt><dd><code>{_esc(packet.get("memory_pr_id"))}</code></dd></div>
      <div><dt>Report</dt><dd><code>{_esc(packet.get("report_id"))}</code></dd></div>
      <div><dt>Proposal</dt><dd><code>{_esc(packet.get("proposal_id"))}</code></dd></div>
    </dl>
  </div></header>
  <main id="main" class="shell" tabindex="-1">
    <section class="section" aria-labelledby="summary-title"><div class="summary-grid">
      <div class="summary-card"><p class="eyebrow">Proposed review</p><h2 id="summary-title">What may change</h2><p>{_esc(packet.get("proposal_summary"))}</p>
        <div class="badge-row">{_badge(f"risk: {packet.get('risk')}", "amber")}{_badge(f"confidence: {packet.get('confidence')}")}{_drift_badge(packet.get("drift_status"))}</div>
      </div>
      <aside class="safety-card" aria-labelledby="boundary-title"><p class="eyebrow">Safety boundary</p><h2 id="boundary-title">Review, never mutation</h2><p>This artifact does not mutate Qdrant, memory payloads, sources, or user files. Only explicitly requested artifact output is permitted.</p></aside>
    </div></section>
    <section class="section" aria-labelledby="changes-title"><p class="eyebrow">Before / after</p><h2 id="changes-title">Proposed status changes</h2>{_status_changes_html(packet.get("proposed_status_changes"))}</section>
    <section class="section" aria-labelledby="evidence-title"><p class="eyebrow">Reloaded at generation time</p><h2 id="evidence-title">Current exact-point evidence</h2><div class="evidence-grid">{_evidence_html(packet.get("current_evidence"))}</div></section>
    <section class="section" aria-labelledby="review-title"><div class="review-grid"><div><p class="eyebrow">Human decision gate</p><h2 id="review-title">Reviewer checklist</h2><ol class="checklist">{checklist_html}</ol></div>
      <aside class="next-step" aria-labelledby="next-title"><p class="eyebrow">Not executed</p><h2 id="next-title">Exact dry-run next step</h2><p>{_esc(next_step.get("statement"))}</p><pre><code>{_esc(next_step_code)}</code></pre></aside>
    </div></section>
  </main>
  <footer class="shell digest-footer"><div><strong>Generated</strong><br>{_esc(packet.get("generated_at"))} · {_esc(packet.get("generation_mode"))}</div><div><strong>Machine-readable content digest</strong><br><code>{_esc(packet.get("content_digest"))}</code></div></footer>
</body>
</html>
"""
    return html_text


def _write_private_file(path: Path, content: bytes, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Memory PR artifact already exists: {path.name}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and not overwrite:
            raise FileExistsError(f"Memory PR artifact already exists: {path.name}")
        os.replace(temporary_path, path)
        path.chmod(0o600)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_path.exists():
            temporary_path.unlink()


def write_memory_pr_artifacts(
    packet: Mapping[str, Any], output_dir: str | os.PathLike[str], *, overwrite: bool = False
) -> dict[str, Any]:
    """Persist private JSON and HTML artifacts only to an explicit directory."""
    memory_pr_id = validate_exact_id(str(packet.get("memory_pr_id") or ""), "memory_pr_id")
    root = Path(output_dir)
    if not str(output_dir):
        raise MemoryPRValidationError("an explicit output directory is required")
    if root.exists() and not root.is_dir():
        raise MemoryPRValidationError("output path must be a directory")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    json_path = root / f"memory-pr-{memory_pr_id}.json"
    html_path = root / f"memory-pr-{memory_pr_id}.html"
    if not overwrite and (json_path.exists() or html_path.exists()):
        raise FileExistsError("Memory PR artifacts already exist; use explicit overwrite")
    json_bytes = (json.dumps(dict(packet), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    html_bytes = render_memory_pr_html(packet).encode("utf-8")
    _write_private_file(json_path, json_bytes, overwrite=overwrite)
    _write_private_file(html_path, html_bytes, overwrite=overwrite)
    return {
        "memory_pr_id": memory_pr_id,
        "content_digest": packet.get("content_digest"),
        "json_path": str(json_path),
        "html_path": str(html_path),
        "persisted": True,
    }


def synthetic_fixture_inputs() -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    """Return a license-safe, private-data-free fact supersession fixture."""
    points = [
        {
            "id": "atlas-runtime-v1",
            "collection_name": "memory",
            "payload": {
                "text": "The Atlas build worker uses runtime v1.",
                "source_type": "release_note",
                "source_uri": "fixture://atlas/releases/2026-01",
                "content_hash": "fixture-atlas-v1",
                "observed_at": "2026-01-10T00:00:00Z",
                "fact_key": "atlas.runtime.version",
                "fact_status": "active",
                "canonical": False,
                "stale": False,
                "requires_review": True,
            },
        },
        {
            "id": "atlas-runtime-v2",
            "collection_name": "memory",
            "payload": {
                "text": "The Atlas build worker uses runtime v2.",
                "source_type": "release_note",
                "source_uri": "fixture://atlas/releases/2026-03",
                "content_hash": "fixture-atlas-v2",
                "observed_at": "2026-03-15T00:00:00Z",
                "fact_key": "atlas.runtime.version",
                "fact_status": "active",
                "canonical": True,
                "stale": False,
                "requires_review": True,
            },
        },
    ]
    proposal_id = "fact-supersession-atlas-runtime"
    proposal = {
        "proposal_id": proposal_id,
        "proposal_type": "fact_supersession_candidate",
        "collection_name": "memory",
        "affected_ids": ["atlas-runtime-v1", "atlas-runtime-v2"],
        "suggested_action": "draft_review_only",
        "risk": "medium",
        "confidence": 0.94,
        "candidate_statement": "Runtime v2 is the newer supported Atlas build-worker fact; runtime v1 may be superseded.",
        "proposed_status_changes": [
            {
                "id": "atlas-runtime-v1",
                "from": "active",
                "to": "superseded",
                "reason": "a newer release observation exists",
                "superseded_by": ["atlas-runtime-v2"],
            },
            {
                "id": "atlas-runtime-v2",
                "from": "active",
                "to": "active",
                "reason": "the newer observation remains current",
            },
        ],
        "source_snippets": [
            {
                "id": "atlas-runtime-v1",
                "source_type": "release_note",
                "snippet": "The Atlas build worker uses runtime v1.",
            },
            {
                "id": "atlas-runtime-v2",
                "source_type": "release_note",
                "snippet": "The Atlas build worker uses runtime v2.",
            },
        ],
        "review_point_snapshots": review_point_snapshots(points),
    }
    report = {
        "schema_version": 1,
        "report_id": "fixture-report-atlas-runtime",
        "scope": "memory",
        "proposals": [proposal],
    }
    return report, proposal_id, points


def generate_fixture_artifacts(output_dir: str | os.PathLike[str], *, overwrite: bool = False) -> dict[str, Any]:
    report, proposal_id, points = synthetic_fixture_inputs()
    packet = build_memory_pr(
        report=report,
        report_id=str(report["report_id"]),
        proposal_id=proposal_id,
        current_points=points,
        generated_at="2026-07-17T12:00:00Z",
        generation_mode="fixture",
    )
    return write_memory_pr_artifacts(packet, output_dir, overwrite=overwrite)


def verify_fixture_determinism() -> dict[str, Any]:
    with (
        tempfile.TemporaryDirectory(prefix="memory-pr-verify-a-") as first_dir,
        tempfile.TemporaryDirectory(prefix="memory-pr-verify-b-") as second_dir,
    ):
        first = generate_fixture_artifacts(first_dir)
        second = generate_fixture_artifacts(second_dir)
        json_match = Path(first["json_path"]).read_bytes() == Path(second["json_path"]).read_bytes()
        html_match = Path(first["html_path"]).read_bytes() == Path(second["html_path"]).read_bytes()
        valid = (
            first["memory_pr_id"] == second["memory_pr_id"]
            and first["content_digest"] == second["content_digest"]
            and json_match
            and html_match
        )
        return {
            "valid": valid,
            "memory_pr_id": first["memory_pr_id"],
            "content_digest": first["content_digest"],
            "json_bytes_match": json_match,
            "html_bytes_match": html_match,
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a review-only Memory PR packet from the offline synthetic fixture."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    fixture = commands.add_parser("fixture", help="Generate deterministic synthetic JSON and HTML artifacts.")
    fixture.add_argument("--output-dir", required=True, help="Explicit caller-selected artifact directory.")
    fixture.add_argument("--overwrite", action="store_true", help="Explicitly replace existing fixture artifacts.")
    commands.add_parser(
        "verify-fixture", help="Generate twice in temporary directories and verify byte-for-byte determinism."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "fixture":
            result = generate_fixture_artifacts(args.output_dir, overwrite=bool(args.overwrite))
        else:
            result = verify_fixture_determinism()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("valid", True) else 1
    except (MemoryPRValidationError, FileExistsError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
