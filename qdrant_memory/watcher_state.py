"""Shared, dependency-free state helpers for Qdrant consolidation watchers.

State contract (v1): ``last_proposal_signature`` is the full SHA-256 digest of
canonical proposal data and ``last_signature`` is its 16-character display and
legacy compatibility prefix.  Both watcher writers must import these helpers
rather than reimplementing normalization.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


WATCHER_STATE_SCHEMA = "qdrant-memory-watcher-state-v1"
PROPOSAL_SIGNATURE_SCHEMA = "qdrant-memory-proposal-signature-v1"


def proposal_signature(proposals: Any) -> str:
    """Return the canonical full digest for a JSON consolidation proposal list.

    Only the proposal identity and action fields that historically controlled
    CLI alerts participate.  Proposal and affected-ID order do not affect the
    result, preserving the previous ``last_proposal_signature`` semantics.
    """
    proposal_list = proposals if isinstance(proposals, list) else []
    normalized: list[dict[str, Any]] = []
    for proposal in proposal_list:
        if not isinstance(proposal, dict):
            continue
        affected = proposal.get("affected_ids")
        affected_list = affected if isinstance(affected, list) else []
        normalized.append(
            {
                "proposal_id": proposal.get("proposal_id"),
                "proposal_type": proposal.get("proposal_type"),
                "suggested_action": proposal.get("suggested_action"),
                "affected_ids": sorted(str(item) for item in affected_list),
            }
        )
    normalized.sort(key=lambda item: json.dumps(item, sort_keys=True))
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def proposal_signature_short(proposals: Any) -> str:
    """Return the stable 16-character compatibility/display signature."""
    return proposal_signature(proposals)[:16]


SCAN_WINDOW_SCHEMA = "qdrant-memory-watcher-scan-window-v1"
SCAN_WINDOW_COLLECTIONS = ("memory", "learning")
_SCAN_OFFSET_MAX_CHARS = 64


def scan_offset_or_none(value: Any) -> str | int | None:
    """Return a usable Qdrant point-ID cursor, or ``None`` to start a new cycle.

    Qdrant point IDs are unsigned integers or UUID strings. Anything else found in the
    persisted state (a hand edit, a truncated write) restarts the cycle instead of
    reaching the scroll call.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str):
        text = value.strip()
        if 0 < len(text) <= _SCAN_OFFSET_MAX_CHARS and all(ch.isalnum() or ch == "-" for ch in text):
            return text
    return None


def next_scan_window_state(previous: Any, report_window: Any) -> dict[str, Any]:
    """Advance the persisted rotating-window cursors from one successful watcher run.

    ``previous`` is the stored ``scan_window`` state (or anything else), ``report_window``
    the ``scan_window`` block of the report. A collection the report did not scan keeps
    its cursor. When a scroll reached the end, the cursor resets to ``None`` and the cycle
    counter advances, so the next run starts again at the first ID.
    """
    stored = previous if isinstance(previous, dict) and previous.get("schema") == SCAN_WINDOW_SCHEMA else {}
    window = report_window if isinstance(report_window, dict) else {}
    state: dict[str, Any] = {"schema": SCAN_WINDOW_SCHEMA}
    for label in SCAN_WINDOW_COLLECTIONS:
        old = stored.get(label) if isinstance(stored.get(label), dict) else {}
        scanned = window.get(label) if isinstance(window.get(label), dict) else None
        if scanned is None:
            if old:
                state[label] = dict(old)
            continue
        collection_name = str(scanned.get("collection_name") or "")
        same_collection = collection_name and old.get("collection_name") == collection_name
        cycle = int(old.get("cycle") or 0) if same_collection else 0
        scanned_in_cycle = int(old.get("scanned_in_cycle") or 0) if same_collection else 0
        runs_in_cycle = int(old.get("runs_in_cycle") or 0) if same_collection else 0
        scanned_in_cycle += max(0, int(scanned.get("scanned") or 0))
        runs_in_cycle += 1
        next_offset = scan_offset_or_none(scanned.get("next_offset"))
        entry = {
            "collection_name": collection_name,
            "next_offset": next_offset,
            "cycle": cycle,
            "runs_in_cycle": runs_in_cycle,
            "scanned_in_cycle": scanned_in_cycle,
            "last_start_offset": scan_offset_or_none(scanned.get("start_offset")),
            "last_scanned": max(0, int(scanned.get("scanned") or 0)),
        }
        if scanned.get("reached_end") or next_offset is None:
            entry.update(
                {
                    "next_offset": None,
                    "cycle": cycle + 1,
                    "runs_in_cycle": 0,
                    "scanned_in_cycle": 0,
                    "last_completed_cycle_scanned": scanned_in_cycle,
                    "last_completed_cycle_runs": runs_in_cycle,
                }
            )
        state[label] = entry
    return state


def scan_offsets_from_state(stored: Any, collection_names: dict[str, str]) -> dict[str, Any]:
    """Return the start cursors to pass to the report for the configured collections.

    A cursor recorded for a different collection name is never reused (profile switch or
    config edit); that collection starts a new cycle.
    """
    window = stored if isinstance(stored, dict) and stored.get("schema") == SCAN_WINDOW_SCHEMA else {}
    offsets: dict[str, Any] = {}
    for label in SCAN_WINDOW_COLLECTIONS:
        entry = window.get(label) if isinstance(window.get(label), dict) else {}
        expected = collection_names.get(label)
        if entry and (not expected or entry.get("collection_name") == expected):
            offsets[label] = scan_offset_or_none(entry.get("next_offset"))
        else:
            offsets[label] = None
    return offsets


def artifact_path(artifact: Any) -> str:
    """Return an artifact path as a string for the shared watcher state."""
    if isinstance(artifact, str):
        return artifact
    if isinstance(artifact, dict):
        value = artifact.get("path")
        return str(value) if value is not None else ""
    return ""


def watcher_state_fields(*, proposals: Any, counts: Any, artifact: Any) -> dict[str, Any]:
    """Build the canonical state fields written by every consolidation watcher.

    ``last_proposal_signature`` remains the full alert-comparison digest used by
    the CLI. ``last_signature`` remains its legacy 16-character form for the
    external watcher. A clean report intentionally returns ``{}``, ``0``, and
    the current artifact path so callers overwrite stale state values.
    """
    proposal_list = proposals if isinstance(proposals, list) else []
    signature = proposal_signature(proposal_list)
    return {
        "watcher_state_schema": WATCHER_STATE_SCHEMA,
        "proposal_signature_schema": PROPOSAL_SIGNATURE_SCHEMA,
        "last_proposal_signature": signature,
        "last_signature": signature[:16],
        "last_artifact": artifact_path(artifact),
        "last_counts": dict(counts) if isinstance(counts, dict) else {},
        "last_total_proposals": len(proposal_list),
    }
