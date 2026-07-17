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
