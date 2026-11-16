"""The write step of a capture item, through one of the provider's existing gated write paths.

Capture adds no write path of its own (design I3 as reworded by the review):

- heuristic items go through ``QdrantMemoryProvider.approve_extraction_candidate`` (producer
  ``source_extraction``, basis ``extractor_source_line``): the extraction write gate on the
  candidate and on the persisted payload, the lineage overwrite refusal, then the locked
  upsert that merges ``session_origin``;
- built-in memory mirror items go through the manual store's write step
  (``QdrantMemoryProvider._manual_store_write``, producer ``manual_store``, basis
  ``agent_proposed``) after the manual write gate (``evaluate_write_candidate``).

Either way the origin is the CAPTURED session and the citation pinned at capture time,
re-verified against ``state.db`` here: a row that changed (``drifted``) or is gone
(``expired``, ``partially_missing``) refuses the write. The approving session only appears as
``approved_session_id`` on review approvals.
"""
from __future__ import annotations

import json
import time
from typing import Any

from qdrant_memory.capture_queue import QueueItem, point_id_for
from qdrant_memory.extraction_candidates import ExtractionCandidate
from qdrant_memory.session_origin import (
    MAX_CITATIONS,
    SessionOriginRefused,
    build_session_origin,
    validate_session_origin,
    verify_citation,
)
from qdrant_memory.write_gate import evaluate_write_candidate

MIRROR_SOURCE = "builtin_memory_mirror"


def _now_z() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def check_citation(provider: Any, item: QueueItem) -> dict[str, Any]:
    """``verify_citation`` of the item's pinned citation against the configured ``state.db``."""
    return verify_citation(item.queued.get("citation") or {}, state_db=provider._state_db_path())


def capture_origin(provider: Any, item: QueueItem, *, producer: str, approval: bool) -> dict[str, Any]:
    """The ``session_origin`` an item records: captured session, re-verified pinned citation(s).

    Raises ``SessionOriginRefused`` when the primary citation no longer verifies. Extra
    citations merged from duplicates are kept only while they still verify.
    """
    queued = item.queued
    state_db = provider._state_db_path()
    check = verify_citation(queued.get("citation") or {}, state_db=state_db)
    if check["status"] != "verified":
        raise SessionOriginRefused(
            f"the cited row changed or is gone since capture: {check['status']} ({check.get('reason', '')}); "
            "nothing was written")
    now = _now_z()
    captured = str(queued.get("session_id") or "")
    origin = build_session_origin(producer=producer, captured_session_id=captured,
                                  captured_at=str(queued.get("at") or now),
                                  citation={**queued["citation"], "verified_at": now})
    seen = {queued["citation"].get("uids_sha256")}
    for extra in item.extra_citations:
        # One citation per distinct message (a compaction clone keeps its message_uid),
        # never more than the mapping's bound; extras that no longer verify are left out.
        if len(origin["citations"]) >= MAX_CITATIONS or extra.get("uids_sha256") in seen:
            continue
        if verify_citation(extra, state_db=state_db)["status"] == "verified":
            origin["citations"].append({**extra, "verified_at": now})
            seen.add(extra.get("uids_sha256"))
    if approval and (provider._session_id or "") != captured:
        origin["approved_session_id"] = provider._session_id or ""
        origin["approved_at"] = now
    validate_session_origin(origin)
    return origin


def extraction_candidate(item: QueueItem, *, approval: bool) -> ExtractionCandidate:
    queued = item.queued
    spec = queued["write"]["candidate"]
    return ExtractionCandidate(
        candidate_id=point_id_for(item.candidate_id), candidate_type=spec["candidate_type"],
        source_uri=spec["source_uri"], locator=dict(spec["locator"]), derived_from=list(spec["derived_from"]),
        proposed_payload=dict(spec["proposed_payload"]), reason=spec["reason"], confidence=float(spec["confidence"]),
        risk=spec.get("risk", "low"), requires_review=True, created_at=str(queued.get("at") or ""),
        origin_capture={"session_id": queued.get("session_id") or "", "captured_at": queued.get("at") or "",
                        "citation": dict(queued.get("citation") or {}),
                        "extra_citations": [dict(c) for c in item.extra_citations], "approval": approval})


def _manual(provider: Any, item: QueueItem, *, dry_run: bool, approval: bool) -> dict[str, Any]:
    from qdrant_memory.writer import ManagedOverwriteRefused

    spec = item.queued["write"]
    text = str(spec.get("text") or "")
    importance = int(spec.get("importance") or 5)
    tags = [str(tag) for tag in spec.get("tags") or []]
    if provider._writer is None:
        return {"error": "Qdrant memory provider is not initialized"}
    preview = provider._writer.preview_text(text, source_type="manual", importance=importance, tags=tags,
                                            source=MIRROR_SOURCE)
    decision = evaluate_write_candidate(text=text, target="memory", source_type="manual", confidence=1.0,
                                        metadata={"importance": importance})
    try:
        origin = capture_origin(provider, item, producer="manual_store", approval=approval)
    except SessionOriginRefused as exc:
        return {"error": str(exc)}
    base = {"path": "manual_store", "point_id": preview.get("id"), "write_decision": decision.to_dict(),
            "would_store": decision.decision == "store",
            "session_origin": {"producer": origin["producer"], "captured_session_id": origin["captured_session_id"],
                               "citation_count": len(origin["citations"]),
                               "approved_session_id": origin.get("approved_session_id")}}
    if dry_run:
        return {"dry_run": True, "saved": False, **base}
    if decision.decision != "store":
        return {"error": f"write gate decided {decision.decision} ({', '.join(decision.reasons)}); nothing written"}
    try:
        point_id = provider._manual_store_write(text, source_type="manual", importance=importance, tags=tags,
                                                session_origin=origin, source=MIRROR_SOURCE,
                                                extra_payload=spec.get("extra_payload") or {})
    except (ManagedOverwriteRefused, SessionOriginRefused) as exc:
        return {"error": str(exc)}
    except Exception as exc:  # lock unavailable, Qdrant down: nothing recorded as applied
        return {"error": f"manual store failed: {exc}"}
    return {"dry_run": False, "saved": bool(point_id), **base, "point_id": point_id}


def _extraction(provider: Any, item: QueueItem, *, dry_run: bool, approve: bool, approval: bool) -> dict[str, Any]:
    candidate = extraction_candidate(item, approval=approval)
    result = json.loads(provider.approve_extraction_candidate(candidate, dry_run=dry_run, approve=approve))
    if "error" in result:
        return {"error": result["error"]}
    out = {"path": "extraction_approval", "dry_run": bool(result.get("dry_run")), "saved": bool(result.get("saved")),
           "point_id": candidate.candidate_id, "write_decision": result.get("write_decision"),
           "would_store": result.get("would_store"), "proposal_created": bool(result.get("proposal_created"))}
    if result.get("session_origin"):
        origin = result["session_origin"]
        out["session_origin"] = {key: origin.get(key) for key in ("producer", "captured_session_id",
                                                                  "approved_session_id", "citation_status")
                                 if key in origin}
    if result.get("proposal_id"):
        out["proposal_id"] = result["proposal_id"]
    return out


def write_item(provider: Any, item: QueueItem, *, dry_run: bool, approve: bool = False,
               approval: bool) -> dict[str, Any]:
    """Preview (``dry_run``) or perform the write of one item; ``{"error": ...}`` when refused."""
    path = (item.queued.get("write") or {}).get("path")
    if path == "extraction_approval":
        if dry_run:
            check = check_citation(provider, item)
            if check["status"] != "verified":
                return {"error": f"the cited row changed or is gone since capture: {check['status']} "
                                 f"({check.get('reason', '')}); nothing was written"}
        return _extraction(provider, item, dry_run=dry_run, approve=approve, approval=approval)
    if path == "manual_store":
        return _manual(provider, item, dry_run=dry_run, approval=approval)
    return {"error": f"capture item {item.short_id} has no known write path"}
