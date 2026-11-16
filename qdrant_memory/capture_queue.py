"""Persisted capture queue: append-only events under ``$HERMES_HOME/qdrant_memory/capture/``.

``queue.jsonl`` (mode 0600) is the only state the capture pipeline keeps between processes.
Every line is one event, appended under an exclusive ``flock`` and fsynced; nothing is ever
rewritten. The state of a candidate is the fold of its events, so any process (the nightly
run, the CLI, a gateway provider, a fresh provider after a restart) sees the same queue.

Events (``EVENTS``):

- ``queued``: a candidate entered the queue with its route (``review``, ``auto`` or
  ``mirror``). It carries the fact text, the pinned citation and the write it would make
  (``write``): this file and the stored point are the only places fact text lives.
- ``approved`` / ``rejected``: the owner's decision on a review item (actor, approving session).
- ``applied``: the write happened; point id, rule id and digests.
- ``expired``: a review item reached ``capture_review_expiry_days`` without a decision; no
  write.
- ``merged``: a duplicate candidate's citation was added to a queued item or a stored point
  instead of becoming a second fact.

Every event other than ``queued`` holds only the keys in ``AUDIT_KEYS``: ids, digests, rule
ids and run ids, never message text. Run reports (``runs/<run_id>.json``) follow the same rule.

A candidate id is ``sha256(session_id, msg_id, kind, normalized_fact)`` (``capture_id``); the
Qdrant point an extraction-path candidate becomes is the UUID of its first 128 bits
(``point_id_for``). The short id the digest shows is its first 12 hex characters.
"""
from __future__ import annotations

import calendar
import fcntl
import hashlib
import json
import logging
import os
import re
import tempfile
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from qdrant_memory.capture_reader import capture_dir
from qdrant_memory.schema import build_payload

logger = logging.getLogger(__name__)

QUEUE_SCHEMA = "qdrant-capture-queue-v1"
RUN_SCHEMA = "qdrant-capture-run-v1"
EVENTS = ("queued", "approved", "rejected", "expired", "merged", "applied")
ROUTES = ("review", "auto", "mirror")
SHORT_ID_LEN = 12
DAY = 86400.0
# Keys an audit event (anything but ``queued``) may carry: ids, digests, rule and run ids.
AUDIT_KEYS = frozenset({
    "schema", "event", "at", "candidate_id", "point_id", "run_id", "route", "rule", "actor",
    "approver_session_id", "approver_platform", "citation", "citation_sha256", "fact_sha256", "into", "outcome",
    "reason", "lane", "kind",
})
# Heuristic kinds -> (extraction candidate type, memory kind) of the extraction write gate.
KIND_TYPES = {
    "preference": ("preference_candidate", "user_preference"),
    "correction": ("preference_candidate", "user_preference"),
    "invariant": ("invariant_candidate", "project_invariant"),
    "decision": ("memory_candidate", "decision"),
    "fact": ("memory_candidate", "manual_fact"),
    "procedure": ("memory_candidate", "workflow_lesson"),
}
# The confidence a heuristic hit alone carries on its extraction candidate: above the gate's
# low-confidence draft threshold (a person approves it), below the auto rule's 0.9.
HEURISTIC_CONFIDENCE = 0.7
_US = "\x1f"


class CaptureQueueError(ValueError):
    """An id that names no queue item (or more than one), or an unusable queue file."""


def utc_iso(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() if ts is None else ts))


def parse_iso(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return float(calendar.timegm(time.strptime(value, "%Y-%m-%dT%H:%M:%SZ")))
    except ValueError:
        return None


def normalize_fact(text: str) -> str:
    """Case-, width- and whitespace-insensitive form of a fact (the id and dedupe key)."""
    folded = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return re.sub(r"\s+", " ", folded).strip().strip(" .;:,!¡?¿\"'`")


def capture_id(session_id: str, msg_id: int, kind: str, fact: str) -> str:
    material = _US.join(("qdrant-capture-v1", str(session_id), str(int(msg_id)), str(kind), normalize_fact(fact)))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def point_id_for(candidate_id: str) -> str:
    return str(uuid.UUID(candidate_id[:32]))


def short_id(candidate_id: str) -> str:
    return str(candidate_id)[:SHORT_ID_LEN]


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def citation_digest(citation: Mapping[str, Any] | None) -> str:
    """One digest for a pinned span: the span digest it already carries."""
    return str((citation or {}).get("span_sha256") or "")


def citation_ids(citation: Mapping[str, Any]) -> dict[str, Any]:
    """The citation without anything that could hold text (no quote): ids and digests only."""
    return {key: value for key, value in dict(citation).items() if key not in ("quote", "quote_sha256")}


# --- writes an item would make (the inputs of an existing gated write path) ------------------


def heuristic_write(*, session_id: str, msg_id: int, kind: str, fact: str, rule_ids: Iterable[str],
                    source: str = "") -> dict[str, Any]:
    """Inputs of the extraction approval path (producer ``source_extraction``) for one row."""
    candidate_type, memory_kind = KIND_TYPES[kind]
    cid = capture_id(session_id, msg_id, kind, fact)
    source_uri = f"hermes-state://{session_id}/{int(msg_id)}"
    locator = {"session_id": session_id, "msg_id": int(msg_id)}
    edge = {"source_uri": source_uri, "source_type": "hermes_state_message", "derivation_type": "source_extraction",
            "relation_type": "EXTRACTED_FROM", "locator": locator}
    payload = build_payload(
        text=fact, source="source_extraction", source_type="source_extraction", chunk_type=memory_kind,
        confidence=HEURISTIC_CONFIDENCE, tags=["capture", f"capture:{kind}"], memory_kind=memory_kind,
        source_uri=source_uri, locator=locator, derived_from=[edge], derivation_type="source_extraction",
        canonical=False, requires_review=False,
    )
    payload.update({"capture_lane": "heuristic", "capture_kind": kind, "capture_rule_ids": sorted(set(rule_ids)),
                    "capture_candidate_id": cid})
    if source:
        payload["capture_source"] = source
    return {"path": "extraction_approval", "candidate": {
        "candidate_type": candidate_type, "source_uri": source_uri, "locator": locator, "derived_from": [edge],
        "proposed_payload": payload, "reason": "capture heuristic: " + ",".join(sorted(set(rule_ids))),
        "confidence": HEURISTIC_CONFIDENCE, "risk": "low"}}


def manual_write(*, text: str, tags: Iterable[str], importance: int = 5,
                 extra_payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Inputs of the manual store path (producer ``manual_store``, basis ``agent_proposed``)."""
    return {"path": "manual_store", "text": text, "tags": list(tags), "importance": int(importance),
            "extra_payload": dict(extra_payload or {})}


def queued_event(*, run_id: str, lane: str, kind: str, route: str, session_id: str, msg_id: int, fact: str,
                 citation: Mapping[str, Any], rule_ids: Iterable[str], write: Mapping[str, Any],
                 target: str = "memory", op: str = "", sensitive: Iterable[str] = (), expiry_days: float = 21,
                 now: float | None = None, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if route not in ROUTES:
        raise ValueError(f"route must be one of {ROUTES}")
    clock = time.time() if now is None else float(now)
    cid = capture_id(session_id, msg_id, kind, fact)
    event: dict[str, Any] = {
        "schema": QUEUE_SCHEMA, "event": "queued", "at": utc_iso(clock), "candidate_id": cid, "run_id": run_id,
        "lane": lane, "kind": kind, "route": route, "session_id": session_id, "msg_id": int(msg_id),
        "target": target, "op": op, "rule_ids": sorted(set(rule_ids)), "sensitive": sorted(set(sensitive)),
        "citation": citation_ids(citation), "fact": fact, "fact_sha256": sha256_text(fact), "write": dict(write),
    }
    if route == "review":
        event["expires_at"] = utc_iso(clock + float(expiry_days) * DAY)
    if extra:
        event.update(dict(extra))
    return event


def audit_event(event_name: str, candidate_id: str, *, now: float | None = None, **fields: Any) -> dict[str, Any]:
    if event_name not in EVENTS or event_name == "queued":
        raise ValueError(f"not an audit event: {event_name}")
    event = {"schema": QUEUE_SCHEMA, "event": event_name, "at": utc_iso(now), "candidate_id": candidate_id,
             **{key: value for key, value in fields.items() if value is not None}}
    unknown = set(event) - AUDIT_KEYS
    if unknown:
        raise ValueError(f"audit event keys must be ids or digests; unknown: {sorted(unknown)}")
    return event


# --- the queue file ------------------------------------------------------------------------


@dataclass
class QueueItem:
    candidate_id: str
    queued: dict[str, Any]
    history: list[dict[str, Any]] = field(default_factory=list)
    status: str = "pending"
    extra_citations: list[dict[str, Any]] = field(default_factory=list)
    point_id: str = ""

    @property
    def short_id(self) -> str:
        return short_id(self.candidate_id)

    @property
    def route(self) -> str:
        return str(self.queued.get("route") or "")

    @property
    def fact(self) -> str:
        return str(self.queued.get("fact") or "")

    @property
    def expires_at(self) -> float | None:
        return parse_iso(self.queued.get("expires_at"))

    def expired(self, now: float | None = None) -> bool:
        expiry = self.expires_at
        return self.status == "pending" and expiry is not None and (time.time() if now is None else now) >= expiry

    def summary(self) -> dict[str, Any]:
        q = self.queued
        return {"id": self.candidate_id, "short_id": self.short_id, "status": self.status, "route": self.route,
                "lane": q.get("lane"), "kind": q.get("kind"), "target": q.get("target"), "op": q.get("op"),
                "fact": self.fact, "session_id": q.get("session_id"), "msg_id": q.get("msg_id"),
                "queued_at": q.get("at"), "expires_at": q.get("expires_at"), "sensitive": q.get("sensitive", []),
                "extra_citations": len(self.extra_citations), "point_id": self.point_id or None}


class CaptureQueue:
    def __init__(self, hermes_home: str) -> None:
        self.dir = capture_dir(hermes_home)
        self.path = self.dir / "queue.jsonl"

    def append(self, *events: Mapping[str, Any]) -> None:
        """Append events as lines under an exclusive lock; one write, fsynced. Mode 0600."""
        if not events:
            return
        data = "".join(json.dumps(dict(e), sort_keys=True, ensure_ascii=False) + "\n" for e in events).encode()
        self.dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.fchmod(fd, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                os.write(fd, data)
                os.fsync(fd)
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except ValueError:
                logger.warning("capture queue: skipping unreadable line %d", number)
                continue
            if isinstance(event, dict) and event.get("schema") == QUEUE_SCHEMA and event.get("event") in EVENTS:
                out.append(event)
        return out

    def items(self, now: float | None = None, events: list[dict[str, Any]] | None = None) -> dict[str, QueueItem]:
        items: dict[str, QueueItem] = {}
        for event in self.events() if events is None else events:
            cid = str(event.get("candidate_id") or "")
            kind = event["event"]
            if kind == "queued":
                if cid not in items:
                    items[cid] = QueueItem(cid, event, [event], "pending" if event.get("route") == "review" else "queued")
                continue
            if kind == "merged":
                into = str(event.get("into") or "")
                if into in items and isinstance(event.get("citation"), dict):
                    items[into].extra_citations.append(event["citation"])
                    items[into].history.append(event)
                if cid in items:
                    items[cid].status = "merged"
                    items[cid].history.append(event)
                continue
            item = items.get(cid)
            if item is None:
                continue
            item.history.append(event)
            item.status = kind
            if kind == "applied":
                item.point_id = str(event.get("point_id") or "")
        clock = time.time() if now is None else now
        for item in items.values():
            if item.expired(clock):
                item.status = "expired"
        return items

    def find(self, ident: str, now: float | None = None) -> QueueItem:
        ident = str(ident or "").strip().lower()
        items = self.items(now)
        if re.fullmatch(r"[0-9a-f]{64}", ident) and ident in items:
            return items[ident]
        if re.fullmatch(r"[0-9a-f]{%d}" % SHORT_ID_LEN, ident):
            matches = [item for cid, item in items.items() if cid.startswith(ident)]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise CaptureQueueError(f"short id {ident} names {len(matches)} items; use the full id")
        raise CaptureQueueError(f"no capture item with id {ident!r} (use the full id or its first "
                                f"{SHORT_ID_LEN} characters)")

    def pending(self, now: float | None = None) -> list[QueueItem]:
        return sorted((item for item in self.items(now).values() if item.status == "pending"),
                      key=lambda item: (str(item.queued.get("at")), item.candidate_id))

    def expire_due(self, now: float | None = None, run_id: str | None = None) -> list[dict[str, Any]]:
        """Append an ``expired`` event for every review item past its expiry; no write elsewhere."""
        clock = time.time() if now is None else now
        due = [item for item in self.items(clock).values() if item.status == "expired"
               and not any(e["event"] == "expired" for e in item.history)]
        events = [audit_event("expired", item.candidate_id, now=clock, run_id=run_id, rule="review:expiry",
                              lane=item.queued.get("lane"), kind=item.queued.get("kind")) for item in due]
        self.append(*events)
        return events


def write_json_private(path: Path, data: Mapping[str, Any]) -> None:
    """Atomic write (temp file in the same directory, fsync, rename), mode 0600."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            json.dump(dict(data), handle, sort_keys=True, indent=1)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def run_report_path(hermes_home: str, run_id: str) -> Path:
    return capture_dir(hermes_home) / "runs" / f"{run_id}.json"


def new_run_id(now: float | None = None) -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(time.time() if now is None else now)) + "-" + os.urandom(3).hex()
