"""``hermes qdrant capture run``: the nightly capture batch over ``state.db``.

One run reads the rows after the capture cursor (capture_reader, read-only), and for each
lane builds candidates that cite their exact source rows:

- lane A (heuristic): ``human`` rows of the owner (``capture_owner_senders``), the heuristic
  detector (capture_detect), a one-row citation (``resolve_span``, basis
  ``extractor_source_line``; anything but a resolved span drops the candidate), the
  extraction write gate, then dedupe and routing;
- lane C (mirror of the built-in ``memory`` tool, ``capture_mirror_builtin_memory``): see
  capture_mirror.

Dedupe, in order: the exact candidate id already in the queue (any state); the same
normalized fact pending in the queue (its citation is merged into that item); the exact point
already stored; a dense near match among stored captures (``capture_lane`` set, not
superseded) at ``capture_dedupe_threshold`` or more (the citation is merged into that point).

Routes: ``review`` (the queue), ``auto`` (lane A, only with ``capture_auto_apply`` and the
design's auto rule, capped per day at ``capture_daily_auto_cap``), ``mirror`` (lane C,
``target=memory``), ``merge_queued`` / ``merge_point``, and ``drop`` (reason in the report).

A dry run (the default) reads Qdrant (search) and writes only its run report. ``--apply``
needs ``capture_mode: on``: it expires overdue review items, appends queue events, performs the
auto and mirror writes through the existing gated paths (capture_apply) and advances the
cursor to the run's snapshot bound.
"""
from __future__ import annotations

import fcntl
import os
import time
from collections import Counter
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Mapping

from qdrant_memory.capture_apply import write_item
from qdrant_memory.capture_detect import detect, sensitive_flags
from qdrant_memory.capture_queue import (
    CaptureQueue,
    QueueItem,
    audit_event,
    capture_id,
    citation_digest,
    heuristic_write,
    new_run_id,
    normalize_fact,
    parse_iso,
    point_id_for,
    queued_event,
    run_report_path,
    utc_iso,
    write_json_private,
    RUN_SCHEMA,
)
from qdrant_memory.capture_reader import (
    CLASS_HUMAN,
    CaptureReadError,
    capture_dir,
    classify_row,
    cursor_path,
    load_cursor,
    read_rows,
    save_cursor,
    snapshot_max_id,
)
from qdrant_memory.extraction_candidates import ExtractionCandidate
from qdrant_memory.lesson_extractor import contains_secret
from qdrant_memory.session_origin import QUOTE_MAX_CHARS, SessionOriginRefused, build_session_origin, resolve_span
from qdrant_memory.source_extraction import evaluate_source_extraction_candidate

# A heuristic hit is the person's own message: one line, at most a quote's length, at least
# three words (a bare "dale" or "ok" says nothing on its own).
MAX_FACT_CHARS = QUOTE_MAX_CHARS
MIN_FACT_WORDS = 3
# Chat types of a one-to-one conversation; anything else with no sender tag is unattributed.
_DIRECT_CHAT_TYPES = frozenset({"", "dm", "private", "direct"})
AUTO_KINDS = frozenset({"preference", "invariant", "correction", "decision"})
AUTO_MIN_SECOND_SIGNAL = 0.9
AUTO_RULE = "auto:hybrid"

SecondOpinion = Callable[[Any, str], "Mapping[str, Any] | None"]


class CaptureRunRefused(RuntimeError):
    """The run may not start (mode, lock, cursor); the message says why."""


@contextmanager
def _run_lock(hermes_home: str) -> Iterator[None]:
    directory = capture_dir(hermes_home)
    directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(directory / "run.lock", os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CaptureRunRefused("another capture run holds the lock") from exc
        yield
    finally:
        os.close(fd)


def owner_row(sender: str, chat_type: str | None, owner_senders: list[str]) -> bool:
    """A row the owner wrote: a listed sender tag, or no tag in a one-to-one conversation."""
    if sender:
        return sender.strip().casefold() in {name.strip().casefold() for name in owner_senders}
    return str(chat_type or "").strip().lower() in _DIRECT_CHAT_TYPES


def one_line(text: str) -> str:
    return " ".join(str(text or "").split())


def _extraction_gate(write: Mapping[str, Any], cid: str) -> Any:
    spec = write["candidate"]
    candidate = ExtractionCandidate(
        candidate_id=point_id_for(cid), candidate_type=spec["candidate_type"], source_uri=spec["source_uri"],
        locator=dict(spec["locator"]), derived_from=list(spec["derived_from"]),
        proposed_payload=dict(spec["proposed_payload"]), reason=spec["reason"], confidence=float(spec["confidence"]),
        risk=spec.get("risk", "low"))
    return evaluate_source_extraction_candidate(candidate)


def auto_verdict(candidate: Mapping[str, Any], opinion: Mapping[str, Any] | None) -> str:
    """``""`` when the design's auto rule holds (cap aside), else the first reason it does not."""
    if opinion is None:
        return "no_second_signal"
    try:
        confidence = float(opinion.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < AUTO_MIN_SECOND_SIGNAL:
        return "second_signal_below_0.9"
    if candidate["kind"] not in AUTO_KINDS:
        return "kind_not_auto"
    if str(opinion.get("scope") or "") == "task":
        return "task_scope"
    if opinion.get("reversal"):
        return "reversal_in_lookahead"
    if candidate.get("sensitive"):
        return "sensitive:" + ",".join(candidate["sensitive"])
    if candidate.get("gate") != "store":
        return f"gate_{candidate.get('gate')}"
    return ""


class _Run:
    def __init__(self, provider: Any, *, apply: bool, since_days: float | None, upto_id: int | None,
                 now: float | None, second_opinion: SecondOpinion | None) -> None:
        self.provider = provider
        self.config = provider._config
        self.apply = apply
        self.since_days = since_days
        self.upto_id = upto_id
        self.now = time.time() if now is None else float(now)
        self.second_opinion = second_opinion
        self.home = str(provider._hermes_home)
        self.state_db = provider._state_db_path()
        self.run_id = new_run_id(self.now)
        self.queue = CaptureQueue(self.home)
        self.candidates: list[dict[str, Any]] = []
        self.drops: Counter = Counter()
        self.rows: Counter = Counter()
        self.errors: list[str] = []
        self.mirror_stats: dict[str, Any] | None = None

    # --- reading ----------------------------------------------------------------------

    def bounds(self) -> tuple[int, int, float | None]:
        cursor = load_cursor(cursor_path(self.home), self.state_db)
        before = int(cursor["last_msg_id"])
        upto = int(self.upto_id) if self.upto_id is not None else snapshot_max_id(self.state_db)
        if upto < before:
            raise CaptureRunRefused(f"capture cursor ({before}) is ahead of the run bound ({upto})")
        since_ts = None if self.since_days is None else self.now - float(self.since_days) * 86400
        return before, upto, since_ts

    def drop(self, reason: str, **fields: Any) -> None:
        self.drops[reason] += 1
        if fields:
            self.candidates.append({"route": "drop", "reason": reason, **fields})

    # --- lane A -----------------------------------------------------------------------

    def lane_a(self, before: int, upto: int, since_ts: float | None, known: dict[str, QueueItem],
               pending_facts: dict[str, tuple[str, Any]]) -> None:
        owners = [str(name) for name in self.config.get("capture_owner_senders") or []]
        expiry = float(self.config.get("capture_review_expiry_days", 21))
        for ctx in read_rows(self.state_db, before, upto, since_ts=since_ts):
            self.rows["read"] += 1
            if ctx.role != "user":
                continue
            self.rows["user"] += 1
            verdict = classify_row(ctx)
            if verdict.classification != CLASS_HUMAN:
                continue
            self.rows["human"] += 1
            if not owner_row(verdict.sender, ctx.session_chat_type, owners):
                self.drops["not_owner"] += 1
                continue
            self.rows["owner"] += 1
            detection = detect(verdict.body)
            if detection is None:
                continue
            self.rows["detected"] += 1
            fact = one_line(verdict.body)
            base = {"lane": "heuristic", "kind": detection.kind, "rule_ids": list(detection.rule_ids),
                    "session_id": ctx.session_id, "msg_id": ctx.msg_id, "source": ctx.source or ""}
            if contains_secret(fact):
                self.drop("possible_secret")  # never in the queue, never in a report
                continue
            if len(fact) > MAX_FACT_CHARS:
                self.drop("too_long", **base)
                continue
            if len(fact.split()) < MIN_FACT_WORDS:
                self.drop("too_short", **base)
                continue
            citation = resolve_span(self.state_db, ctx.session_id, ctx.msg_id, ctx.msg_id,
                                    basis="extractor_source_line")
            if not isinstance(citation, dict):
                self.drop(f"citation_{citation}", **base)
                continue
            cid = capture_id(ctx.session_id, ctx.msg_id, detection.kind, fact)
            write = heuristic_write(session_id=ctx.session_id, msg_id=ctx.msg_id, kind=detection.kind, fact=fact,
                                    rule_ids=detection.rule_ids, source=ctx.source or "")
            gate = _extraction_gate(write, cid)
            candidate = {**base, "candidate_id": cid, "point_id": point_id_for(cid), "fact": fact,
                         "citation": citation, "write": write, "gate": gate.decision,
                         "sensitive": list(sensitive_flags(fact)), "target": "memory", "op": "",
                         "expiry_days": expiry}
            if gate.decision in ("reject", "skip"):
                self.drop(f"gate_{gate.decision}:{','.join(gate.reasons)}", **{**base, "candidate_id": cid})
                continue
            self.route_lane_a(candidate, ctx, known, pending_facts)

    def route_lane_a(self, candidate: dict[str, Any], ctx: Any, known: dict[str, QueueItem],
                     pending_facts: dict[str, tuple[str, Any]]) -> None:
        cid = candidate["candidate_id"]
        if cid in known or any(c.get("candidate_id") == cid and c["route"] != "drop" for c in self.candidates):
            self.drop("duplicate_id", **{k: candidate[k] for k in ("lane", "kind", "session_id", "msg_id",
                                                                  "candidate_id")})
            return
        key = normalize_fact(candidate["fact"])
        if key in pending_facts:
            into, content = pending_facts[key]
            if content == candidate["citation"].get("uids_sha256"):
                # The host re-copies a compacted tail under new ids keeping each message_uid:
                # the same message, not a repetition. Nothing to merge.
                self.drop("compaction_clone", **{k: candidate[k] for k in ("lane", "kind", "session_id", "msg_id",
                                                                          "candidate_id")})
                return
            self.candidates.append({**candidate, "route": "merge_queued", "into": into})
            return
        if self.stored(candidate["point_id"]):
            self.drop("already_stored", **{k: candidate[k] for k in ("lane", "kind", "session_id", "msg_id",
                                                                    "candidate_id")})
            return
        near = self.near_capture(candidate["fact"])
        if near:
            cited = {c.get("uids_sha256") for c in ((near[2].get("session_origin") or {}).get("citations") or [])}
            if candidate["citation"].get("uids_sha256") in cited:
                self.drop("compaction_clone", **{k: candidate[k] for k in ("lane", "kind", "session_id", "msg_id",
                                                                          "candidate_id")})
                return
            self.candidates.append({**candidate, "route": "merge_point", "into": near[0], "score": near[1]})
            return
        candidate["route"] = "review"
        if self.config.get("capture_auto_apply"):
            opinion = self.second_opinion(ctx, candidate["kind"]) if self.second_opinion else None
            candidate["auto_verdict"] = auto_verdict(candidate, opinion) or AUTO_RULE
            if candidate["auto_verdict"] == AUTO_RULE:
                candidate["route"] = "auto"
        pending_facts.setdefault(key, (cid, candidate["citation"].get("uids_sha256")))
        self.candidates.append(candidate)

    # --- Qdrant reads -----------------------------------------------------------------

    def stored(self, point_id: str) -> bool:
        try:
            return bool(self.provider._qdrant.retrieve(self.config["collection_name"], [point_id], with_payload=False))
        except Exception as exc:
            self.errors.append(f"retrieve failed: {type(exc).__name__}")
            return False

    def near_capture(self, fact: str) -> tuple[str, float, dict[str, Any]] | None:
        """The closest stored capture at or above the dedupe threshold (dense, captures only)."""
        threshold = float(self.config.get("capture_dedupe_threshold", 0.92))
        flt = {"must": [{"key": "capture_lane", "match": {"any": ["heuristic", "mirror"]}}],
               "must_not": [{"key": "fact_status", "match": {"value": "superseded"}}]}
        try:
            embed = getattr(self.provider._embeddings, "embed_query", None) or self.provider._embeddings.embed_document
            hits = self.provider._qdrant.search(self.config["collection_name"], embed(fact), limit=3, filter=flt,
                                                with_payload=True)
        except Exception as exc:
            self.errors.append(f"dense dedupe unavailable: {type(exc).__name__}")
            return None
        for hit in hits or []:
            score = float(hit.get("score", 0.0))
            if score >= threshold:
                return str(hit.get("id")), score, dict(hit.get("payload") or {})
        return None

    # --- apply ------------------------------------------------------------------------

    def auto_applied_today(self) -> int:
        today = time.strftime("%Y-%m-%d", time.localtime(self.now))
        count = 0
        for event in self.queue.events():
            if event["event"] == "applied" and event.get("rule") == AUTO_RULE:
                ts = parse_iso(event.get("at"))
                if ts is not None and time.strftime("%Y-%m-%d", time.localtime(ts)) == today:
                    count += 1
        return count

    def queued(self, candidate: dict[str, Any], route: str) -> dict[str, Any]:
        return queued_event(run_id=self.run_id, lane=candidate["lane"], kind=candidate["kind"], route=route,
                            session_id=candidate["session_id"], msg_id=candidate["msg_id"], fact=candidate["fact"],
                            citation=candidate["citation"], rule_ids=candidate["rule_ids"],
                            write=candidate["write"], target=candidate.get("target", "memory"),
                            op=candidate.get("op", ""), sensitive=candidate.get("sensitive", ()),
                            expiry_days=candidate.get("expiry_days", 21), now=self.now,
                            extra=candidate.get("queue_extra"))

    def apply_write(self, candidate: dict[str, Any], route: str, rule: str) -> None:
        """Write an auto or mirror candidate now; on refusal it falls back to the review queue."""
        event = self.queued(candidate, route)
        item = QueueItem(candidate["candidate_id"], event, [event], "queued")
        result = write_item(self.provider, item, dry_run=True, approval=False)
        if "error" not in result:
            result = write_item(self.provider, item, dry_run=False, approve=True, approval=False)
        if "error" in result or not result.get("saved"):
            candidate["outcome"] = "fell_back_to_review"
            candidate["route"] = "review"
            self.errors.append(f"{candidate['candidate_id'][:12]}: {str(result.get('error') or 'not saved')[:160]}")
            self.queue.append(self.queued(candidate, "review"))
            return
        candidate["outcome"] = "applied"
        candidate["point_id"] = str(result.get("point_id") or candidate.get("point_id") or "")
        self.queue.append(event, audit_event(
            "applied", candidate["candidate_id"], now=self.now, point_id=candidate["point_id"], route=route,
            rule=rule, run_id=self.run_id, citation_sha256=citation_digest(candidate["citation"]),
            fact_sha256=event["fact_sha256"], lane=candidate["lane"], kind=candidate["kind"]))

    def merge_into_point(self, candidate: dict[str, Any]) -> None:
        """Add the duplicate's citation to a stored capture through the extraction locked upsert."""
        provider, collection = self.provider, self.config["collection_name"]
        try:
            got = provider._qdrant.retrieve(collection, [candidate["into"]], with_payload=True, with_vector=True)
        except Exception as exc:
            got = []
            self.errors.append(f"merge retrieve failed: {type(exc).__name__}")
        if not got or not got[0].get("vector"):
            candidate["outcome"] = "merge_target_gone"
            self.queue.append(self.queued(candidate, "review"))
            candidate["route"] = "review"
            return
        payload = {k: v for k, v in (got[0].get("payload") or {}).items() if k != "session_origin"}
        origin = build_session_origin(producer="source_extraction", captured_session_id=candidate["session_id"],
                                      captured_at=utc_iso(self.now), citation=candidate["citation"])
        try:
            error = provider._upsert_extraction_with_origin(candidate["into"], got[0]["vector"], payload, origin)
        except SessionOriginRefused as exc:
            error = str(exc)
        if error:
            candidate["outcome"] = "merge_refused"
            self.errors.append(f"{candidate['candidate_id'][:12]}: merge refused: {error[:160]}")
            return
        candidate["outcome"] = "merged"
        self.queue.append(audit_event("merged", candidate["candidate_id"], now=self.now, into=candidate["into"],
                                      run_id=self.run_id, citation={**candidate["citation"]}, lane=candidate["lane"],
                                      kind=candidate["kind"], rule="dedupe:dense"))

    def apply_all(self) -> None:
        cap = int(self.config.get("capture_daily_auto_cap", 5))
        used = self.auto_applied_today()
        for candidate in self.candidates:
            route = candidate["route"]
            if route == "review":
                self.queue.append(self.queued(candidate, "review"))
                candidate["outcome"] = "queued"
            elif route == "auto":
                if used >= cap:
                    candidate["route"], candidate["auto_verdict"] = "review", "daily_cap"
                    self.queue.append(self.queued(candidate, "review"))
                    candidate["outcome"] = "queued"
                    continue
                self.apply_write(candidate, "auto", AUTO_RULE)
                used += candidate.get("outcome") == "applied"
            elif route == "mirror":
                self.apply_write(candidate, "mirror", candidate.get("rule", "mirror:target_memory"))
            elif route == "merge_queued":
                self.queue.append(audit_event("merged", candidate["candidate_id"], now=self.now,
                                              into=candidate["into"], run_id=self.run_id,
                                              citation={**candidate["citation"]}, lane=candidate["lane"],
                                              kind=candidate["kind"], rule="dedupe:normalized_fact"))
                candidate["outcome"] = "merged"
            elif route == "merge_point":
                self.merge_into_point(candidate)

    def preview_caps(self) -> None:
        """Dry run: show the cap the apply would hit (same order as apply_all)."""
        cap = int(self.config.get("capture_daily_auto_cap", 5))
        used = self.auto_applied_today()
        for candidate in self.candidates:
            if candidate["route"] == "auto":
                if used >= cap:
                    candidate["route"], candidate["auto_verdict"] = "review", "daily_cap"
                else:
                    used += 1

    # --- report -----------------------------------------------------------------------

    def report(self, before: int, upto: int, cursor_after: int, expired: list[str], started: float) -> dict[str, Any]:
        kept = [c for c in self.candidates if c["route"] != "drop"]
        safe_keys = ("candidate_id", "lane", "kind", "route", "rule_ids", "session_id", "msg_id", "point_id", "into",
                     "sensitive", "outcome", "reason", "auto_verdict", "target", "op", "gate", "score", "rule")
        rows = []
        for c in self.candidates:
            row = {k: c[k] for k in safe_keys if k in c and c[k] not in (None, "", [])}
            if c.get("citation"):
                row["citation_sha256"] = citation_digest(c["citation"])
                row["citation_span"] = [c["citation"]["first_msg_id"], c["citation"]["last_msg_id"]]
            rows.append(row)
        return {
            "schema": RUN_SCHEMA, "run_id": self.run_id, "started_at": utc_iso(started), "finished_at": utc_iso(),
            "mode": self.config.get("capture_mode"), "apply": self.apply, "since_days": self.since_days,
            "mirror_enabled": bool(self.config.get("capture_mirror_builtin_memory")),
            "auto_apply": bool(self.config.get("capture_auto_apply")),
            "daily_auto_cap": int(self.config.get("capture_daily_auto_cap", 5)),
            "cursor": {"before": before, "upto": upto, "after": cursor_after, "advanced": cursor_after != before},
            "rows": dict(self.rows),
            "counts": {"by_route": dict(Counter(c["route"] for c in kept)),
                       "by_kind": dict(Counter(f"{c['lane']}:{c['kind']}" for c in kept)),
                       "by_lane_route": dict(Counter(f"{c['lane']}:{c['route']}" for c in kept)),
                       "drops": dict(self.drops)},
            "candidates": rows, "expired": expired, "errors": self.errors,
            **({"mirror": {key: dict(value) if isinstance(value, Counter) else value
                           for key, value in self.mirror_stats.items()}} if self.mirror_stats is not None else {}),
        }


def run_capture(provider: Any, *, apply: bool = False, since_days: float | None = None, upto_id: int | None = None,
                now: float | None = None, second_opinion: SecondOpinion | None = None) -> dict[str, Any]:
    """Run one capture batch. Returns ``{"report": <ids only>, "candidates": <with text>}``.

    ``upto_id`` pins the snapshot bound (to replay a receipt); ``second_opinion`` is the second
    signal the auto rule needs (none ships yet, so without it nothing auto-applies).
    """
    mode = str(provider._config.get("capture_mode") or "off")
    if mode == "off":
        raise CaptureRunRefused("capture is off (qdrant_memory.capture_mode: off)")
    if apply and mode != "on":
        raise CaptureRunRefused(f"--apply needs qdrant_memory.capture_mode: on (it is {mode})")
    run = _Run(provider, apply=apply, since_days=since_days, upto_id=upto_id, now=now, second_opinion=second_opinion)
    started = time.time()
    lock = _run_lock(run.home) if apply else _no_lock()
    with lock:
        try:
            before, upto, since_ts = run.bounds()
        except CaptureReadError as exc:
            raise CaptureRunRefused(str(exc)) from exc
        items = run.queue.items(run.now)
        expired = [item.candidate_id for item in items.values() if item.status == "expired"
                   and not any(e["event"] == "expired" for e in item.history)]
        if apply:
            run.queue.expire_due(run.now, run.run_id)
            items = run.queue.items(run.now)
        pending_facts = {normalize_fact(item.fact): (cid, (item.queued.get("citation") or {}).get("uids_sha256"))
                         for cid, item in items.items()
                         if item.status == "pending" and item.queued.get("lane") == "heuristic"}
        run.lane_a(before, upto, since_ts, items, pending_facts)
        if provider._config.get("capture_mirror_builtin_memory"):
            from qdrant_memory.capture_mirror import mirror_lane

            mirror_lane(run, before, upto, since_ts, items)
        after = before
        if apply:
            run.apply_all()
            after = save_cursor(cursor_path(run.home), run.state_db, upto)["last_msg_id"]
        else:
            run.preview_caps()
        report = run.report(before, upto, after, expired, started)
        write_json_private(run_report_path(run.home, run.run_id), report)
    return {"report": report, "candidates": [c for c in run.candidates if c["route"] != "drop"]}


@contextmanager
def _no_lock() -> Iterator[None]:
    yield
