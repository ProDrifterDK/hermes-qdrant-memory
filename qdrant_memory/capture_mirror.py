"""Mirror of the agent's built-in ``memory`` tool writes into Qdrant (capture lane C, design CP5).

The built-in memory (MEMORY.md / USER.md) is a capped working set that the agent prunes to
make room. The mirror turns every write the agent made with the built-in ``memory`` tool into
a cited capture, so an entry that is later replaced or removed stays recallable.

Parsing (``parse_tool_calls``) accepts every argument shape the host's ``memory`` tool has
accepted: an ``operations`` batch (with or without a top-level ``target``; each op may carry
its own ``target``), and the single-op shapes ``action`` + ``content`` / ``old_text`` /
``new_text`` (``new_text`` stands in for an empty ``content``, as in the host). A call whose
arguments do not parse still counts as one op, so the op count always equals the number of
operations the agent sent.

Only ops of calls the tool accepted are mirrored (``success: true`` in the tool result; a
persisted oversized result is read from its preview); a batch is atomic, so a failed batch
mirrors nothing. The full entry texts come from the result when the host returned them
(``replaced_entry(ies)``, ``removed_entry(ies)``, ``patched_entr(y|ies)_before/after``);
otherwise from an entry the mirror already knows that contains ``old_text`` exactly once;
otherwise the op is dropped as ``entry_text_unknown`` (a patch is never stored as its bare span).

Routes: ``target=memory`` add, replace and patch results are stored automatically through the
manual store's write step (producer ``manual_store``, basis ``agent_proposed``): the agent asked
for them. The old text of a replace, patch or remove is stored as a superseded capture
(``fact_status: superseded``), never deleted. ``target=user`` (identity) ops go to the review
queue. Every citation spans the last human row before the call to the last eligible row at or
before the calling row (the calling row itself has no text when it only carries tool calls).
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from typing import Any

from qdrant_memory.capture_apply import MIRROR_SOURCE
from qdrant_memory.capture_queue import capture_id, manual_write
from qdrant_memory.capture_reader import CLASS_HOST, CLASS_HUMAN, CaptureReadError, _context, _select_sql, classify_row
from qdrant_memory.lesson_extractor import contains_secret
from qdrant_memory.session_origin import MAX_SPAN_MESSAGES, _Unavailable, _read_retrying, resolve_span
from qdrant_memory.write_gate import evaluate_write_candidate

MEMORY_TOOL = "memory"
MIRROR_IMPORTANCE = 5
_SUCCESS_RE = re.compile(r'"success"\s*:\s*(true|false)')
_LOOK_BACK_ROWS = 400


@dataclass(frozen=True)
class MemoryOp:
    index: int  # 1-based position in the call (the host's "Operation N")
    action: str
    target: str
    content: str
    old_text: str


@dataclass(frozen=True)
class MemoryCall:
    call_id: str
    shape: str
    batch: bool
    ops: tuple[MemoryOp, ...]
    # sha256 of the call id and its raw arguments: the same call copied into a continuation
    # session (compaction copies the calling row under a new message_uid) has the same digest.
    digest: str = ""


def _op(index: int, raw: Any, default_target: str) -> MemoryOp:
    if not isinstance(raw, dict):
        return MemoryOp(index, "", default_target, "", "")
    content = raw.get("content") or raw.get("new_text") or ""
    return MemoryOp(index, str(raw.get("action") or ""), str(raw.get("target") or default_target),
                    str(content), str(raw.get("old_text") or ""))


def parse_tool_calls(raw: Any) -> list[MemoryCall]:
    """Every ``memory`` call in one assistant row's ``tool_calls`` JSON, in order."""
    try:
        calls = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    except ValueError:
        return []
    out: list[MemoryCall] = []
    for call in calls if isinstance(calls, list) else []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        if (function.get("name") or call.get("name")) != MEMORY_TOOL:
            continue
        call_id = str(call.get("id") or call.get("call_id") or "")
        arguments = function.get("arguments", call.get("arguments"))
        raw_args = arguments if isinstance(arguments, str) else json.dumps(arguments, sort_keys=True, default=str)
        digest = hashlib.sha256(f"{call_id}\x1f{raw_args}".encode("utf-8", "surrogatepass")).hexdigest()
        try:
            args = arguments if isinstance(arguments, dict) else json.loads(arguments or "{}")
        except (TypeError, ValueError):
            args = None
        if not isinstance(args, dict):
            out.append(MemoryCall(call_id, "<unparseable>", False, (MemoryOp(1, "", "memory", "", ""),), digest))
            continue
        shape = "+".join(sorted(str(key) for key in args)) or "<empty>"
        target = str(args.get("target") or "memory")
        operations = args.get("operations")
        if isinstance(operations, list):
            out.append(MemoryCall(call_id, shape, True, tuple(_op(i, op, target) for i, op in
                                                              enumerate(operations, start=1)), digest))
        else:
            out.append(MemoryCall(call_id, shape, False, (_op(1, args, target),), digest))
    return out


def call_outcome(text: Any) -> tuple[bool | None, dict[str, Any]]:
    """``(accepted, result)`` from a tool result row: True/False, or None when unknown."""
    if text is None:
        return None, {}
    body = text.decode("utf-8", "replace") if isinstance(text, bytes) else str(text)
    body = body.strip()
    if not body.startswith("<persisted-output>"):
        try:
            result, _ = json.JSONDecoder().raw_decode(body)
        except ValueError:
            result = None
        if isinstance(result, dict) and isinstance(result.get("success"), bool):
            return result["success"], result
    match = _SUCCESS_RE.search(body)
    return (match.group(1) == "true" if match else None), {}


def _result_entry(result: dict[str, Any], single: str, batch: str, op: MemoryOp, is_batch: bool) -> str:
    value = (result.get(batch) or {}).get(str(op.index)) if is_batch else result.get(single)
    if not is_batch and value is None and isinstance(result.get(batch), dict):
        value = result[batch].get(str(op.index))
    return value.strip() if isinstance(value, str) and value.strip() else ""


class KnownEntries:
    """The entries the mirror has seen written, per target (fallback for results without text)."""

    def __init__(self) -> None:
        self.entries: dict[str, list[str]] = {}

    def lookup(self, target: str, old_text: str) -> str:
        needle = old_text.strip()
        if not needle:
            return ""
        hits = [entry for entry in self.entries.get(target, []) if needle in entry]
        return hits[0] if len(hits) == 1 else ""

    def add(self, target: str, entry: str) -> None:
        bucket = self.entries.setdefault(target, [])
        if entry not in bucket:
            bucket.append(entry)

    def remove(self, target: str, entry: str) -> None:
        bucket = self.entries.setdefault(target, [])
        if entry in bucket:
            bucket.remove(entry)


def resolve_op(op: MemoryOp, result: dict[str, Any], is_batch: bool, known: KnownEntries) -> tuple[str, str, str]:
    """``(new_entry, old_entry, problem)`` for one accepted op; updates ``known``."""
    action = op.action
    if action == "add":
        new = op.content.strip()
        if not new:
            return "", "", "empty_content"
        known.add(op.target, new)
        return new, "", ""
    if action == "replace":
        new = op.content.strip()
        if not new:
            return "", "", "empty_content"
        old = _result_entry(result, "replaced_entry", "replaced_entries", op, is_batch) or known.lookup(op.target,
                                                                                                        op.old_text)
        if old:
            known.remove(op.target, old)
        known.add(op.target, new)
        return new, old, "" if old else "old_entry_unknown"
    if action == "patch":
        before = (_result_entry(result, "patched_entry_before", "patched_entries", op, is_batch)
                  or known.lookup(op.target, op.old_text))
        after = _result_entry(result, "patched_entry_after", "patched_entries_after", op, is_batch)
        if not after and before and op.old_text and before.count(op.old_text) == 1:
            after = before.replace(op.old_text, op.content, 1).strip()
        if not after:
            return "", "", "entry_text_unknown"
        if before:
            known.remove(op.target, before)
        known.add(op.target, after)
        return after, before, ""
    if action == "remove":
        old = _result_entry(result, "removed_entry", "removed_entries", op, is_batch) or known.lookup(op.target,
                                                                                                      op.old_text)
        if not old:
            return "", "", "entry_text_unknown"
        known.remove(op.target, old)
        return "", old, ""
    return "", "", "no_action" if not action else f"unknown_action:{action[:20]}"


# --- reading state.db -------------------------------------------------------------------


def read_memory_call_rows(state_db: str, after_id: int, upto_id: int, since_ts: float | None,
                          page_size: int = 500) -> list[tuple[int, str, float | None, Any, Any]]:
    """Assistant rows in ``(after_id, upto_id]`` whose ``tool_calls`` mention the memory tool.

    ``(id, session_id, timestamp, tool_calls, message_uid)``; JSON parsing decides which ones
    really call it. Read-only through the C' reader.
    """
    out: list[tuple[int, str, float | None, Any, Any]] = []
    cursor = int(after_id)
    while True:
        def work(conn: sqlite3.Connection) -> list[tuple]:
            columns = {row[1].decode() if isinstance(row[1], bytes) else row[1]
                       for row in conn.execute("PRAGMA table_info(messages)")}
            if "tool_calls" not in columns:
                return []
            sql = ("SELECT m.id, m.session_id, m.timestamp, m.tool_calls, m.message_uid FROM messages m "
                   "WHERE m.role = 'assistant' AND m.id > ? AND m.id <= ? AND m.tool_calls IS NOT NULL "
                   "AND instr(m.tool_calls, '\"memory\"') > 0" + (" AND m.timestamp >= ?" if since_ts is not None else "")
                   + " ORDER BY m.id LIMIT ?")
            params: list[Any] = [cursor, int(upto_id)] + ([float(since_ts)] if since_ts is not None else [])
            return conn.execute(sql, [*params, page_size]).fetchall()

        try:
            page = _read_retrying(state_db, work)
        except _Unavailable as exc:
            raise CaptureReadError(f"state.db unreadable: {exc.reason}") from exc
        for raw in page:
            out.append((int(raw[0]), _text(raw[1]), float(raw[2]) if raw[2] is not None else None, raw[3], raw[4]))
        if len(page) < page_size:
            return out
        cursor = int(page[-1][0])


def _text(value: Any) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else ("" if value is None else str(value))


def call_context(state_db: str, session_id: str, msg_id: int, call_ids: list[str],
                 message_uid: Any) -> dict[str, Any]:
    """Tool results of the calls, clone check, and the citation span of the calling turn."""
    def work(conn: sqlite3.Connection) -> dict[str, Any]:
        results = {}
        for call_id in call_ids:
            row = conn.execute("SELECT content FROM messages WHERE session_id = ? AND role = 'tool' AND tool_call_id = ? "
                               "ORDER BY id LIMIT 1", (session_id, call_id)).fetchone() if call_id else None
            results[call_id] = row[0] if row else None
        clone = bool(message_uid) and conn.execute(
            "SELECT 1 FROM messages WHERE message_uid = ? AND id < ? LIMIT 1", (message_uid, msg_id)).fetchone() is not None
        rows = [_context(raw) for raw in conn.execute(
            _select_sql(conn, "m.session_id = ? AND m.id <= ?", newest_first=True), [session_id, msg_id, _LOOK_BACK_ROWS])]
        return {"results": results, "clone": clone, "rows": rows}

    try:
        found = _read_retrying(state_db, work)
    except _Unavailable as exc:
        raise CaptureReadError(f"state.db unreadable: {exc.reason}") from exc
    rows = found.pop("rows")
    found["span"] = None
    if rows:
        end = rows[0].msg_id
        start, prompt_class = None, ""
        for row in rows:
            if row.role != "user":
                continue
            verdict = classify_row(row)
            if verdict.classification == CLASS_HUMAN:
                start, prompt_class = row.msg_id, "human"
                break
            if verdict.classification != CLASS_HOST and start is None:
                start, prompt_class = row.msg_id, "machine"  # keep looking for a human row
        if start is None:
            # No prompt row in this session (a compaction continuation opens with the host's
            # summary and the person's message stayed in the parent session): cite the calling
            # turn inside its own session, from as far back as one span may reach.
            start, prompt_class = rows[min(len(rows), MAX_SPAN_MESSAGES) - 1].msg_id, "none"
        found["span"] = (start, end, prompt_class)
    return found


# --- the lane ---------------------------------------------------------------------------


def _known_from_queue(items: dict[str, Any]) -> KnownEntries:
    known = KnownEntries()
    for item in sorted(items.values(), key=lambda i: (str(i.queued.get("at")), int(i.queued.get("msg_id") or 0))):
        queued = item.queued
        if queued.get("lane") != "mirror" or item.status in ("rejected", "expired"):
            continue
        target = str(queued.get("target") or "memory")
        if queued.get("kind") == "mirror_superseded":
            known.remove(target, item.fact)
        else:
            known.add(target, item.fact)
    return known


def mirror_lane(run: Any, before: int, upto: int, since_ts: float | None, items: dict[str, Any]) -> None:
    """Add lane C candidates to ``run`` (a ``capture_run._Run``)."""
    stats = {"calls": 0, "ops": 0, "shapes": Counter(), "actions": Counter(), "outcomes": Counter(),
             "prompt_class": Counter()}
    run.mirror_stats = stats
    known = _known_from_queue(items)
    expiry = float(run.config.get("capture_review_expiry_days", 21))
    seen_ids = set(items)
    # call digest -> the row that first carried it; a later row with the same call is a copy.
    seen_calls = {str(item.queued["call_digest"]): int(item.queued.get("msg_id") or 0) for item in items.values()
                  if item.queued.get("call_digest")}
    for msg_id, session_id, _ts, tool_calls, message_uid in read_memory_call_rows(run.state_db, before, upto, since_ts):
        calls = parse_tool_calls(tool_calls)
        if not calls:
            continue
        context = call_context(run.state_db, session_id, msg_id, [c.call_id for c in calls], message_uid)
        for call in calls:
            stats["calls"] += 1
            stats["shapes"][call.shape] += 1
            stats["ops"] += len(call.ops)
            accepted, result = call_outcome(context["results"].get(call.call_id))
            stats["outcomes"][{True: "accepted", False: "refused", None: "unknown"}[accepted]] += len(call.ops)
            copied = seen_calls.setdefault(call.digest, msg_id) != msg_id
            for op in call.ops:
                stats["actions"][f"{op.target}:{op.action or '-'}"] += 1
                if context["clone"]:
                    run.drop("compaction_clone")
                    continue
                if copied:
                    run.drop("copied_call")
                    continue
                if accepted is not True:
                    run.drop("tool_failed" if accepted is False else "tool_result_missing")
                    continue
                new, old, problem = resolve_op(op, result, call.batch, known)
                if problem and not new and not old:
                    run.drop(problem)
                    continue
                if problem == "old_entry_unknown":
                    run.drop("old_entry_unknown")
                span = context["span"]
                if span is None:
                    run.drop("no_prompt_row")
                    continue
                stats["prompt_class"][span[2]] += 1
                for text, kind in ((old, "mirror_superseded"), (new, "mirror_add" if op.action == "add"
                                                                 else "mirror_replace")):
                    if not text or (kind == "mirror_superseded" and text == new):
                        continue
                    _candidate(run, session_id, msg_id, op, call, text, kind, span, expiry, seen_ids)


def _candidate(run: Any, session_id: str, msg_id: int, op: MemoryOp, call: MemoryCall, text: str, kind: str,
               span: tuple[int, int, str], expiry: float, seen_ids: set[str]) -> None:
    if contains_secret(text):
        run.drop("possible_secret")  # never in the queue, never in a report
        return
    base = {"lane": "mirror", "kind": kind, "session_id": session_id, "msg_id": msg_id, "target": op.target,
            "op": op.action}
    cid = capture_id(session_id, msg_id, kind, text)
    if cid in seen_ids:
        run.drop("duplicate_id", **base, candidate_id=cid)
        return
    seen_ids.add(cid)
    decision = evaluate_write_candidate(text=text, target="memory", source_type="manual", confidence=1.0,
                                        metadata={"importance": MIRROR_IMPORTANCE})
    if decision.decision in ("reject", "skip"):
        run.drop(f"gate_{decision.decision}:{','.join(decision.reasons)}", **base, candidate_id=cid)
        return
    citation = resolve_span(run.state_db, session_id, span[0], span[1], basis="agent_proposed")
    if not isinstance(citation, dict):
        run.drop(f"citation_{citation}", **base, candidate_id=cid)
        return
    identity = op.target == "user"
    rule = "mirror:target_user" if identity else ("mirror:superseded" if kind == "mirror_superseded"
                                                  else "mirror:target_memory")
    extra = {"capture_lane": "mirror", "capture_kind": kind, "capture_candidate_id": cid,
             "capture_target": op.target, "capture_op": op.action}
    if kind == "mirror_superseded":
        extra["fact_status"] = "superseded"
    write = manual_write(text=text, tags=["capture", "builtin_memory", f"target:{op.target}"],
                         importance=MIRROR_IMPORTANCE, extra_payload=extra)
    point_id = run.provider._writer.preview_text(text, source_type="manual", importance=MIRROR_IMPORTANCE,
                                                 tags=write["tags"], source=MIRROR_SOURCE)["id"]
    run.candidates.append({
        **base, "candidate_id": cid, "point_id": point_id, "fact": text, "citation": citation, "write": write,
        "rule_ids": [rule], "rule": rule, "gate": decision.decision, "route": "review" if identity else "mirror",
        "sensitive": ["identity"] if identity else [], "expiry_days": expiry,
        "queue_extra": {"call_id": call.call_id, "call_digest": call.digest, "op_index": op.index,
                        "prompt_class": span[2]},
    })
