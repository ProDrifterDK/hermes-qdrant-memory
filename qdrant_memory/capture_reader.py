"""Read-only capture reader over ``state.db``: new rows since a cursor, each with its authorship.

Capture (docs/capture.md) may only treat a row as "the user said this" when a person wrote
it. In ``state.db`` most ``user`` rows are not that: the host stores its own scaffolding as
user rows (background-process notices, compaction handoffs, task lists, notifications), and
agents write prompts that land as the first user row of a session (delegation briefs,
scheduled and scripted runs). ``classify_row`` separates the three deterministically, from
structure first and text shapes second, and returns the body with gateway envelopes stripped.
No learned model, no threshold tuned on anyone's data.

Classes (``Classification.classification``):

- ``host``: text the host wrote (``kind`` names the shape, see ``HOST_KINDS``).
- ``machine``: text a program or an agent wrote: every ``assistant`` row, and ``user`` rows of
  non-interactive sessions or agent briefs (``kind`` names the rule).
- ``human``: a ``user`` row a person typed (``kind`` is ``""``). ``sender`` carries the gateway
  sender tag when there is one (shared channels have several people).

Features and decision order, all verifiable on the row and its session:

1. Role. ``assistant`` rows are ``machine`` (``assistant``) unless they are host text (step 3)
   or carry ``display_kind`` ``failed_turn`` / ``hidden`` (host, ``system_notice``).
2. ``messages.display_kind`` set by the host: ``process_complete`` -> host
   ``background_process``; ``internal_notification`` -> host ``notification``;
   ``async_delegation_complete`` -> host ``delegation_report``; ``model_switch`` -> host
   ``system_notice``. ``steer`` marks a message the user typed mid-turn: it is unwrapped
   (step 3) and judged like any other row.
3. Envelopes are stripped (``strip_envelopes``): the out-of-band steer wrapper, a leading
   ``[Note: model was just switched ...]`` line, ``[Triggering message id: ...]``, the
   ``Gateway message origin (JSON data ...)`` block, a ``[Recent channel messages]`` transcript
   up to ``[New message]``, a ``[Replying to: "..."]`` quote with its
   ``[Context around the replied-to message]`` block,
   host media descriptions (``[The user sent ...~]`` and the ``vision_analyze`` hint), the
   ``--- Attached Context ---`` / ``--- Context Warnings ---`` tail and the LCM
   ``[Note: This conversation uses Lossless Context Management ...]`` tail. What remains is
   host text when ``host_markers.is_host_injected`` says so, or when it matches one of
   ``EXTRA_HOST_SHAPES`` (shapes this module adds without editing ``host_markers``). A row
   with nothing left after the envelopes is host ``envelope_only``.
4. Session source. ``user`` rows of a session whose ``sessions.source`` is not one a person
   types into (``HUMAN_SOURCES``: cron, subagent, oneshot, tool, api_server, a test canary,
   or no session row at all) are ``machine`` ``non_interactive_source``.
5. Delegation brief. A ``user`` row is ``machine`` ``delegation_brief`` when ALL hold:
   the session source is a messaging gateway (``GATEWAY_SOURCES``); the session has no
   ``parent_session_id``; the session carries no gateway identity (``user_id``, ``chat_type``
   and ``origin_json`` all NULL: no platform delivered a message into it); the session has no
   title; and the row is the session's first eligible row (no earlier user or assistant row
   with content). That is how older hosts stored a child agent's task: a gateway-labelled
   session that no gateway message ever reached, opened by the parent's brief. A person's
   message in such a session that answers an assistant row is never the first row, and the
   titled identity-less sessions are context-management continuations (titled ``<topic> #<n>``) whose
   first row is the person's or the host's.
6. Otherwise ``human``.

Features evaluated and left out: the language of the brief (briefs are written in English
and in Spanish, and so are people's messages), length (one-line briefs and long pasted
human messages both exist), and ``display_metadata.gateway_input_owner`` (present on
gateway-delivered input only in recent rows, so its absence proves nothing).

The reader (``read_rows`` / ``scan``) opens ``state.db`` only through the C' read-only reader
(``session_origin._read_retrying``: ``mode=ro``, or ``immutable=1`` when no Hermes process
has it open, one retry on a transient lock) with ``session_recall._ELIGIBLE`` as the row
filter, and pages by message id inside a snapshot bound (the largest id at the start of the
scan), so no row is skipped or read twice. The cursor (the last message id handed out) is
persisted atomically under ``$HERMES_HOME/qdrant_memory/capture/``.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

from qdrant_memory.host_markers import is_host_injected
from qdrant_memory.session_origin import _Unavailable, _read_retrying
from qdrant_memory.session_recall import _ELIGIBLE

CLASS_HOST, CLASS_MACHINE, CLASS_HUMAN = "host", "machine", "human"
CLASSES = (CLASS_HOST, CLASS_MACHINE, CLASS_HUMAN)

# Sources a person types into. Anything else is a program's session.
HUMAN_SOURCES = frozenset({
    "cli", "tui", "desktop", "gui", "local", "acp",
    "telegram", "discord", "slack", "whatsapp", "signal", "matrix", "mattermost", "email", "sms",
})
# Messaging gateways: a real message into one of these sessions gives it a gateway identity.
GATEWAY_SOURCES = frozenset({"telegram", "discord", "slack", "whatsapp", "signal", "matrix", "mattermost",
                             "email", "sms"})

_USER_HOST_DISPLAY_KINDS = {
    "process_complete": "background_process",
    "internal_notification": "notification",
    "async_delegation_complete": "delegation_report",
    "model_switch": "system_notice",
}
_ASSISTANT_HOST_DISPLAY_KINDS = frozenset({"failed_turn", "hidden"})

# Shapes ``host_markers`` does not know (yet). Reconcile at merge; see docs/capture.md.
EXTRA_HOST_SHAPES: tuple[tuple[str, str], ...] = (
    ("agent_loop", "You've reached the maximum number of tool-calling iterations allowed."),
    ("skill_or_cron_prompt", "You are running as a scheduled cron job"),
    ("skill_or_cron_prompt", 'has invoked the "'),
)

# Host kinds, matched in order on the text left after the envelopes (substring tests).
HOST_KINDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("compaction_summary", ("[CONTEXT COMPACTION", "[PRIOR CONTEXT", "[END OF PRIOR CONTEXT",
                            "--- END OF CONTEXT SUMMARY", "[Current user objective preserved",
                            "[STILL IN PROGRESS")),
    # A steer frame that is not the row's own wrapper (merged behind other host text).
    ("steer_frame", ("[OUT-OF-BAND USER MESSAGE", "[/OUT-OF-BAND USER MESSAGE")),
    ("task_list", ("[Your active task list", "[Planning state preserved")),
    ("background_process", ("[IMPORTANT: Background process", "[Background process ", "\u2705 Background task",
                            "background processes completed")),
    ("delegation_report", ("[ASYNC DELEGATION", "[ASYNC COMPLETE")),
    ("notification", ("[INTERNAL NOTIFICATION", "[Cron delivery:", "Cronjob Response:", "[The user reacted ",
                      "[Continuing toward your standing goal]", "[Recent channel messages]")),
    ("memory_fence", ("<memory-context>", "[System note: The following is recalled memory context")),
    ("media_note", ("[The user sent ",)),
    ("recovered_context", ("[HDFP ", "[Internal HDFP")),
    ("system_notice", ("[System:", "[System note:", "[SYSTEM]", "[Runtime note:", "[Note: ",
                       "[IMPORTANT:")),
)
_LCM_SUMMARY_RE = re.compile(r"\[(?:Recent|Session Arc|Durable|Depth-\d+) Summary \(d\d+, node ")

# --- envelopes ---------------------------------------------------------------------------

_STEER_OPEN = "[OUT-OF-BAND USER MESSAGE"
_STEER_CLOSE = "[/OUT-OF-BAND USER MESSAGE]"
_MODEL_SWITCH_NOTE_RE = re.compile(r"\A\[Note: model was just switched[^\n]*\]\s*")
_TRIGGER_RE = re.compile(r"\A\[Triggering message id: [^\n]*\]\s*")
_REPLY_OPEN = '[Replying to: "'
_REPLY_CONTEXT = "[Context around the replied-to message]"
_REPLY_NEW = "[New message]"
# gateway/run_inbound.py: the delivery metadata block in front of a gateway message.
_ORIGIN_RE = re.compile(r"\AGateway message origin \(JSON data[^\n]*\n\{[^\n]*\}\n"
                        r"(?:Do not guess a reply destination[^\n]*\n?)?\s*")
_CHANNEL_OPEN = "[Recent channel messages]"
_MEDIA_RE = re.compile(r"\A\[The user sent [^\n]*?~[\s\S]*?~\]\s*")
_VISION_HINT_RE = re.compile(r"\A\[If you need a closer look, use vision_analyze[^\n]*\]\s*")
_TAIL_MARKERS = ("\n--- Attached Context ---", "\n--- Context Warnings ---",
                 "[Note: This conversation uses Lossless Context Management")
# ``[Name] `` gateway sender tag (gateway/session.py). A colon in the brackets is a host label.
_SENDER_RE = re.compile(r"\A\[([^\[\]\n:]{1,80})\] (?=\S)")


@dataclass(frozen=True)
class Classification:
    classification: str
    kind: str
    body: str
    sender: str = ""
    envelopes: tuple[str, ...] = ()
    features: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RowContext:
    """One eligible ``messages`` row with the session fields the classifier reads."""

    msg_id: int
    session_id: str
    role: str
    content: str
    timestamp: float | None = None
    source: str | None = None
    parent_session_id: str | None = None
    session_user_id: str | None = None
    session_chat_type: str | None = None
    session_has_origin: bool = False
    session_titled: bool = False
    display_kind: str | None = None
    first_in_session: bool = False
    session_known: bool = True


def _strip_reply_quote(text: str) -> str | None:
    """The text after a leading ``[Replying to: "..."]`` quote (and its context block)."""
    if not text.startswith(_REPLY_OPEN):
        return None
    end = text.find('"]\n')
    if end < 0:
        end = text.find('"]', len(_REPLY_OPEN))
        if end < 0 or text[end + 2:].strip():
            return None
    rest = text[end + 2:].lstrip()
    if rest.startswith(_REPLY_CONTEXT):
        marker = rest.find(_REPLY_NEW)
        rest = rest[marker + len(_REPLY_NEW):].lstrip() if marker >= 0 else ""
    return rest


def strip_envelopes(content: str) -> tuple[str, tuple[str, ...]]:
    """``content`` without the gateway/host envelopes around what a person typed.

    Returns ``(body, envelopes)``; ``envelopes`` names what was removed, in order. The sender
    tag is not removed here (``classify_row`` judges host labels that share its shape first).
    """
    text = str(content or "").strip()
    found: list[str] = []
    if text.startswith(_STEER_OPEN):
        newline = text.find("\n")
        inner = text[newline + 1:] if newline >= 0 else ""
        close = inner.rfind(_STEER_CLOSE)
        text = (inner[:close] if close >= 0 else inner).strip()
        found.append("steer")
    for _ in range(8):
        before = text
        for name, pattern in (("model_switch_note", _MODEL_SWITCH_NOTE_RE), ("trigger_id", _TRIGGER_RE),
                              ("gateway_origin", _ORIGIN_RE), ("media_note", _MEDIA_RE),
                              ("vision_hint", _VISION_HINT_RE)):
            stripped = pattern.sub("", text, count=1)
            if stripped != text:
                text = stripped.lstrip()
                found.append(name)
        rest = _strip_reply_quote(text)
        if rest is not None:
            text = rest
            found.append("reply_quote")
        if text.startswith(_CHANNEL_OPEN) and _REPLY_NEW in text:
            text = text[text.rfind(_REPLY_NEW) + len(_REPLY_NEW):].lstrip()
            found.append("channel_transcript")
        if text == before:
            break
    for marker in _TAIL_MARKERS:
        cut = text.find(marker)
        if cut >= 0:
            text = text[:cut].rstrip()
            found.append("attached_context" if "Context" in marker and "---" in marker else "lcm_note")
    return text.strip(), tuple(found)


def host_kind(text: str) -> str:
    """The host shape of ``text``, or ``""`` when it is not host text."""
    body = str(text or "")
    for kind, needle in EXTRA_HOST_SHAPES:
        if needle in body:
            return kind
    if not is_host_injected(body):
        return ""
    if _LCM_SUMMARY_RE.search(body):
        return "compaction_summary"
    for kind, needles in HOST_KINDS:
        if any(needle in body for needle in needles):
            return kind
    return "other_host"


def _row(row: RowContext | Mapping[str, Any]) -> RowContext:
    if isinstance(row, RowContext):
        return row
    return RowContext(**{key: row[key] for key in RowContext.__dataclass_fields__ if key in row})


def classify_row(row: RowContext | Mapping[str, Any]) -> Classification:
    """``host`` / ``machine`` / ``human`` for one eligible row, with its envelope-free body."""
    r = _row(row)
    source = (r.source or "").strip().lower()
    features = {
        "role": r.role, "source": source or None, "display_kind": r.display_kind,
        "has_parent": bool(r.parent_session_id), "gateway_identity": bool(
            r.session_user_id or r.session_chat_type or r.session_has_origin),
        "titled": bool(r.session_titled), "first_in_session": bool(r.first_in_session),
    }
    body, envelopes = strip_envelopes(r.content)
    if r.role == "assistant":
        if r.display_kind in _ASSISTANT_HOST_DISPLAY_KINDS:
            return Classification(CLASS_HOST, "system_notice", body, "", envelopes, features)
        kind = host_kind(body)
        if kind:
            return Classification(CLASS_HOST, kind, body, "", envelopes, features)
        return Classification(CLASS_MACHINE, "assistant", body, "", envelopes, features)
    if r.display_kind in _USER_HOST_DISPLAY_KINDS:
        return Classification(CLASS_HOST, _USER_HOST_DISPLAY_KINDS[r.display_kind], body, "", envelopes, features)
    kind = host_kind(body)
    if kind:
        return Classification(CLASS_HOST, kind, body, "", envelopes, features)
    sender = ""
    match = _SENDER_RE.match(body)
    if match:
        sender = match.group(1).strip()
        body = body[match.end():].strip()
        envelopes = (*envelopes, "sender_tag")
        kind = host_kind(body)
        if kind:
            return Classification(CLASS_HOST, kind, body, sender, envelopes, features)
    if not body:
        kind = "media_note" if "media_note" in envelopes else "envelope_only"
        return Classification(CLASS_HOST, kind, body, sender, envelopes, features)
    if not r.session_known or source not in HUMAN_SOURCES:
        return Classification(CLASS_MACHINE, "non_interactive_source", body, sender, envelopes, features)
    if (source in GATEWAY_SOURCES and not features["has_parent"] and not features["gateway_identity"]
            and not r.session_titled and r.first_in_session):
        return Classification(CLASS_MACHINE, "delegation_brief", body, sender, envelopes, features)
    return Classification(CLASS_HUMAN, "", body, sender, envelopes, features)


# --- reading state.db ------------------------------------------------------------------------


class CaptureReadError(RuntimeError):
    """``state.db`` could not be read, or the cursor does not belong to it."""


@dataclass(frozen=True)
class CaptureRecord:
    """One eligible row as capture sees it. ``body`` is message text: never put it in a receipt."""

    session_id: str
    msg_id: int
    role: str
    source: str
    classification: str
    kind: str
    body: str
    sender: str = ""
    timestamp: float | None = None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1].decode() if isinstance(row[1], bytes) else str(row[1])
            for row in conn.execute(f"PRAGMA table_info({table})")}


def _select_sql(conn: sqlite3.Connection, where: str, *, newest_first: bool = False) -> str:
    """The row query, tolerant of host schemas without the optional columns."""
    sessions, messages = _columns(conn, "sessions"), _columns(conn, "messages")

    def s(column: str, expr: str | None = None) -> str:
        return (expr or f"s.{column}") if column in sessions else "NULL"

    display_kind = "m.display_kind" if "display_kind" in messages else "NULL"
    titled = s("title", "COALESCE(trim(s.title), '') <> ''")
    has_origin = s("origin_json", "s.origin_json IS NOT NULL")
    first = ("NOT EXISTS (SELECT 1 FROM messages p WHERE p.session_id = m.session_id AND p.id < m.id AND "
             + _ELIGIBLE.replace("m.", "p.") + ")")
    return (
        f"SELECT m.id, m.session_id, m.role, m.content, m.timestamp, {s('source')}, {s('parent_session_id')}, "
        f"{s('user_id')}, {s('chat_type')}, {has_origin}, {titled}, {display_kind}, {first}, s.id IS NOT NULL "
        f"FROM messages m LEFT JOIN sessions s ON s.id = m.session_id WHERE {_ELIGIBLE} AND {where} "
        f"ORDER BY m.id{' DESC' if newest_first else ''} LIMIT ?"
    )


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return "" if value is None else str(value)


def _context(raw: tuple) -> RowContext:
    return RowContext(
        msg_id=int(raw[0]), session_id=_text(raw[1]), role=_text(raw[2]), content=_text(raw[3]),
        timestamp=float(raw[4]) if raw[4] is not None else None, source=_text(raw[5]) or None,
        parent_session_id=_text(raw[6]) or None, session_user_id=_text(raw[7]) or None,
        session_chat_type=_text(raw[8]) or None, session_has_origin=bool(raw[9]), session_titled=bool(raw[10]),
        display_kind=_text(raw[11]) or None, first_in_session=bool(raw[12]), session_known=bool(raw[13]),
    )


def record_for(context: RowContext) -> CaptureRecord:
    verdict = classify_row(context)
    return CaptureRecord(context.session_id, context.msg_id, context.role, context.source or "",
                         verdict.classification, verdict.kind, verdict.body, verdict.sender, context.timestamp)


def snapshot_max_id(state_db: str) -> int:
    """The largest ``messages.id`` now: the upper bound a scan reads up to."""
    try:
        return int(_read_retrying(state_db, lambda conn: conn.execute("SELECT max(id) FROM messages").fetchone()[0])
                   or 0)
    except _Unavailable as exc:
        raise CaptureReadError(f"state.db unreadable: {exc.reason}") from exc


def read_rows(state_db: str, after_id: int, upto_id: int, *, since_ts: float | None = None,
              page_size: int = 2000) -> Iterator[RowContext]:
    """Eligible rows with ``after_id < id <= upto_id`` (and ``timestamp >= since_ts``), id order.

    Each page is one read transaction through the C' read-only reader; ids never change, so
    paging under a fixed upper bound neither skips nor repeats a row even while the host writes.
    """
    cursor = int(after_id)
    while True:
        where = "? < m.id AND m.id <= ?" + (" AND m.timestamp >= ?" if since_ts is not None else "")
        params: list[Any] = [cursor, int(upto_id)] + ([float(since_ts)] if since_ts is not None else [])

        def work(conn: sqlite3.Connection) -> list[tuple]:
            return conn.execute(_select_sql(conn, where), [*params, int(page_size)]).fetchall()

        try:
            page = _read_retrying(state_db, work)
        except _Unavailable as exc:
            raise CaptureReadError(f"state.db unreadable: {exc.reason}") from exc
        if not page:
            return
        for raw in page:
            yield _context(raw)
        cursor = int(page[-1][0])
        if len(page) < page_size:
            return


def window_user_rows(conn: sqlite3.Connection, session_id: str, first_msg_id: int, last_msg_id: int
                     ) -> list[RowContext]:
    """The eligible ``user`` rows of one session-window span, on an open read-only connection."""
    where = "m.session_id = ? AND m.id BETWEEN ? AND ? AND m.role = 'user'"
    rows = conn.execute(_select_sql(conn, where), [session_id, int(first_msg_id), int(last_msg_id), 10_000]).fetchall()
    return [_context(raw) for raw in rows]


def _prompt_before(conn: sqlite3.Connection, session_id: str, msg_id: int, *, look_back: int = 50) -> list[RowContext]:
    """The last non-host ``user`` row of the session before ``msg_id``: the prompt being answered."""
    sql = _select_sql(conn, "m.session_id = ? AND m.role = 'user' AND m.id < ?", newest_first=True)
    for raw in conn.execute(sql, [session_id, int(msg_id), int(look_back)]).fetchall():
        row = _context(raw)
        if classify_row(row).classification != CLASS_HOST:
            return [row]
    return []


def window_is_machine_only(conn: sqlite3.Connection, session_id: str, first_msg_id: int, last_msg_id: int) -> bool:
    """True when the user side that prompts a session window is machine text only.

    The user side is the window's eligible ``user`` rows that are not host text (the lane cuts
    host text out of windows already, so a notice inside a person's conversation decides
    nothing). When no such row is in the window, it is the last non-host ``user`` row before
    it (what the window's assistant text answers). No such row at all: False.
    """
    rows = [row for row in window_user_rows(conn, session_id, first_msg_id, last_msg_id)
            if classify_row(row).classification != CLASS_HOST]
    rows = rows or _prompt_before(conn, session_id, first_msg_id)
    return bool(rows) and all(classify_row(row).classification == CLASS_MACHINE for row in rows)


# --- cursor ------------------------------------------------------------------------------------

CURSOR_SCHEMA = "qdrant-capture-cursor-v1"


def capture_dir(hermes_home: str) -> Path:
    return Path(hermes_home) / "qdrant_memory" / "capture"


def cursor_path(hermes_home: str) -> Path:
    return capture_dir(hermes_home) / "cursor.json"


def load_cursor(path: Path, state_db: str) -> dict[str, Any]:
    """The persisted cursor for ``state_db``; a fresh one (``last_msg_id`` 0) when none exists."""
    resolved = str(Path(state_db).expanduser().resolve())
    if not path.exists():
        return {"schema": CURSOR_SCHEMA, "state_db": resolved, "last_msg_id": 0, "updated_at": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CaptureReadError(f"capture cursor unreadable: {type(exc).__name__}") from exc
    valid = isinstance(data, dict) and data.get("schema") == CURSOR_SCHEMA and isinstance(data.get("last_msg_id"), int)
    if not valid:
        raise CaptureReadError("capture cursor has an unknown format")
    if data.get("state_db") != resolved:
        raise CaptureReadError("capture cursor belongs to another state.db; refusing to reuse it")
    return data


def save_cursor(path: Path, state_db: str, last_msg_id: int) -> dict[str, Any]:
    """Write the cursor atomically (temp file in the same directory, fsync, rename), mode 0600."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"schema": CURSOR_SCHEMA, "state_db": str(Path(state_db).expanduser().resolve()),
            "last_msg_id": int(last_msg_id), "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    fd, tmp = tempfile.mkstemp(prefix=".cursor-", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            json.dump(data, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return data


# --- scan ----------------------------------------------------------------------------------------


def scan(state_db: str, hermes_home: str, *, since_days: float | None = None, advance: bool = False,
         now: float | None = None) -> dict[str, Any]:
    """Classify every row after the cursor (optionally only the last ``since_days``) and count.

    Read-only on ``state.db``. Advances the cursor to the scan's snapshot bound only with
    ``advance=True``. The result holds ids and counts, never message text.
    """
    path = cursor_path(hermes_home)
    cursor = load_cursor(path, state_db)
    before = int(cursor["last_msg_id"])
    upto = snapshot_max_id(state_db)
    if upto < before:
        raise CaptureReadError(f"capture cursor ({before}) is ahead of state.db ({upto}); refusing to scan")
    clock = float(now if now is not None else time.time())
    since_ts = None if since_days is None else clock - float(since_days) * 86400
    by_class: Counter = Counter()
    by_kind: Counter = Counter()
    by_role: Counter = Counter()
    rows = 0
    first_id = last_id = None
    for context in read_rows(state_db, before, upto, since_ts=since_ts):
        verdict = classify_row(context)
        rows += 1
        first_id = context.msg_id if first_id is None else first_id
        last_id = context.msg_id
        by_class[verdict.classification] += 1
        by_role[f"{context.role}:{verdict.classification}"] += 1
        by_kind[f"{verdict.classification}:{verdict.kind or '-'}"] += 1
    result: dict[str, Any] = {
        "rows": rows, "by_class": {name: by_class.get(name, 0) for name in CLASSES},
        "by_role": dict(sorted(by_role.items())), "by_kind": dict(sorted(by_kind.items())),
        "first_msg_id": first_id, "last_msg_id": last_id, "snapshot_max_id": upto,
        "since_days": since_days,
        "cursor": {"path": str(path), "before": before, "after": before, "advanced": False},
    }
    if advance:
        saved = save_cursor(path, state_db, upto)
        result["cursor"].update({"after": saved["last_msg_id"], "advanced": True, "updated_at": saved["updated_at"]})
    return result
