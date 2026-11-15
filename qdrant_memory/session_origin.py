"""Read-only resolver and verifier for ``state.db`` spans (C'1, ``session-origin-v1``).

A conversation-born fact cites its origin as a digest-pinned span of Hermes' own
``state.db``: ``session_id`` plus the first and last message row ids. ``resolve_span``
pins a span, ``verify_citation`` re-checks one later. Both only read the database and
never raise to the caller.

Span selection is session recall's eligibility rule (``session_recall._ELIGIBLE``: user
and assistant rows with non-blank content) over ``id`` in ``[first_msg_id, last_msg_id]``
of the one session, in ``id`` order. Every digest covers the stored bytes (no timestamp
rendering, no redaction), so it is independent of the host timezone and the redactor.

Digests (sha256, hex; one record per row, concatenated):

- ``span_sha256`` (session-span-v1): ``"hermes-span-v1" 0x1F session_id 0x1F id 0x1F role
  0x1F content 0x1E``. Salted by session and row id.
- ``uids_sha256``: ``"hermes-uids-v1" 0x1F message_uid 0x1E``, the ordered uid list.
- ``content_sha256``: ``"hermes-span-content-v1" 0x1F session_id 0x1F message_uid 0x1F role
  0x1F content 0x1E``, with the CITED session id. It carries no row id, so a span whose rows
  Hermes re-sequenced under fresh ids (compaction clones keep ``message_uid`` and the
  payload byte-exact) can still be checked byte for byte. Without it a uid-resolved span
  could only be matched by uids, and a rewritten clone would pass.

Verification never writes, never demotes anything, and treats an expired session (host
retention) as a normal outcome, not a retraction.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar

from qdrant_memory.lesson_extractor import contains_secret, redact_secret_spans
from qdrant_memory.session_recall import _ELIGIBLE, MAX_CHAIN_DEPTH

STORE = "hermes-state.db"
STATUSES = ("verified", "verified_via_uid", "drifted", "partially_missing", "expired", "unavailable")
UNRESOLVED_REASONS = ("no_state_db", "schema_unsupported", "session_absent", "no_match", "ambiguous_match",
                      "span_too_large")
SPAN_BASES = ("extractor_source_line", "agent_proposed")
MAX_SPAN_MESSAGES = 200
QUOTE_MAX_CHARS = 1000
BUSY_TIMEOUT_SECONDS = 2.0
# Hermes' fence key for a deleted session (hermes_state_mutation_retirement.RETIRED_PREFIX).
RETIRED_PREFIX = "gateway.retired_session.v1."
_REQUIRED_COLUMNS = {
    "messages": ("id", "session_id", "role", "content", "timestamp", "message_uid"),
    "sessions": ("id",),
}
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_US, _RS = b"\x1f", b"\x1e"

T = TypeVar("T")


class _Unavailable(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# --- digests ------------------------------------------------------------------


def _b(value: Any) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    return str(value).encode("utf-8", "surrogatepass")


def span_sha256(session_id: str, rows: list[tuple]) -> str:
    """``rows``: ``(id, role, content, ...)`` in ``id`` order."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(b"hermes-span-v1" + _US + _b(session_id) + _US + str(int(row[0])).encode()
                      + _US + _b(row[1]) + _US + _b(row[2]) + _RS)
    return digest.hexdigest()


def _uid_record(uid: Any) -> bytes:
    return b"hermes-uids-v1" + _US + _b(uid) + _RS


def uids_sha256(uids: list[Any]) -> str:
    return hashlib.sha256(b"".join(_uid_record(uid) for uid in uids)).hexdigest()


def content_sha256(session_id: str, rows: list[tuple]) -> str:
    """``rows``: ``(id, role, content, message_uid, ...)``; ``session_id`` is the cited one."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(b"hermes-span-content-v1" + _US + _b(session_id) + _US + _b(row[3]) + _US + _b(row[1])
                      + _US + _b(row[2]) + _RS)
    return digest.hexdigest()


# --- read-only access ---------------------------------------------------------


def _is_wal(path: Path) -> bool:
    with open(path, "rb") as handle:
        header = handle.read(20)
    return len(header) == 20 and header[:16] == b"SQLite format 3\x00" and header[18] == 2 and header[19] == 2


def _fingerprint(path: Path) -> tuple:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size, Path(f"{path}-wal").exists(), Path(f"{path}-shm").exists()


def _read(state_db: str, work: Callable[[sqlite3.Connection], T]) -> T:
    """Run ``work`` in one read transaction on a strictly read-only connection.

    A ``mode=ro`` open of a WAL database whose ``-wal``/``-shm`` files are absent (no Hermes
    process has it open) CREATES both files next to it. That case is opened ``immutable=1``
    instead, which takes no locks and creates nothing, and the read is discarded if the file
    or its sidecars changed meanwhile (a writer appeared).
    """
    path = Path(str(state_db or "")).expanduser()
    try:
        if not str(state_db or "") or not path.is_file():
            raise _Unavailable("no_state_db")
        path = path.resolve()
        wal_files = (Path(f"{path}-wal").exists(), Path(f"{path}-shm").exists())
        quiescent = False
        if _is_wal(path):
            if wal_files == (False, False):
                quiescent = True
            elif wal_files != (True, True):
                raise _Unavailable("wal_sidecars_incomplete")
        before = _fingerprint(path) if quiescent else None
        query = "mode=ro&immutable=1" if quiescent else "mode=ro"
        conn = sqlite3.connect(f"file:{urllib.parse.quote(str(path))}?{query}", uri=True,
                               timeout=BUSY_TIMEOUT_SECONDS, isolation_level=None, check_same_thread=False)
    except _Unavailable:
        raise
    except (OSError, sqlite3.Error) as exc:
        raise _Unavailable(f"state_db_unreadable: {type(exc).__name__}") from exc
    try:
        conn.text_factory = bytes
        conn.execute("BEGIN")
        for table, columns in _REQUIRED_COLUMNS.items():
            have = {row[1].decode() if isinstance(row[1], bytes) else row[1]
                    for row in conn.execute(f"PRAGMA table_info({table})")}
            if not set(columns) <= have:
                raise _Unavailable("schema_unsupported")
        result = work(conn)
    except _Unavailable:
        raise
    except sqlite3.OperationalError as exc:
        message = str(exc).lower()
        reason = "locked" if "locked" in message or "busy" in message else "state_db_unreadable"
        raise _Unavailable(f"{reason}: {exc}") from exc
    except sqlite3.Error as exc:
        raise _Unavailable(f"state_db_unreadable: {exc}") from exc
    finally:
        conn.close()
    if quiescent and _fingerprint(path) != before:
        raise _Unavailable("state_db_changed_during_read")
    return result


def _session_exists(conn: sqlite3.Connection, session_id: str) -> bool:
    return conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone() is not None


def _span_rows(conn: sqlite3.Connection, session_id: str, first: int, last: int, limit: int | None = None) -> list:
    sql = (f"SELECT m.id, m.role, m.content, m.message_uid, m.timestamp FROM messages m "
           f"WHERE m.session_id = ? AND m.id BETWEEN ? AND ? AND {_ELIGIBLE} ORDER BY m.id")
    params: list[Any] = [session_id, first, last]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return conn.execute(sql, params).fetchall()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_id(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


# --- resolve ------------------------------------------------------------------


def resolve_span(state_db: str, session_id: str, first_msg_id: int, last_msg_id: int, *, basis: str,
                 window_hint: str | None = None) -> dict[str, Any] | str:
    """Pin ``[first_msg_id, last_msg_id]`` of ``session_id`` as a ``session-origin-v1`` citation.

    Returns the citation dict, or ``"unresolved:<reason>"`` with a reason from
    ``UNRESOLVED_REASONS``. Both ids must be eligible rows of the session, so the span has
    exact edges. ``basis`` must be one of ``SPAN_BASES`` (``tool_position`` has no span);
    an unknown basis is a caller programming error and raises ``ValueError``.
    """
    if basis not in SPAN_BASES:
        raise ValueError(f"basis must be one of {SPAN_BASES}, got {basis!r}")
    if not isinstance(session_id, str) or not session_id:
        return "unresolved:session_absent"
    if not (_is_id(first_msg_id) and _is_id(last_msg_id)) or first_msg_id > last_msg_id:
        return "unresolved:no_match"

    def work(conn: sqlite3.Connection) -> dict[str, Any] | str:
        if not _session_exists(conn, session_id):
            return "unresolved:session_absent"
        rows = _span_rows(conn, session_id, first_msg_id, last_msg_id, limit=MAX_SPAN_MESSAGES + 1)
        if not rows or int(rows[0][0]) != first_msg_id:
            return "unresolved:no_match"
        if len(rows) > MAX_SPAN_MESSAGES:
            return "unresolved:span_too_large"
        if int(rows[-1][0]) != last_msg_id:
            return "unresolved:no_match"
        citation: dict[str, Any] = {
            "store": STORE,
            "session_id": session_id,
            "first_msg_id": first_msg_id,
            "last_msg_id": last_msg_id,
            "msg_count": len(rows),
            "uids_sha256": uids_sha256([row[3] for row in rows]),
            "span_sha256": span_sha256(session_id, rows),
            "content_sha256": content_sha256(session_id, rows),
            "basis": basis,
            "verified_at": _utc_now(),
        }
        if window_hint:
            citation["window_hint"] = str(window_hint)
        return citation

    try:
        return _read(state_db, work)
    except _Unavailable as exc:
        return "unresolved:schema_unsupported" if exc.reason == "schema_unsupported" else "unresolved:no_state_db"
    except Exception:  # never raise to the caller
        return "unresolved:no_state_db"


# --- verify -------------------------------------------------------------------


def _malformed(citation: Any) -> str:
    if not isinstance(citation, Mapping):
        return "citation is not a mapping"
    if citation.get("store") != STORE:
        return "store is not hermes-state.db"
    if not isinstance(citation.get("session_id"), str) or not citation.get("session_id"):
        return "session_id missing"
    first, last, count = citation.get("first_msg_id"), citation.get("last_msg_id"), citation.get("msg_count")
    if not (_is_id(first) and _is_id(last) and _is_id(count)) or first > last:
        return "first_msg_id/last_msg_id/msg_count invalid"
    for key in ("span_sha256", "uids_sha256", "content_sha256"):
        if not isinstance(citation.get(key), str) or not _SHA256_HEX.match(citation[key]):
            return f"{key} missing or not a sha256 hex digest"
    return ""


def _family(conn: sqlite3.Connection, session_id: str, has_parent: bool) -> list[str]:
    """The cited session plus its continuations (sessions whose parent chain reaches it)."""
    family = [session_id]
    frontier = [session_id]
    while frontier and has_parent and len(family) < MAX_CHAIN_DEPTH:
        marks = ", ".join("?" for _ in frontier)
        children = [row[0].decode() if isinstance(row[0], bytes) else row[0] for row in conn.execute(
            f"SELECT id FROM sessions WHERE parent_session_id IN ({marks}) ORDER BY id", frontier)]
        frontier = [sid for sid in children if sid not in family][: MAX_CHAIN_DEPTH - len(family)]
        family.extend(frontier)
    return family


def _resolve_by_uid(conn: sqlite3.Connection, citation: Mapping[str, Any]) -> list | None:
    """Find the cited rows by ``message_uid`` when row ids are gone.

    Candidates are the eligible rows of the cited session and its continuations with
    ``id >= first_msg_id`` (clones always get higher ids), one row per uid (the original
    where it survives, else the lowest id), in ``(timestamp, id)`` order (clones keep the
    original timestamp). The cited rows are the run of ``msg_count`` candidates whose
    ordered uids hash to ``uids_sha256``. Any ordering doubt yields no match, never a
    false one.
    """
    sid, first = citation["session_id"], citation["first_msg_id"]
    has_parent = any((row[1].decode() if isinstance(row[1], bytes) else row[1]) == "parent_session_id"
                     for row in conn.execute("PRAGMA table_info(sessions)"))
    family = _family(conn, sid, has_parent)
    marks = ", ".join("?" for _ in family)
    rows = conn.execute(
        f"SELECT m.id, m.role, m.content, m.message_uid, m.timestamp, m.session_id FROM messages m "
        f"WHERE m.session_id IN ({marks}) AND m.id >= ? AND {_ELIGIBLE}", [*family, first]).fetchall()
    chosen: dict[bytes, tuple] = {}
    # A copy always has a higher id than its source, so the lowest id is the original.
    for row in sorted(rows, key=lambda r: int(r[0])):
        chosen.setdefault(_b(row[3]), row)
    sequence = sorted(chosen.values(), key=lambda r: (float(r[4]), int(r[0])))
    width = int(citation["msg_count"])
    records = [_uid_record(row[3]) for row in sequence]
    target = citation["uids_sha256"]
    for start in range(len(sequence) - width + 1):
        if hashlib.sha256(b"".join(records[start:start + width])).hexdigest() == target:
            return sequence[start:start + width]
    return None


def _verify(conn: sqlite3.Connection, citation: Mapping[str, Any]) -> dict[str, Any]:
    sid, first, last = citation["session_id"], citation["first_msg_id"], citation["last_msg_id"]
    width = int(citation["msg_count"])
    if not _session_exists(conn, sid):
        tombstone = None
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'state_meta'").fetchone():
            key = RETIRED_PREFIX + sid
            if conn.execute("SELECT 1 FROM state_meta WHERE key = ?", (key,)).fetchone():
                tombstone = key
        return {"status": "expired", "reason": "session absent from sessions", "retired_tombstone": tombstone}

    present = _span_rows(conn, sid, first, last)
    if len(present) == width and span_sha256(sid, present) == citation["span_sha256"]:
        return {"status": "verified", "reason": "span digest matches"}
    if len(present) > width:
        return {"status": "drifted", "reason": "more eligible rows in the span than were cited"}
    if len(present) == width and uids_sha256([row[3] for row in present]) == citation["uids_sha256"]:
        return {"status": "drifted", "reason": "same messages at the same ids, bytes differ"}

    resolved = _resolve_by_uid(conn, citation)
    if resolved is not None:
        ids = [[(row[5].decode() if isinstance(row[5], bytes) else row[5]), int(row[0])] for row in resolved]
        if content_sha256(sid, resolved) == citation["content_sha256"]:
            return {"status": "verified_via_uid", "reason": "row ids changed, uids and content match",
                    "resolved_ids": ids}
        return {"status": "drifted", "reason": "rows resolved by uid, bytes differ", "resolved_ids": ids}
    return {"status": "partially_missing",
            "reason": f"{len(present)} of {width} cited rows present at their ids; uids do not resolve"}


def verify_citation(citation: Mapping[str, Any], *, state_db: str) -> dict[str, Any]:
    """Re-check a citation against ``state_db``; read-only, never raises.

    Returns ``{"status": <one of STATUSES>, "reason": str, ...}``. ``expired`` adds
    ``retired_tombstone`` (the host's fence key, or ``None``); ``verified_via_uid`` adds
    ``resolved_ids`` as ``[session_id, id]`` pairs. A malformed citation is ``unavailable``.
    """
    try:
        problem = _malformed(citation)
        if problem:
            return {"status": "unavailable", "reason": f"citation_malformed: {problem}"}
        return _read(state_db, lambda conn: _verify(conn, citation))
    except _Unavailable as exc:
        return {"status": "unavailable", "reason": exc.reason}
    except Exception as exc:  # never raise to the caller
        return {"status": "unavailable", "reason": f"internal_error: {type(exc).__name__}"}


# --- quote (owner decision D1) ------------------------------------------------


def make_quote(text: str) -> dict[str, str]:
    """A bounded redacted quote: ``redact_secret_spans``, at most ``QUOTE_MAX_CHARS``.

    Redaction runs before the cut (a cut can hide a secret from the detector) and again
    after it (a cut can also create a match). If the text never settles, the quote is
    empty. Deciding whether a citation carries a quote belongs to the producer (C'2).
    """
    quote = redact_secret_spans(str(text or ""))
    for _ in range(4):
        quote = redact_secret_spans(quote[:QUOTE_MAX_CHARS])
        if len(quote) <= QUOTE_MAX_CHARS and not contains_secret(quote):
            break
    else:
        quote = ""
    return {"quote": quote, "quote_sha256": hashlib.sha256(quote.encode("utf-8")).hexdigest()}


__all__ = [
    "STATUSES", "UNRESOLVED_REASONS", "SPAN_BASES", "STORE", "MAX_SPAN_MESSAGES", "QUOTE_MAX_CHARS",
    "resolve_span", "verify_citation", "make_quote", "span_sha256", "uids_sha256", "content_sha256",
]
