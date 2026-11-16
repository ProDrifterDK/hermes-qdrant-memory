"""Opt-in filter for automatic recall: stored conversation turns of test harnesses and probes.

``auto_recall_exclude_machine`` (default off) makes the legacy block and the hybrid dense lane
leave out a conversation-turn point (``memory_kind=conversation_turn``, or
``source_type=conversation`` with ``chunk_type=turn``) when one of these holds:

- ``one_shot_session``: its session (payload ``session_id``) is a non-interactive one-shot run:
  ``sessions.source`` is in ``capture_reader.ONE_SHOT_SOURCES``. Its text does not matter.
- ``host_user_side``: its user side is host text under ``capture_reader.classify_row`` (the
  session-lane filter's classifier), judged with the session's fields when its row exists.
- ``echo_probe``: its session has no row in ``state.db`` (sessions leave it, turn points
  outlive them) or ``state.db`` cannot be read, and its own text proves a scripted check: the
  user side demands an exact reply and the whole assistant side is that literal, copied from
  the user side (``is_echo_probe``).

Everything else is kept. A session with a non-interactive source other than one-shot (a
delegated child is a ``subagent`` session; cron, tool, API clients) is kept unless its user
side is host text: agent-prompted sessions hold findings the user asks about later. A session
whose row exists is judged by structure only, so a probe typed in a person's interactive
session is kept. Documents, manual facts and every other point kind are never judged.

The session's fields come from ``state.db`` (``session_recall_state_db``), read through the C'
read-only reader without its lock retry (a busy database means "no structural evidence", never a
wait on the turn), at most one read per recall for the session ids not yet cached. Found rows are
cached for the process (a session's source and parent never change); absent ones for
``MISSING_TTL_SECONDS``. Query time only: nothing is written to Qdrant or to ``state.db``.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from qdrant_memory.capture_reader import (
    CLASS_HOST,
    CLASS_MACHINE,
    LANE_NOISE_KINDS,
    ONE_SHOT_SOURCES,
    RowContext,
    classify_row,
)
from qdrant_memory.session_origin import _read, _Unavailable

logger = logging.getLogger(__name__)

REASON_ONE_SHOT_SESSION = "one_shot_session"
REASON_HOST_USER_SIDE = "host_user_side"
REASON_ECHO_PROBE = "echo_probe"
REASONS = (REASON_ONE_SHOT_SESSION, REASON_HOST_USER_SIDE, REASON_ECHO_PROBE)

MISSING_TTL_SECONDS = 600.0
CACHE_MAX_ENTRIES = 4096
# A recall asks for at most 20 memories; this caps one ``IN (...)`` query either way.
_LOOKUP_CHUNK = 200

# The demand for an exact reply, in the languages the prompts on record use.
_EXACT_REPLY_RE = re.compile(r"(?i)\b(?:exactly|exactamente|verbatim|literally|literalmente)\b")
ECHO_MAX_PROMPT_CHARS = 500
ECHO_MAX_ANSWER_CHARS = 80


@dataclass(frozen=True)
class SessionFacts:
    """The ``sessions`` fields the rule reads."""

    source: str | None
    parent_session_id: str | None


def is_conversation_turn(payload: Mapping[str, Any]) -> bool:
    return payload.get("memory_kind") == "conversation_turn" or (
        payload.get("source_type") == "conversation" and payload.get("chunk_type") == "turn")


def split_turn(text: str) -> tuple[str, str]:
    """``(user, assistant)`` of a stored ``User: ...\\nAssistant: ...`` turn; else ``(text, "")``."""
    text = str(text or "")
    if text.startswith("User: ") and "\nAssistant: " in text:
        user, assistant = text[len("User: "):].split("\nAssistant: ", 1)
        return user, assistant
    return text, ""


def is_echo_probe(user_body: str, assistant: str) -> bool:
    """True when the turn is a scripted check: an exact reply demanded, and given verbatim.

    All must hold: the prompt (envelopes stripped) is at most ``ECHO_MAX_PROMPT_CHARS`` and
    says ``exactly`` (or ``exactamente``, ``verbatim``, ``literally``, ``literalmente``); the
    answer is one line of at most ``ECHO_MAX_ANSWER_CHARS``; and the answer, without trailing
    ``.``/``!`` and surrounding quotes or backticks, occurs in the prompt (case-insensitive).
    A long prompt (a group-chat transcript, a brief) or an answer that says anything the
    prompt did not is never a probe.
    """
    prompt = str(user_body or "").strip()
    answer = str(assistant or "").strip()
    if not prompt or len(prompt) > ECHO_MAX_PROMPT_CHARS or not _EXACT_REPLY_RE.search(prompt):
        return False
    if not answer or "\n" in answer or len(answer) > ECHO_MAX_ANSWER_CHARS:
        return False
    core = answer.rstrip(".!").strip().strip("`'\"").strip()
    return bool(core) and core.casefold() in prompt.casefold()


def turn_noise_reason(payload: Mapping[str, Any], facts: SessionFacts | None) -> str:
    """The reason a conversation-turn point is left out of automatic recall, or ``""``.

    ``facts`` is the point's session row, or None when ``state.db`` has none (or is unreadable).
    """
    if not is_conversation_turn(payload):
        return ""
    user, assistant = split_turn(str(payload.get("text") or ""))
    known = facts is not None
    source = ((facts.source if facts else None) or "").strip().lower()
    parent = facts.parent_session_id if facts else None
    verdict = classify_row(RowContext(
        msg_id=0, session_id=str(payload.get("session_id") or ""), role="user", content=user,
        source=source or None, parent_session_id=parent, session_known=known))
    if verdict.classification == CLASS_HOST:
        return REASON_HOST_USER_SIDE
    if known:
        # Structure decides. Delegated children are ``subagent`` sessions (kept); a one-shot
        # run's compression continuation keeps the ``oneshot`` source (left out with it).
        if source in ONE_SHOT_SOURCES or (verdict.classification == CLASS_MACHINE
                                          and verdict.kind in LANE_NOISE_KINDS):
            return REASON_ONE_SHOT_SESSION
        return ""
    return REASON_ECHO_PROBE if is_echo_probe(verdict.body, assistant) else ""


class SessionFactsCache:
    """Read-only, cached ``sessions`` lookups by id; thread-safe."""

    def __init__(self, state_db: str, *, max_entries: int = CACHE_MAX_ENTRIES,
                 missing_ttl: float = MISSING_TTL_SECONDS, clock=time.monotonic):
        self.state_db = str(state_db or "")
        self.max_entries = int(max_entries)
        self.missing_ttl = float(missing_ttl)
        self._clock = clock
        self._lock = threading.Lock()
        # id -> (SessionFacts or None, expires_at or None)
        self._entries: OrderedDict[str, tuple[SessionFacts | None, float | None]] = OrderedDict()
        self.reads = 0
        self.last_error = ""

    def _cached(self, session_id: str, now: float) -> tuple[bool, SessionFacts | None]:
        entry = self._entries.get(session_id)
        if entry is None:
            return False, None
        facts, expires = entry
        if expires is not None and now >= expires:
            del self._entries[session_id]
            return False, None
        self._entries.move_to_end(session_id)
        return True, facts

    def lookup(self, session_ids: Iterable[str]) -> dict[str, SessionFacts | None]:
        """``{id: SessionFacts}`` for rows found, ``{id: None}`` for ids with no row.

        Ids that could not be checked (``state.db`` missing, locked, unreadable) are absent
        from the result and not cached.
        """
        wanted = sorted({str(s) for s in session_ids if s})
        out: dict[str, SessionFacts | None] = {}
        now = self._clock()
        with self._lock:
            missing = []
            for sid in wanted:
                hit, facts = self._cached(sid, now)
                if hit:
                    out[sid] = facts
                else:
                    missing.append(sid)
        if not missing:
            return out
        try:
            found = _read(self.state_db, lambda conn: _select_sessions(conn, missing))
        except _Unavailable as exc:
            self.last_error = exc.reason
            logger.debug("auto_recall_exclude_machine: state.db not read (%s); text rule only", exc.reason)
            return out
        self.reads += 1
        with self._lock:
            for sid in missing:
                facts = found.get(sid)
                out[sid] = facts
                self._entries[sid] = (facts, None if facts is not None else now + self.missing_ttl)
                self._entries.move_to_end(sid)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return out


def _text(value: Any) -> str | None:
    if value is None:
        return None
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def _select_sessions(conn, session_ids: list[str]) -> dict[str, SessionFacts]:
    columns = {_text(row[1]) for row in conn.execute("PRAGMA table_info(sessions)")}
    source = "source" if "source" in columns else "NULL"
    parent = "parent_session_id" if "parent_session_id" in columns else "NULL"
    found: dict[str, SessionFacts] = {}
    for start in range(0, len(session_ids), _LOOKUP_CHUNK):
        chunk = session_ids[start:start + _LOOKUP_CHUNK]
        sql = f"SELECT id, {source}, {parent} FROM sessions WHERE id IN ({','.join('?' * len(chunk))})"
        for row in conn.execute(sql, chunk):
            found[_text(row[0]) or ""] = SessionFacts(_text(row[1]) or None, _text(row[2]) or None)
    return found


class TurnNoiseFilter:
    """Drops probe and harness turns from a ranked chunk list (``RetrievedMemory``-like)."""

    def __init__(self, sessions: SessionFactsCache):
        self.sessions = sessions
        self.last_excluded: list[tuple[str, str]] = []

    def judge(self, chunks: list[Any]) -> list[tuple[Any, str]]:
        """``(chunk, reason)`` pairs in input order; ``reason`` is ``""`` for a kept chunk."""
        turns = [c for c in chunks if is_conversation_turn(getattr(c, "payload", None) or {})]
        facts = self.sessions.lookup(str((c.payload or {}).get("session_id") or "") for c in turns) if turns else {}
        out = []
        for chunk in chunks:
            payload = getattr(chunk, "payload", None) or {}
            sid = str(payload.get("session_id") or "")
            out.append((chunk, turn_noise_reason(payload, facts.get(sid)) if is_conversation_turn(payload) else ""))
        return out

    def apply(self, chunks: list[Any]) -> list[Any]:
        judged = self.judge(list(chunks))
        self.last_excluded = [(str(getattr(c, "id", "")), reason) for c, reason in judged if reason]
        return [c for c, reason in judged if not reason]


class NoiseFilteredRecall:
    """A retriever (``search(query, top_k, **kwargs)``) whose results pass ``TurnNoiseFilter``.

    It asks the wrapped retriever for ``widened_top_k`` results, drops the noise, and returns
    the first ``top_k`` that are left, so a dropped turn frees its slot for the next memory.
    The provider passes ``max(top_k, min(20, search_candidates))``: the Qdrant request is then
    the one the unwrapped retriever sends (its limit is ``max(top_k, search_candidates)``), so the
    scores of the kept memories and their order are unchanged.
    """

    def __init__(self, inner: Any, noise: TurnNoiseFilter, widened_top_k: int):
        self.inner = inner
        self.noise = noise
        self.widened_top_k = int(widened_top_k)

    def search(self, query: str, top_k: int = 5, **kwargs: Any) -> list[Any]:
        limit = max(1, min(20, int(top_k)))
        chunks = self.inner.search(query, top_k=max(limit, self.widened_top_k), **kwargs)
        return self.noise.apply(chunks)[:limit]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)
