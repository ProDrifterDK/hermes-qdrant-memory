"""Phase 6H: runtime shadow recorder for the hybrid auto-recall path.

This module is **stdlib-only** (no new dependencies) and implements a
privacy-safe, aggregate-only append-only JSONL recorder for shadow
hybrid retrieval events.

Design constraints (see Phase 6H spec):

* **Never** persists raw query, raw packet, result text, point IDs,
  source_uri, file_path, headings, warnings text, exception strings,
  matched tokens, or payload excerpts.
* Uses sha256[:16] digests for query correlation, matching the
  existing ``_redact_query_metadata`` practice in
  :mod:`qdrant_memory.hybrid.router`.
* Per-session event count is bounded by ``max_per_session``.
* Fail-closed: any error during recording is swallowed silently so the
  shadow path can never crash the real prefetch path.
* No cron, no config mutation, no auto promotion.

The recorder is intentionally separate from ``__init__.py`` so the
provider file does not bloat and the recorder can be unit-tested in
isolation.
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = "qdrant_shadow_event.v1"

# Allowlist for ``status`` — anything else falls back to ``"error"``.
_ALLOWED_STATUSES = frozenset({"ok", "error", "skipped"})
_DEFAULT_ERROR_STATUS = "error"
_DEFAULT_OK_STATUS = "ok"

# Allowlist for ``trigger`` — anything else falls back to ``"invalid"``.
# Phase 6H only emits shadow events from the synchronous ``prefetch``
# path. ``queue_prefetch`` is cache priming only and must never reach
# the recorder — its own caller-side sanitization must keep the
# recorded trigger to ``"prefetch"``. Any caller-supplied lowercase
# ``[a-z0-9_-]`` string that resembles a secret-shaped fragment
# therefore CANNOT persist verbatim: it must collapse to ``"invalid"``
# via this allowlist. This is the
# privacy-sensitive boundary the reviewer flagged.
_ALLOWED_TRIGGERS = frozenset({"prefetch"})
_DEFAULT_TRIGGER = _GENERIC_INVALID_CODE = "invalid"

# Allowlist for ``error_code`` — anything non-empty that is not in the
# set falls back to ``"exception"``. The empty string ``""`` is the
# caller's explicit "no error code" sentinel and is preserved as-is so
# the OK-path statistics stay meaningful. ``None`` / non-string /
# unsafe raw text collapses to ``"exception"``.
_ALLOWED_ERROR_CODES = frozenset({"", "router_unavailable", "exception"})
_DEFAULT_ERROR_CODE = _GENERIC_EXCEPTION_CODE = "exception"


def _sanitize_status(value: Any) -> str:
    """Allowlist ``status`` to ``ok``/``error``/``skipped``.

    Any unrecognized input (including raw strings) collapses to
    ``"error"`` (the safe default for unknown caller-supplied states).
    """
    if not isinstance(value, str):
        return _DEFAULT_ERROR_STATUS
    candidate = value.strip().lower()
    if candidate in _ALLOWED_STATUSES:
        return candidate
    return _DEFAULT_ERROR_STATUS


def _sanitize_trigger(value: Any) -> str:
    """Allowlist ``trigger`` to a finite set of known safe values.

    Phase 6H currently only emits events from the synchronous
    ``prefetch`` path, so the only valid recorded trigger is
    ``"prefetch"``. Any non-string input (including ``None``) or any
    string that is not in the allowlist collapses to ``"invalid"`` so
    a caller-supplied lowercase ``[a-z0-9_-]`` secret-shaped fragment
    cannot persist verbatim in JSONL or ``get_status_summary()``.

    The allowlist check runs FIRST, before any character-level
    sanitization, so a safe-alphabet string such as ``"abcdef0123"``
    that is not in the allowlist is treated identically to a raw
    exception string: both collapse to ``"invalid"``.
    """
    if not isinstance(value, str):
        return _DEFAULT_TRIGGER
    candidate = value.strip()
    if not candidate:
        return _DEFAULT_TRIGGER
    if candidate in _ALLOWED_TRIGGERS:
        return candidate
    return _DEFAULT_TRIGGER


def _sanitize_error_code(value: Any) -> str:
    """Allowlist ``error_code`` to a finite set of known safe values.

    The empty string ``""`` is the caller's explicit "no error code"
    sentinel and is preserved as-is so OK-path statistics remain
    meaningful (``error_code == ""`` clearly distinguishes success
    rows from error rows in aggregate queries). Any non-string input
    (including ``None``), any whitespace-only string, or any
    non-empty string that is not in the allowlist collapses to
    ``"exception"`` so a caller-supplied lowercase ``[a-z0-9_-]``
    secret-shaped fragment (or an exception class name, file path
    fragment, etc.) cannot persist verbatim in JSONL or
    ``get_status_summary()``.

    Notes on why this uses a hard allowlist rather than the older
    ``[a-z0-9_-]`` regex sanitizer:

    * The older sanitizer was a *generic* "machine-safe token"
      filter. It silently accepted any safe-alphabet caller input —
      which means an attacker-controlled string such as
      ``"abcdef0123456789_xyz"`` would have been recorded verbatim.
    * Phase 6H error_code values are always one of the small known
      set in :data:`_ALLOWED_ERROR_CODES`. Anything else is by
      definition unknown and must collapse to the generic fallback.
    """
    if value is None or not isinstance(value, str):
        return _DEFAULT_ERROR_CODE
    candidate = value.strip()
    if not candidate:
        # Empty / whitespace-only caller input collapses to "" — the
        # caller's explicit "no error" sentinel.
        return ""
    if candidate in _ALLOWED_ERROR_CODES:
        return candidate
    return _DEFAULT_ERROR_CODE


def _query_digest(query: str) -> str:
    """Return sha256[:16] of the raw query for correlation.

    Matches the digest practice in
    :func:`qdrant_memory.hybrid.router._redact_query_metadata`.
    """
    return hashlib.sha256(str(query or "").encode("utf-8")).hexdigest()[:16]


def _session_hash(session_id: str) -> str:
    """Return sha256[:16] of the session id for per-session tracking."""
    return hashlib.sha256(str(session_id or "").encode("utf-8")).hexdigest()[:16]


def _default_artifact_dir(hermes_home: str) -> Path:
    """Return the default artifact directory under HERMES_HOME."""
    return Path(hermes_home) / "qdrant_memory" / "auto_recall_shadow"


class ShadowRecorder:
    """Append-only JSONL shadow event recorder.

    The recorder is thread-safe via a single lock. All public methods
    swallow exceptions so the shadow path can never disrupt the real
    prefetch path.

    Parameters
    ----------
    hermes_home
        The HERMES_HOME directory. Used to resolve the default artifact
        dir when ``artifact_dir`` is empty.
    max_per_session
        Maximum number of shadow events per session id. Once reached,
        subsequent ``record_event`` calls return ``False`` without
        writing.
    artifact_dir
        Override directory for JSONL artifacts. When empty, defaults to
        ``$HERMES_HOME/qdrant_memory/auto_recall_shadow``.
    """

    def __init__(
        self,
        hermes_home: str,
        max_per_session: int = 20,
        artifact_dir: str = "",
    ) -> None:
        self._hermes_home = str(hermes_home or "")
        self._max_per_session = max(0, int(max_per_session))
        if artifact_dir:
            self._artifact_dir = Path(artifact_dir)
        else:
            self._artifact_dir = _default_artifact_dir(self._hermes_home)
        self._lock = threading.Lock()
        self._session_counts: dict[str, int] = {}
        self._last_event: dict[str, Any] | None = None
        self._recorded_total = 0

    @property
    def artifact_dir(self) -> Path:
        return self._artifact_dir

    @property
    def max_per_session(self) -> int:
        return self._max_per_session

    @property
    def recorded_total(self) -> int:
        """Total events recorded across all sessions (lifetime of recorder)."""
        return self._recorded_total

    def _events_file(self) -> Path:
        return self._artifact_dir / "shadow_events.jsonl"

    def _state_file(self) -> Path:
        return self._artifact_dir / "shadow_state.json"

    def _load_existing_counts(self) -> None:
        """Load per-session counts from the state file if it exists.

        Called lazily once so a freshly-constructed recorder knows the
        counts from a prior process.
        """
        try:
            state_path = self._state_file()
            if state_path.exists():
                raw = json.loads(state_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    counts = raw.get("session_counts", {})
                    if isinstance(counts, dict):
                        for k, v in counts.items():
                            try:
                                self._session_counts[str(k)] = int(v)
                            except Exception:
                                pass
                    total = raw.get("recorded_total")
                    if isinstance(total, int):
                        self._recorded_total = total
        except Exception:
            pass

    def _save_state(self) -> None:
        """Persist the compact state JSON."""
        state = {
            "schema": "qdrant_shadow_state.v1",
            "session_counts": dict(self._session_counts),
            "recorded_total": self._recorded_total,
            "max_per_session": self._max_per_session,
        }
        self._state_file().write_text(
            json.dumps(state, sort_keys=True),
            encoding="utf-8",
        )

    def _append_event(self, event: dict[str, Any]) -> None:
        """Append one JSONL line."""
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        with open(self._events_file(), "a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")

    def record_event(
        self,
        *,
        query: str,
        session_id: str,
        trigger: str,
        latency_ms: float,
        legacy_chars: int,
        legacy_empty: bool,
        hybrid_summaries_count: int,
        hybrid_cited_leaves_count: int,
        hybrid_exact_hits_count: int,
        hybrid_graph_relations_count: int,
        hybrid_warning_count: int,
        hybrid_context_used_chars: int,
        status: str = "ok",
        error_code: str = "",
    ) -> bool:
        """Record one sanitized shadow event.

        Returns ``True`` if the event was written, ``False`` if the
        per-session cap was reached or an error occurred.

        **Privacy contract**: the only query-derived value persisted is
        ``query_digest`` (sha256[:16]). The raw query is NEVER written.
        No result text, point IDs, source URIs, file paths, headings,
        warning text, or exception strings are persisted — only counts
        and sanitized fields.

        ``status``, ``error_code`` and ``trigger`` are **always**
        sanitized via per-field allowlists before reaching JSONL:

        * ``status`` is restricted to ``"ok"`` / ``"error"`` / ``"skipped"``.
        * ``trigger`` is restricted to ``"prefetch"`` for Phase 6H; any
          other value (including safe-alphabet caller-supplied strings
          such as ``"abcdef0123456789"`` that would have been silently
          accepted by the old generic ``[a-z0-9_-]`` token filter)
          collapses to ``"invalid"``.
        * ``error_code`` is restricted to ``""`` / ``"router_unavailable"``
          / ``"exception"``; any other value (including safe-alphabet
          caller-supplied strings) collapses to ``"exception"``. The
          ``""`` sentinel is preserved as-is because callers use it to
          mean "no error code" on the OK path.
        """
        try:
            sh = _session_hash(session_id)
            sanitized_status = _sanitize_status(status)
            sanitized_trigger = _sanitize_trigger(trigger)
            sanitized_error_code = _sanitize_error_code(error_code)
            with self._lock:
                if not self._session_counts and self._recorded_total == 0:
                    self._load_existing_counts()
                current = self._session_counts.get(sh, 0)
                if current >= self._max_per_session:
                    return False
                event: dict[str, Any] = {
                    "schema": _SCHEMA_VERSION,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "trigger": sanitized_trigger,
                    "session_hash": sh,
                    "query_length": len(str(query or "")),
                    "query_digest": _query_digest(query),
                    "latency_ms": round(float(latency_ms), 3),
                    "legacy_chars": int(legacy_chars),
                    "legacy_empty": bool(legacy_empty),
                    "hybrid_summaries_count": int(hybrid_summaries_count),
                    "hybrid_cited_leaves_count": int(hybrid_cited_leaves_count),
                    "hybrid_exact_hits_count": int(hybrid_exact_hits_count),
                    "hybrid_graph_relations_count": int(hybrid_graph_relations_count),
                    "hybrid_warning_count": int(hybrid_warning_count),
                    "hybrid_context_used_chars": int(hybrid_context_used_chars),
                    "status": sanitized_status,
                    "error_code": sanitized_error_code,
                }
                self._append_event(event)
                self._session_counts[sh] = current + 1
                self._recorded_total += 1
                self._last_event = event
                self._save_state()
                return True
        except Exception:
            return False

    def get_status_summary(self) -> dict[str, Any]:
        """Return aggregate-only status fields for operator visibility.

        Never includes raw query, session_id, text, or payload data.
        """
        with self._lock:
            if not self._session_counts and self._recorded_total == 0:
                self._load_existing_counts()
            last = dict(self._last_event) if self._last_event else None
            if last:
                # Strip fields that could theoretically carry data —
                # though the event is already aggregate-only, we keep
                # the status block minimal.
                last_summary = {
                    "status": last.get("status", ""),
                    "error_code": last.get("error_code", ""),
                    "latency_ms": last.get("latency_ms", 0),
                    "legacy_chars": last.get("legacy_chars", 0),
                    "legacy_empty": last.get("legacy_empty", False),
                    "hybrid_summaries_count": last.get("hybrid_summaries_count", 0),
                    "hybrid_cited_leaves_count": last.get("hybrid_cited_leaves_count", 0),
                    "hybrid_exact_hits_count": last.get("hybrid_exact_hits_count", 0),
                    "hybrid_graph_relations_count": last.get("hybrid_graph_relations_count", 0),
                    "hybrid_warning_count": last.get("hybrid_warning_count", 0),
                    "hybrid_context_used_chars": last.get("hybrid_context_used_chars", 0),
                    "timestamp": last.get("timestamp", ""),
                }
            else:
                last_summary = None
            return {
                "shadow_recorded_count": self._recorded_total,
                "shadow_session_count": len(self._session_counts),
                "shadow_last_event": last_summary,
            }


def _safe_hybrid_counts(result: Any) -> tuple[int, int, int, int, int, int]:
    """Extract safe aggregate counts from a HybridRouteResult.

    Returns ``(summaries, cited_leaves, exact_hits, graph_relations,
    warning_count, context_used_chars)``. Never accesses text/ids.
    """
    summaries = 0
    cited_leaves = 0
    exact_hits = 0
    graph_relations = 0
    warning_count = 0
    context_used_chars = 0
    try:
        summaries = len(getattr(result, "summaries", []) or [])
        cited_leaves = len(getattr(result, "cited_leaves", []) or [])
        exact_hits = len(getattr(result, "exact_hits", []) or [])
        graph_relations = len(getattr(result, "graph_relations", []) or [])
        warnings = getattr(result, "warnings", []) or []
        warning_count = len(warnings)
        debug = getattr(result, "debug", {}) or {}
        context_used_chars = int(debug.get("context_used_chars", 0) or 0)
    except Exception:
        pass
    return (summaries, cited_leaves, exact_hits, graph_relations, warning_count, context_used_chars)
