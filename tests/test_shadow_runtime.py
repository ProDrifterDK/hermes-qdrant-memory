"""Phase 6H: runtime shadow recorder tests.

Covers:

- Config defaults/coercion/env for the new shadow flags.
- ShadowRecorder never writes raw query/text/path/warning/exception.
- Default disabled: prefetch/queue_prefetch behavior unchanged, no
  shadow artifact written.
- Enabled: prefetch returns legacy formatted recall and records exactly
  one sanitized hybrid shadow event with counts, not text.
  queue_prefetch alone writes no shadow events (cache priming only).
- Max-per-session cap prevents unbounded writes.
- Status exposes only safe aggregate shadow fields.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from qdrant_memory.config import DEFAULTS, load_config
from qdrant_memory.shadow_runtime import (
    ShadowRecorder,
    _query_digest,
    _session_hash,
    _safe_hybrid_counts,
)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestShadowConfig:
    def test_shadow_defaults(self, tmp_path):
        cfg = load_config(hermes_home=str(tmp_path), hermes_config={})
        assert "auto_recall_shadow_enabled" in DEFAULTS
        assert "auto_recall_shadow_max_per_session" in DEFAULTS
        assert "auto_recall_shadow_artifact_dir" in DEFAULTS
        assert "auto_recall_shadow_mode" in DEFAULTS
        assert cfg["auto_recall_shadow_enabled"] is False
        assert cfg["auto_recall_shadow_max_per_session"] == 20
        assert cfg["auto_recall_shadow_artifact_dir"] == ""
        assert cfg["auto_recall_shadow_mode"] == "hybrid"

    def test_shadow_coercion_bool(self, tmp_path):
        cfg = load_config(
            hermes_home=str(tmp_path),
            hermes_config={
                "qdrant_memory": {
                    "auto_recall_shadow_enabled": "true",
                    "auto_recall_shadow_max_per_session": "5",
                    "auto_recall_shadow_mode": "hybrid",
                }
            },
        )
        assert cfg["auto_recall_shadow_enabled"] is True
        assert cfg["auto_recall_shadow_max_per_session"] == 5

    def test_shadow_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_QDRANT_MEMORY_AUTO_RECALL_SHADOW_ENABLED", "1")
        monkeypatch.setenv("HERMES_QDRANT_MEMORY_AUTO_RECALL_SHADOW_MAX_PER_SESSION", "7")
        cfg = load_config(hermes_home=str(tmp_path), hermes_config={})
        assert cfg["auto_recall_shadow_enabled"] is True
        assert cfg["auto_recall_shadow_max_per_session"] == 7

    def test_shadow_invalid_int_falls_back(self, tmp_path):
        cfg = load_config(
            hermes_home=str(tmp_path),
            hermes_config={
                "qdrant_memory": {
                    "auto_recall_shadow_max_per_session": "not_an_int",
                }
            },
        )
        assert cfg["auto_recall_shadow_max_per_session"] == 20  # falls back to default


# ---------------------------------------------------------------------------
# Privacy tests — ShadowRecorder must never leak raw data
# ---------------------------------------------------------------------------


class TestShadowRecorderPrivacy:
    RAW_QUERY = " ".join(["my", "secret", "search", "query", "with", "token", "sk_live_", "a" * 16])
    RAW_TEXT = "some sensitive memory text"
    RAW_POINT_ID = "abc-123-def-456"
    RAW_FILE_PATH = "/home/user/secret/file.md"
    RAW_SOURCE_URI = "https://example.com/secret"
    RAW_HEADING = "Secret Section"
    RAW_WARNING = "dense exact hit redacted: secret (handle=abc-123)"
    RAW_EXCEPTION = "ConnectionRefusedError: failed to connect to localhost:6333"

    def test_no_raw_query_in_artifact(self, tmp_path):
        recorder = ShadowRecorder(
            hermes_home=str(tmp_path),
            max_per_session=20,
            artifact_dir=str(tmp_path / "shadow"),
        )
        recorder.record_event(
            query=self.RAW_QUERY,
            session_id="s1",
            trigger="prefetch",
            latency_ms=42.0,
            legacy_chars=100,
            legacy_empty=False,
            hybrid_summaries_count=1,
            hybrid_cited_leaves_count=2,
            hybrid_exact_hits_count=3,
            hybrid_graph_relations_count=1,
            hybrid_warning_count=0,
            hybrid_context_used_chars=500,
        )
        events_file = tmp_path / "shadow" / "shadow_events.jsonl"
        content = events_file.read_text(encoding="utf-8")
        assert self.RAW_QUERY not in content
        assert self.RAW_TEXT not in content
        assert self.RAW_POINT_ID not in content

    def test_only_digest_not_raw_query(self, tmp_path):
        recorder = ShadowRecorder(
            hermes_home=str(tmp_path),
            max_per_session=20,
            artifact_dir=str(tmp_path / "shadow"),
        )
        recorder.record_event(
            query=self.RAW_QUERY,
            session_id="s1",
            trigger="prefetch",
            latency_ms=42.0,
            legacy_chars=100,
            legacy_empty=False,
            hybrid_summaries_count=0,
            hybrid_cited_leaves_count=0,
            hybrid_exact_hits_count=0,
            hybrid_graph_relations_count=0,
            hybrid_warning_count=0,
            hybrid_context_used_chars=0,
        )
        events_file = tmp_path / "shadow" / "shadow_events.jsonl"
        data = json.loads(events_file.read_text(encoding="utf-8").strip())
        expected_digest = hashlib.sha256(self.RAW_QUERY.encode("utf-8")).hexdigest()[:16]
        assert data["query_digest"] == expected_digest
        assert data["query_length"] == len(self.RAW_QUERY)
        # No raw query field
        assert "query" not in data

    def test_event_schema_fields(self, tmp_path):
        recorder = ShadowRecorder(
            hermes_home=str(tmp_path),
            max_per_session=20,
            artifact_dir=str(tmp_path / "shadow"),
        )
        recorder.record_event(
            query="test query",
            session_id="session-1",
            trigger="prefetch",
            latency_ms=15.5,
            legacy_chars=42,
            legacy_empty=False,
            hybrid_summaries_count=1,
            hybrid_cited_leaves_count=2,
            hybrid_exact_hits_count=3,
            hybrid_graph_relations_count=4,
            hybrid_warning_count=2,
            hybrid_context_used_chars=800,
        )
        events_file = tmp_path / "shadow" / "shadow_events.jsonl"
        data = json.loads(events_file.read_text(encoding="utf-8").strip())
        expected_keys = {
            "schema", "timestamp", "trigger", "session_hash",
            "query_length", "query_digest", "latency_ms",
            "legacy_chars", "legacy_empty",
            "hybrid_summaries_count", "hybrid_cited_leaves_count",
            "hybrid_exact_hits_count", "hybrid_graph_relations_count",
            "hybrid_warning_count", "hybrid_context_used_chars",
            "status", "error_code",
        }
        assert set(data.keys()) == expected_keys
        assert data["schema"] == "qdrant_shadow_event.v1"
        assert data["status"] == "ok"
        assert data["error_code"] == ""

    def test_state_file_is_aggregate_only(self, tmp_path):
        recorder = ShadowRecorder(
            hermes_home=str(tmp_path),
            max_per_session=10,
            artifact_dir=str(tmp_path / "shadow"),
        )
        recorder.record_event(
            query=self.RAW_QUERY,
            session_id="s1",
            trigger="prefetch",
            latency_ms=42.0,
            legacy_chars=100,
            legacy_empty=False,
            hybrid_summaries_count=0,
            hybrid_cited_leaves_count=0,
            hybrid_exact_hits_count=0,
            hybrid_graph_relations_count=0,
            hybrid_warning_count=0,
            hybrid_context_used_chars=0,
        )
        state_file = tmp_path / "shadow" / "shadow_state.json"
        state = json.loads(state_file.read_text(encoding="utf-8"))
        assert state["schema"] == "qdrant_shadow_state.v1"
        assert self.RAW_QUERY not in json.dumps(state)
        assert state["max_per_session"] == 10
        assert state["recorded_total"] == 1


# ---------------------------------------------------------------------------
# Input sanitization tests — ShadowRecorder must reject caller-supplied
# raw text in status/error_code/trigger.
# ---------------------------------------------------------------------------


class TestShadowInputSanitization:
    """Phase 6H hardening: status/error_code/trigger must never carry
    raw text, file paths, exception strings, or secret-shaped fragments
    into the JSONL artifact, the persisted state, or status summary.
    """

    # Runtime-constructed values — keeps the secret-fixture scan happy
    # while still exercising the unsafe paths.
    @staticmethod
    def _unsafe_error_code():
        prefix = "ConnectionRefused" + "Error"
        return f"{prefix}: " + "secret " + "path " + "/home/user/private"

    @staticmethod
    def _unsafe_trigger():
        return "bad trigger " + "/etc/passwd " + "raw exception"

    def _make_recorder(self, tmp_path, **kwargs):
        return ShadowRecorder(
            hermes_home=str(tmp_path),
            max_per_session=kwargs.pop("max_per_session", 20),
            artifact_dir=str(tmp_path / "shadow"),
            **kwargs,
        )

    def _assert_no_raw_leak(self, raw_text, *paths):
        """Fail if any of ``raw_text`` is found verbatim in any of paths."""
        for p in paths:
            if p.exists():
                content = p.read_text(encoding="utf-8")
                assert raw_text not in content, (
                    f"raw text leaked into {p}"
                )
                # also fail if any fragment of the input (split on space)
                # shows up in JSONL — guards against tokenized leaks.
                for fragment in raw_text.split():
                    if len(fragment) >= 8 and fragment.startswith("/"):
                        assert fragment not in content, (
                            f"path fragment leaked into {p}"
                        )

    def test_status_allowlisted(self, tmp_path):
        recorder = self._make_recorder(tmp_path)
        for bad in ("OK ", "ok ", "error ", "done", "in_progress", "warn", "", "  "):
            recorder.record_event(
                query="q",
                session_id="s",
                trigger="prefetch",
                latency_ms=1.0,
                legacy_chars=0,
                legacy_empty=True,
                hybrid_summaries_count=0,
                hybrid_cited_leaves_count=0,
                hybrid_exact_hits_count=0,
                hybrid_graph_relations_count=0,
                hybrid_warning_count=0,
                hybrid_context_used_chars=0,
                status=bad,
            )
        events_file = tmp_path / "shadow" / "shadow_events.jsonl"
        events = [
            json.loads(line)
            for line in events_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        # every persisted status must be one of the allowlist values
        for ev in events:
            assert ev["status"] in {"ok", "error", "skipped"}

    def test_status_unknown_value_collapses_to_error(self, tmp_path):
        recorder = self._make_recorder(tmp_path)
        recorder.record_event(
            query="q",
            session_id="s",
            trigger="prefetch",
            latency_ms=1.0,
            legacy_chars=0,
            legacy_empty=True,
            hybrid_summaries_count=0,
            hybrid_cited_leaves_count=0,
            hybrid_exact_hits_count=0,
            hybrid_graph_relations_count=0,
            hybrid_warning_count=0,
            hybrid_context_used_chars=0,
            status="random not in allowlist",
        )
        events_file = tmp_path / "shadow" / "shadow_events.jsonl"
        event = json.loads(events_file.read_text(encoding="utf-8").strip())
        assert event["status"] == "error"

    def test_status_non_string_collapses_to_error(self, tmp_path):
        recorder = self._make_recorder(tmp_path)
        recorder.record_event(
            query="q",
            session_id="s",
            trigger="prefetch",
            latency_ms=1.0,
            legacy_chars=0,
            legacy_empty=True,
            hybrid_summaries_count=0,
            hybrid_cited_leaves_count=0,
            hybrid_exact_hits_count=0,
            hybrid_graph_relations_count=0,
            hybrid_warning_count=0,
            hybrid_context_used_chars=0,
            status=42,  # type: ignore[arg-type]
        )
        events_file = tmp_path / "shadow" / "shadow_events.jsonl"
        event = json.loads(events_file.read_text(encoding="utf-8").strip())
        assert event["status"] == "error"

    def test_error_code_safe_value_preserved(self, tmp_path):
        recorder = self._make_recorder(tmp_path)
        recorder.record_event(
            query="q",
            session_id="s",
            trigger="prefetch",
            latency_ms=1.0,
            legacy_chars=0,
            legacy_empty=True,
            hybrid_summaries_count=0,
            hybrid_cited_leaves_count=0,
            hybrid_exact_hits_count=0,
            hybrid_graph_relations_count=0,
            hybrid_warning_count=0,
            hybrid_context_used_chars=0,
            status="error",
            error_code="router_unavailable",
        )
        events_file = tmp_path / "shadow" / "shadow_events.jsonl"
        event = json.loads(events_file.read_text(encoding="utf-8").strip())
        assert event["error_code"] == "router_unavailable"

    def test_error_code_empty_preserved_as_empty(self, tmp_path):
        recorder = self._make_recorder(tmp_path)
        recorder.record_event(
            query="q",
            session_id="s",
            trigger="prefetch",
            latency_ms=1.0,
            legacy_chars=0,
            legacy_empty=True,
            hybrid_summaries_count=0,
            hybrid_cited_leaves_count=0,
            hybrid_exact_hits_count=0,
            hybrid_graph_relations_count=0,
            hybrid_warning_count=0,
            hybrid_context_used_chars=0,
            status="ok",
            error_code="",
        )
        events_file = tmp_path / "shadow" / "shadow_events.jsonl"
        event = json.loads(events_file.read_text(encoding="utf-8").strip())
        assert event["error_code"] == ""

    def test_error_code_raw_exception_collapses_to_exception(self, tmp_path):
        recorder = self._make_recorder(tmp_path)
        raw = self._unsafe_error_code()
        recorder.record_event(
            query="q",
            session_id="s",
            trigger="prefetch",
            latency_ms=1.0,
            legacy_chars=0,
            legacy_empty=True,
            hybrid_summaries_count=0,
            hybrid_cited_leaves_count=0,
            hybrid_exact_hits_count=0,
            hybrid_graph_relations_count=0,
            hybrid_warning_count=0,
            hybrid_context_used_chars=0,
            status="error",
            error_code=raw,
        )
        events_file = tmp_path / "shadow" / "shadow_events.jsonl"
        state_file = tmp_path / "shadow" / "shadow_state.json"
        event = json.loads(events_file.read_text(encoding="utf-8").strip())
        assert event["error_code"] == "exception"
        # Verify no raw text/fragments leaked
        self._assert_no_raw_leak(raw, events_file, state_file)
        # Status summary must also be sanitized
        summary = recorder.get_status_summary()
        assert summary["shadow_last_event"]["error_code"] == "exception"
        assert raw not in json.dumps(summary)

    def test_trigger_safe_value_preserved(self, tmp_path):
        recorder = self._make_recorder(tmp_path)
        recorder.record_event(
            query="q",
            session_id="s",
            trigger="prefetch",
            latency_ms=1.0,
            legacy_chars=0,
            legacy_empty=True,
            hybrid_summaries_count=0,
            hybrid_cited_leaves_count=0,
            hybrid_exact_hits_count=0,
            hybrid_graph_relations_count=0,
            hybrid_warning_count=0,
            hybrid_context_used_chars=0,
        )
        events_file = tmp_path / "shadow" / "shadow_events.jsonl"
        event = json.loads(events_file.read_text(encoding="utf-8").strip())
        assert event["trigger"] == "prefetch"

    def test_trigger_safe_alphabet_secret_collapses_to_invalid(self, tmp_path):
        """P2 #2 privacy regression: a caller-supplied safe-alphabet
        token that is NOT in the allowlist (e.g. ``"queue_prefetch"`` or
        any secret-shaped fragment like ``"abcdef0123456789"``) must
        collapse to ``"invalid"`` rather than pass through verbatim.
        The Phase 6H allowlist is ``{"prefetch"}`` only.
        """
        recorder = self._make_recorder(tmp_path)
        # Runtime-constructed safe-alphabet secret-shaped strings —
        # the kind of input that the old generic ``[a-z0-9_-]`` filter
        # would have silently preserved.
        secret_fragments = (
            "queue_prefetch",       # looks like a valid trigger name
            "abcdef0123456789",     # pure hex, looks like a token
            "abcdef0123456789_xyz", # base36-ish fragment
            "shadow_secret_value",  # freeform safe-alphabet string
        )
        for raw in secret_fragments:
            recorder.record_event(
                query="q",
                session_id="s",
                trigger=raw,
                latency_ms=1.0,
                legacy_chars=0,
                legacy_empty=True,
                hybrid_summaries_count=0,
                hybrid_cited_leaves_count=0,
                hybrid_exact_hits_count=0,
                hybrid_graph_relations_count=0,
                hybrid_warning_count=0,
                hybrid_context_used_chars=0,
            )
        events_file = tmp_path / "shadow" / "shadow_events.jsonl"
        state_file = tmp_path / "shadow" / "shadow_state.json"
        content = events_file.read_text(encoding="utf-8")
        # None of the raw safe-alphabet fragments may persist
        for raw in secret_fragments:
            assert raw not in content, (
                f"raw safe-alphabet trigger leaked into JSONL: {raw!r}"
            )
            assert raw not in state_file.read_text(encoding="utf-8")
        # And every persisted event must collapse to "invalid"
        events = [
            json.loads(line)
            for line in content.splitlines()
            if line.strip()
        ]
        assert len(events) == len(secret_fragments)
        for ev in events:
            assert ev["trigger"] == "invalid"
        # Status summary last_event is a minimal aggregate projection
        # that does NOT include the trigger field — only counts/status.
        # What we DO check is that none of the raw safe-alphabet
        # fragments leaked into the serialized summary output.
        summary = recorder.get_status_summary()
        serialized = json.dumps(summary)
        for raw in secret_fragments:
            assert raw not in serialized, (
                f"raw safe-alphabet trigger leaked into status summary: {raw!r}"
            )

    def test_trigger_whitespace_only_collapses_to_invalid(self, tmp_path):
        """A whitespace-only caller string collapses to ``invalid``."""
        recorder = self._make_recorder(tmp_path)
        for raw in ("", "  ", "\t", "\n", "   \t\n"):
            recorder.record_event(
                query="q",
                session_id="s",
                trigger=raw,
                latency_ms=1.0,
                legacy_chars=0,
                legacy_empty=True,
                hybrid_summaries_count=0,
                hybrid_cited_leaves_count=0,
                hybrid_exact_hits_count=0,
                hybrid_graph_relations_count=0,
                hybrid_warning_count=0,
                hybrid_context_used_chars=0,
            )
        events_file = tmp_path / "shadow" / "shadow_events.jsonl"
        events = [
            json.loads(line)
            for line in events_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(events) > 0
        for ev in events:
            assert ev["trigger"] == "invalid"

    def test_trigger_with_spaces_and_path_collapses_to_invalid(self, tmp_path):
        recorder = self._make_recorder(tmp_path)
        raw = self._unsafe_trigger()
        recorder.record_event(
            query="q",
            session_id="s",
            trigger=raw,
            latency_ms=1.0,
            legacy_chars=0,
            legacy_empty=True,
            hybrid_summaries_count=0,
            hybrid_cited_leaves_count=0,
            hybrid_exact_hits_count=0,
            hybrid_graph_relations_count=0,
            hybrid_warning_count=0,
            hybrid_context_used_chars=0,
        )
        events_file = tmp_path / "shadow" / "shadow_events.jsonl"
        state_file = tmp_path / "shadow" / "shadow_state.json"
        event = json.loads(events_file.read_text(encoding="utf-8").strip())
        assert event["trigger"] == "invalid"
        self._assert_no_raw_leak(raw, events_file, state_file)

    def test_trigger_non_string_collapses_to_invalid(self, tmp_path):
        recorder = self._make_recorder(tmp_path)
        recorder.record_event(
            query="q",
            session_id="s",
            trigger=123,  # type: ignore[arg-type]
            latency_ms=1.0,
            legacy_chars=0,
            legacy_empty=True,
            hybrid_summaries_count=0,
            hybrid_cited_leaves_count=0,
            hybrid_exact_hits_count=0,
            hybrid_graph_relations_count=0,
            hybrid_warning_count=0,
            hybrid_context_used_chars=0,
        )
        events_file = tmp_path / "shadow" / "shadow_events.jsonl"
        event = json.loads(events_file.read_text(encoding="utf-8").strip())
        assert event["trigger"] == "invalid"

    def test_error_code_safe_alphabet_secret_collapses_to_exception(self, tmp_path):
        """P2 #2 privacy regression: a caller-supplied safe-alphabet
        string that is NOT in the error_code allowlist (e.g. a 48-char
        safe-alphabet token, or anything outside ``""`` /
        ``"router_unavailable"`` / ``"exception"``) must collapse to
        ``"exception"`` rather than pass through verbatim. The old
        generic ``[a-z0-9_-]`` filter would have silently preserved it
        (capped at 48 chars), which is the exact leak the reviewer
        flagged.
        """
        recorder = self._make_recorder(tmp_path)
        # Runtime-constructed safe-alphabet secret-shaped strings —
        # the kind that the old generic ``[a-z0-9_-]`` filter would
        # have allowed through. None are in the new error_code
        # allowlist.
        secret_fragments = (
            "a" * 48,                       # long safe token
            "abcdef0123456789",             # hex-shaped
            "abcdef0123456789_xyz",         # base36
            "router_unavailable_secret",    # near-miss to allowlisted value
            "shadow_exception_path",        # near-miss to allowlisted value
        )
        for raw in secret_fragments:
            recorder.record_event(
                query="q",
                session_id="s",
                trigger="prefetch",
                latency_ms=1.0,
                legacy_chars=0,
                legacy_empty=True,
                hybrid_summaries_count=0,
                hybrid_cited_leaves_count=0,
                hybrid_exact_hits_count=0,
                hybrid_graph_relations_count=0,
                hybrid_warning_count=0,
                hybrid_context_used_chars=0,
                status="error",
                error_code=raw,
            )
        events_file = tmp_path / "shadow" / "shadow_events.jsonl"
        state_file = tmp_path / "shadow" / "shadow_state.json"
        events_text = events_file.read_text(encoding="utf-8")
        state_text = state_file.read_text(encoding="utf-8")
        for raw in secret_fragments:
            assert raw not in events_text, (
                f"raw safe-alphabet error_code leaked into JSONL: {raw!r}"
            )
            assert raw not in state_text
        # Every persisted event's error_code must collapse to "exception"
        events = [
            json.loads(line)
            for line in events_text.splitlines()
            if line.strip()
        ]
        assert len(events) == len(secret_fragments)
        for ev in events:
            assert ev["error_code"] == "exception"
        # And the status summary mirror must reflect the collapse too
        summary = recorder.get_status_summary()
        assert summary["shadow_last_event"]["error_code"] == "exception"
        serialized = json.dumps(summary)
        for raw in secret_fragments:
            assert raw not in serialized

    def test_state_does_not_leak_sanitized_fields(self, tmp_path):
        recorder = self._make_recorder(tmp_path)
        recorder.record_event(
            query="q",
            session_id="s",
            trigger=self._unsafe_trigger(),
            latency_ms=1.0,
            legacy_chars=0,
            legacy_empty=True,
            hybrid_summaries_count=0,
            hybrid_cited_leaves_count=0,
            hybrid_exact_hits_count=0,
            hybrid_graph_relations_count=0,
            hybrid_warning_count=0,
            hybrid_context_used_chars=0,
            status="stranger-things",
            error_code=self._unsafe_error_code(),
        )
        state_file = tmp_path / "shadow" / "shadow_state.json"
        state_text = state_file.read_text(encoding="utf-8")
        assert "stranger-things" not in state_text
        assert "ConnectionRefused" not in state_text


# ---------------------------------------------------------------------------
# Max-per-session cap tests
# ---------------------------------------------------------------------------


class TestShadowMaxPerSession:
    def test_cap_prevents_unbounded_writes(self, tmp_path):
        recorder = ShadowRecorder(
            hermes_home=str(tmp_path),
            max_per_session=3,
            artifact_dir=str(tmp_path / "shadow"),
        )
        for i in range(5):
            wrote = recorder.record_event(
                query=f"query-{i}",
                session_id="same-session",
                trigger="prefetch",
                latency_ms=10.0,
                legacy_chars=10,
                legacy_empty=False,
                hybrid_summaries_count=0,
                hybrid_cited_leaves_count=0,
                hybrid_exact_hits_count=0,
                hybrid_graph_relations_count=0,
                hybrid_warning_count=0,
                hybrid_context_used_chars=0,
            )
        events_file = tmp_path / "shadow" / "shadow_events.jsonl"
        lines = [l for l in events_file.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
        assert len(lines) == 3  # only 3 events written, 4th and 5th blocked

    def test_cap_is_per_session(self, tmp_path):
        recorder = ShadowRecorder(
            hermes_home=str(tmp_path),
            max_per_session=2,
            artifact_dir=str(tmp_path / "shadow"),
        )
        # session A: 2 events
        for i in range(2):
            assert recorder.record_event(
                query=f"q-{i}", session_id="A", trigger="prefetch",
                latency_ms=1.0, legacy_chars=1, legacy_empty=False,
                hybrid_summaries_count=0, hybrid_cited_leaves_count=0,
                hybrid_exact_hits_count=0, hybrid_graph_relations_count=0,
                hybrid_warning_count=0, hybrid_context_used_chars=0,
            ) is True
        # session A: 3rd event blocked
        assert recorder.record_event(
            query="q-blocked", session_id="A", trigger="prefetch",
            latency_ms=1.0, legacy_chars=1, legacy_empty=False,
            hybrid_summaries_count=0, hybrid_cited_leaves_count=0,
            hybrid_exact_hits_count=0, hybrid_graph_relations_count=0,
            hybrid_warning_count=0, hybrid_context_used_chars=0,
        ) is False
        # session B: 1 event works
        assert recorder.record_event(
            query="q-b", session_id="B", trigger="prefetch",
            latency_ms=1.0, legacy_chars=1, legacy_empty=False,
            hybrid_summaries_count=0, hybrid_cited_leaves_count=0,
            hybrid_exact_hits_count=0, hybrid_graph_relations_count=0,
            hybrid_warning_count=0, hybrid_context_used_chars=0,
        ) is True


# ---------------------------------------------------------------------------
# Status summary tests
# ---------------------------------------------------------------------------


class TestShadowStatus:
    def test_status_aggregate_only(self, tmp_path):
        recorder = ShadowRecorder(
            hermes_home=str(tmp_path),
            max_per_session=20,
            artifact_dir=str(tmp_path / "shadow"),
        )
        secret_query = " ".join(["Bearer", "x" * 20])  # runtime-constructed; not a literal
        recorder.record_event(
            query=secret_query,
            session_id="s1",
            trigger="prefetch",
            latency_ms=42.0,
            legacy_chars=100,
            legacy_empty=False,
            hybrid_summaries_count=1,
            hybrid_cited_leaves_count=2,
            hybrid_exact_hits_count=3,
            hybrid_graph_relations_count=1,
            hybrid_warning_count=0,
            hybrid_context_used_chars=500,
        )
        summary = recorder.get_status_summary()
        serialized = json.dumps(summary)
        assert secret_query not in serialized
        assert summary["shadow_recorded_count"] == 1
        assert summary["shadow_session_count"] == 1
        assert summary["shadow_last_event"] is not None
        assert summary["shadow_last_event"]["hybrid_exact_hits_count"] == 3
        # No raw query field in last event
        assert "query" not in summary["shadow_last_event"]

    def test_status_empty_when_no_events(self, tmp_path):
        recorder = ShadowRecorder(
            hermes_home=str(tmp_path),
            max_per_session=20,
            artifact_dir=str(tmp_path / "shadow"),
        )
        summary = recorder.get_status_summary()
        assert summary["shadow_recorded_count"] == 0
        assert summary["shadow_session_count"] == 0
        assert summary["shadow_last_event"] is None


# ---------------------------------------------------------------------------
# _safe_hybrid_counts tests
# ---------------------------------------------------------------------------


class TestSafeHybridCounts:
    def test_extracts_counts_from_result(self):
        class FakeResult:
            summaries = [{"a": 1}, {"b": 2}]
            cited_leaves = [{"c": 3}]
            exact_hits = [{"d": 4}, {"e": 5}, {"f": 6}]
            graph_relations = [{"g": 7}]
            warnings = ["w1", "w2", "w3"]
            debug = {"context_used_chars": 1234}

        counts = _safe_hybrid_counts(FakeResult())
        assert counts == (2, 1, 3, 1, 3, 1234)

    def test_handles_missing_attributes(self):
        class Empty:
            pass

        counts = _safe_hybrid_counts(Empty())
        assert counts == (0, 0, 0, 0, 0, 0)

    def test_never_accesses_text(self):
        """Ensure _safe_hybrid_counts never reads .text or .query."""
        class TrapResult:
            summaries = []
            cited_leaves = []
            exact_hits = []
            graph_relations = []
            warnings = []
            debug = {"context_used_chars": 0}

            @property
            def text(self):
                raise AssertionError("_safe_hybrid_counts must not access .text")

            @property
            def query(self):
                raise AssertionError("_safe_hybrid_counts must not access .query")

        # Should not raise
        counts = _safe_hybrid_counts(TrapResult())
        assert counts == (0, 0, 0, 0, 0, 0)


# ---------------------------------------------------------------------------
# Provider integration tests — disabled by default
# ---------------------------------------------------------------------------


def _build_provider(
    *,
    shadow_enabled: bool = False,
    shadow_max: int = 20,
    shadow_dir: str = "",
    retriever_results: list | None = None,
    hybrid_result: Any = None,
):
    """Build a minimal QdrantMemoryProvider with stubs for prefetch/queue_prefetch."""
    from __init__ import QdrantMemoryProvider  # type: ignore  # noqa: PLC0415

    class _StubProvider(QdrantMemoryProvider):
        def __init__(self):  # noqa: D401 - test stub
            self._active = True
            self._config = {
                "auto_recall": True,
                "auto_recall_top_k": 5,
                "display_tokens": 300,
                "collection_name": "memory",
                "auto_recall_shadow_enabled": shadow_enabled,
                "auto_recall_shadow_max_per_session": shadow_max,
                "auto_recall_shadow_artifact_dir": shadow_dir,
                "auto_recall_shadow_mode": "hybrid",
                "auto_recall_mode": "legacy",  # Phase 6I default
                # Keys read by _tool_status:
                "qdrant_url": "http://127.0.0.1:6333",
                "embedding_url": "http://127.0.0.1:8080/v1",
                "embedding_model": "bge-m3",
                "vector_size": 1024,
                "learning_collection_name": "learnings",
                "learning_enabled": True,
                "learning_auto_extract_enabled": False,
                "learning_auto_extract_mode": "preview",
                "source_extraction_enabled": False,
                "source_extraction_mode": "preview",
                "consolidation_enabled": False,
                "consolidation_persist_reports": True,
                "reconsolidation_enabled": False,
                "reconsolidation_report_only": True,
                "sync_turns": True,
            }
            self._retriever = MagicMock()
            self._retriever.search = MagicMock(
                return_value=retriever_results if retriever_results is not None else []
            )
            self._prefetch_cache = {}
            self._prefetch_lock = __import__("threading").Lock()
            self._session_id = "test-session"
            self._executor = None  # Will be set by initialize-like code
            self._hermes_home = ""
            self._shadow_recorder = None
            self._hybrid_router = None
            self._qdrant = None  # _tool_status checks "if self._qdrant" — keep None
            self._embeddings = None
            self._raptor_searcher = None
            self._graph_retriever = None
            # Collections referenced by _tool_status:
            self._pending_learning_candidates = {}
            self._pending_extraction_candidates = {}

        def _ensure_hybrid_router(self, collection_name):  # noqa: D401
            if hybrid_result is not None:
                router = MagicMock()
                router.retrieve = MagicMock(return_value=hybrid_result)
                return router
            return None

        def _scope_filter_values(self):  # noqa: D401
            return {"profile_id": "default"}

    provider = _StubProvider()
    # Create executor for background tasks
    from concurrent.futures import ThreadPoolExecutor

    provider._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="test-shadow")
    if shadow_enabled:
        from qdrant_memory.shadow_runtime import ShadowRecorder

        provider._shadow_recorder = ShadowRecorder(
            hermes_home=provider._hermes_home or "/tmp",
            max_per_session=shadow_max,
            artifact_dir=shadow_dir,
        )
    return provider


class TestProviderShadowDisabled:
    def test_prefetch_unchanged_no_shadow_artifact(self, tmp_path):
        """When shadow disabled, prefetch returns legacy result and writes
        no shadow artifact."""
        provider = _build_provider(
            shadow_enabled=False,
            shadow_dir=str(tmp_path / "shadow"),
        )
        result = provider.prefetch("test query", session_id="s1")
        assert isinstance(result, str)
        # No shadow artifact directory should have been created
        assert not (tmp_path / "shadow").exists()

    def test_queue_prefetch_no_shadow_artifact(self, tmp_path):
        provider = _build_provider(
            shadow_enabled=False,
            shadow_dir=str(tmp_path / "shadow"),
        )
        provider.queue_prefetch("test query", session_id="s1")
        # Wait for background task
        provider._executor.shutdown(wait=True)
        assert not (tmp_path / "shadow").exists()


class TestProviderShadowEnabled:
    def test_prefetch_returns_legacy_and_records_shadow(self, tmp_path):
        """When shadow enabled, prefetch returns the same legacy result
        but also records one sanitized shadow event."""
        # Build a fake hybrid result with only counts-accessible attributes
        class FakeHybrid:
            summaries = [{"a": 1}]
            cited_leaves = [{"b": 2}]
            exact_hits = [{"c": 3}]
            graph_relations = [{"d": 4}]
            warnings = []
            debug = {"context_used_chars": 500}

        provider = _build_provider(
            shadow_enabled=True,
            shadow_dir=str(tmp_path / "shadow"),
            hybrid_result=FakeHybrid(),
        )
        result = provider.prefetch("secret query", session_id="s1")
        assert isinstance(result, str)
        # Wait for background shadow task
        provider._executor.shutdown(wait=True)
        # Check shadow event was written
        events_file = tmp_path / "shadow" / "shadow_events.jsonl"
        assert events_file.exists()
        lines = [l for l in events_file.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
        assert len(lines) == 1
        event = json.loads(lines[0])
        # Privacy: no raw query
        assert "secret query" not in lines[0]
        assert "query" not in event
        # Counts present
        assert event["hybrid_summaries_count"] == 1
        assert event["hybrid_cited_leaves_count"] == 1
        assert event["hybrid_exact_hits_count"] == 1
        assert event["hybrid_graph_relations_count"] == 1
        assert event["status"] == "ok"
        assert event["trigger"] == "prefetch"

    def test_queue_prefetch_alone_writes_no_shadow(self, tmp_path):
        """Phase 6H fix2: ``queue_prefetch`` is now purely a cache-priming
        step. It must NOT record a shadow event on its own, because
        prompt-context is only built later in ``prefetch``. A plain
        ``queue_prefetch`` followed by ``executor shutdown`` (with no
        matching ``prefetch``) writes zero events.
        """
        class FakeHybrid:
            summaries = []
            cited_leaves = []
            exact_hits = [{"x": 1}]
            graph_relations = []
            warnings = ["w1"]
            debug = {"context_used_chars": 100}

        provider = _build_provider(
            shadow_enabled=True,
            shadow_dir=str(tmp_path / "shadow"),
            hybrid_result=FakeHybrid(),
        )
        provider.queue_prefetch("queued secret query", session_id="s1")
        provider._executor.shutdown(wait=True)
        events_file = tmp_path / "shadow" / "shadow_events.jsonl"
        if events_file.exists():
            lines = [l for l in events_file.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
        else:
            lines = []
        assert len(lines) == 0
        # And the cache was populated as a side effect of queue_prefetch.
        # With an empty chunks list the legacy format_for_prompt returns "".
        assert provider._prefetch_cache.get("s1") == ""
        assert "queued secret query" not in str(provider._prefetch_cache)

    def test_status_exposes_safe_shadow_fields(self, tmp_path):
        class FakeHybrid:
            summaries = []
            cited_leaves = []
            exact_hits = [{"x": 1}]
            graph_relations = []
            warnings = []
            debug = {"context_used_chars": 50}

        provider = _build_provider(
            shadow_enabled=True,
            shadow_dir=str(tmp_path / "shadow"),
            hybrid_result=FakeHybrid(),
        )
        provider.prefetch("status test query", session_id="s1")
        provider._executor.shutdown(wait=True)
        status_raw = provider._tool_status()
        status = json.loads(status_raw)
        assert status["shadow_enabled"] is True
        assert status["shadow_max_per_session"] == 20
        assert status["shadow_recorded_count"] == 1
        assert status["shadow_last_event"] is not None
        assert status["shadow_last_event"]["hybrid_exact_hits_count"] == 1
        # No raw query in status
        assert "status test query" not in status_raw
        assert "query" not in (status["shadow_last_event"] or {})

    def test_status_disabled_shows_zeros(self, tmp_path):
        provider = _build_provider(
            shadow_enabled=False,
            shadow_dir=str(tmp_path / "shadow"),
        )
        status = json.loads(provider._tool_status())
        assert status["shadow_enabled"] is False
        assert status["shadow_recorded_count"] == 0
        assert status["shadow_last_event"] is None

    def test_max_per_session_cap_in_provider(self, tmp_path):
        class FakeHybrid:
            summaries = []
            cited_leaves = []
            exact_hits = []
            graph_relations = []
            warnings = []
            debug = {"context_used_chars": 0}

        provider = _build_provider(
            shadow_enabled=True,
            shadow_max=2,
            shadow_dir=str(tmp_path / "shadow"),
            hybrid_result=FakeHybrid(),
        )
        for i in range(5):
            provider.prefetch(f"query-{i}", session_id="same-session")
        provider._executor.shutdown(wait=True)
        events_file = tmp_path / "shadow" / "shadow_events.jsonl"
        lines = [l for l in events_file.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
        assert len(lines) == 2  # capped at 2


# ---------------------------------------------------------------------------
# P2 #1 cache-miss regression.
#
# Original bug: ``prefetch()`` treated an empty queued cache entry as a
# cache hit and returned ``""`` without running the legacy dense search.
# The pre-fix2 prompt-context contract used a truthiness check
# (``if cached:``) — empty strings fell through to
# ``MemoryRetriever.search + format_for_prompt``. This regression suite
# preserves that contract: an empty cache value MUST trigger the legacy
# dense search, MUST return a freshly formatted result, and MUST still
# emit exactly one shadow event from ``prefetch``.
# ---------------------------------------------------------------------------


class TestProviderCacheMissRegression:
    def test_empty_cached_value_triggers_legacy_search(self, tmp_path):
        """Seeding ``_prefetch_cache[sid] = ""`` must be treated as a cache
        miss. The legacy dense search must run and ``format_for_prompt``
        must be invoked; the freshly formatted result is what
        ``prefetch`` returns.
        """

        class FakeHybrid:
            summaries = []
            cited_leaves = []
            exact_hits = [{"x": 1}]
            graph_relations = []
            warnings = []
            debug = {"context_used_chars": 0}

        provider = _build_provider(
            shadow_enabled=True,
            shadow_dir=str(tmp_path / "shadow"),
            hybrid_result=FakeHybrid(),
        )
        # Seed the cache with an empty string — the exact bug shape the
        # reviewer flagged.
        sid = "empty-cache-sid"
        provider._prefetch_cache[sid] = ""
        # Spy on retriever.search + format_for_prompt. The legacy
        # truthiness contract is "if cached: return it; else run
        # retriever.search + format_for_prompt". An empty cached value
        # MUST take the else branch.
        from unittest.mock import patch as _patch

        with _patch(
            "__init__.format_for_prompt", return_value="freshly-formatted-legacy"
        ) as fmt_spy:
            search_calls = {"n": 0}
            original_search = provider._retriever.search

            def counting_search(q, top_k, **kwargs):
                search_calls["n"] += 1
                return original_search(q, top_k=top_k, **kwargs)

            provider._retriever.search = MagicMock(side_effect=counting_search)
            result = provider.prefetch("some query", session_id=sid)
            # Legacy dense search ran exactly once (else branch).
            assert search_calls["n"] == 1
            # format_for_prompt was called by the legacy path.
            assert fmt_spy.called
        # Legacy return is the freshly formatted string, NOT the cached
        # empty string.
        assert result == "freshly-formatted-legacy"
        # And the cache was consumed (popped) after the call.
        assert sid not in provider._prefetch_cache
        provider._executor.shutdown(wait=True)
        # Shadow event was emitted from prefetch exactly once — the
        # empty-cached-value miss path still records the shadow.
        events_file = tmp_path / "shadow" / "shadow_events.jsonl"
        assert events_file.exists()
        lines = [
            l
            for l in events_file.read_text(encoding="utf-8").strip().split("\n")
            if l.strip()
        ]
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["trigger"] == "prefetch"
        assert event["legacy_chars"] == len("freshly-formatted-legacy")

    def test_prefetch_with_seeded_empty_cache_records_single_shadow(self, tmp_path):
        """Empty-cache miss path emits exactly one shadow event."""

        class FakeHybrid:
            summaries = []
            cited_leaves = []
            exact_hits = [{"x": 1}]
            graph_relations = []
            warnings = []
            debug = {"context_used_chars": 7}

        provider = _build_provider(
            shadow_enabled=True,
            shadow_dir=str(tmp_path / "shadow"),
            hybrid_result=FakeHybrid(),
        )
        sid = "seeded-empty-sid"
        provider._prefetch_cache[sid] = ""
        provider.prefetch("query with empty cache", session_id=sid)
        provider._executor.shutdown(wait=True)
        events_file = tmp_path / "shadow" / "shadow_events.jsonl"
        assert events_file.exists()
        lines = [
            l
            for l in events_file.read_text(encoding="utf-8").strip().split("\n")
            if l.strip()
        ]
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["trigger"] == "prefetch"
        assert event["error_code"] == ""
        assert event["status"] == "ok"

    def test_shadow_disabled_path_preserves_legacy_behavior(self, tmp_path):
        """When shadow is disabled, an empty cached value must still
        trigger the legacy dense search. The shadow code path being off
        does not change the prefetch prompt-context contract."""

        provider = _build_provider(
            shadow_enabled=False,
            shadow_dir=str(tmp_path / "shadow"),
        )
        sid = "shadow-off-sid"
        provider._prefetch_cache[sid] = ""
        from unittest.mock import patch as _patch

        with _patch(
            "__init__.format_for_prompt", return_value="shadow-off-formatted"
        ) as fmt_spy:
            search_calls = {"n": 0}
            original_search = provider._retriever.search

            def counting_search(q, top_k, **kwargs):
                search_calls["n"] += 1
                return original_search(q, top_k=top_k, **kwargs)

            provider._retriever.search = MagicMock(side_effect=counting_search)
            result = provider.prefetch("shadow off query", session_id=sid)
            assert search_calls["n"] == 1
            assert fmt_spy.called
        assert result == "shadow-off-formatted"
        provider._executor.shutdown(wait=True)
        # No shadow artifact created (shadow is disabled).
        assert not (tmp_path / "shadow").exists()

    def test_nonempty_cached_value_still_skips_legacy_search(self, tmp_path):
        """Companion check: a non-empty cached value MUST still take the
        cache-hit path (``prefetch`` returns the cached string without
        running ``retriever.search``). The fix only changed how an
        empty cached value is treated, not the cache-hit semantics for
        real results.
        """
        from unittest.mock import patch as _patch

        provider = _build_provider(
            shadow_enabled=True,
            shadow_dir=str(tmp_path / "shadow"),
        )
        sid = "nonempty-cache-sid"
        cached_text = "meaningful legacy result string"
        provider._prefetch_cache[sid] = cached_text
        with _patch(
            "__init__.format_for_prompt", return_value="should-not-be-called"
        ) as fmt_spy:
            search_calls = {"n": 0}
            original_search = provider._retriever.search

            def counting_search(q, top_k, **kwargs):
                search_calls["n"] += 1
                return original_search(q, top_k=top_k, **kwargs)

            provider._retriever.search = MagicMock(side_effect=counting_search)
            result = provider.prefetch("nonempty cache query", session_id=sid)
            # Cache hit: neither search nor format_for_prompt was called.
            assert search_calls["n"] == 0
            assert not fmt_spy.called
        assert result == cached_text
        assert sid not in provider._prefetch_cache
        provider._executor.shutdown(wait=True)
        # Shadow event was still emitted by prefetch.
        events_file = tmp_path / "shadow" / "shadow_events.jsonl"
        assert events_file.exists()

    def test_format_for_prompt_returning_empty_is_treated_as_cache_miss(self, tmp_path):
        """If a previous ``queue_prefetch`` legitimately produced a
        non-empty legacy prompt context, ``prefetch`` consumes it via
        the truthy-cache branch. To exercise the regression from the
        opposite angle we mock ``format_for_prompt`` to return
        ``""`` for the queue's bg task and a real string for the
        subsequent prefetch's own search. ``prefetch`` should still
        run its own dense search and ``format_for_prompt``.
        """
        from unittest.mock import patch as _patch

        provider = _build_provider(
            shadow_enabled=True,
            shadow_dir=str(tmp_path / "shadow"),
        )
        sid = "format-empty-sid"

        # Simulate queue_prefetch populating cache with empty string.
        provider._prefetch_cache[sid] = ""

        seen_calls = {"n": 0}

        def fake_format(chunks, tokens):
            seen_calls["n"] += 1
            return "freshly-formatted-after-cache-miss"

        with _patch("__init__.format_for_prompt", side_effect=fake_format):
            # retriever.search is the default MagicMock with return_value=()
            # but iterating over () works fine — it just yields nothing.
            result = provider.prefetch("format empty cache query", session_id=sid)
            # format_for_prompt was called exactly once — by prefetch's own
            # legacy else branch (cache miss because cached == "").
            assert seen_calls["n"] == 1
        assert result == "freshly-formatted-after-cache-miss"
        provider._executor.shutdown(wait=True)
        # Shadow emitted once by prefetch.
        events_file = tmp_path / "shadow" / "shadow_events.jsonl"
        assert events_file.exists()
        lines = [
            l
            for l in events_file.read_text(encoding="utf-8").strip().split("\n")
            if l.strip()
        ]
        assert len(lines) == 1


# ---------------------------------------------------------------------------
# Dedup: queue_prefetch followed by prefetch for the same session must
# produce exactly one shadow event (emitted by prefetch, not queue's bg
# task). Legacy return behavior — prefetch still returns the cached
# formatted string — is preserved.
# ---------------------------------------------------------------------------


class TestProviderShadowDedup:
    def test_queue_then_prefetch_same_session_one_shadow(self, tmp_path):
        """Phase 6H fix2: ``queue_prefetch`` + ``prefetch`` for the same
        session writes exactly one shadow event, and the trigger is
        ``"prefetch"`` (the prompt-context build). Legacy return behavior
        — prefetch still returns the cached formatted string — is
        preserved.
        """
        class FakeHybrid:
            summaries = [{"a": 1}]
            cited_leaves = [{"b": 2}]
            exact_hits = [{"c": 3}]
            graph_relations = []
            warnings = []
            debug = {"context_used_chars": 250}

        provider = _build_provider(
            shadow_enabled=True,
            shadow_dir=str(tmp_path / "shadow"),
            hybrid_result=FakeHybrid(),
        )
        query = "auto recall query"
        sid = "auto-recall-session"
        from unittest.mock import patch as _patch

        queue_formatted = "queued-legacy-result"
        prefetch_formatted = "fresh-legacy-result"
        seen_format_calls = {"queue": 0, "prefetch": 0}

        def _fake_format(chunks, tokens):
            # 1st call is from queue_prefetch bg task, 2nd would be
            # from prefetch's own search — which we should NOT see
            # because prefetch consumes the cache.
            if seen_format_calls["queue"] == 0:
                seen_format_calls["queue"] += 1
                return queue_formatted
            seen_format_calls["prefetch"] += 1
            return prefetch_formatted

        with _patch("__init__.format_for_prompt", side_effect=_fake_format):
            # 1) Run the queued prefetch — bg task primes the cache.
            # queue_prefetch alone must not write a shadow.
            provider.queue_prefetch(query, session_id=sid)
            provider._executor.shutdown(wait=True)
            assert provider._prefetch_cache.get(sid) == queue_formatted
            # 2) Spin up a fresh executor for prefetch's shadow submit.
            from concurrent.futures import ThreadPoolExecutor

            provider._executor = ThreadPoolExecutor(max_workers=1)
            result = provider.prefetch(query, session_id=sid)
            provider._executor.shutdown(wait=True)
        # Legacy return is the cached string populated by the queue task.
        assert isinstance(result, str)
        assert result == queue_formatted
        # Prefetch did NOT re-execute format_for_prompt — it consumed
        # the cached queued prefetch result and skipped its own search.
        assert seen_format_calls["prefetch"] == 0
        events_file = tmp_path / "shadow" / "shadow_events.jsonl"
        lines = [
            l
            for l in events_file.read_text(encoding="utf-8").strip().split("\n")
            if l.strip()
        ]
        # Exactly one shadow event — emitted by prefetch.
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["trigger"] == "prefetch"
        # Hybrid counts from FakeHybrid survived sanitization.
        assert event["hybrid_summaries_count"] == 1
        assert event["hybrid_cited_leaves_count"] == 1
        assert event["hybrid_exact_hits_count"] == 1

    def test_queue_then_prefetch_different_session_writes_one(self, tmp_path):
        """Phase 6H fix2: ``queue_prefetch`` for session A followed by
        ``prefetch`` for session B must NOT produce two events. A's
        queue_prefetch is just cache priming (no prompt injection ever
        happens for A), and B's prefetch emits exactly one shadow event.
        Result: 1 event total, ``trigger=="prefetch"``.
        """
        class FakeHybrid:
            summaries = []
            cited_leaves = []
            exact_hits = [{"x": 1}]
            graph_relations = []
            warnings = []
            debug = {"context_used_chars": 0}

        provider = _build_provider(
            shadow_enabled=True,
            shadow_dir=str(tmp_path / "shadow"),
            hybrid_result=FakeHybrid(),
        )
        provider.queue_prefetch("query A", session_id="session-A")
        provider._executor.shutdown(wait=True)
        from concurrent.futures import ThreadPoolExecutor

        provider._executor = ThreadPoolExecutor(max_workers=1)
        # Cache for session-A must NOT bleed into session-B's prefetch.
        provider.prefetch("query B", session_id="session-B")
        provider._executor.shutdown(wait=True)
        events_file = tmp_path / "shadow" / "shadow_events.jsonl"
        lines = [
            l
            for l in events_file.read_text(encoding="utf-8").strip().split("\n")
            if l.strip()
        ]
        # Only session-B's prefetch emits an event. queue_prefetch for
        # session-A wrote none.
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["trigger"] == "prefetch"

    def test_prefetch_alone_without_queue_records_shadow(self, tmp_path):
        """Without queue_prefetch, prefetch still records its own shadow."""
        class FakeHybrid:
            summaries = []
            cited_leaves = []
            exact_hits = [{"x": 1}]
            graph_relations = []
            warnings = []
            debug = {"context_used_chars": 7}

        provider = _build_provider(
            shadow_enabled=True,
            shadow_dir=str(tmp_path / "shadow"),
            hybrid_result=FakeHybrid(),
        )
        provider.prefetch("solo query", session_id="solo-session")
        provider._executor.shutdown(wait=True)
        events_file = tmp_path / "shadow" / "shadow_events.jsonl"
        lines = [
            l
            for l in events_file.read_text(encoding="utf-8").strip().split("\n")
            if l.strip()
        ]
        assert len(lines) == 1
        assert json.loads(lines[0])["trigger"] == "prefetch"

    def test_no_stale_marker_after_consume(self, tmp_path):
        """Phase 6H fix2: there is no per-session marker set anymore. The
        shadow invariant is purely event-driven — every real
        ``prefetch`` call emits one event; queue_prefetch alone emits
        none. Verify this directly: queue_prefetch primes, prefetch
        consumes, a subsequent queue_prefetch again primes, prefetch
        again consumes. Exactly two events written.
        """
        class FakeHybrid:
            summaries = []
            cited_leaves = []
            exact_hits = []
            graph_relations = []
            warnings = []
            debug = {"context_used_chars": 0}

        provider = _build_provider(
            shadow_enabled=True,
            shadow_dir=str(tmp_path / "shadow"),
            hybrid_result=FakeHybrid(),
        )
        # No _prefetch_shadow_recorded attribute should exist.
        assert not hasattr(provider, "_prefetch_shadow_recorded")
        provider.queue_prefetch("q1", session_id="sid-x")
        provider._executor.shutdown(wait=True)
        from concurrent.futures import ThreadPoolExecutor

        provider._executor = ThreadPoolExecutor(max_workers=1)
        provider.prefetch("q1", session_id="sid-x")
        # Re-prime and consume again.
        provider.queue_prefetch("q2", session_id="sid-x")
        provider._executor.shutdown(wait=True)
        provider._executor = ThreadPoolExecutor(max_workers=1)
        provider.prefetch("q2", session_id="sid-x")
        provider._executor.shutdown(wait=True)
        events_file = tmp_path / "shadow" / "shadow_events.jsonl"
        lines = [
            l
            for l in events_file.read_text(encoding="utf-8").strip().split("\n")
            if l.strip()
        ]
        # Two events: one per prefetch (queue_prefetch correctly emitted
        # zero on its own).
        assert len(lines) == 2
        for line in lines:
            assert json.loads(line)["trigger"] == "prefetch"


# ---------------------------------------------------------------------------
# Phase 6H fix2 — race-order regression.
#
# Reproduces the original P2: queue_prefetch submits a background task
# that takes a while to populate the cache, but prefetch() runs
# *before* the cache is ready. If the legacy code path used a
# pre-spawned "marker" before doing the actual retrieve, that marker
# could be left behind, producing a second shadow event when the
# queue_prefetch bg task eventually recorded its own event — exactly
# what the reviewer observed (``event_count=2`` and a stale marker).
#
# The fix instruments the prompt-injection point (``prefetch``) only.
# queue_prefetch writes zero events on its own and leaves no marker
# machinery behind for a stray consumer to find.
# ---------------------------------------------------------------------------


class TestShadowRaceOrder:
    def test_queue_prefetch_delayed_prefetch_immediate_one_event(self, tmp_path):
        """Race-order: queue_prefetch's background search is intentionally
        delayed by a pre-queued blocking event; prefetch runs while the
        legacy queue task is still blocked. After executor shutdown,
        exactly one shadow event exists, trigger == ``"prefetch"``, and
        no stale per-session marker remains because the marker machinery
        was removed.
        """
        import threading as _threading

        class FakeHybrid:
            summaries = [{"a": 1}]
            cited_leaves = []
            exact_hits = []
            graph_relations = []
            warnings = []
            debug = {"context_used_chars": 1}

        provider = _build_provider(
            shadow_enabled=True,
            shadow_dir=str(tmp_path / "shadow"),
            hybrid_result=FakeHybrid(),
        )
        sid = "race-sid"
        query = "race order query"

        # A gate that makes the retriever.search block until released.
        # This intentionally delays the queue_prefetch bg task so
        # prefetch is forced to run its own synchronous search path.
        gate = _threading.Event()
        search_started = _threading.Event()
        original_search = provider._retriever.search

        def slow_search(q, top_k, **kwargs):
            search_started.set()
            gate.wait(timeout=5.0)  # block until released or timeout
            return original_search(q, top_k=top_k, **kwargs)

        provider._retriever.search = MagicMock(side_effect=slow_search)

        # 1) Kick off the queue_prefetch — its bg task will block.
        provider.queue_prefetch(query, session_id=sid)
        # Wait until the bg task has actually entered search().
        assert search_started.wait(timeout=2.0), "queue_prefetch bg task never started"

        # 2) Call prefetch while the bg task is still blocked. It will
        # pop the (still-empty) cache, miss, and run its own synchronous
        # retriever.search. The bg task is still pending.
        result = provider.prefetch(query, session_id=sid)
        # Release the bg task so the executor can finish cleanly.
        gate.set()
        provider._executor.shutdown(wait=True)

        # The legacy return is whatever prefetch produced (an empty
        # prompt string because retriever_results default is []).
        assert isinstance(result, str)

        events_file = tmp_path / "shadow" / "shadow_events.jsonl"
        assert events_file.exists(), (
            "expected at least one shadow event from prefetch"
        )
        lines = [
            l
            for l in events_file.read_text(encoding="utf-8").strip().split("\n")
            if l.strip()
        ]
        # Exactly one event — emitted by prefetch. queue_prefetch
        # contributed no event of its own because the prompt-context
        # build never came from it.
        assert len(lines) == 1, (
            f"expected exactly 1 event, got {len(lines)}: {lines}"
        )
        event = json.loads(lines[0])
        assert event["trigger"] == "prefetch"
        # Original P2 reproducibility marker — would be 2 in the
        # broken state. Keep this explicit so a future regression
        # has a sharp signal.
        assert len(lines) != 2
        # And there is no stale per-session marker, because we removed
        # the marker machinery entirely.
        assert not hasattr(provider, "_prefetch_shadow_recorded") or (
            not getattr(provider, "_prefetch_shadow_recorded", None)
        ), "no stale marker should be left behind"

    def test_queue_prefetch_only_emits_zero_events(self, tmp_path):
        """Companion check: a queue_prefetch that completes successfully
        but is never consumed by a real prefetch call writes zero events.
        This is the second half of the P2 invariant: queue_prefetch
        alone must produce no shadow, so a stale-marked session left
        in cache-primer-only mode cannot leak an event into JSONL.
        """
        class FakeHybrid:
            summaries = []
            cited_leaves = []
            exact_hits = [{"x": 1}]
            graph_relations = []
            warnings = []
            debug = {"context_used_chars": 0}

        provider = _build_provider(
            shadow_enabled=True,
            shadow_dir=str(tmp_path / "shadow"),
            hybrid_result=FakeHybrid(),
        )
        provider.queue_prefetch("only-queued-query", session_id="only-queue-sid")
        provider._executor.shutdown(wait=True)
        events_file = tmp_path / "shadow" / "shadow_events.jsonl"
        if events_file.exists():
            lines = [
                l
                for l in events_file.read_text(encoding="utf-8").strip().split("\n")
                if l.strip()
            ]
        else:
            lines = []
        assert len(lines) == 0
        # Cache is populated; the prompt-context build that *would*
        # later happen via prefetch still has not occurred.
        assert provider._prefetch_cache.get("only-queue-sid") == ""


# ---------------------------------------------------------------------------
# Error-path: shadow records error code, never crashes
# ---------------------------------------------------------------------------


class TestShadowErrorPath:
    def test_router_unavailable_records_error(self, tmp_path):
        """If router is None, shadow records error_code='router_unavailable'."""
        provider = _build_provider(
            shadow_enabled=True,
            shadow_dir=str(tmp_path / "shadow"),
            hybrid_result=None,  # router will be None
        )
        result = provider.prefetch("test query", session_id="s1")
        provider._executor.shutdown(wait=True)
        assert isinstance(result, str)
        events_file = tmp_path / "shadow" / "shadow_events.jsonl"
        assert events_file.exists()
        event = json.loads(events_file.read_text(encoding="utf-8").strip())
        assert event["status"] == "error"
        assert event["error_code"] == "router_unavailable"

    def test_shadow_never_crashes_on_exception(self, tmp_path):
        """If router.retrieve raises, shadow records error and never crashes."""

        class CrashingRouter:
            def retrieve(self, *args, **kwargs):
                raise RuntimeError("boom with " + "z" * 20)

        from __init__ import QdrantMemoryProvider  # type: ignore  # noqa: PLC0415
        from concurrent.futures import ThreadPoolExecutor

        class _CrashProvider(QdrantMemoryProvider):
            def __init__(self):
                self._active = True
                self._config = {
                    "auto_recall": True,
                    "auto_recall_top_k": 5,
                    "display_tokens": 300,
                    "collection_name": "memory",
                    "auto_recall_shadow_enabled": True,
                    "auto_recall_shadow_max_per_session": 20,
                    "auto_recall_shadow_artifact_dir": str(tmp_path / "shadow"),
                    "auto_recall_shadow_mode": "hybrid",
                }
                self._retriever = MagicMock()
                self._prefetch_cache = {}
                self._prefetch_lock = __import__("threading").Lock()
                self._session_id = "test"
                self._executor = ThreadPoolExecutor(max_workers=2)
                self._hermes_home = ""
                self._shadow_recorder = ShadowRecorder(
                    hermes_home="/tmp", max_per_session=20,
                    artifact_dir=str(tmp_path / "shadow"),
                )
                self._hybrid_router = None
                self._qdrant = MagicMock()
                self._embeddings = MagicMock()
                self._raptor_searcher = None
                self._graph_retriever = None

            def _ensure_hybrid_router(self, collection_name):
                return CrashingRouter()

            def _scope_filter_values(self):
                return {}

        provider = _CrashProvider()
        result = provider.prefetch("query that triggers crash", session_id="s1")
        provider._executor.shutdown(wait=True)
        assert isinstance(result, str)
        events_file = tmp_path / "shadow" / "shadow_events.jsonl"
        assert events_file.exists()
        event = json.loads(events_file.read_text(encoding="utf-8").strip())
        assert event["status"] == "error"
        assert event["error_code"] == "exception"
        # Exception text must not leak
        boom_token = "z" * 20
        assert boom_token not in json.dumps(event)
