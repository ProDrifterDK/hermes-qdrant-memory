#!/usr/bin/env python3
"""End-to-end activation check for ``session_origin_mode: recorded`` (C'6).

Runs the provider in recorded mode through its real tool handlers (plus the real CLI
``store --cite`` path and the real backup restore) against a real Qdrant and a real
embedding server, citing real sessions and spans picked from the ``state.db`` you give it.

    python3 scripts/e2e_session_origin.py --state-db ~/.hermes/state.db \\
        [--qdrant-url http://127.0.0.1:6333] [--embedding-url http://127.0.0.1:8080/v1] \\
        [--embedding-model bge-m3] [--prefix tmp_c6_] [--out receipt.json]

    python3 scripts/e2e_session_origin.py --make-synthetic-state-db PATH   # a fixture to try it on

Safety:
- Writes only collections it creates, named ``<prefix><run>_memory`` / ``_learnings``; the
  prefix must start with ``tmp_c6_``. They are deleted at the end, also on failure, and
  their absence is verified. Every Qdrant client it uses refuses writes to the production
  collections (``hermes_memory``, ``hermes_learnings``, ``hermes_session_windows``).
- ``state.db`` is only ever opened read-only (``mode=ro``, or ``immutable=1`` when no
  Hermes process has it open), through the plugin's own reader. Nothing is written next
  to it; the directory listing is compared before and after.
- No Hermes config is read: the provider gets an explicit config (printed in the receipt)
  and a temporary HERMES_HOME; ``HERMES_QDRANT_MEMORY_*`` variables are ignored.
- The receipt carries ids, digests and statuses only, never message text.

Capture CP1 removed the session-hook producers (learning heuristic, source-line hook
extractor) and the extraction/learning preview and approve tools that read their in-memory
queue. The two checks that went through them now build the candidate the hook used to build
(same extractor, capture frozen in the source session) and call the surviving write paths,
``approve_extraction_candidate`` / ``approve_learning_candidate``. The receipt lists them under
``cp1_changed_checks``.

Prints a JSON receipt with one entry per check and exits non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import traceback
import urllib.parse
import uuid
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Callable

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PREFIX = "tmp_c6_"
FORBIDDEN = frozenset({"hermes_memory", "hermes_learnings", "hermes_session_windows", "resystbot_learnings",
                       "picoclaw_memory", "hermes_memory_local27b", "hermes_learnings_local27b",
                       "hermes_session_windows_local27b"})
PROFILE = "c6-e2e"
MAX_SCAN_SESSIONS = 400
MAX_HOOK_ROWS = 80
MAX_MESSAGE_CHARS = 6000
CP1_CHANGED_CHECKS = {
    "extraction_after_switch_captured_session_wins": (
        "was hook on_pre_compress -> qdrant_memory_extraction_preview/_approve tools; now the same source "
        "extractor builds the candidate from a real row, the capture is frozen in its session, and "
        "approve_extraction_candidate (the surviving write path) approves it after the session switch"),
    "learning_approval_then_store_merges": (
        "was the removed learning heuristic -> qdrant_learning_preview/_approve tools; now a learning candidate "
        "whose source is one real user row (lesson text synthetic) goes through approve_learning_candidate"),
    "off_mode_restores_keep_origin": (
        "the extraction replay goes through approve_extraction_candidate instead of the removed approve tool"),
}


# --- independent digests (design 5.2 / G1 byte formats, not the module under test) ------


def _b(value: Any) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    return str(value).encode("utf-8", "surrogatepass")


def expected_span_digests(sid: str, rows: list[tuple]) -> dict[str, Any]:
    """``rows``: ``(id, role, content, message_uid)`` of the cited eligible rows, id order."""
    span, content, uids = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    for row_id, role, text, uid in rows:
        span.update(b"hermes-span-v1\x1f" + _b(sid) + b"\x1f" + str(int(row_id)).encode() + b"\x1f" + _b(role)
                    + b"\x1f" + _b(text) + b"\x1e")
        content.update(b"hermes-span-content-v1\x1f" + _b(sid) + b"\x1f" + _b(uid) + b"\x1f" + _b(role) + b"\x1f"
                       + _b(text) + b"\x1e")
        uids.update(b"hermes-uids-v1\x1f" + _b(uid) + b"\x1e")
    return {"session_id": sid, "first_msg_id": int(rows[0][0]), "last_msg_id": int(rows[-1][0]),
            "msg_count": len(rows), "span_sha256": span.hexdigest(), "content_sha256": content.hexdigest(),
            "uids_sha256": uids.hexdigest()}


def pinned(citation: dict[str, Any]) -> dict[str, Any]:
    keys = ("session_id", "first_msg_id", "last_msg_id", "msg_count", "span_sha256", "content_sha256", "uids_sha256")
    return {key: citation.get(key) for key in keys}


def short(value: Any) -> Any:
    return value[:16] if isinstance(value, str) and len(value) == 64 else value


# --- synthetic state.db ----------------------------------------------------------------


SYN_SESSIONS = ["local-c6e2e-aaaa", "local-c6e2e-bbbb", "local-c6e2e-xxxx", "local-c6e2e-llll", "local-c6e2e-iiii",
                "local-c6e2e-tttt"]


def make_synthetic_state_db(path: Path) -> None:
    """A state.db with the host's columns (tests/fixtures/state_db_schema.sql) and synthetic rows."""
    schema = PLUGIN_ROOT / "tests" / "fixtures" / "state_db_schema.sql"
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    a, b, x, lrn, imp, trn = SYN_SESSIONS
    rows = [
        (a, "user", "Where should the nightly backup land?"),
        (a, "assistant", "On the second disk, under /srv/backup, rotated weekly."),
        (a, "user", "Good, keep that."),
        (b, "user", "Remind me where the nightly backup lands."),
        (b, "assistant", "It lands on the second disk under /srv/backup."),
        (x, "user", "Let us settle the proxy plan."),
        (x, "assistant", "Decision: route staging traffic through the blue proxy pool.\nI will note it."),
        (x, "user", "Thanks."),
        (lrn, "assistant", "I restarted the worker with the default queue."),
        (lrn, "user", "Actually, always restart the worker with the priority queue, not the default one."),
        (lrn, "assistant", "Understood, the priority queue from now on."),
        (imp, "user", "What about the scheduler?"),
        (imp, "assistant", "Decision: pin the scheduler to two workers.\nThat keeps it stable."),
        (trn, "user", "And the cache?"),
        (trn, "assistant", "Decision: keep the cache warm for ten minutes after deploys."),
    ]
    conn = sqlite3.connect(path)
    try:
        conn.executescript(schema.read_text(encoding="utf-8"))
        t0 = 1_794_000_000.5
        for index, sid in enumerate(SYN_SESSIONS):
            conn.execute("INSERT INTO sessions (id, source, started_at) VALUES (?, 'cli', ?)", (sid, t0 + index))
        for index, (sid, role, content) in enumerate(rows):
            conn.execute("INSERT INTO messages (session_id, role, content, timestamp, message_uid) VALUES (?, ?, ?, ?, ?)",
                         (sid, role, content, t0 + 60.0 * index, f"c6-{index:03d}"))
        conn.commit()
    finally:
        conn.close()


def learning_candidate(run: str, line: str):
    """A learning candidate whose source is ``line`` (one real user row); the lesson is synthetic."""
    from qdrant_memory.lesson_extractor import LearningCandidate

    return LearningCandidate(lesson=f"C6 e2e learning {run}: keep the citation on the exact source row.",
                             learning_type="environment_quirk", trigger=f"c6 e2e {run}",
                             correction="cite the exact source row", evidence="C6 e2e synthetic lesson.",
                             confidence=0.9, importance=7, source_hook="on_pre_compress", source_text=line,
                             source_role="user")


# --- receipt ---------------------------------------------------------------------------


class Receipt:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(self, name: str, passed: bool, **observed: Any) -> bool:
        self.checks.append({"check": name, "pass": bool(passed), "observed": observed})
        return bool(passed)

    def crash(self, name: str, exc: BaseException) -> None:
        self.checks.append({"check": name, "pass": False,
                            "observed": {"exception": f"{type(exc).__name__}: {exc}",
                                         "traceback_tail": traceback.format_exc().splitlines()[-6:]}})


def run_check(receipt: Receipt, name: str, fn: Callable[[], None]) -> None:
    try:
        fn()
    except (Exception, SystemExit) as exc:  # one broken check must not hide the others or skip cleanup
        receipt.crash(name, exc)


# --- the run ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--state-db", default="")
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--embedding-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--embedding-model", default="bge-m3")
    parser.add_argument("--prefix", default=REQUIRED_PREFIX)
    parser.add_argument("--out", default="")
    parser.add_argument("--make-synthetic-state-db", default="")
    args = parser.parse_args(argv)

    if args.make_synthetic_state_db:
        make_synthetic_state_db(Path(args.make_synthetic_state_db).expanduser())
        print(json.dumps({"synthetic_state_db": str(Path(args.make_synthetic_state_db).expanduser())}))
        return 0
    if not args.state_db:
        parser.error("--state-db is required")
    if not args.prefix.startswith(REQUIRED_PREFIX):
        parser.error(f"--prefix must start with {REQUIRED_PREFIX!r}")
    state_db = str(Path(args.state_db).expanduser().resolve())
    if not Path(state_db).is_file():
        parser.error(f"state.db not found: {state_db}")

    for key in list(os.environ):
        if key.startswith("HERMES_QDRANT_MEMORY_"):
            del os.environ[key]
    home = Path(tempfile.mkdtemp(prefix="c6-e2e-home-", dir=os.environ.get("TMPDIR") or None))
    os.environ["HERMES_HOME"] = str(home)
    sys.path.insert(0, str(PLUGIN_ROOT))

    run = uuid.uuid4().hex[:10]
    names = {"memory": f"{args.prefix}{run}_memory", "learnings": f"{args.prefix}{run}_learnings"}
    for name in names.values():
        if not name.startswith(REQUIRED_PREFIX) or name in FORBIDDEN:
            raise SystemExit(f"refusing collection name {name!r}")

    receipt = Receipt()
    report: dict[str, Any] = {"e2e": "C'6 session_origin activation check", "run": run,
                              "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                              "state_db": state_db, "collections": names, "hermes_home": str(home),
                              "checks": receipt.checks}
    state_dir_before = sorted(p.name for p in Path(state_db).parent.iterdir() if p.name.startswith(Path(state_db).name))
    state_sha_before = _small_file_sha256(state_db)

    from qdrant_memory.client import QdrantClient

    admin = QdrantClient(args.qdrant_url, timeout=60.0, read_only_collections=FORBIDDEN)
    created: list[str] = []
    try:
        existing = set(admin.get_collections())
        clash = sorted(set(names.values()) & existing)
        if clash:
            raise SystemExit(f"refusing to reuse existing collections {clash}")
        ctx = Context(args, state_db, home, names, admin, receipt, report)
        ctx.prepare()
        for name in (names["memory"], names["learnings"]):
            if admin.ensure_collection(name, ctx.vector_size, "Cosine").get("exists"):
                raise SystemExit(f"collection {name} appeared concurrently; refusing")
            created.append(name)
        ctx.run_all()
    except BaseException as exc:  # noqa: BLE001 - report, then always clean up
        receipt.crash("run", exc)
    finally:
        cleanup = _cleanup(admin, created)
        receipt.check("cleanup_collections_deleted", cleanup["absent"], **cleanup)
        state_dir_after = sorted(p.name for p in Path(state_db).parent.iterdir()
                                 if p.name.startswith(Path(state_db).name))
        state_sha_after = _small_file_sha256(state_db)
        receipt.check("state_db_untouched", state_dir_before == state_dir_after
                      and (state_sha_before is None or state_sha_before == state_sha_after),
                      files_before=state_dir_before, files_after=state_dir_after,
                      sha256_compared=state_sha_before is not None,
                      sha256_equal=None if state_sha_before is None else state_sha_before == state_sha_after,
                      opened="read-only through qdrant_memory.session_origin._read (mode=ro / immutable=1)")
        shutil.rmtree(home, ignore_errors=True)

    passed = sum(1 for item in receipt.checks if item["pass"])
    report.update({"finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "passed": passed, "failed": len(receipt.checks) - passed, "total": len(receipt.checks),
                   "ok": passed == len(receipt.checks)})
    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["ok"] else 1


def _small_file_sha256(path: str) -> str | None:
    """sha256 of a state.db small enough to hash quickly (fixtures); ``None`` for a live one."""
    if Path(path).stat().st_size > 64 * 1024 * 1024:
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cleanup(admin: Any, created: list[str]) -> dict[str, Any]:
    errors = []
    for name in reversed(created):
        if not name.startswith(REQUIRED_PREFIX) or name in FORBIDDEN:
            errors.append(f"refused to delete {name}")
            continue
        try:
            admin._request("DELETE", f"/collections/{urllib.parse.quote(name, safe='')}")
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    try:
        remaining = sorted(set(created) & set(admin.get_collections()))
    except Exception as exc:
        return {"absent": False, "created": created, "errors": errors + [f"listing failed: {exc}"]}
    return {"absent": not remaining and not errors, "created": created, "remaining": remaining, "errors": errors}


class Context:
    """Everything one run needs; each ``check_*`` method records exactly one check."""

    def __init__(self, args, state_db: str, home: Path, names: dict[str, str], admin: Any, receipt: Receipt,
                 report: dict[str, Any]) -> None:
        self.args, self.state_db, self.home, self.names = args, state_db, home, names
        self.admin, self.receipt, self.report = admin, receipt, report
        self.points: dict[str, str] = {}
        self.spans: dict[str, Any] = {}

    # --- setup ---------------------------------------------------------------------

    def config(self, mode: str = "recorded") -> dict[str, Any]:
        return {
            "qdrant_url": self.args.qdrant_url,
            "embedding_url": self.args.embedding_url,
            "embedding_model": self.args.embedding_model,
            "collection_name": self.names["memory"],
            "learning_collection_name": self.names["learnings"],
            "session_recall_collection": self.names["memory"] + "_windows_unused",
            "session_recall_state_db": self.state_db,
            "session_origin_mode": mode,
            "vector_size": self.vector_size,
            "lineage_lock_dir": str(self.home / "locks"),
            "sync_turns": False,
        }

    def prepare(self) -> None:
        from qdrant_memory.embeddings import EmbeddingClient

        self.embeddings = EmbeddingClient(self.args.embedding_url, self.args.embedding_model, timeout=120.0)
        self.vector_size = len(self.embeddings.embed_query("c6 e2e probe"))
        effective = self.provider("e2e-config-probe")._config
        shown = {key: effective.get(key) for key in sorted(self.config())}
        self.report["effective_config"] = shown
        self.report["hermes_config_read"] = False
        from qdrant_memory.session_origin import origin_size  # noqa: F401 - import proves the plugin path
        self.report["plugin_root"] = str(PLUGIN_ROOT)
        self.report["cp1_changed_checks"] = CP1_CHANGED_CHECKS
        self.picks = Picker(self.state_db, self).pick()
        self.report["picked"] = self.picks["summary"]

    def provider(self, session_id: str, mode: str = "recorded"):
        import importlib

        from qdrant_memory.client import QdrantClient
        from qdrant_memory.config import load_config
        from qdrant_memory.writer import ConversationWriter

        plugin = importlib.import_module("__init__")
        provider = plugin.QdrantMemoryProvider()
        provider._active = True
        provider._hermes_home = str(self.home)
        provider._profile_id = PROFILE
        provider._platform = "e2e"
        provider._session_id = session_id
        provider._config = load_config(hermes_home=str(self.home), hermes_config={"qdrant_memory": self.config(mode)})
        for key in ("collection_name", "learning_collection_name"):
            if not str(provider._config[key]).startswith(REQUIRED_PREFIX):
                raise SystemExit(f"{key} resolved to {provider._config[key]!r}; refusing")
        provider._qdrant = QdrantClient(self.args.qdrant_url, timeout=60.0, read_only_collections=FORBIDDEN)
        provider._embeddings = getattr(self, "embeddings", None)
        provider._writer = ConversationWriter(qdrant=provider._qdrant, embeddings=provider._embeddings,
                                              collection_name=self.names["memory"], profile_id=PROFILE,
                                              platform="e2e", session_id=session_id)
        return provider

    @staticmethod
    def call(provider, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        return json.loads(provider.handle_tool_call(tool, args))

    def payload(self, collection: str, point_id: str) -> dict[str, Any] | None:
        points = self.admin.retrieve(self.names[collection], [point_id], with_payload=True)
        return points[0]["payload"] if points else None

    def origin(self, collection: str, point_id: str) -> dict[str, Any]:
        return (self.payload(collection, point_id) or {}).get("session_origin") or {}

    def approver_for(self, captured: str) -> str:
        """A session other than ``captured`` to approve from (the compression continuation)."""
        span_b = self.picks["span_b"]["session_id"]
        return span_b if span_b != captured else self.picks["span_a"]["session_id"]

    @staticmethod
    def citation_as_expected(origin: dict[str, Any], pick: dict[str, Any]) -> bool:
        """A verified line citation equal to the independent one, or the honest fallback status."""
        cited = [pinned(c) for c in origin.get("citations") or []]
        if pick["expected_status"] == "verified":
            return origin.get("citation_status") == "verified" and cited == [pick["expected"]]
        return origin.get("citation_status") == pick["expected_status"] and cited == []

    def store_with_cite(self, provider, text: str, span: dict[str, Any]) -> tuple[dict, dict]:
        cite = {"session_id": span["session_id"], "first_msg_id": span["first_msg_id"],
                "last_msg_id": span["last_msg_id"]}
        dry = self.call(provider, "qdrant_memory_store", {"text": text, "cite": cite})
        live = self.call(provider, "qdrant_memory_store", {"text": text, "cite": cite, "dry_run": False,
                                                           "approve": True})
        return dry, live

    # --- checks --------------------------------------------------------------------

    def run_all(self) -> None:
        steps = [
            ("manual_store_cite_exact_digest", self.check_manual_store_cite),
            ("cite_without_preview_refused", self.check_cite_without_preview),
            ("tampered_cite_refused", self.check_tampered_cite),
            ("backup_snapshot_taken", self.check_backup_snapshot),
            ("restore_from_second_session_merges", self.check_restore_second_session),
            ("restore_keeps_origin", self.check_restore),
            ("store_without_cite_tool_position", self.check_tool_position),
            ("extraction_after_switch_captured_session_wins", self.check_extraction_switch),
            ("learning_approval_then_store_merges", self.check_learning),
            ("off_mode_restores_keep_origin", self.check_off_mode_restores),
            ("improve_source_text_cites_after_restart", self.check_improve_real),
            ("improve_legacy_turn_no_citation", self.check_improve_legacy_turn),
            ("consolidation_merge_merges_citations", self.check_consolidation),
            ("cli_store_cite_preview_then_approve", self.check_cli_cite),
            ("origin_check_counts_and_ids", self.check_origin_check),
            ("trace_and_inspect_disclose_origin", self.check_trace_inspect),
            ("forget_deletes_cited_point", self.check_forget),
        ]
        for name, fn in steps:
            run_check(self.receipt, name, fn)

    def check_manual_store_cite(self) -> None:
        span = self.picks["span_a"]
        provider = self.provider(span["session_id"])
        text = f"C6 e2e fact {self.report['run']}: the nightly backup lands on the second disk."
        dry, live = self.store_with_cite(provider, text, span)
        point_id = str(live.get("id") or "")
        self.points["manual"], self.spans["manual_text"] = point_id, text
        origin = self.origin("memory", point_id)
        stored = (origin.get("citations") or [{}])[0]
        expected = span["expected"]
        payload = self.payload("memory", point_id) or {}
        self.receipt.check(
            "manual_store_cite_exact_digest",
            dry.get("session_origin", {}).get("citation_status") == "verified"
            and pinned(dry["session_origin"]["citation"]) == expected
            and live.get("saved") is True and pinned(stored) == expected and stored.get("basis") == "agent_proposed"
            and origin.get("producer") == "manual_store" and origin.get("captured_session_id") == span["session_id"]
            and origin.get("citation_status") == "verified" and payload.get("session_id") == span["session_id"],
            point_id=point_id, preview_status=dry.get("session_origin", {}).get("citation_status"),
            stored={key: short(value) for key, value in pinned(stored).items()},
            expected={key: short(value) for key, value in expected.items()},
            captured_session_id=origin.get("captured_session_id"), legacy_session_id=payload.get("session_id"))

    def check_cite_without_preview(self) -> None:
        span = self.picks["span_b"]
        provider = self.provider(span["session_id"])
        text = f"C6 e2e unpreviewed {self.report['run']}: never stored."
        cite = {"session_id": span["session_id"], "first_msg_id": span["first_msg_id"],
                "last_msg_id": span["last_msg_id"]}
        live = self.call(provider, "qdrant_memory_store", {"text": text, "cite": cite, "dry_run": False,
                                                           "approve": True})
        preview_id = str(self.call(provider, "qdrant_memory_store", {"text": text}).get("id") or "")
        self.receipt.check("cite_without_preview_refused",
                           "needs a preview" in str(live.get("error") or "") and not self.payload("memory", preview_id),
                           error=live.get("error"), point_written=bool(self.payload("memory", preview_id)))

    def check_tampered_cite(self) -> None:
        span_a, span_b = self.picks["span_a"], self.picks["span_b"]
        provider = self.provider(span_b["session_id"])
        text = f"C6 e2e tampered {self.report['run']}: never stored."
        # Session B with session A's message ids: those rows are not B's.
        cite = {"session_id": span_b["session_id"], "first_msg_id": span_a["first_msg_id"],
                "last_msg_id": span_a["last_msg_id"]}
        dry = self.call(provider, "qdrant_memory_store", {"text": text, "cite": cite})
        live = self.call(provider, "qdrant_memory_store", {"text": text, "cite": cite, "dry_run": False,
                                                           "approve": True})
        preview_id = str(self.call(provider, "qdrant_memory_store", {"text": text}).get("id") or "")
        self.receipt.check("tampered_cite_refused",
                           "cite refused (unresolved:no_match)" in str(dry.get("error") or "")
                           and bool(live.get("error")) and not self.payload("memory", preview_id),
                           preview_error=dry.get("error"), approve_error=live.get("error"))

    def check_backup_snapshot(self) -> None:
        from qdrant_memory.backup import create_backup

        provider = self.provider(self.picks["span_a"]["session_id"])
        with redirect_stdout(io.StringIO()):
            backup = create_backup(provider._qdrant, provider._config, hermes_home=str(self.home), scope="both")
        self.spans["backup_id"] = backup.get("backup_id")
        self.receipt.check("backup_snapshot_taken", bool(self.spans["backup_id"]),
                           backup_id=self.spans["backup_id"],
                           snapshot_citations=len(self.origin("memory", self.points["manual"]).get("citations") or []))

    def check_restore_second_session(self) -> None:
        span_a, span_b = self.picks["span_a"], self.picks["span_b"]
        before = self.origin("memory", self.points["manual"])
        time.sleep(1.1)  # a later captured_at would win if the first capture were not kept
        provider = self.provider(span_a["session_id"])
        provider.on_session_switch(span_b["session_id"], reset=True)
        dry, live = self.store_with_cite(provider, self.spans["manual_text"], span_b)
        after = self.origin("memory", self.points["manual"])
        cited = [pinned(c) for c in after.get("citations") or []]
        self.receipt.check(
            "restore_from_second_session_merges",
            live.get("saved") is True and live.get("id") == self.points["manual"]
            and cited == [span_a["expected"], span_b["expected"]]
            and after.get("captured_session_id") == span_a["session_id"]
            and after.get("captured_at") == before.get("captured_at")
            and (self.payload("memory", self.points["manual"]) or {}).get("session_id") == span_a["session_id"],
            citations=[(c["session_id"], c["first_msg_id"], c["last_msg_id"]) for c in cited],
            captured_session_id=after.get("captured_session_id"),
            captured_at_kept=after.get("captured_at") == before.get("captured_at"),
            preview_merged_count=dry.get("session_origin", {}).get("merged_citation_count"))

    def check_restore(self) -> None:
        from qdrant_memory.backup import restore_backup

        provider = self.provider(self.picks["span_a"]["session_id"])
        before = self.origin("memory", self.points["manual"])
        with redirect_stdout(io.StringIO()):
            plan = restore_backup(provider._qdrant, provider._config, hermes_home=str(self.home),
                                  backup_id=self.spans["backup_id"], dry_run=True)
            applied = restore_backup(provider._qdrant, provider._config, hermes_home=str(self.home),
                                     backup_id=self.spans["backup_id"], dry_run=False)
        after = self.origin("memory", self.points["manual"])
        memory_plan = applied.get("collections", {}).get("memory", {})
        self.receipt.check(
            "restore_keeps_origin",
            applied.get("applied") is True and self.points["manual"] in memory_plan.get("changed_ids", [])
            and self.points["manual"] in memory_plan.get("session_origin_kept_ids", [])
            and after == before and len(after.get("citations") or []) == 2,
            restored_ids=memory_plan.get("changed_ids"), kept_ids=memory_plan.get("session_origin_kept_ids"),
            dry_run_kept_ids=plan.get("collections", {}).get("memory", {}).get("session_origin_kept_ids"),
            citations_after=len(after.get("citations") or []), mapping_unchanged=after == before)

    def check_tool_position(self) -> None:
        span = self.picks["span_a"]
        provider = self.provider(span["session_id"])
        text = f"C6 e2e uncited fact {self.report['run']}: the build cache is pruned on Sundays."
        dry = self.call(provider, "qdrant_memory_store", {"text": text})
        live = self.call(provider, "qdrant_memory_store", {"text": text, "dry_run": False, "approve": True})
        self.points["position"] = str(live.get("id") or "")
        origin = self.origin("memory", self.points["position"])
        citation = (origin.get("citations") or [{}])[0]
        self.receipt.check(
            "store_without_cite_tool_position",
            live.get("saved") is True and citation.get("basis") == "tool_position"
            and citation.get("session_id") == span["session_id"]
            and citation.get("after_msg_id") >= span["max_eligible_id"] and origin.get("citation_status") == "verified",
            point_id=self.points["position"], basis=citation.get("basis"), after_msg_id=citation.get("after_msg_id"),
            max_eligible_id_at_pick=span["max_eligible_id"], preview_status=dry.get("session_origin", {}).get(
                "citation_status"))

    def check_extraction_switch(self) -> None:
        pick = self.picks["extraction"]
        approver = self.approver_for(pick["session_id"])
        provider = self.provider(pick["session_id"])
        candidate = dataclasses.replace(pick["candidate"], origin_capture=provider._freeze_origin_capture(
            "on_pre_compress", pick["line"], pick["role"]))
        provider.on_session_switch(approver, parent_session_id=pick["session_id"], reset=False, reason="compression")
        # Candidate ids carry the payload's created_at, so a re-extraction mints a new id; keep
        # this one to replay the same id later (the off-mode overwrite case).
        self.spans["extraction_candidate"] = candidate
        dry = json.loads(provider.approve_extraction_candidate(candidate))
        live = json.loads(provider.approve_extraction_candidate(candidate, dry_run=False, approve=True))
        self.points["extraction"] = str(live.get("id") or "")
        origin = self.origin("memory", self.points["extraction"])
        cited = [pinned(c) for c in origin.get("citations") or []]
        payload = self.payload("memory", self.points["extraction"]) or {}
        self.receipt.check(
            "extraction_after_switch_captured_session_wins",
            live.get("saved") is True and origin.get("producer") == "source_extraction"
            and origin.get("captured_session_id") == pick["session_id"] and origin.get("approved_session_id") == approver
            and self.citation_as_expected(origin, pick)
            and dry.get("session_origin", {}).get("captured_session_id") == pick["session_id"]
            and payload.get("session_id") == pick["session_id"],
            point_id=self.points["extraction"], captured_session_id=origin.get("captured_session_id"),
            approved_session_id=origin.get("approved_session_id"), legacy_session_id=payload.get("session_id"),
            citation_status=origin.get("citation_status"), expected_status=pick["expected_status"],
            fallback=pick.get("fallback"),
            cited_row=(cited[0]["session_id"], cited[0]["first_msg_id"]) if cited else None,
            expected_row=(pick["expected"]["session_id"], pick["expected"]["first_msg_id"]) if pick["expected"] else None)

    def check_learning(self) -> None:
        pick = self.picks["learning"]
        approver = self.approver_for(pick["session_id"])
        provider = self.provider(pick["session_id"])
        candidate = learning_candidate(self.report["run"], pick["line"])
        candidate.origin_capture = provider._freeze_origin_capture("on_pre_compress", pick["line"], "user")
        provider.on_session_switch(approver, parent_session_id=pick["session_id"], reset=False, reason="compression")
        dry = json.loads(provider.approve_learning_candidate(candidate))
        live = json.loads(provider.approve_learning_candidate(candidate, dry_run=False))
        point_id = str(live.get("id") or "")
        self.points["learning"] = point_id
        first = self.origin("learnings", point_id)
        time.sleep(1.1)
        store_args = dict(dry["learning_args"])
        restored = self.call(provider, "qdrant_learning_store", store_args)
        merged = self.origin("learnings", point_id)
        citations = merged.get("citations") or []
        self.self_learning_args = store_args
        self.receipt.check(
            "learning_approval_then_store_merges",
            live.get("saved") is True and restored.get("id") == point_id
            and first.get("captured_session_id") == pick["session_id"] and first.get("approved_session_id") == approver
            and [pinned(c) for c in first.get("citations") or []] == [pick["expected"]]
            and len(citations) == 2 and pinned(citations[0]) == pick["expected"]
            and citations[1].get("basis") == "tool_position" and citations[1].get("session_id") == approver
            and merged.get("producer") == "learning_approval" and merged.get("captured_at") == first.get("captured_at"),
            point_id=point_id, captured_session_id=first.get("captured_session_id"),
            approved_session_id=first.get("approved_session_id"),
            citations=[(c.get("session_id"), c.get("basis")) for c in citations],
            captured_at_kept=merged.get("captured_at") == first.get("captured_at"))

    def check_off_mode_restores(self) -> None:
        span = self.picks["span_a"]
        off = self.provider(span["session_id"], mode="off")
        manual_before = self.origin("memory", self.points["manual"])
        learning_before = self.origin("learnings", self.points["learning"])
        extraction_before = self.origin("memory", self.points["extraction"])
        stored = self.call(off, "qdrant_memory_store", {"text": self.spans["manual_text"], "dry_run": False,
                                                        "approve": True})
        learned = self.call(off, "qdrant_learning_store", dict(self.self_learning_args))
        pick = self.picks["extraction"]
        off_x = self.provider(pick["session_id"], mode="off")
        replay = self.spans["extraction_candidate"]  # the same candidate id lands on the stored point
        dry = json.loads(off_x.approve_extraction_candidate(replay))
        approved = json.loads(off_x.approve_extraction_candidate(replay, dry_run=False, approve=True))
        self.receipt.check(
            "off_mode_restores_keep_origin",
            stored.get("saved") is True and learned.get("saved") is True and approved.get("saved") is True
            and dry.get("dry_run") is True and approved.get("id") == self.points["extraction"]
            and self.origin("memory", self.points["manual"]) == manual_before
            and self.origin("learnings", self.points["learning"]) == learning_before
            and self.origin("memory", self.points["extraction"]) == extraction_before
            and "session_origin" not in stored and "session_origin" not in approved,
            manual_kept=self.origin("memory", self.points["manual"]) == manual_before,
            learning_kept=self.origin("learnings", self.points["learning"]) == learning_before,
            extraction_kept=self.origin("memory", self.points["extraction"]) == extraction_before)

    def check_improve_real(self) -> None:
        pick = self.picks["improve"]
        first = self.provider(pick["session_id"])
        report = self.call(first, "qdrant_memory_improve_preview",
                           {"source_text": pick["line"], "session_id": pick["session_id"],
                            "source_uri": "session://current/improve-source"})
        item = next(c for c in report["candidates"]
                    if c.get("would_store") and c["proposed_payload"].get("text") == pick["text"])
        previewed_by = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        time.sleep(1.1)  # an apply that lost the capture would stamp a later captured_at
        # A fresh provider: the in-memory capture is gone, as after a process restart.
        second = self.provider(pick["session_id"])
        dry = self.call(second, "qdrant_memory_improve_apply", {"report_id": report["report_id"],
                                                                "candidate_id": item["candidate_id"]})
        live = self.call(second, "qdrant_memory_improve_apply", {"report_id": report["report_id"],
                                                                 "candidate_id": item["candidate_id"],
                                                                 "dry_run": False, "approve": True})
        point_id = str(live.get("id") or item.get("target_point_id") or "")
        self.points["improve"] = point_id
        origin = self.origin("memory", point_id)
        cited = [pinned(c) for c in origin.get("citations") or []]
        self.receipt.check(
            "improve_source_text_cites_after_restart",
            live.get("applied") is True and origin.get("producer") == "improve"
            and origin.get("captured_session_id") == pick["session_id"] and self.citation_as_expected(origin, pick)
            and dry.get("session_origin", {}).get("citation_status") == pick["expected_status"]
            and str(origin.get("captured_at") or "9") <= previewed_by,
            point_id=point_id, report_id=report["report_id"], dry_run_status=dry.get("session_origin", {}).get(
                "citation_status"), stored_status=origin.get("citation_status"), expected_status=pick["expected_status"],
            fallback=pick.get("fallback"), captured_at=origin.get("captured_at"), previewed_by=previewed_by,
            capture_survived_restart=str(origin.get("captured_at") or "9") <= previewed_by,
            cited_row=(cited[0]["session_id"], cited[0]["first_msg_id"]) if cited else None)

    def check_improve_legacy_turn(self) -> None:
        pick = self.picks["turn"]
        provider = self.provider(pick["session_id"])
        turn_id = provider._writer.store_turn(pick["user"], pick["assistant"])
        report = self.call(provider, "qdrant_memory_improve_preview", {"point_ids": [turn_id]})
        candidates = report.get("candidates") or []
        observed: list[dict[str, Any]] = []
        ok = bool(candidates)
        for item in candidates[:2]:
            dry = self.call(provider, "qdrant_memory_improve_apply", {"report_id": report["report_id"],
                                                                      "candidate_id": item["candidate_id"]})
            live = self.call(provider, "qdrant_memory_improve_apply", {"report_id": report["report_id"],
                                                                       "candidate_id": item["candidate_id"],
                                                                       "dry_run": False, "approve": True})
            written = self.payload("memory", str(live.get("id") or "")) if live.get("applied") else None
            entry = {"dry_run_session_origin": dry.get("session_origin", "absent"), "applied": live.get("applied"),
                     "error": live.get("error"), "stored_has_session_origin": bool(written and "session_origin" in written)}
            observed.append(entry)
            ok = ok and dry.get("session_origin", "absent") is None and not entry["stored_has_session_origin"]
        turn_payload = self.payload("memory", turn_id) or {}
        self.receipt.check("improve_legacy_turn_no_citation", ok and "session_origin" not in turn_payload,
                           turn_point_id=turn_id, candidates=len(candidates), applies=observed)

    def check_consolidation(self) -> None:
        span_a, span_b = self.picks["span_a"], self.picks["span_b"]
        run = self.report["run"]
        text_one = f"C6 e2e duplicate {run}: the staging proxy pool is blue."
        text_two = f"c6 e2e duplicate {run}: the staging proxy pool is blue"
        one = self.provider(span_a["session_id"])
        _, stored_one = self.store_with_cite(one, text_one, span_a)
        time.sleep(1.1)  # the first capture is decided by captured_at (second resolution)
        two = self.provider(span_b["session_id"])
        _, stored_two = self.store_with_cite(two, text_two, span_b)
        ids = {str(stored_one.get("id")), str(stored_two.get("id"))}
        report = self.call(one, "qdrant_memory_consolidate", {"scope": "memory", "max_points": 1000})
        proposals = [p for p in report.get("proposals") or [] if p.get("proposal_type") == "duplicate_cluster"]
        proposal = next((p for p in proposals if set(p.get("affected_ids") or []) == ids), None)
        if proposal is None:
            self.receipt.check("consolidation_merge_merges_citations", False, stored_ids=sorted(ids),
                               saved=[stored_one.get("saved"), stored_two.get("saved")],
                               duplicate_proposals=[sorted(p.get("affected_ids") or []) for p in proposals],
                               report_error=report.get("error"))
            return
        dry = self.call(one, "qdrant_memory_consolidation_apply", {"proposal_id": proposal["proposal_id"],
                                                                   "report_id": report["report_id"]})
        live = self.call(one, "qdrant_memory_consolidation_apply", {"proposal_id": proposal["proposal_id"],
                                                                    "report_id": report["report_id"],
                                                                    "dry_run": False, "approve": True,
                                                                    "action": "merge"})
        canonical = str(live.get("canonical_id") or "")
        origin = self.origin("memory", canonical)
        cited = sorted((c["session_id"], c["first_msg_id"], c["last_msg_id"]) for c in origin.get("citations") or [])
        expected = sorted((s["session_id"], s["first_msg_id"], s["last_msg_id"]) for s in (span_a, span_b))
        deleted = [pid for pid in ids if pid != canonical]
        self.receipt.check(
            "consolidation_merge_merges_citations",
            live.get("applied") is True and cited == expected and origin.get("captured_session_id") == span_a["session_id"]
            and all(self.payload("memory", pid) is None for pid in deleted)
            and set(live.get("session_origin_merge", {}).get("from_ids") or []) == ids
            and set(dry.get("session_origin_merge", {}).get("from_ids") or []) == ids,
            canonical_id=canonical, deleted_ids=deleted, citations=cited,
            merge_summary=live.get("session_origin_merge"), dry_run_summary=dry.get("session_origin_merge"))
        self.points["consolidated"] = canonical

    def check_cli_cite(self) -> None:
        import argparse as _argparse
        import importlib.util

        from qdrant_memory.cli_core import execute_command

        spec = importlib.util.spec_from_file_location("c6_plugin_cli", PLUGIN_ROOT / "cli.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load the plugin cli.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        parser = _argparse.ArgumentParser(prog="hermes qdrant")
        cli.register_cli(parser)
        span = self.picks["span_b"]
        text = f"C6 e2e CLI fact {self.report['run']}: the CLI can cite a span."
        cite = ["--cite", f"{span['first_msg_id']}:{span['last_msg_id']}", "--session-id", span["session_id"]]

        def run_cli(argv: list[str]) -> tuple[int, dict[str, Any] | None, str]:
            out, err = io.StringIO(), io.StringIO()
            # One fresh provider per command, as one CLI process per command would have.
            code = execute_command(parser.parse_args(argv), provider_factory=lambda: self.provider("cli"),
                                   stdout=out, stderr=err)
            raw = out.getvalue().strip() or err.getvalue().strip()
            try:
                return code, json.loads(raw), raw
            except Exception:
                return code, None, raw

        code_dry, dry, _ = run_cli(["store", text, *cite, "--json"])
        code_live, live, _ = run_cli(["store", text, *cite, "--no-dry-run", "--approve", "--json"])
        other = ["--cite", f"{self.picks['span_a']['first_msg_id']}:{self.picks['span_a']['last_msg_id']}",
                 "--session-id", self.picks["span_a"]["session_id"]]
        code_unpreviewed, unpreviewed, _ = run_cli(["store", text + " (unpreviewed)", *other, "--no-dry-run",
                                                    "--approve", "--json"])
        code_usage, _, usage_text = run_cli(["store", text, "--cite", "5:3", "--session-id", "x", "--json"])
        point_id = str((live or {}).get("id") or "")
        origin = self.origin("memory", point_id) if point_id else {}
        cited = [pinned(c) for c in origin.get("citations") or []]
        self.points["cli"] = point_id
        self.receipt.check(
            "cli_store_cite_preview_then_approve",
            code_dry == 0 and (dry or {}).get("session_origin", {}).get("citation_status") == "verified"
            and code_live == 0 and (live or {}).get("saved") is True and cited == [span["expected"]]
            and code_unpreviewed == 1 and "needs a preview" in json.dumps(unpreviewed or {})
            and code_usage == 2,
            exit_codes={"dry_run": code_dry, "approve": code_live, "approve_without_preview": code_unpreviewed,
                        "bad_range": code_usage},
            point_id=point_id, captured_session_id=origin.get("captured_session_id"),
            cited=[(c["session_id"], c["first_msg_id"], c["last_msg_id"]) for c in cited])

    def _expected_origin_counts(self, collection: str) -> tuple[dict[str, int], list[str], int]:
        """What origin-check must report, from the raw payloads and an independent re-check."""
        points = self.admin.scroll_by_filter(self.names[collection], {}, limit=1000, with_payload=True)
        with_citations, total = [], 0
        for point in points:
            origin = (point.get("payload") or {}).get("session_origin")
            if isinstance(origin, dict) and origin.get("citations"):
                with_citations.append(str(point["id"]))
                total += len(origin["citations"])
        return {"verified": total} if total else {}, sorted(with_citations), total

    def check_origin_check(self) -> None:
        provider = self.provider(self.picks["span_a"]["session_id"])
        observed: dict[str, Any] = {}
        ok = True
        for collection, flag in (("memory", "memory"), ("learnings", "learning")):
            result = self.call(provider, "qdrant_memory_origin_check", {"collection": flag, "limit": 500,
                                                                        "max_ids": 500})
            counts, ids, total = self._expected_origin_counts(collection)
            got_ids = sorted((result.get("point_ids") or {}).get("verified") or [])
            same = (result.get("counts") == counts and got_ids == ids and result.get("citations_checked") == total
                    and result.get("point_counts", {}).get("verified", 0) == len(ids) and result.get("complete") is True
                    and result.get("invalid_count") == 0)
            ok = ok and same
            observed[collection] = {"counts": result.get("counts"), "expected_counts": counts,
                                    "point_counts": result.get("point_counts"), "point_ids": got_ids,
                                    "expected_point_ids": ids, "uncited_counts": result.get("uncited_counts"),
                                    "invalid_count": result.get("invalid_count"), "match": same}
        self.receipt.check("origin_check_counts_and_ids", ok, **observed)

    def check_trace_inspect(self) -> None:
        provider = self.provider(self.picks["span_a"]["session_id"])
        inspected = self.call(provider, "qdrant_memory_inspect", {"point_id": self.points["manual"]})
        traced = self.call(provider, "qdrant_memory_trace", {"point_id": self.points["manual"]})
        inspect_origin = inspected.get("session_origin") or {}
        trace_origin = traced.get("session_origin") or {}
        self.receipt.check(
            "trace_and_inspect_disclose_origin",
            inspect_origin.get("current_status_counts") == {"verified": 2} and inspect_origin.get("valid") is True
            and [c.get("current_status") for c in trace_origin.get("citations") or []] == ["verified", "verified"],
            inspect_counts=inspect_origin.get("current_status_counts"),
            trace_statuses=[c.get("current_status") for c in trace_origin.get("citations") or []])

    def check_forget(self) -> None:
        provider = self.provider(self.picks["span_a"]["session_id"])
        dry = self.call(provider, "qdrant_memory_forget", {"ids": [self.points["manual"]]})
        still_there = self.payload("memory", self.points["manual"]) is not None
        result = self.call(provider, "qdrant_memory_forget", {"ids": [self.points["manual"]], "dry_run": False})
        self.receipt.check("forget_deletes_cited_point",
                           dry.get("dry_run") is True and still_there and not result.get("error")
                           and self.payload("memory", self.points["manual"]) is None,
                           dry_run_kept_point=still_there,
                           result={key: result.get(key) for key in ("deleted", "error") if key in result},
                           point_gone=self.payload("memory", self.points["manual"]) is None)


class Picker:
    """Chooses real sessions, spans and source lines from ``state.db``, read-only.

    Every read goes through the plugin's own read-only reader. Lines are chosen by running
    the provider's own hook extractors on a session's rows and keeping a candidate whose
    source line resolves to exactly one message (``verified``), so the e2e cites what a
    live approval would cite. Nothing chosen here is printed except ids.
    """

    def __init__(self, state_db: str, ctx: Context) -> None:
        self.state_db, self.ctx = state_db, ctx

    def read(self, work):
        from qdrant_memory.session_origin import _read

        def wrapped(conn):
            conn.text_factory = str
            return work(conn)
        return _read(self.state_db, wrapped)

    def sessions(self) -> list[str]:
        from qdrant_memory.session_recall import _ELIGIBLE

        return self.read(lambda conn: [row[0] for row in conn.execute(
            f"SELECT s.id FROM sessions s WHERE EXISTS (SELECT 1 FROM messages m WHERE m.session_id = s.id AND "
            f"{_ELIGIBLE}) ORDER BY s.started_at DESC LIMIT ?", (MAX_SCAN_SESSIONS,))])

    def rows(self, sid: str) -> list[tuple]:
        from qdrant_memory.session_recall import _ELIGIBLE

        return self.read(lambda conn: conn.execute(
            f"SELECT m.id, m.role, m.content, m.message_uid FROM messages m WHERE m.session_id = ? AND {_ELIGIBLE} "
            f"ORDER BY m.id", (sid,)).fetchall())

    def span(self, sid: str, rows: list[tuple]) -> dict[str, Any] | None:
        from qdrant_memory.lesson_extractor import contains_secret

        for left, right in zip(rows, rows[1:]):
            if all(len(r[2]) <= MAX_MESSAGE_CHARS and not contains_secret(r[2]) for r in (left, right)):
                expected = expected_span_digests(sid, [left, right])
                return {"session_id": sid, "first_msg_id": int(left[0]), "last_msg_id": int(right[0]),
                        "expected": expected, "max_eligible_id": int(rows[-1][0])}
        return None

    @staticmethod
    def source_lines(sid: str, rows: list[tuple]):
        """``(candidate, line, role)`` for every line of the session's last rows that the source
        extractor recognises: the rule the removed hook producer applied to the same rows."""
        from qdrant_memory.source_extraction import extract_source_candidates_from_text

        for _row_id, role, content, _uid in rows[-MAX_HOOK_ROWS:]:
            if len(content) > MAX_MESSAGE_CHARS:
                continue
            lines = content.splitlines()
            for candidate in extract_source_candidates_from_text(
                    content, source_uri=f"session://{sid}/on_pre_compress", source_type="completed_turn",
                    derivation_type="completed_turn", lifecycle_id=f"{sid}:on_pre_compress", min_confidence=0.0):
                line_no = int((candidate.locator or {}).get("line_start") or 0)
                if 1 <= line_no <= len(lines):
                    yield candidate, lines[line_no - 1], role

    def line_citation(self, sid: str, line: str, role: str | None) -> dict[str, Any] | None:
        """The one-row citation the product resolves, re-derived independently from the row."""
        from qdrant_memory.session_origin import resolve_source_line

        resolved = resolve_source_line(self.state_db, sid, line, role=role)
        if not isinstance(resolved, dict):
            return None
        row = self.read(lambda conn: conn.execute(
            "SELECT id, role, content, message_uid FROM messages WHERE id = ?", (resolved["first_msg_id"],)).fetchone())
        return expected_span_digests(resolved["session_id"], [tuple(row)])

    def pick(self) -> dict[str, Any]:
        from qdrant_memory.source_extraction import evaluate_source_extraction_candidate
        from qdrant_memory.lesson_extractor import contains_secret

        picks: dict[str, Any] = {}
        scanned: list[tuple[str, list[tuple]]] = []
        used_lines: set[str] = set()
        scratch = self.ctx.provider("picker")
        scratch._qdrant = None  # the picker never touches Qdrant
        run = self.ctx.report["run"]
        for sid in self.sessions():
            rows = self.rows(sid)
            if len(rows) < 2:
                continue
            if "span_a" not in picks or "span_b" not in picks:
                span = self.span(sid, rows)
                if span:
                    picks["span_a" if "span_a" not in picks else "span_b"] = span
            scratch._session_id = sid
            if "extraction" not in picks or "improve" not in picks or "turn" not in picks:
                for candidate, line, role in self.source_lines(sid, rows):
                    if not line.strip() or line in used_lines or contains_secret(line):
                        continue
                    if evaluate_source_extraction_candidate(candidate).decision != "store":
                        continue
                    expected = self.line_citation(sid, line, role or None)
                    if not expected:
                        continue
                    if "extraction" not in picks:
                        picks["extraction"] = {"session_id": sid, "candidate": candidate, "line": line, "role": role,
                                               "text": candidate.proposed_payload.get("text"),
                                               "expected": expected, "expected_status": "verified"}
                    elif "improve" not in picks:
                        improve = self.improve_candidate(scratch, sid, line)
                        if not improve:
                            continue
                        picks["improve"] = {"session_id": sid, "line": line, "text": improve,
                                            "expected": expected, "expected_status": "verified"}
                    elif "turn" not in picks:
                        picks["turn"] = {"session_id": sid, "user": "What did we settle?",
                                         "assistant": "Noted.\n" + line}
                    else:
                        break
                    used_lines.add(line)
            if "learning" not in picks:
                # The learning heuristic that chose a row here was removed (capture CP1). Any real,
                # one-line human-looking user row that resolves to exactly one message serves:
                # the check is about the origin the approval records, not about the lesson.
                for _row_id, role, content, _uid in rows:
                    line = content.strip()
                    if role != "user" or "\n" in line or not 12 <= len(line) <= 300 or line.startswith("["):
                        continue
                    if contains_secret(line) or line in used_lines:
                        continue
                    candidate = learning_candidate(run, line)
                    dry = json.loads(scratch.approve_learning_candidate(candidate))
                    if (dry.get("write_decision") or {}).get("decision") not in {"learning_candidate", "store",
                                                                                 "skill_candidate"}:
                        continue
                    expected = self.line_citation(sid, line, "user")
                    if expected:
                        picks["learning"] = {"session_id": sid, "line": line, "expected": expected,
                                             "expected_status": "verified"}
                        used_lines.add(line)
                        break
            if all(key in picks for key in ("span_a", "span_b", "extraction", "improve", "turn", "learning")):
                break
            scanned.append((sid, rows))
        self.fallbacks(picks, scanned, scratch)
        missing = [key for key in ("span_a", "span_b", "extraction", "improve", "turn", "learning") if key not in picks]
        if missing:
            raise RuntimeError(f"state.db has no usable {', '.join(missing)} in its {MAX_SCAN_SESSIONS} most "
                               "recent sessions")
        picks["summary"] = {
            "fallbacks": {key: picks[key].get("fallback") for key in ("extraction", "improve", "turn", "learning")
                          if picks[key].get("fallback")},
            "span_a": {k: picks["span_a"][k] for k in ("session_id", "first_msg_id", "last_msg_id")},
            "span_b": {k: picks["span_b"][k] for k in ("session_id", "first_msg_id", "last_msg_id")},
            **{key: {"session_id": picks[key]["session_id"], "expected_status": picks[key]["expected_status"],
                     "cited_row": (picks[key]["expected"] or {}).get("first_msg_id")}
               for key in ("extraction", "learning", "improve")},
            "legacy_turn_session": picks["turn"]["session_id"],
        }
        return picks

    def fallbacks(self, picks: dict[str, Any], scanned: list[tuple[str, list[tuple]]], scratch) -> None:
        """When no row of state.db carries a line the extractors recognise (a live database whose
        conversations never write "Decision: ..." lines has none), exercise the routes on a real
        session anyway, with one synthetic signal line that state.db does not hold. The citation
        must then be ``unresolved:no_match``, the store must still happen (design 5.8) and the
        captured session must still win. The receipt names every fallback."""
        from qdrant_memory.source_extraction import evaluate_source_extraction_candidate, \
            extract_source_candidates_from_text

        run = self.ctx.report["run"]
        reason = (f"no line in the {len(scanned) + 2} scanned sessions matches the source extractor; "
                  "a synthetic signal line (not in state.db) was used")
        others = [(sid, rows) for sid, rows in scanned if sid not in {picks.get("span_b", {}).get("session_id")}]
        if not others:
            return
        if "extraction" not in picks:
            sid, _rows = others[0]
            line = f"Decision: C6 e2e synthetic signal {run}, route staging traffic through the blue pool."
            for candidate in extract_source_candidates_from_text(
                    line, source_uri=f"session://{sid}/on_pre_compress", source_type="completed_turn",
                    derivation_type="completed_turn", lifecycle_id=f"{sid}:on_pre_compress", min_confidence=0.0):
                if evaluate_source_extraction_candidate(candidate).decision == "store":
                    picks["extraction"] = {"session_id": sid, "candidate": candidate, "line": line, "role": "assistant",
                                           "text": candidate.proposed_payload.get("text"), "expected": None,
                                           "expected_status": "unresolved:no_match", "fallback": reason}
                    break
        if "improve" not in picks:
            sid = others[min(1, len(others) - 1)][0]
            line = f"Decision: C6 e2e synthetic signal {run}, pin the scheduler to two workers."
            text = self.improve_candidate(scratch, sid, line)
            if text:
                picks["improve"] = {"session_id": sid, "line": line, "text": text, "expected": None,
                                    "expected_status": "unresolved:no_match", "fallback": reason}
        if "turn" not in picks:
            picks["turn"] = {"session_id": others[0][0], "user": "What did we settle?",
                             "assistant": f"Noted.\nDecision: C6 e2e synthetic turn {run}, keep the cache warm.",
                             "fallback": reason}

    def improve_candidate(self, scratch, sid: str, line: str) -> str | None:
        """A store-eligible improve candidate for ``line`` in ``sid`` (dry preview, nothing kept)."""
        with redirect_stderr(io.StringIO()):
            report = json.loads(scratch._tool_improve_preview({"source_text": line, "session_id": sid,
                                                               "source_uri": "session://current/improve-source",
                                                               "persist": False}))
        for item in report.get("candidates") or []:
            if item.get("would_store") and (item.get("proposed_payload") or {}).get("text"):
                return str(item["proposed_payload"]["text"])
        return None


if __name__ == "__main__":
    sys.exit(main())
