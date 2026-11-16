#!/usr/bin/env python3
"""End-to-end check of the CP6 watcher: rotating scan window and guarded-auto safety.

Runs the real ``hermes qdrant watcher run`` code path (``cli_core.execute_command`` with the
real provider) for N simulated nights against a real Qdrant, in two arms:

- ``prod``: report-only against the production collections, read-only. Every Qdrant request
  is fenced: a write to any collection whose name does not start with ``tmp_watch_`` raises.
- ``tmp``: guarded-auto against ``tmp_watch_<run>_memory`` / ``_learnings``, restored from a
  plugin backup of the production collections taken at the start (the backup/restore flow
  the guarded-auto policy relies on).

    python3 scripts/e2e_watcher_rotating.py --out DIR [--nights 7] [--max-points 300] \\
        [--arms prod,tmp] [--qdrant-url http://127.0.0.1:6333] [--keep] [--tmp-prefix tmp_watch_] \\
        [--fail-backup-night N] [--backup-keep-last 7] [--backup-keep-days 30] [--embed-seeds]

Safety:
- Every run uses a temporary HERMES_HOME under ``--out``: the real watcher state, log,
  consolidation artifacts and backups are never touched. No Hermes config is read; the
  provider gets an explicit config (written to the receipt).
- Temporary collections are deleted at the end, also on failure, and their absence is
  verified (``--keep`` skips the deletion for debugging).
- The receipt (``receipt.json``) carries ids, digests, counts and statuses. Point text is
  written only to ``samples-private.json`` under ``--out`` for human review; never commit it.

Oracles (exit non-zero if any fails):
- the union of memory ids the watcher read grows by ``--max-points`` per night until the
  collection is covered (observed at the Qdrant client, cross-checked against the report's
  ``scanned_ids_sha256``); the learning window covers the learning collection;
- every applied action has an application record, a full pre-mutation backup and a partial
  snapshot backup whose point digests equal the sealed proposal digests;
- diffing the temporary collection against the starting backup: the changed ids are
  exactly the applied ids, nothing was deleted, no point carrying ``session_origin`` and no
  identity-bearing point changed;
- one applied action is undone by restoring its snapshot backup, and the point returns to
  its starting payload digest;
- the production arm issued no write (the fence counts attempts);
- backup retention (tmp arm, seeded with two old watcher backups and one manual backup): the
  watcher's full backups left are exactly the configured set, the deletions are reported, and
  the manual backup and every per-action snapshot are kept;
- with ``--fail-backup-night N``: that night applies nothing and leaves the cursor unchanged,
  and night N+1 reads the same window again;
- with ``--embed-seeds``: every quarantined control seed is absent from a legacy recall query
  made of its own text and present in ``qdrant_memory_inspect``; after the snapshot restore
  the restored point is recalled by the same query (control).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
import traceback
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

TMP_PREFIX = "tmp_watch_"
PROD_MEMORY = "hermes_memory"
PROD_LEARNING = "hermes_learnings"
PROFILE = "default"
# Production values that change watcher behaviour (observed in the live config, read-only).
ARM_CONFIG = {
    "lineage_mode": "capture",
    "session_origin_mode": "recorded",
    "consolidation_enabled": True,
    "reconsolidation_enabled": False,
    "consolidation_persist_reports": True,
    "vector_size": 1024,
    "distance": "Cosine",
}
_READ_POST = ("/points", "/points/search", "/points/scroll", "/points/count", "/points/query")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


class Fence:
    """Process-wide write fence on the plugin's Qdrant client."""

    def __init__(self) -> None:
        self.refused: list[str] = []
        self.writes: Counter = Counter()

    def install(self) -> None:
        import urllib.parse

        from qdrant_memory import client as client_module

        original = client_module.QdrantClient._send
        fence = self

        def guarded_send(client_self, method, path, body=None):
            route = urllib.parse.urlsplit(path if path.startswith("/") else "/" + path).path
            parts = route.split("/", 3)
            verb = method.upper()
            rest = "/" + parts[3] if len(parts) > 3 else ""
            is_read = verb == "GET" or (verb == "POST" and rest.rstrip("/") in _READ_POST)
            if not is_read:
                name = urllib.parse.unquote(parts[2]) if len(parts) > 2 and parts[1] == "collections" else ""
                if not name.startswith(TMP_PREFIX):
                    fence.refused.append(f"{verb} {route}")
                    raise RuntimeError(f"e2e write fence: refused {verb} {route}")
                fence.writes[name] += 1
            return original(client_self, method, path, body)

        client_module.QdrantClient._send = guarded_send


class WindowRecorder:
    """Records the ids every ``scroll_window`` call returned (what the watcher actually read)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def install(self) -> None:
        from qdrant_memory import client as client_module

        original = client_module.QdrantClient.scroll_window
        recorder = self

        def recording(client_self, name, filter, **kwargs):
            points, offset = original(client_self, name, filter, **kwargs)
            recorder.calls.append((name, [str(point.get("id")) for point in points]))
            return points, offset

        client_module.QdrantClient.scroll_window = recording


def _client(url: str):
    from qdrant_memory.client import QdrantClient

    return QdrantClient(url, timeout=120.0)


def _config(home: Path, memory: str, learning: str, url: str) -> dict[str, Any]:
    from qdrant_memory.config import load_config

    # HERMES_QDRANT_MEMORY_<KEY> variables override any explicit config: pin the two names
    # (the CLI resolves them too) and refuse if anything else wins.
    os.environ["HERMES_QDRANT_MEMORY_COLLECTION_NAME"] = memory
    os.environ["HERMES_QDRANT_MEMORY_LEARNING_COLLECTION_NAME"] = learning
    config = load_config(
        hermes_home=str(home),
        hermes_config={"qdrant_memory": {**ARM_CONFIG, "qdrant_url": url, "collection_name": memory, "learning_collection_name": learning}},
    )
    if (config["collection_name"], config["learning_collection_name"]) != (memory, learning):
        raise SystemExit(f"config resolved to {config['collection_name']}/{config['learning_collection_name']}; refusing")
    return config


def _provider(home: Path, config: dict[str, Any], url: str):
    import importlib

    plugin = importlib.import_module("__init__")
    provider = plugin.QdrantMemoryProvider()
    provider._active = True
    provider._hermes_home = str(home)
    provider._profile_id = PROFILE
    provider._platform = "cli"
    provider._session_id = "cp6-e2e"
    provider._config = dict(config)
    provider._qdrant = _client(url)
    return provider


def _watcher_args(mode: str, max_points: int):
    import argparse as _argparse
    import importlib.util

    # By path: a bare `import cli` can resolve to Hermes' own cli.py.
    spec = importlib.util.spec_from_file_location("qdrant_plugin_cli_e2e", PLUGIN_ROOT / "cli.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    parser = _argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command", required=True)
    qdrant_parser = sub.add_parser("qdrant")
    cli.register_cli(qdrant_parser)
    return parser.parse_args([
        "qdrant", "watcher", "run", "--scope", "both", "--max-points", str(max_points), "--max-groups", "20",
        "--reconsolidation-max-candidates", "10", "--scan-window", "rotating", "--autonomy-mode", mode, "--json",
    ])


def _iter_backup_records(backup_dir: Path, scope: str):
    path = backup_dir / f"{scope}.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _scroll_all(client, name: str, *, with_vector: bool = False):
    offset = None
    while True:
        batch, offset = client.scroll_page(name, {}, limit=512, offset=offset, with_payload=True, with_vector=with_vector)
        yield from batch
        if offset is None or not batch:
            return


class Run:
    def __init__(self, args: argparse.Namespace) -> None:
        global TMP_PREFIX
        if not args.tmp_prefix.startswith("tmp_"):
            raise SystemExit("--tmp-prefix must start with tmp_")
        TMP_PREFIX = args.tmp_prefix  # the fence reads the module value
        self.args = args
        self.out = Path(args.out).expanduser().resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.run_id = uuid.uuid4().hex[:8]
        self.tmp_memory = f"{TMP_PREFIX}{self.run_id}_memory"
        self.tmp_learning = f"{TMP_PREFIX}{self.run_id}_learnings"
        self.fence = Fence()
        self.recorder = WindowRecorder()
        self.receipt: dict[str, Any] = {
            "run_id": self.run_id,
            "plugin_root": str(PLUGIN_ROOT),
            "arm_config": ARM_CONFIG,
            "nights": args.nights,
            "max_points": args.max_points,
            "tmp_collections": [self.tmp_memory, self.tmp_learning],
            "checks": [],
            "arms": {},
        }
        self.samples: list[dict[str, Any]] = []

    def check(self, name: str, ok: bool, **detail: Any) -> None:
        self.receipt["checks"].append({"name": name, "ok": bool(ok), **detail})
        print(f"[{'ok' if ok else 'FAIL'}] {name} {json.dumps(detail, default=str)[:300]}", flush=True)

    # -- setup ---------------------------------------------------------------------------
    def snapshot_production(self) -> None:
        from qdrant_memory.backup import create_backup, restore_backup

        url = self.args.qdrant_url
        home = self.out / "home_restore"
        home.mkdir(parents=True, exist_ok=True)
        prod_config = _config(home, PROD_MEMORY, PROD_LEARNING, url)
        started = time.time()
        backup = create_backup(_client(url), prod_config, hermes_home=str(home), scope="both")
        self.backup_dir = Path(backup["backup_dir"])
        self.receipt["source_backup"] = {
            "backup_id": backup["backup_id"],
            "collections": {scope: {"collection_name": d["collection_name"], "count": d["count"], "sha256": d["sha256"]} for scope, d in backup["collections"].items()},
            "seconds": round(time.time() - started, 1),
            "bytes": sum(path.stat().st_size for path in self.backup_dir.iterdir()),
        }
        admin = _client(url)
        for name in (self.tmp_memory, self.tmp_learning):
            admin.ensure_collection(name, 1024, "Cosine")
        tmp_config = _config(home, self.tmp_memory, self.tmp_learning, url)
        started = time.time()
        restored = restore_backup(admin, tmp_config, hermes_home=str(home), backup_id=backup["backup_id"], dry_run=False)
        self.receipt["restore"] = {
            "upserted": restored.get("upserted"),
            "collections": {scope: {k: v for k, v in d.items() if k not in {"changed_ids", "missing_ids", "session_origin_kept_ids"}} for scope, d in restored["collections"].items()},
            "seconds": round(time.time() - started, 1),
        }
        counts = {"memory": admin.count(self.tmp_memory), "learning": admin.count(self.tmp_learning)}
        expected = {scope: d["count"] for scope, d in backup["collections"].items()}
        self.check(f"snapshot restored into {TMP_PREFIX} collections through the plugin backup/restore flow", counts == expected, counts=counts, expected=expected)
        # Starting state of every point, by digest, for the final diff.
        self.start: dict[str, dict[str, Any]] = {}
        for scope in ("memory", "learning"):
            for record in _iter_backup_records(self.backup_dir, scope):
                payload = record.get("payload") or {}
                self.start[str(record["id"])] = {
                    "scope": scope,
                    "payload_sha256": _sha(_canon(payload)),
                    "session_origin": payload.get("session_origin") is not None,
                    "identity": self._identity(payload),
                }
        self.receipt["start_population"] = {
            "points": len(self.start),
            "with_session_origin": sum(1 for v in self.start.values() if v["session_origin"]),
            "identity_bearing": sum(1 for v in self.start.values() if v["identity"]),
        }

    def seed_controls(self) -> None:
        """Insert labelled synthetic control points into the tmp memory collection.

        Production data rarely produces a guarded-auto action in a 300-point window, which
        would leave the per-action oracles vacuous. The seeds give each night some eligible
        actions and, next to them, decoys that must never be touched (cited, money, security,
        identity). Their ids sit between real ids inside the first nights' windows; their
        payloads carry ``cp6_e2e_seed`` and no real memory text.
        """
        admin = _client(self.args.qdrant_url)
        ordered = sorted(pid for pid, info in self.start.items() if info["scope"] == "memory")
        window = self.args.max_points
        # A held night (forced backup failure) re-reads its window, so one window fewer is read.
        windows_read = self.args.nights - (1 if 0 < self.args.fail_backup_night < self.args.nights else 0)
        span = min(len(ordered) - 2, window * windows_read - 2)
        old = "2025-01-01T00:00:00+00:00"
        recent = "2026-11-01T00:00:00+00:00"
        stale = {"importance": 1, "confidence": 0.3, "access_count": 0, "created_at": old}
        cited = {"captured_session_id": "20261101_000000_cp6e2e", "citation_status": "verified", "citations": []}
        kinds = [
            ("heading_ok", "Tareas", {}, True),
            ("heading_cited", "Notas", {"session_origin": cited}, False),
            ("dup_ok_a", "CP6 seed: run the indexer in dry-run mode before a live pass", {"importance": 6, "created_at": recent}, True),
            ("dup_ok_b", "CP6 seed: run the indexer in dry-run mode before a live pass", {"importance": 5, "created_at": recent}, True),
            ("dup_cited_a", "CP6 seed: prefer ripgrep for repository searches", {"created_at": recent}, False),
            ("dup_cited_b", "CP6 seed: prefer ripgrep for repository searches", {"created_at": recent, "session_origin": cited}, False),
            ("stale_ok", "CP6 seed: scratch remark about an old folder layout", dict(stale), True),
            ("stale_money", "CP6 seed: paid the hosting invoice for the month", dict(stale), False),
            ("stale_security", "CP6 seed: rotated the ssh deploy key on the build box", dict(stale), False),
            ("stale_identity", "CP6 seed: remark kept in the user profile", {**stale, "source_type": "user_memory"}, False),
            ("stale_cited", "CP6 seed: chat remark about lunch plans", {**stale, "session_origin": cited}, False),
            ("heading_ok_2", "## Next handoff", {}, True),
        ]
        vector = next(_iter_backup_records(self.backup_dir, "memory"))["vector"]
        embed = None
        if self.args.embed_seeds:
            from qdrant_memory.embeddings import EmbeddingClient

            config = _config(self.out / "home_restore", self.tmp_memory, self.tmp_learning, self.args.qdrant_url)
            client = EmbeddingClient(config["embedding_url"], config["embedding_model"], query_prefix=config["query_prefix"],
                                     document_prefix=config["document_prefix"])
            embed = client.embed_document
        points, seeds = [], []
        for index, (kind, text, extra, expect_touched) in enumerate(kinds):
            # Spread over the nights; the two members of a duplicate pair share a window.
            anchor = ordered[max(0, min(span, int(span * (index // 2 * 2 + 1) / (len(kinds) + 1))))]
            seed_id = str(uuid.UUID(int=uuid.UUID(anchor).int + 1 + index))
            if seed_id in self.start:
                raise SystemExit(f"seed id collides with a real point: {seed_id}")
            payload = {"text": text, "profile_id": PROFILE, "source_type": "conversation", "cp6_e2e_seed": kind, **extra}
            points.append({"id": seed_id, "vector": embed(text) if embed else vector, "payload": payload})
            seeds.append({"id": seed_id, "kind": kind, "expect_touched": expect_touched, "text": text})
            self.start[seed_id] = {
                "scope": "memory",
                "payload_sha256": _sha(_canon(payload)),
                "session_origin": "session_origin" in extra,
                "identity": self._identity(payload),
                "seed": kind,
            }
        admin.upsert(self.tmp_memory, points)
        self.seeds = seeds
        self.receipt["seeds"] = seeds

    RETENTION_OLD = ("backup-20000101T000000Z-e2eold01", "backup-20000102T000000Z-e2eold02")
    RETENTION_MANUAL = "backup-20000101T000000Z-e2emanual"

    def seed_retention_fixtures(self, home: Path) -> None:
        """Old watcher-tagged full backups (unreferenced) and an old manual backup in the arm's home.

        Placeholder files only (a manifest and a one-line collection file, no memory text).
        Retention must delete the old watcher backups once newer ones take the newest slots and
        must never delete the manual one.
        """
        from qdrant_memory.backup import SCHEMA_VERSION
        from qdrant_memory.backup_retention import WATCHER_BACKUP_ORIGIN

        root = home / "qdrant_memory" / "backups"
        for index, backup_id in enumerate((*self.RETENTION_OLD, self.RETENTION_MANUAL)):
            path = root / backup_id
            path.mkdir(parents=True, exist_ok=False)
            (path / "memory.jsonl").write_text('{"schema_version": 1}\n', encoding="utf-8")
            manifest = {"schema_version": SCHEMA_VERSION, "artifact_type": "qdrant_memory_backup", "backup_id": backup_id,
                        "created_at": f"2000-01-0{index + 1}T00:00:00Z", "scope": "memory",
                        "collections": {"memory": {"collection_name": self.tmp_memory, "file": "memory.jsonl", "count": 0}}}
            if backup_id != self.RETENTION_MANUAL:
                manifest["created_by"] = WATCHER_BACKUP_ORIGIN
            (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def verify_retention_and_hold(self, records: list[dict[str, Any]]) -> None:
        from qdrant_memory.backup import _load_manifest_from_dir
        from qdrant_memory.backup_retention import WATCHER_BACKUP_ORIGIN

        home = self.out / "home_tmp"
        nights = self.receipt["arms"]["tmp"]["nights"]
        root = home / "qdrant_memory" / "backups"
        remaining: dict[str, dict[str, Any]] = {}
        for child in root.iterdir():
            try:
                remaining[child.name] = _load_manifest_from_dir(child)
            except Exception:  # noqa: BLE001
                remaining[child.name] = {}
        taken = [n["backup"]["backup_id"] for n in nights if n.get("backup")]
        tagged_full = sorted(name for name, m in remaining.items()
                             if m.get("created_by") == WATCHER_BACKUP_ORIGIN and m.get("partial") is not True)
        referenced = {r.get("pre_apply_backup_id") for r in records}
        snapshots = {(r.get("pre_mutation_snapshot") or {}).get("backup_id") for r in records}
        # The rule applied to every tagged full backup the arm ever had (deletions are final).
        created = {name: str(remaining.get(name, {}).get("created_at") or "") for name in taken}
        created.update({name: f"2000-01-0{i + 1}T00:00:00Z" for i, name in enumerate(self.RETENTION_OLD)})
        newest = [name for _c, name in sorted(((c, n) for n, c in created.items()), reverse=True)][: self.args.backup_keep_last]
        recent_refs = referenced if self.args.backup_keep_days >= 1 else set()
        expected = sorted(set(newest) | (set(taken) & recent_refs))
        deleted_reported = sorted({d for n in nights for d in ((n.get("backup_retention") or {}).get("deleted") or [])})
        self.receipt["arms"]["tmp"]["retention"] = {
            "keep_last": self.args.backup_keep_last, "keep_days": self.args.backup_keep_days, "full_backups_taken": taken,
            "tagged_full_remaining": tagged_full, "expected_remaining": expected, "deleted_reported": deleted_reported,
            "manual_present": self.RETENTION_MANUAL in remaining,
            "snapshots_present": sorted(s for s in snapshots if s in remaining), "snapshots_total": len(snapshots),
        }
        self.check("retention: the watcher's full backups left are exactly the configured set",
                   tagged_full == expected, remaining=tagged_full, expected=expected)
        self.check("retention: what it deleted is reported in the run output",
                   set(deleted_reported) == (set(self.RETENTION_OLD) | set(taken)) - set(expected),
                   deleted=deleted_reported)
        self.check("retention: the manual backup and every per-action snapshot are kept",
                   self.RETENTION_MANUAL in remaining and all(s in remaining for s in snapshots),
                   snapshots=len(snapshots))
        fail_night = self.args.fail_backup_night
        if fail_night:
            failed = next((n for n in nights if n["night"] == fail_night), None)
            following = next((n for n in nights if n["night"] == fail_night + 1), None)
            ok = bool(failed) and any(e.get("code") == "backup_failed" for e in failed["errors"]) \
                and not failed["applied"] and failed["held"] == "backup_failed" \
                and failed["cursor_after"] == failed["cursor_before"]
            self.check("a failed pre-mutation backup applies nothing and leaves the cursor unchanged", ok,
                       night=fail_night, cursor_before=failed and failed["cursor_before"],
                       cursor_after=failed and failed["cursor_after"], errors=failed and failed["errors"])
            retried = bool(failed) and bool(following) \
                and following["start_offset"]["memory"] == failed["start_offset"]["memory"] and not following["held"]
            self.check("the held window is read again the next night", retried,
                       failed_start=failed and failed["start_offset"], next_start=following and following["start_offset"],
                       next_applied=following and [a.get("proposal_type") for a in following["applied"]])
            others = [n["night"] for n in nights if n["held"] and n["night"] != fail_night]
            self.check("no other night held the cursor", not others, held_nights=others)

    def verify_quarantine_recall(self, records: list[dict[str, Any]]) -> None:
        """Quarantined seeds are absent from a legacy recall query with their own text; inspect reaches them."""
        import importlib

        home = self.out / "home_recall"
        home.mkdir(parents=True, exist_ok=True)
        (home / "qdrant_memory.json").write_text(json.dumps({
            **ARM_CONFIG, "qdrant_url": self.args.qdrant_url, "collection_name": self.tmp_memory,
            "learning_collection_name": self.tmp_learning, "auto_recall_mode": "legacy", "auto_recall_top_k": 10,
            "session_recall_enabled": False, "session_recall_auto_index": False, "sync_turns": False}), encoding="utf-8")
        previous_home = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = str(home)
        try:
            plugin = importlib.import_module("__init__")
            provider = plugin.QdrantMemoryProvider()
            provider.initialize("close-e2e-recall", platform="cli", agent_context="primary", hermes_home=str(home))
        finally:
            if previous_home is not None:
                os.environ["HERMES_HOME"] = previous_home
        effective = {k: provider._config.get(k) for k in ("collection_name", "learning_collection_name", "auto_recall_mode",
                                                          "embedding_url", "embedding_model", "qdrant_url")}
        print("RECALL EFFECTIVE CONFIG:", json.dumps(effective, sort_keys=True), flush=True)
        seeds = {seed["id"]: seed for seed in getattr(self, "seeds", [])}
        quarantined = sorted({str(pid) for r in records for pid in (r.get("quarantined_ids") or []) if str(pid) in seeds})
        admin = _client(self.args.qdrant_url)
        result: dict[str, Any] = {"effective_config": effective, "quarantined_seeds": []}

        def legacy_ids(text: str) -> list[str]:
            chain = provider._memory_recall() or provider._retriever
            return [str(chunk.id) for chunk in chain.search(text, top_k=20, update_access=False)]

        self._recall = (provider, legacy_ids, seeds, result)
        bad = []
        for pid in quarantined:
            seed = seeds[pid]
            payload = (admin.retrieve(self.tmp_memory, [pid], with_payload=True) or [{}])[0].get("payload") or {}
            ids = legacy_ids(seed["text"])
            block = provider.prefetch(seed["text"], session_id="close-e2e-recall") or ""
            inspected = json.loads(provider.handle_tool_call("qdrant_memory_inspect", {"point_id": pid}))
            entry = {"id": pid, "kind": seed["kind"], "marker": payload.get("consolidation_quarantined") is True,
                     "in_legacy_top20": pid in ids, "inspect_found": inspected.get("found") is True,
                     "prefetch_block_chars": len(block)}
            if seed["kind"] == "dup_ok_b":  # the canonical copy has the same text and must still be recalled
                canonical = next(s["id"] for s in seeds.values() if s["kind"] == "dup_ok_a")
                entry["canonical_in_legacy_top20"] = canonical in ids
                if not entry["canonical_in_legacy_top20"]:
                    bad.append(entry)
            result["quarantined_seeds"].append(entry)
            if not entry["marker"] or entry["in_legacy_top20"] or not entry["inspect_found"]:
                bad.append(entry)
        self.check("quarantined points are absent from a legacy recall query with their own text and present in inspect",
                   bool(quarantined) and not bad, checked=len(result["quarantined_seeds"]), bad=bad[:3])
        self.receipt["arms"]["tmp"]["quarantine_recall"] = result

    def recall_control(self, restored_id: str) -> None:
        """After the snapshot restore: the restored point (marker gone) is recalled by its own text."""
        provider, legacy_ids, seeds, result = self._recall
        try:
            if restored_id in seeds:
                ids = legacy_ids(seeds[restored_id]["text"])
                result["restored_control"] = {"id": restored_id, "kind": seeds[restored_id]["kind"],
                                              "in_legacy_top20": restored_id in ids}
                self.check("control: the point restored from its snapshot (marker gone) is recalled by the same query",
                           restored_id in ids, id=restored_id, kind=seeds[restored_id]["kind"])
        finally:
            provider.shutdown()

    @staticmethod
    def _identity(payload: dict[str, Any]) -> bool:
        from qdrant_memory.consolidation import identity_bearing_payload, sensitive_topic_categories

        text = str(payload.get("text") or payload.get("lesson") or "")
        return identity_bearing_payload(payload) or "identity" in sensitive_topic_categories(text)

    # -- nights --------------------------------------------------------------------------
    def run_arm(self, arm: str) -> None:
        from qdrant_memory.cli_core import execute_command

        url = self.args.qdrant_url
        home = self.out / f"home_{arm}"
        home.mkdir(parents=True, exist_ok=True)
        os.environ["HERMES_HOME"] = str(home)
        os.environ["HERMES_QDRANT_MEMORY_LINEAGE_LOCK_DIR"] = str(self.out / "locks")
        memory, learning = (PROD_MEMORY, PROD_LEARNING) if arm == "prod" else (self.tmp_memory, self.tmp_learning)
        config = _config(home, memory, learning, url)
        mode = "report-only" if arm == "prod" else "guarded-auto"
        total_nights = self.args.prod_nights if arm == "prod" else self.args.nights
        if arm == "tmp":
            config.update({"guarded_auto_backup_keep_last": self.args.backup_keep_last,
                           "guarded_auto_backup_keep_days": self.args.backup_keep_days})
            self.seed_retention_fixtures(home)
        state_path = home / "qdrant_memory" / "consolidation" / "watcher_state.json"

        def memory_cursor() -> Any:
            if not state_path.exists():
                return None
            return ((json.loads(state_path.read_text(encoding="utf-8")).get("scan_window") or {}).get("memory") or {}).get("next_offset")
        # The watcher only reads points in its own scope (profile filter, no structural
        # lineage records): count with the very same filter.
        from qdrant_memory.consolidation import make_filter

        def scoped_count(name: str, source_type: str | None = None) -> int:
            flt = make_filter({"profile_id": PROFILE}, source_type=source_type)
            data = _client(url)._request("POST", f"/collections/{name}/points/count", {"exact": True, "filter": flt})
            return int(data["result"]["count"])

        collection_total = scoped_count(memory)
        learning_total = scoped_count(learning, "learning")
        nights: list[dict[str, Any]] = []
        union: dict[str, set[str]] = {PROD_MEMORY: set(), PROD_LEARNING: set(), self.tmp_memory: set(), self.tmp_learning: set()}
        for night in range(1, total_nights + 1):
            provider = _provider(home, config, url)
            forced_failure = arm == "tmp" and night == self.args.fail_backup_night
            if forced_failure:
                def failing_backup(scope):
                    raise OSError("e2e: forced pre-mutation backup failure")

                provider.guarded_auto_backup = failing_backup
            cursor_before = memory_cursor()
            self.recorder.calls.clear()
            out, err = io.StringIO(), io.StringIO()
            started = time.time()
            code = execute_command(_watcher_args(mode, self.args.max_points), provider_factory=lambda: provider, stdout=out, stderr=err)
            seconds = round(time.time() - started, 1)
            if code != 0:
                self.check(f"{arm} night {night} watcher exit code", False, code=code, stderr=err.getvalue()[-2000:])
                break
            report = json.loads(out.getvalue())
            window = report.get("scan_window") or {}
            observed = {name: ids for name, ids in self.recorder.calls}
            for label, name in (("memory", memory), ("learning", learning)):
                ids = observed.get(name, [])
                union[name] |= set(ids)
                claimed = (window.get(label) or {}).get("scanned_ids_sha256")
                if claimed != _sha("\n".join(sorted(ids))):
                    self.check(f"{arm} night {night} {label} report digest matches observed ids", False)
            guarded = report.get("guarded_auto") or {}
            by_type = Counter(p.get("proposal_type") for p in report.get("proposals", []))
            would = Counter(e.get("proposal_type") for e in guarded.get("would_apply", []))
            applied = Counter(e.get("proposal_type") for e in guarded.get("applied", []))
            nights.append({
                "night": night,
                "seconds": seconds,
                "report_id": report.get("report_id"),
                "scanned": {label: (window.get(label) or {}).get("scanned") for label in ("memory", "learning")},
                "start_offset": {label: (window.get(label) or {}).get("start_offset") for label in ("memory", "learning")},
                "reached_end": {label: (window.get(label) or {}).get("reached_end") for label in ("memory", "learning")},
                "union_memory": len(union[memory]),
                "union_learning": len(union[learning]),
                "proposals_by_type": dict(by_type),
                "would_apply_by_type": dict(would),
                "applied_by_type": dict(applied),
                "errors": guarded.get("errors", []),
                "skipped_reasons": dict(Counter(e.get("reason") for e in guarded.get("skipped", []))),
                "backup": guarded.get("backup"),
                "applied": guarded.get("applied", []),
                "would_apply": guarded.get("would_apply", []),
                "artifact": (report.get("artifact") or {}).get("path"),
                "forced_backup_failure": forced_failure,
                "cursor_before": cursor_before,
                "cursor_after": memory_cursor(),
                "held": (json.loads(state_path.read_text(encoding="utf-8")).get("scan_window_held") or {}).get("reason"),
                "backup_retention": guarded.get("backup_retention"),
            })
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        self.receipt["arms"][arm] = {
            "mode": mode,
            "collections": [memory, learning],
            "nights": nights,
            "final_scan_window": state.get("scan_window"),
            "notifier_fields": {key: state.get(key) for key in (
                "last_alerted", "last_alert_signature", "last_report_id", "last_scope", "last_total_proposals",
                "last_counts", "last_guarded_auto_mode", "last_guarded_auto_applied_count",
                "last_guarded_auto_error_count", "last_artifact")},
        }
        self.notifier_check(arm, home)
        # A held night re-reads its window the next night: that next night adds no new ids.
        expected_union = []
        windows = 0
        for index, entry in enumerate(nights):
            if index == 0 or not nights[index - 1]["held"]:
                windows += 1
            expected_union.append(min(collection_total, self.args.max_points * windows))
        self.check(
            f"{arm}: union of analysed memory ids grows by max-points per night",
            [n["union_memory"] for n in nights] == expected_union and len(nights) == total_nights,
            observed=[n["union_memory"] for n in nights], expected=expected_union, collection_points=collection_total,
        )
        if arm == "prod":
            self.collect_prod_samples(nights)
        self.check(
            f"{arm}: learning window covers the learning collection",
            bool(nights) and nights[-1]["union_learning"] == min(learning_total, self.args.max_points * len(nights)),
            observed=[n["union_learning"] for n in nights], learning_points=learning_total,
        )

    def collect_prod_samples(self, nights: list[dict[str, Any]]) -> None:
        """Read (read-only) the points behind the report-only arm's would-apply actions."""
        admin = _client(self.args.qdrant_url)
        for night in nights:
            for entry in night["would_apply"]:
                ids = [str(i) for i in entry.get("affected_ids") or []]
                collection = PROD_LEARNING if entry.get("proposal_type") == "learning_promotion_candidate" else PROD_MEMORY
                points = admin.retrieve(collection, ids, with_payload=True) if ids else []
                self.samples.append({
                    "arm": "prod (report-only, real data)",
                    "night": night["night"],
                    "proposal_type": entry.get("proposal_type"),
                    "action": entry.get("action"),
                    "points": [self._sample_point(point) for point in points],
                })

    @staticmethod
    def _sample_point(point: dict[str, Any]) -> dict[str, Any]:
        payload = point.get("payload") or {}
        return {"id": str(point.get("id")), "source_type": payload.get("source_type"), "created_at": payload.get("created_at"),
                "importance": payload.get("importance"), "confidence": payload.get("confidence"),
                "access_count": payload.get("access_count"), "seed": payload.get("cp6_e2e_seed"),
                "text": str(payload.get("text") or payload.get("lesson") or "")[:400]}

    def notifier_check(self, arm: str, home: Path) -> None:
        """Run the installed notifier script read-only against the arm's state copy."""
        script = Path.home() / ".hermes" / "scripts" / "qdrant_native_watcher_notify.py"
        if not script.exists():
            self.check(f"{arm}: notifier script present", False, path=str(script))
            return
        import subprocess

        env = {**os.environ, "HERMES_HOME": str(home), "QDRANT_WATCHER_NOTIFY_STATE_PATH": str(home / "notify_state.json")}
        result = subprocess.run([sys.executable, "-I", str(script)], capture_output=True, text=True, env=env, timeout=60)
        self.receipt["arms"][arm]["notifier"] = {"exit_code": result.returncode, "message": result.stdout}
        self.check(f"{arm}: 03:05 notifier reads the new state", result.returncode == 0, exit_code=result.returncode, stderr=result.stderr[-500:])

    # -- oracles on the guarded arm ---------------------------------------------------------
    def verify_guarded(self) -> None:
        from qdrant_memory.backup import read_backup_manifest, restore_backup

        url = self.args.qdrant_url
        home = self.out / "home_tmp"
        config = _config(home, self.tmp_memory, self.tmp_learning, url)
        nights = self.receipt["arms"]["tmp"]["nights"]
        app_dir = home / "qdrant_memory" / "consolidation" / "applications"
        records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(app_dir.glob("application-*.json"))] if app_dir.exists() else []
        applied_total = sum(len(n["applied"]) for n in nights)
        self.check("every applied action has an application record", len(records) == applied_total, records=len(records), applied=applied_total)
        applied_ids: set[str] = set()
        reports: dict[str, dict[str, Any]] = {}
        bad: list[str] = []
        for record in records:
            report_id = str(record.get("report_id"))
            if report_id not in reports:
                reports[report_id] = json.loads((home / "qdrant_memory" / "consolidation" / f"report-{report_id}.json").read_text(encoding="utf-8"))
            sealed = next(p for p in reports[report_id]["proposals"] if p["proposal_id"] == record["proposal_id"])
            ids = [str(i) for i in record.get("affected_ids") or []]
            applied_ids |= set(ids)
            try:
                full = read_backup_manifest(config, hermes_home=str(home), backup_id=record["pre_apply_backup_id"])
                assert full.get("partial") is not True
                assert any(d.get("collection_name") == record["collection_name"] for d in full["collections"].values())
                snap = record["pre_mutation_snapshot"]
                part = read_backup_manifest(config, hermes_home=str(home), backup_id=snap["backup_id"])
                assert part.get("partial") is True
                assert snap["point_digests"] == sealed["guarded_auto_snapshot"]["point_digests"]
                assert part["snapshot"]["point_digests"] == snap["point_digests"]
                assert sorted(snap["point_ids"]) == sorted(ids)
            except Exception as exc:  # noqa: BLE001 - recorded as a failed oracle
                bad.append(f"{record.get('application_id')}: {exc!r}")
        self.check("every applied action has a full backup and a digest-bound snapshot", not bad, problems=bad[:5])
        backups_taken = sorted({n["backup"]["backup_id"] for n in nights if n.get("backup")})
        self.receipt["arms"]["tmp"]["full_backups"] = backups_taken

        # Diff the tmp collections against the starting backup.
        admin = _client(url)
        now: dict[str, str] = {}
        for name in (self.tmp_memory, self.tmp_learning):
            for point in _scroll_all(admin, name):
                now[str(point["id"])] = _sha(_canon(point.get("payload") or {}))
        deleted = sorted(set(self.start) - set(now))
        changed = sorted(pid for pid, digest in now.items() if pid in self.start and self.start[pid]["payload_sha256"] != digest)
        added = sorted(set(now) - set(self.start))
        self.check("no point was deleted or added", not deleted and not added, deleted=len(deleted), added=len(added))
        self.check("changed ids are exactly the applied ids", set(changed) == applied_ids, changed=len(changed), applied=len(applied_ids),
                   unexplained=sorted(set(changed) - applied_ids)[:5], unchanged_applied=sorted(applied_ids - set(changed))[:5])
        cited = [pid for pid in changed if self.start[pid]["session_origin"]]
        identity = [pid for pid in changed if self.start[pid]["identity"]]
        self.check("no point carrying session_origin was touched", not cited, touched=cited[:5])
        self.check("no identity-bearing point was touched", not identity, touched=identity[:5])
        seed_outcomes = {seed["kind"]: (seed["id"] in set(changed)) for seed in getattr(self, "seeds", [])}
        wrong = [seed["kind"] for seed in getattr(self, "seeds", []) if seed_outcomes[seed["kind"]] != seed["expect_touched"]]
        self.check("control seeds: eligible ones applied, decoys (cited, money, security, identity) untouched", not wrong,
                   outcomes=seed_outcomes, wrong=wrong)
        real_changed = [pid for pid in changed if not self.start[pid].get("seed")]
        self.receipt["arms"]["tmp"]["real_points_changed"] = len(real_changed)
        self.receipt["arms"]["tmp"]["diff"] = {"changed": len(changed), "deleted": len(deleted), "added": len(added)}

        # Samples for human review (outputs only, never committed).
        for record in records:
            snap = record["pre_mutation_snapshot"]
            backup_dir = Path(snap["backup_dir"])
            scope = "memory" if record["collection_name"] == self.tmp_memory else "learning"
            texts = [self._sample_point(point) for point in _iter_backup_records(backup_dir, scope)]
            self.samples.append({"arm": "tmp (guarded-auto, applied)", "report_id": record["report_id"],
                                 "proposal_type": record["proposal_type"], "action": record["action"], "points": texts})

        self.verify_retention_and_hold(records)
        if self.args.embed_seeds:
            self.verify_quarantine_recall(records)

        # Undo one action through its snapshot backup (a quarantine, when there is one).
        if records:
            record = next((r for r in records if r.get("action") == "quarantine"), records[0])
            target = str(record["affected_ids"][-1])
            restored = restore_backup(admin, config, hermes_home=str(home), backup_id=record["pre_mutation_snapshot"]["backup_id"], dry_run=False)
            point = admin.retrieve(record["collection_name"], [target], with_payload=True)[0]
            ok = _sha(_canon(point.get("payload") or {})) == self.start[target]["payload_sha256"]
            self.check("restoring one action's snapshot backup returns the point to its starting payload", ok,
                       upserted=restored.get("upserted"), action=record["action"], proposal_type=record["proposal_type"])
            if self.args.embed_seeds:
                self.recall_control(target)

    def cleanup(self) -> None:
        if self.args.keep:
            return
        admin = _client(self.args.qdrant_url)
        existing = set(admin.get_collections())
        for name in (self.tmp_memory, self.tmp_learning):
            if name in existing:
                admin._request("DELETE", f"/collections/{name}")
        remaining = [name for name in admin.get_collections() if name.startswith(TMP_PREFIX)]
        self.check("temporary collections deleted", not remaining, remaining=remaining)

    def main(self) -> int:
        self.fence.install()
        self.recorder.install()
        arms = [arm.strip() for arm in self.args.arms.split(",") if arm.strip()]
        try:
            if "prod" in arms:
                self.run_arm("prod")
            if "tmp" in arms:
                self.snapshot_production()
                self.seed_controls()
                self.run_arm("tmp")
                self.verify_guarded()
        except Exception:
            self.check("harness completed", False, error=traceback.format_exc()[-3000:])
        finally:
            self.cleanup()
            self.check(f"production write fence: no write attempt outside {TMP_PREFIX}", not self.fence.refused,
                       refused=self.fence.refused[:5], tmp_writes=dict(self.fence.writes))
            (self.out / "receipt.json").write_text(json.dumps(self.receipt, indent=2, sort_keys=True, default=str), encoding="utf-8")
            (self.out / "samples-private.json").write_text(json.dumps(self.samples, indent=2, ensure_ascii=False), encoding="utf-8")
            os.chmod(self.out / "samples-private.json", 0o600)
        failed = [c["name"] for c in self.receipt["checks"] if not c["ok"]]
        print(json.dumps({"receipt": str(self.out / "receipt.json"), "failed": failed}, indent=2))
        return 1 if failed else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--nights", type=int, default=7, help="Nights for the guarded-auto tmp arm.")
    parser.add_argument("--prod-nights", type=int, default=7, help="Nights for the report-only production arm.")
    parser.add_argument("--max-points", type=int, default=300)
    parser.add_argument("--arms", default="prod,tmp")
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--tmp-prefix", default=TMP_PREFIX,
                        help="Prefix of the temporary collections (the only ones the fence lets the run write).")
    parser.add_argument("--fail-backup-night", type=int, default=0,
                        help="Tmp arm: make the pre-mutation backup fail on this night (0 = never). The oracle "
                             "checks that the cursor is held and the window is read again the next night.")
    parser.add_argument("--backup-keep-last", type=int, default=7, help="Tmp arm: guarded_auto_backup_keep_last.")
    parser.add_argument("--backup-keep-days", type=int, default=30, help="Tmp arm: guarded_auto_backup_keep_days.")
    parser.add_argument("--embed-seeds", action="store_true",
                        help="Embed the control seeds' text with the configured embedding server, so a recall "
                             "query with a seed's text matches it (needed for the quarantine recall oracle).")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(Run(parse_args()).main())
