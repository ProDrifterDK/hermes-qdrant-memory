"""C'6: routes that used to lose a recorded session_origin, and the CLI ``store --cite``.

Real path: the provider's tools, the real backup restore and the real CLI command path,
against a real Qdrant in disposable collections that must start with ``tmp_c6_``, and a
synthetic ``state.db`` built from the dumped host schema (``tests/fixtures/state_db_schema.sql``)
that the product opens read-only. Vectors come from the deterministic integration embedder.
Expected digests are computed here from the literal rows (design 5.2 / G1 byte formats).

One test per route class that lost an origin before C'6, plus the CLI:
  1 re-store with the mode switched back to off (learning store, extraction approve replay)
                                     -- the off-mode upsert replacing the payload (C'3 M1)
  2 backup restore of a point whose snapshot predates a citation
                                     -- restore putting the snapshot's mapping back
  3 consolidation merge of two cited duplicates
                                     -- the duplicates' citations deleted with them
  4 CLI store --cite: preview in one process, approve in another; no preview refuses
                                     -- no span citation from the CLI, or a skipped preview

Run: RUN_QDRANT_INTEGRATION=1 QDRANT_TEST_URL=http://127.0.0.1:6333 pytest tests/integration/test_live_session_origin_c6.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pytest

from conftest import DeterministicEmbedding, _delete_and_verify
from qdrant_memory.client import QdrantClient
from qdrant_memory.config import load_config

TRUTHY = {"1", "true", "yes", "on"}
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_QDRANT_INTEGRATION", "").strip().lower() not in TRUTHY,
    reason="set RUN_QDRANT_INTEGRATION=1 to run live Qdrant integration tests",
)

REPO = Path(__file__).resolve().parents[2]
SCHEMA = REPO / "tests" / "fixtures" / "state_db_schema.sql"
PREFIX = "tmp_c6_"
_FORBIDDEN = {"hermes_memory", "hermes_learnings", "hermes_session_windows"}
VECTOR_SIZE = 64

SID_A = "20261115_090000_c6aaaa"
SID_B = "20261115_100000_c6bbbb"
SID_X = "20261115_110000_c6xxxx"
T0 = 1_794_000_000.5
DECISION = "Decision: rotate the staging certificates every sixty days."
ROWS = [  # (session, role, content, message_uid); ids come from AUTOINCREMENT in this order
    (SID_A, "user", "Where do the nightly backups go?", "a-u1"),
    (SID_A, "assistant", "To the second disk, under /srv/backup.", "a-a2"),
    (SID_B, "user", "Remind me where the nightly backups go.", "b-u1"),
    (SID_B, "assistant", "They go to the second disk under /srv/backup.", "b-a2"),
    (SID_X, "user", "What did we agree on certificates?", "x-u1"),
    (SID_X, "assistant", DECISION + "\nI will note it.", "x-a2"),
]
HOOK_X = [{"role": "user", "content": "What did we agree on certificates?"},
          {"role": "assistant", "content": DECISION + "\nI will note it."}]


def _build_state_db(path: Path) -> dict[str, int]:
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA.read_text(encoding="utf-8"))
    for sid in (SID_A, SID_B, SID_X):
        conn.execute("INSERT INTO sessions (id, source, started_at) VALUES (?, 'cli', ?)", (sid, T0))
    ids: dict[str, int] = {}
    for index, (sid, role, content, uid) in enumerate(ROWS):
        cur = conn.execute("INSERT INTO messages (session_id, role, content, timestamp, message_uid) "
                           "VALUES (?, ?, ?, ?, ?)", (sid, role, content, T0 + 60.0 * index, uid))
        ids[uid] = int(cur.lastrowid)
    conn.commit()
    conn.close()
    return ids


def _span(sid: str, uids: list[str], ids: dict[str, int]) -> dict[str, Any]:
    """The pinned fields of a span citation, from the literal rows."""
    by_uid = {uid: (role, content) for _sid, role, content, uid in ROWS}
    span, content, uid_list = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    for uid in uids:
        role, text = by_uid[uid]
        span.update(b"hermes-span-v1\x1f" + sid.encode() + b"\x1f" + str(ids[uid]).encode() + b"\x1f"
                    + role.encode() + b"\x1f" + text.encode("utf-8") + b"\x1e")
        content.update(b"hermes-span-content-v1\x1f" + sid.encode() + b"\x1f" + uid.encode() + b"\x1f"
                       + role.encode() + b"\x1f" + text.encode("utf-8") + b"\x1e")
        uid_list.update(b"hermes-uids-v1\x1f" + uid.encode() + b"\x1e")
    return {"session_id": sid, "first_msg_id": ids[uids[0]], "last_msg_id": ids[uids[-1]], "msg_count": len(uids),
            "span_sha256": span.hexdigest(), "content_sha256": content.hexdigest(),
            "uids_sha256": uid_list.hexdigest()}


def _pinned(citation: dict[str, Any]) -> dict[str, Any]:
    keys = ("session_id", "first_msg_id", "last_msg_id", "msg_count", "span_sha256", "content_sha256", "uids_sha256")
    return {key: citation.get(key) for key in keys}


@dataclass(frozen=True)
class Live:
    qdrant: QdrantClient
    memory: str
    learnings: str


@pytest.fixture
def live() -> Iterator[Live]:
    qdrant_url = os.environ.get("QDRANT_TEST_URL", "").strip()
    if not qdrant_url:
        pytest.skip("set QDRANT_TEST_URL to run live Qdrant integration tests")
    qdrant = QdrantClient(qdrant_url, timeout=10.0, api_key=os.environ.get("QDRANT_TEST_API_KEY", ""))
    suffix = uuid.uuid4().hex[:12]
    names = (f"{PREFIX}{suffix}_memory", f"{PREFIX}{suffix}_learnings")
    existing = set(qdrant.get_collections())
    for name in names:
        if not name.startswith(PREFIX) or name in _FORBIDDEN or name in existing:
            pytest.fail(f"refusing test collection {name!r}")
    created: list[str] = []
    try:
        for name in names:
            if qdrant.ensure_collection(name, VECTOR_SIZE, "Cosine").get("exists"):
                pytest.fail(f"refusing to reuse concurrently created collection {name!r}")
            created.append(name)
        yield Live(qdrant=qdrant, memory=names[0], learnings=names[1])
    finally:
        _delete_and_verify(qdrant, created, PREFIX)


@pytest.fixture
def state(tmp_path) -> tuple[Path, dict[str, int]]:
    db = tmp_path / "state.db"
    return db, _build_state_db(db)


def _provider(live: Live, db: Path, home: Path, session_id: str, mode: str = "recorded"):
    from __init__ import QdrantMemoryProvider
    from qdrant_memory.writer import ConversationWriter

    provider = QdrantMemoryProvider()
    provider._active = True
    provider._hermes_home = str(home)
    provider._profile_id = "c6-oracle"
    provider._platform = "pytest"
    provider._session_id = session_id
    provider._config = load_config(hermes_home=str(home), hermes_config={"qdrant_memory": {
        "collection_name": live.memory,
        "learning_collection_name": live.learnings,
        "session_recall_collection": live.memory + "_windows_unused",
        "session_recall_state_db": str(db),
        "session_origin_mode": mode,
        "vector_size": VECTOR_SIZE,
        "lineage_lock_dir": str(home / "locks"),
        "source_extraction_enabled": True,
    }})
    for key in ("collection_name", "learning_collection_name"):
        assert str(provider._config[key]).startswith(PREFIX), f"{key} resolved to {provider._config[key]!r}"
    provider._qdrant = live.qdrant
    provider._embeddings = DeterministicEmbedding(VECTOR_SIZE)
    provider._writer = ConversationWriter(qdrant=live.qdrant, embeddings=provider._embeddings,
                                          collection_name=live.memory, profile_id="c6-oracle", platform="pytest",
                                          session_id=session_id)
    return provider


def _call(provider, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    return json.loads(provider.handle_tool_call(tool, args))


def _origin(live: Live, collection: str, point_id: str) -> dict[str, Any] | None:
    points = live.qdrant.retrieve(collection, [point_id], with_payload=True)
    return (points[0]["payload"] if points else {}).get("session_origin")


def _store_cited(provider, text: str, sid: str, first: int, last: int) -> str:
    cite = {"session_id": sid, "first_msg_id": first, "last_msg_id": last}
    dry = _call(provider, "qdrant_memory_store", {"text": text, "cite": cite})
    assert dry["session_origin"]["citation_status"] == "verified", dry
    live_result = _call(provider, "qdrant_memory_store", {"text": text, "cite": cite, "dry_run": False,
                                                          "approve": True})
    assert live_result.get("saved") is True, live_result
    return str(live_result["id"])


# --- the cases ----------------------------------------------------------------


def test_restores_with_the_mode_back_off_keep_the_recorded_origin(live, state, tmp_path):
    db, _ids = state
    lesson = {"lesson": "Rotate the staging certificates before they expire.", "learning_type": "workflow_lesson",
              "trigger": "certificate expiry", "correction": "rotate every sixty days"}
    recorded = _provider(live, db, tmp_path, SID_A)
    learning_id = _call(recorded, "qdrant_learning_store", dict(lesson))["id"]
    recorded_x = _provider(live, db, tmp_path, SID_X)
    recorded_x.on_pre_compress(HOOK_X)
    (candidate_id, candidate), = recorded_x._pending_extraction_candidates.items()
    _call(recorded_x, "qdrant_memory_extraction_approve", {"candidate_id": candidate_id})
    approved = _call(recorded_x, "qdrant_memory_extraction_approve",
                     {"candidate_id": candidate_id, "dry_run": False, "approve": True})
    learning_before = _origin(live, live.learnings, learning_id)
    extraction_before = _origin(live, live.memory, candidate_id)
    assert learning_before and learning_before["citations"], learning_before
    assert extraction_before and extraction_before["citation_status"] == "verified", extraction_before

    off = _provider(live, db, tmp_path, SID_B, mode="off")
    restored = _call(off, "qdrant_learning_store", {**lesson, "lesson": lesson["lesson"]})
    # Extraction candidate ids carry the payload's created_at, so only a replay of the same
    # candidate lands on the stored point: exactly the overwrite this route must survive.
    off_x = _provider(live, db, tmp_path, SID_X, mode="off")
    off_x._pending_extraction_candidates[candidate_id] = candidate
    _call(off_x, "qdrant_memory_extraction_approve", {"candidate_id": candidate_id})
    replayed = _call(off_x, "qdrant_memory_extraction_approve",
                     {"candidate_id": candidate_id, "dry_run": False, "approve": True})

    assert approved["saved"] is True and restored["id"] == learning_id and replayed["id"] == candidate_id
    assert _origin(live, live.learnings, learning_id) == learning_before
    assert _origin(live, live.memory, candidate_id) == extraction_before


def test_backup_restore_keeps_citations_recorded_after_the_snapshot(live, state, tmp_path):
    from qdrant_memory.backup import create_backup, restore_backup

    db, ids = state
    text = "The nightly backups go to the second disk under /srv/backup."
    provider_a = _provider(live, db, tmp_path, SID_A)
    point_id = _store_cited(provider_a, text, SID_A, ids["a-u1"], ids["a-a2"])
    backup = create_backup(live.qdrant, provider_a._config, hermes_home=str(tmp_path), scope="both")
    time.sleep(1.1)
    provider_b = _provider(live, db, tmp_path, SID_B)
    _store_cited(provider_b, text, SID_B, ids["b-u1"], ids["b-a2"])
    live_origin = _origin(live, live.memory, point_id)

    plan = restore_backup(live.qdrant, provider_a._config, hermes_home=str(tmp_path),
                          backup_id=backup["backup_id"], dry_run=True)
    result = restore_backup(live.qdrant, provider_a._config, hermes_home=str(tmp_path),
                            backup_id=backup["backup_id"], dry_run=False)

    origin = _origin(live, live.memory, point_id)
    assert result["applied"] is True and point_id in result["collections"]["memory"]["changed_ids"]
    assert plan["collections"]["memory"]["session_origin_kept_ids"] == [point_id]
    assert [_pinned(c) for c in origin["citations"]] == [_span(SID_A, ["a-u1", "a-a2"], ids),
                                                        _span(SID_B, ["b-u1", "b-a2"], ids)]
    assert origin == live_origin


def test_consolidation_merge_moves_the_duplicates_citations_to_the_canonical_point(live, state, tmp_path):
    db, ids = state
    provider_a = _provider(live, db, tmp_path, SID_A)
    first = _store_cited(provider_a, "The nightly backups go to the second disk.", SID_A, ids["a-u1"], ids["a-a2"])
    time.sleep(1.1)
    provider_b = _provider(live, db, tmp_path, SID_B)
    second = _store_cited(provider_b, "the nightly backups go to the second disk", SID_B, ids["b-u1"], ids["b-a2"])
    report = _call(provider_a, "qdrant_memory_consolidate", {"scope": "memory"})
    proposal = next(p for p in report["proposals"] if set(p.get("affected_ids") or []) == {first, second})

    dry = _call(provider_a, "qdrant_memory_consolidation_apply", {"proposal_id": proposal["proposal_id"],
                                                                  "report_id": report["report_id"]})
    applied = _call(provider_a, "qdrant_memory_consolidation_apply", {
        "proposal_id": proposal["proposal_id"], "report_id": report["report_id"], "dry_run": False,
        "approve": True, "action": "merge"})

    canonical = applied["canonical_id"]
    origin = _origin(live, live.memory, canonical)
    assert dry["session_origin_merge"]["citation_count"] == 2, dry
    assert [_pinned(c) for c in origin["citations"]] == [_span(SID_A, ["a-u1", "a-a2"], ids),
                                                        _span(SID_B, ["b-u1", "b-a2"], ids)]
    assert origin["captured_session_id"] == SID_A  # the earliest capture stays the first capture
    assert live.qdrant.retrieve(live.memory, [p for p in (first, second) if p != canonical]) == []


def _cli_parser() -> argparse.ArgumentParser:
    spec = importlib.util.spec_from_file_location("c6_plugin_cli", REPO / "cli.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = argparse.ArgumentParser(prog="hermes qdrant")
    module.register_cli(parser)
    return parser


def test_cli_store_cite_previews_in_one_process_and_approves_in_another(live, state, tmp_path):
    from qdrant_memory.cli_core import execute_command

    db, ids = state
    parser = _cli_parser()
    text = "The nightly backups go to the second disk, per the CLI."

    def run(argv: list[str]) -> tuple[int, dict[str, Any]]:
        out, err = io.StringIO(), io.StringIO()
        # A fresh provider per command, as each CLI command is its own process.
        code = execute_command(parser.parse_args(argv), provider_factory=lambda: _provider(live, db, tmp_path, "cli"),
                               stdout=out, stderr=err)
        raw = (out.getvalue() or err.getvalue()).strip()
        return code, json.loads(raw) if raw.startswith("{") else {"raw": raw}

    cite_a = ["--cite", f"{ids['a-u1']}:{ids['a-a2']}", "--session-id", SID_A]
    preview_code, preview = run(["store", text, *cite_a, "--json"])
    approve_code, approved = run(["store", text, *cite_a, "--no-dry-run", "--approve", "--json"])
    cite_b = ["--cite", f"{ids['b-u1']}:{ids['b-a2']}", "--session-id", SID_B]
    blind_code, blind = run(["store", text + " Blind.", *cite_b, "--no-dry-run", "--approve", "--json"])
    usage_code, _ = run(["store", text, "--cite", f"{ids['a-u1']}:{ids['a-a2']}", "--json"])

    assert preview_code == 0 and preview["session_origin"]["citation_status"] == "verified", preview
    assert approve_code == 0 and approved["saved"] is True, approved
    origin = _origin(live, live.memory, approved["id"])
    assert [_pinned(c) for c in origin["citations"]] == [_span(SID_A, ["a-u1", "a-a2"], ids)]
    assert blind_code == 1 and "needs a preview" in json.dumps(blind), blind
    assert usage_code == 2  # --cite without --session-id
