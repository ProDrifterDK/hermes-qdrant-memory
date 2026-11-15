"""C'1 oracle: the read-only state.db span resolver and verifier.

The fixture database is built from the dumped production schema
(``tests/fixtures/state_db_schema.sql``: real columns, triggers and FTS tables). Each
case mutates it the way Hermes itself does (in-place UPDATE, prune DELETE, compaction
clone, destructive delete, schema drift) and asserts the one status the contract
requires. Expected digests are computed here from the literal rows and the byte format
of design section 5.2, never through the module under test.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from qdrant_memory.session_origin import resolve_span, verify_citation

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = Path(__file__).resolve().parent / "fixtures" / "state_db_schema.sql"
SID = "20261101_101500_c1a2b3"
OTHER = "20261101_101700_d4e5f6"
T0 = 1_793_000_000.25
# Built at runtime so the literal-secret scan never sees a token in the source. A digest
# over redacted text would not match the independent digest of this raw row.
SECRET_SHAPED = "deploy with sk-" + "".join(["abcd", "efgh", "ijkl", "mnop"])

# (session, role, content, message_uid). Ids are assigned by AUTOINCREMENT in this order.
ROWS = [
    (SID, "user", "Pin the retention at 180 days, please.", "uid-a1"),
    (SID, "assistant", "", "uid-a2"),  # tool-call-only assistant turn: not eligible
    (OTHER, "user", "unrelated session row inside the id range", "uid-o1"),
    (SID, "tool", '{"ok": true}', "uid-a3"),  # tool output: not eligible
    (SID, "assistant", "Agreed: retention_days 180, caf\u00e9 \u2713\n", "uid-a4"),
    (SID, "user", f"  {SECRET_SHAPED}  ", "uid-a5"),
    (SID, "assistant", "Done.", "uid-a6"),
    (SID, "user", "after the span", "uid-a7"),
]
CITED_UIDS = ["uid-a1", "uid-a4", "uid-a5", "uid-a6"]


def _build(path: Path) -> dict[str, int]:
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA.read_text(encoding="utf-8"))
    for sid in (SID, OTHER):
        conn.execute("INSERT INTO sessions (id, source, started_at) VALUES (?, 'cli', ?)", (sid, T0))
    ids: dict[str, int] = {}
    for index, (sid, role, content, uid) in enumerate(ROWS):
        cur = conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp, message_uid) VALUES (?, ?, ?, ?, ?)",
            (sid, role, content, T0 + 60.0 * index, uid),
        )
        ids[uid] = int(cur.lastrowid)
    conn.commit()
    conn.close()
    return ids


def _expected_span_sha256(ids: dict[str, int]) -> str:
    by_uid = {uid: (role, content) for _sid, role, content, uid in ROWS}
    digest = hashlib.sha256()
    for uid in CITED_UIDS:
        role, content = by_uid[uid]
        digest.update(
            b"hermes-span-v1\x1f" + SID.encode() + b"\x1f" + str(ids[uid]).encode() + b"\x1f"
            + role.encode() + b"\x1f" + content.encode("utf-8") + b"\x1e"
        )
    return digest.hexdigest()


def _cite(tmp_path: Path):
    db = tmp_path / "state.db"
    ids = _build(db)
    citation = resolve_span(str(db), SID, ids["uid-a1"], ids["uid-a6"], basis="agent_proposed")
    assert isinstance(citation, dict), citation
    return db, ids, citation


def test_intact_span_verifies_with_independent_digest(tmp_path):
    db, ids, citation = _cite(tmp_path)

    assert citation["span_sha256"] == _expected_span_sha256(ids)
    assert citation["msg_count"] == 4
    assert citation["store"] == "hermes-state.db"
    assert citation["verified_at"].endswith("Z")
    assert verify_citation(citation, state_db=str(db))["status"] == "verified"


def test_in_place_update_is_drifted(tmp_path):
    db, ids, citation = _cite(tmp_path)
    conn = sqlite3.connect(db)
    conn.execute("UPDATE messages SET content = 'Agreed: retention_days 90' WHERE id = ?", (ids["uid-a4"],))
    conn.commit()
    conn.close()

    assert verify_citation(citation, state_db=str(db))["status"] == "drifted"


def test_pruned_session_is_expired(tmp_path):
    db, _ids, citation = _cite(tmp_path)
    conn = sqlite3.connect(db)
    # hermes_state_maintenance.prune_sessions: fence, orphan children, delete messages, delete session.
    conn.execute("INSERT INTO state_meta (key, value) VALUES (?, '{}')", ("gateway.retired_session.v1." + SID,))
    conn.execute("UPDATE sessions SET parent_session_id = NULL WHERE parent_session_id = ?", (SID,))
    conn.execute("DELETE FROM messages WHERE session_id = ?", (SID,))
    conn.execute("DELETE FROM sessions WHERE id = ?", (SID,))
    conn.commit()
    conn.close()

    result = verify_citation(citation, state_db=str(db))
    assert result["status"] == "expired"
    assert result["retired_tombstone"] == "gateway.retired_session.v1." + SID


def test_compaction_clone_with_original_deleted_is_verified_via_uid(tmp_path):
    db, ids, citation = _cite(tmp_path)
    conn = sqlite3.connect(db)
    # Same column copy as hermes_state_messages._clone_message_rows, then drop the original.
    columns = [r[1] for r in conn.execute("PRAGMA table_info(messages)")]
    copied = ", ".join(c for c in columns if c not in ("id", "active", "compacted", "display_order"))
    conn.execute(
        f"INSERT INTO messages ({copied}, active, compacted) SELECT {copied}, 1, 0 FROM messages WHERE id = ?",
        (ids["uid-a4"],),
    )
    conn.execute("DELETE FROM messages WHERE id = ?", (ids["uid-a4"],))
    conn.commit()
    clone_id = conn.execute("SELECT id FROM messages WHERE message_uid = 'uid-a4'").fetchone()[0]
    conn.close()
    assert clone_id > ids["uid-a7"]

    assert verify_citation(citation, state_db=str(db))["status"] == "verified_via_uid"


def test_destructive_delete_is_partially_missing(tmp_path):
    db, ids, citation = _cite(tmp_path)
    conn = sqlite3.connect(db)
    conn.execute("DELETE FROM messages WHERE id = ?", (ids["uid-a5"],))
    conn.commit()
    conn.close()

    assert verify_citation(citation, state_db=str(db))["status"] == "partially_missing"


def test_dropped_required_column_is_unavailable(tmp_path):
    db, ids, citation = _cite(tmp_path)
    conn = sqlite3.connect(db)
    conn.execute("DROP TRIGGER messages_message_uid_insert")
    conn.execute("DROP INDEX idx_messages_session_uid")
    conn.execute("ALTER TABLE messages DROP COLUMN message_uid")
    conn.commit()
    conn.close()

    assert verify_citation(citation, state_db=str(db))["status"] == "unavailable"
    assert resolve_span(str(db), SID, ids["uid-a1"], ids["uid-a6"], basis="agent_proposed") == (
        "unresolved:schema_unsupported"
    )


_TZ_PROBE = """
import json, sys
from datetime import datetime
from qdrant_memory.session_origin import resolve_span
c = resolve_span(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), basis="agent_proposed")
print(json.dumps({"span": c["span_sha256"], "local": datetime.fromtimestamp(float(sys.argv[5])).isoformat()}))
"""


def test_digest_is_identical_across_host_timezones(tmp_path):
    db = tmp_path / "state.db"
    ids = _build(db)
    out = {}
    for tz in ("UTC", "America/Santiago"):
        proc = subprocess.run(
            [sys.executable, "-c", _TZ_PROBE, str(db), SID, str(ids["uid-a1"]), str(ids["uid-a6"]), str(T0)],
            cwd=ROOT, env={"TZ": tz, "PYTHONPATH": str(ROOT)}, capture_output=True, text=True, timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        out[tz] = json.loads(proc.stdout)

    # Precondition: the two processes really render local time differently.
    assert out["UTC"]["local"] != out["America/Santiago"]["local"]
    assert out["UTC"]["span"] == out["America/Santiago"]["span"]
    assert out["UTC"]["span"] == _expected_span_sha256(ids)
