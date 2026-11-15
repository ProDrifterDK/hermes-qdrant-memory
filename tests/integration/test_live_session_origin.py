"""C'2 oracle: session_origin on the manual store, end to end.

Real path: the provider's ``qdrant_memory_store`` / ``_inspect`` / ``_forget`` tools
against a real Qdrant, in disposable collections that must start with ``tmp_c2_``, and a
synthetic ``state.db`` built from the dumped production schema
(``tests/fixtures/state_db_schema.sql``) that the product opens read-only. The test edits
that fixture directly where a case needs Hermes to have rewritten a row. Vectors come from
the deterministic integration embedder: these cases are about payloads, not ranking.

Expected digests are computed here from the literal rows and the byte formats of design
5.2 / amendment G1, never through the module under test.

Cases (design 5.9, C'2) and the failure each one names:
  1 store with ``cite`` records the exact digest      -- a wrong or recomputed pin
  2 a wrong ``last_msg_id`` is refused                 -- a tampered cite stored as pinned
  3 a row mutated between preview and approve refuses  -- approve without re-verification
  4 the same text from session B gives ``[A, B]`` and keeps the first capture time
                                                        -- replace semantics on re-store
  5 ``forget`` deletes a C' point (control)            -- C' fields making points unforgettable
  6 an oversized mapping is refused by the write path  -- the 4 KiB cap not enforced
  7 ``session_origin_mode: off`` is byte-identical to the pre-C'2 tree
                                                        -- off mode leaking schema/payload/output

Run: RUN_QDRANT_INTEGRATION=1 QDRANT_TEST_URL=http://127.0.0.1:6333 pytest tests/integration/test_live_session_origin.py
Set SESSION_ORIGIN_RECEIPT=<path> to write the JSON receipt of the run there.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
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
DRIVER = Path(__file__).resolve().parent / "session_origin_offmode_driver.py"
HERMES_AGENT = Path.home() / ".hermes" / "hermes-agent"
PREFIX = "tmp_c2_"
# main at the branch point of C'2: the tree off mode must be byte-identical to.
C2_BASE = "5904628"
_FORBIDDEN = {"hermes_memory", "hermes_learnings", "hermes_session_windows"}
VECTOR_SIZE = 64

SID_A = "20261115_090000_aaaa01"
SID_B = "20261115_100000_bbbb02"
SID_Q = "20261115_110000_cccc03"
T0 = 1_794_000_000.5
TEXT = "The staging proxy listens on port 8443."
_FILLER = ("The rollout notes for the staging proxy mention ports, certificates, health checks and the "
           "weekly maintenance window in plain words. ")
# (session, role, content, message_uid); ids come from AUTOINCREMENT in this order.
ROWS = [
    (SID_A, "user", "Which port does the staging proxy listen on?", "a-u1"),
    (SID_A, "assistant", "The staging proxy listens on port 8443.", "a-a2"),
    (SID_A, "tool", '{"ok": true}', "a-t3"),
    (SID_A, "user", "Good, write that down.", "a-u4"),
    (SID_A, "assistant", "Stored.", "a-a5"),
    (SID_B, "user", "Remind me of the staging proxy port.", "b-u1"),
    (SID_B, "assistant", "The staging proxy listens on port 8443.", "b-a2"),
    (SID_Q, "user", "first long note: " + _FILLER * 12, "q-1"),
    (SID_Q, "assistant", "second long note: " + _FILLER * 12, "q-2"),
    *[(SID_Q, "user" if n % 2 else "assistant", f"short note number {n}", f"q-{n}") for n in range(3, 11)],
]
CITE_A_UIDS = ["a-u1", "a-a2", "a-u4", "a-a5"]  # the tool row inside the range is not eligible
CITE_B_UIDS = ["b-u1", "b-a2"]

RECEIPT: dict[str, Any] = {"oracle": "C'2 session_origin manual store", "cases": {}}


# --- fixtures -----------------------------------------------------------------


def _build_state_db(path: Path) -> dict[str, int]:
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA.read_text(encoding="utf-8"))
    for sid in (SID_A, SID_B, SID_Q):
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


def _expected_digests(sid: str, uids: list[str], ids: dict[str, int]) -> dict[str, str]:
    """design 5.2 span_sha256, G1 content_sha256 and uids_sha256, from the literal rows."""
    by_uid = {uid: (role, content) for _sid, role, content, uid in ROWS}
    span, content, uid_list = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    for uid in uids:
        role, text = by_uid[uid]
        span.update(b"hermes-span-v1\x1f" + sid.encode() + b"\x1f" + str(ids[uid]).encode() + b"\x1f"
                    + role.encode() + b"\x1f" + text.encode("utf-8") + b"\x1e")
        content.update(b"hermes-span-content-v1\x1f" + sid.encode() + b"\x1f" + uid.encode() + b"\x1f"
                       + role.encode() + b"\x1f" + text.encode("utf-8") + b"\x1e")
        uid_list.update(b"hermes-uids-v1\x1f" + uid.encode() + b"\x1e")
    return {"span_sha256": span.hexdigest(), "content_sha256": content.hexdigest(),
            "uids_sha256": uid_list.hexdigest()}


@dataclass(frozen=True)
class Live:
    qdrant: QdrantClient
    qdrant_url: str
    memory: str
    learnings: str


@pytest.fixture
def live() -> Iterator[Live]:
    qdrant_url = os.environ.get("QDRANT_TEST_URL", "").strip()
    if not qdrant_url:
        pytest.skip("set QDRANT_TEST_URL to run live Qdrant integration tests")
    qdrant = QdrantClient(qdrant_url, timeout=10.0, **{"api_key": os.environ.get("QDRANT_TEST_API_KEY", "")})
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
        yield Live(qdrant=qdrant, qdrant_url=qdrant_url, memory=names[0], learnings=names[1])
    finally:
        _delete_and_verify(qdrant, created, PREFIX)


@pytest.fixture
def state(tmp_path) -> tuple[Path, dict[str, int]]:
    db = tmp_path / "state.db"
    return db, _build_state_db(db)


@pytest.fixture(scope="module", autouse=True)
def _receipt() -> Iterator[None]:
    RECEIPT["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    yield
    target = os.environ.get("SESSION_ORIGIN_RECEIPT", "").strip()
    if target:
        Path(target).write_text(json.dumps(RECEIPT, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _provider(live: Live, db: Path, home: Path, session_id: str, mode: str = "recorded"):
    from __init__ import QdrantMemoryProvider
    from qdrant_memory.writer import ConversationWriter

    provider = QdrantMemoryProvider()
    provider._active = True
    provider._hermes_home = str(home)
    provider._profile_id = "c2-oracle"
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
    }})
    for key in ("collection_name", "learning_collection_name"):
        assert str(provider._config[key]).startswith(PREFIX), f"{key} resolved to {provider._config[key]!r}"
    provider._qdrant = live.qdrant
    provider._embeddings = DeterministicEmbedding(VECTOR_SIZE)
    provider._writer = ConversationWriter(qdrant=live.qdrant, embeddings=provider._embeddings,
                                          collection_name=live.memory, profile_id="c2-oracle", platform="pytest",
                                          session_id=session_id)
    return provider


def _call(provider, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    return json.loads(provider.handle_tool_call(tool, args))


def _preview(provider, text: str, cite: dict[str, Any]) -> dict[str, Any]:
    return _call(provider, "qdrant_memory_store", {"text": text, "cite": cite})


def _approve(provider, text: str, cite: dict[str, Any]) -> dict[str, Any]:
    return _call(provider, "qdrant_memory_store", {"text": text, "cite": cite, "dry_run": False, "approve": True})


def _payload(live: Live, point_id: str) -> dict[str, Any] | None:
    points = live.qdrant.retrieve(live.memory, [point_id], with_payload=True)
    return points[0]["payload"] if points else None


def _point_id(provider, text: str) -> str:
    return provider._writer.preview_text(text, source_type="manual")["id"]


def _cite(sid: str, ids: dict[str, int], first: str, last: str, **extra: Any) -> dict[str, Any]:
    return {"session_id": sid, "first_msg_id": ids[first], "last_msg_id": ids[last], **extra}


# --- the oracle -------------------------------------------------------------


def test_store_with_cite_records_the_exact_digest(live, state, tmp_path):
    db, ids = state
    provider = _provider(live, db, tmp_path, SID_A)
    cite = _cite(SID_A, ids, "a-u1", "a-a5")
    expected = _expected_digests(SID_A, CITE_A_UIDS, ids)

    preview = _preview(provider, TEXT, cite)
    assert preview["session_origin"]["citation_status"] == "verified", preview
    for key, value in expected.items():
        assert preview["session_origin"]["citation"][key] == value, key
    stored = _approve(provider, TEXT, cite)
    assert stored.get("saved") is True, stored

    origin = _payload(live, stored["id"])["session_origin"]
    assert origin["schema"] == "session-origin-v1"
    assert origin["producer"] == "manual_store"
    assert origin["captured_session_id"] == SID_A
    assert origin["citation_status"] == "verified"
    assert len(origin["citations"]) == 1
    citation = origin["citations"][0]
    assert {k: citation[k] for k in expected} == expected
    assert (citation["session_id"], citation["first_msg_id"], citation["last_msg_id"], citation["msg_count"],
            citation["basis"]) == (SID_A, ids["a-u1"], ids["a-a5"], 4, "agent_proposed")
    inspected = _call(provider, "qdrant_memory_inspect", {"point_id": stored["id"]})
    assert inspected["session_origin"]["current_status_counts"] == {"verified": 1}, inspected
    RECEIPT["cases"]["exact_digest"] = {
        "expected": expected, "stored": {k: citation[k] for k in expected}, "msg_count": citation["msg_count"],
        "inspect_status_counts": inspected["session_origin"]["current_status_counts"], "pass": True,
    }


def test_a_wrong_last_msg_id_is_refused(live, state, tmp_path):
    db, ids = state
    provider = _provider(live, db, tmp_path, SID_A)
    wrong = _cite(SID_A, ids, "a-u1", "a-t3")  # the last id is a tool row: not a span edge

    preview = _preview(provider, TEXT, wrong)
    approve = _approve(provider, TEXT, wrong)

    assert preview["error"].startswith("cite refused (unresolved:no_match)"), preview
    assert "error" in approve, approve
    assert _payload(live, _point_id(provider, TEXT)) is None
    RECEIPT["cases"]["wrong_last_msg_id"] = {"preview_error": preview["error"], "approve_error": approve["error"],
                                             "point_written": False, "pass": True}


def test_a_row_mutated_between_preview_and_approve_is_refused(live, state, tmp_path):
    db, ids = state
    provider = _provider(live, db, tmp_path, SID_A)
    cite = _cite(SID_A, ids, "a-u1", "a-a5")
    preview = _preview(provider, TEXT, cite)
    assert preview["session_origin"]["citation_status"] == "verified", preview

    conn = sqlite3.connect(db)
    conn.execute("UPDATE messages SET content = ? WHERE id = ?", ("The staging proxy listens on port 9443.",
                                                                   ids["a-a2"]))
    conn.commit()
    conn.close()
    approve = _approve(provider, TEXT, cite)

    assert approve.get("error", "").startswith("cited span changed since preview: drifted"), approve
    assert _payload(live, _point_id(provider, TEXT)) is None
    RECEIPT["cases"]["mutated_between_preview_and_approve"] = {"approve_error": approve["error"],
                                                               "point_written": False, "pass": True}


def test_restore_from_session_b_merges_citations_and_keeps_first_capture(live, state, tmp_path):
    db, ids = state
    provider = _provider(live, db, tmp_path, SID_A)
    cite_a = _cite(SID_A, ids, "a-u1", "a-a5")
    _preview(provider, TEXT, cite_a)
    first = _approve(provider, TEXT, cite_a)
    assert first.get("saved") is True, first
    captured_at = _payload(live, first["id"])["session_origin"]["captured_at"]

    time.sleep(1.1)  # a replaced mapping would carry a later captured_at
    provider.on_session_switch(SID_B)
    cite_b = _cite(SID_B, ids, "b-u1", "b-a2")
    _preview(provider, TEXT, cite_b)
    second = _approve(provider, TEXT, cite_b)
    assert second.get("saved") is True and second["id"] == first["id"], second

    origin = _payload(live, first["id"])["session_origin"]
    assert [c["session_id"] for c in origin["citations"]] == [SID_A, SID_B]
    assert origin["citations"][0]["span_sha256"] == _expected_digests(SID_A, CITE_A_UIDS, ids)["span_sha256"]
    assert origin["citations"][1]["span_sha256"] == _expected_digests(SID_B, CITE_B_UIDS, ids)["span_sha256"]
    assert origin["captured_at"] == captured_at
    assert origin["captured_session_id"] == SID_A
    assert live.qdrant.count(live.memory) == 1
    RECEIPT["cases"]["restore_merges"] = {
        "citation_sessions": [c["session_id"] for c in origin["citations"]],
        "first_captured_at": captured_at, "stored_captured_at": origin["captured_at"],
        "captured_session_id": origin["captured_session_id"], "point_count": 1, "pass": True,
    }


def test_forget_deletes_a_c_prime_point(live, state, tmp_path):
    db, ids = state
    provider = _provider(live, db, tmp_path, SID_A)
    cite = _cite(SID_A, ids, "a-u1", "a-a5")
    _preview(provider, TEXT, cite)
    stored = _approve(provider, TEXT, cite)
    assert "session_origin" in _payload(live, stored["id"])

    forgot = _call(provider, "qdrant_memory_forget", {"ids": [stored["id"]], "dry_run": False})

    assert forgot == {"dry_run": False, "ids": [stored["id"]], "deleted": 1}, forgot
    assert _payload(live, stored["id"]) is None
    RECEIPT["cases"]["forget_deletes"] = {"forget": forgot, "point_after": None, "pass": True}


def test_an_oversized_mapping_is_refused_by_the_write_path(live, state, tmp_path):
    """Two quoted citations, then single-message citations previewed together against the
    same starting mapping (each fits alone) and approved in turn: the approve that would
    push the mapping past 4 KiB is refused by the writer's validator and writes nothing."""
    db, ids = state
    provider = _provider(live, db, tmp_path, SID_Q)
    quoted = [_cite(SID_Q, ids, uid, uid, quote=True) for uid in ("q-1", "q-2")]
    for cite in quoted:
        preview = _preview(provider, TEXT, cite)
        assert "quote" in (preview["session_origin"]["citation"] or {}), (
            f"precondition: a quote needs the host egress redactor: {preview['session_origin']}")
        assert _approve(provider, TEXT, cite).get("saved") is True
    point_id = _point_id(provider, TEXT)

    singles = [_cite(SID_Q, ids, f"q-{n}", f"q-{n}") for n in range(3, 11)]
    previews = [_preview(provider, TEXT, cite) for cite in singles]
    assert all("session_origin" in p for p in previews), previews
    sizes, refusal = [], None
    for cite in singles:
        before = _payload(live, point_id)
        result = _approve(provider, TEXT, cite)
        if "error" in result:
            refusal = result["error"]
            assert _payload(live, point_id) == before, "a refused approve must not write"
            break
        sizes.append(len(json.dumps(_payload(live, point_id)["session_origin"], sort_keys=True,
                                    separators=(",", ":"), ensure_ascii=False).encode("utf-8")))

    assert refusal is not None, f"no approve was refused; stored sizes {sizes}"
    assert "bytes; the cap is 4096 bytes" in refusal, refusal
    assert sizes and max(sizes) <= 4096
    RECEIPT["cases"]["oversized_refused"] = {"stored_sizes_after_quotes": sizes, "refusal": refusal,
                                             "payload_unchanged": True, "pass": True}


def test_off_mode_is_byte_identical_to_base(live, state, tmp_path):
    db, ids = state
    base = tmp_path / "base"
    base.mkdir()
    archive = subprocess.run(["git", "-C", str(REPO), "archive", C2_BASE], capture_output=True)
    if archive.returncode != 0:
        pytest.skip(f"base commit {C2_BASE} is not available here: {archive.stderr.decode()[:200]}")
    subprocess.run(["tar", "-x", "-C", str(base)], input=archive.stdout, check=True)
    cite = json.dumps(_cite(SID_A, ids, "a-u1", "a-a5"))
    outputs = {}
    for label, tree in (("base", base), ("head", REPO)):
        home = tmp_path / f"home-{label}"
        home.mkdir()
        out = tmp_path / f"{label}.json"
        run = subprocess.run(
            [sys.executable, str(DRIVER), str(tree), str(HERMES_AGENT), live.qdrant_url, live.memory,
             live.learnings, str(db), str(home), cite, str(out)],
            capture_output=True, text=True, cwd=str(tmp_path), timeout=120,
        )
        assert run.returncode == 0, f"{label} driver failed: {run.stderr[-2000:]}"
        outputs[label] = json.loads(out.read_text(encoding="utf-8"))
        assert outputs[label]["after_forget"] == "[]", outputs[label]["after_forget"]

    differing = sorted(key for key in outputs["base"] if outputs["base"][key] != outputs["head"].get(key))
    assert set(outputs["base"]) == set(outputs["head"])
    assert not differing, {key: (outputs["base"][key], outputs["head"][key]) for key in differing}
    assert "session_origin" not in outputs["head"]["payload_b"]
    assert '"cite":' not in outputs["head"]["schemas"]
    RECEIPT["cases"]["off_mode_byte_identical"] = {
        "base": C2_BASE,
        "observables": {key: hashlib.sha256(value.encode("utf-8")).hexdigest() for key, value in
                        sorted(outputs["head"].items())},
        "differing": differing, "pass": True,
    }
