"""C'3 oracle: hook candidates and learnings record the session they were captured in.

Real path: ``on_session_switch``, the learning / extraction approval backends
(``approve_learning_candidate`` / ``approve_extraction_candidate``) and the learning store
tool against a real Qdrant, in
disposable collections that must start with ``tmp_c3_``, and a synthetic ``state.db``
built from the dumped host schema (``tests/fixtures/state_db_schema.sql``) that the product
opens read-only. Vectors come from the deterministic integration embedder.

The fixture mirrors what the host does on compression: a child session whose
``parent_session_id`` is the old one, carrying the protected tail as copies under new row
ids with the same ``message_uid``. Expected digests are computed here from the literal rows
and the design 5.2 / amendment G1 byte formats, never through the module under test.

Cases (design 5.9, C'3) and the failure each one names:
  1 a hook in A, approval after a compression switch to A', gives origin A
                                          -- the approver's session stamped as origin (R8)
  2 a source line found in two messages gives unresolved:ambiguous_match, still stored
                                          -- first match wins
  3 re-storing a learning from session B merges its citations
                                          -- replace semantics on the learning upsert
  4 a learning re-store keeps the existing origin byte-equal
                                          -- the upsert dropping session_origin

Capture CP1 removed the hook producer and the preview/approve tools. A candidate is now built
here the way the hook built it (``conftest.captured_extraction_candidate`` /
``captured_learning_candidate``: the same extractor and the capture frozen in the hook's
session) and handed to the backend. Case 5 (off mode byte-identical to the pre-C'3 tree)
replayed the hook -> preview -> approve tool journey, which no longer exists on head; it was
retired with it. Off mode on the backends stays pinned by the C'6 oracle
(``test_restores_with_the_mode_back_off_keep_the_recorded_origin``), and CP4 re-pins it for the
queue-backed routes.

Run: RUN_QDRANT_INTEGRATION=1 QDRANT_TEST_URL=http://127.0.0.1:6333 pytest tests/integration/test_live_session_origin_c3.py
Set SESSION_ORIGIN_C3_RECEIPT=<path> to write the JSON receipt of the run there.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pytest

from conftest import DeterministicEmbedding, _delete_and_verify, captured_extraction_candidate, captured_learning_candidate
from qdrant_memory.client import QdrantClient
from qdrant_memory.config import load_config

TRUTHY = {"1", "true", "yes", "on"}
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_QDRANT_INTEGRATION", "").strip().lower() not in TRUTHY,
    reason="set RUN_QDRANT_INTEGRATION=1 to run live Qdrant integration tests",
)

REPO = Path(__file__).resolve().parents[2]
SCHEMA = REPO / "tests" / "fixtures" / "state_db_schema.sql"
PREFIX = "tmp_c3_"
_FORBIDDEN = {"hermes_memory", "hermes_learnings", "hermes_session_windows"}
VECTOR_SIZE = 64

SID_A0 = "20261115_110000_c3aa00"  # A's own compression parent
SID_A = "20261115_120000_c3aaaa"   # the session the hook ran in
SID_A2 = "20261115_130000_c3aaa2"  # A' : the compression continuation of A
SID_B = "20261115_140000_c3bbbb"
SID_D = "20261115_150000_c3dddd"
T0 = 1_794_000_000.5

DECISION = "Decision: route staging traffic through the blue proxy pool."
CORRECTION = "Actually, use the blue proxy pool, not the green one."
REPEATED = "Decision: pin the scheduler to two workers."
SESSIONS = [(SID_A0, None), (SID_A, SID_A0), (SID_A2, SID_A), (SID_B, None), (SID_D, None)]
# (session, role, content, message_uid); ids come from AUTOINCREMENT in this order.
ROWS = [
    (SID_A0, "user", "Let us plan the proxy rollout.", "a0-u1"),
    (SID_A0, "assistant", DECISION + "\nI will note it.", "a0-a2"),
    (SID_A, "assistant", DECISION + "\nI will note it.", "a0-a2"),  # tail carried from A0
    (SID_A, "user", CORRECTION, "a-u1"),
    (SID_A, "assistant", "Understood, the blue pool.", "a-a2"),
    (SID_A2, "user", CORRECTION, "a-u1"),                            # tail carried into A'
    (SID_A2, "assistant", "Continuing after compaction.", "a2-a1"),
    (SID_B, "user", "Anything else about the proxy?", "b-u1"),
    (SID_D, "assistant", REPEATED, "d-a1"),
    (SID_D, "user", "Say it again.", "d-u2"),
    (SID_D, "assistant", REPEATED, "d-a3"),
]
LESSON = {"lesson": "User correction: use the blue proxy pool, not the green one.", "learning_type": "user_correction",
          "trigger": CORRECTION, "mistake": "the green one", "correction": "use the blue proxy pool",
          "evidence": "Explicit user correction followed by assistant acknowledgement.", "confidence": 0.9,
          "importance": 8}

RECEIPT: dict[str, Any] = {"oracle": "C'3 session_origin on extraction approval, learning approval and learning store",
                           "cases": {}}


# --- fixtures -----------------------------------------------------------------


def _build_state_db(path: Path) -> dict[tuple[str, str], int]:
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA.read_text(encoding="utf-8"))
    for sid, parent in SESSIONS:
        conn.execute("INSERT INTO sessions (id, source, parent_session_id, started_at) VALUES (?, 'cli', ?, ?)",
                     (sid, parent, T0))
    ids: dict[tuple[str, str], int] = {}
    for index, (sid, role, content, uid) in enumerate(ROWS):
        cur = conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp, message_uid) VALUES (?, ?, ?, ?, ?)",
            (sid, role, content, T0 + 60.0 * index, uid),
        )
        ids[(sid, uid)] = int(cur.lastrowid)
    conn.commit()
    conn.close()
    return ids


def _expected_citation(sid: str, uid: str, ids: dict[tuple[str, str], int]) -> dict[str, Any]:
    """One-row span citation per design 5.2 / G1, from the literal row."""
    row_id = ids[(sid, uid)]
    role, text = next((r, c) for s, r, c, u in ROWS if (s, u) == (sid, uid))
    span = hashlib.sha256(b"hermes-span-v1\x1f" + sid.encode() + b"\x1f" + str(row_id).encode() + b"\x1f"
                          + role.encode() + b"\x1f" + text.encode("utf-8") + b"\x1e").hexdigest()
    content = hashlib.sha256(b"hermes-span-content-v1\x1f" + sid.encode() + b"\x1f" + uid.encode() + b"\x1f"
                             + role.encode() + b"\x1f" + text.encode("utf-8") + b"\x1e").hexdigest()
    uids = hashlib.sha256(b"hermes-uids-v1\x1f" + uid.encode() + b"\x1e").hexdigest()
    return {"store": "hermes-state.db", "session_id": sid, "first_msg_id": row_id, "last_msg_id": row_id,
            "msg_count": 1, "span_sha256": span, "content_sha256": content, "uids_sha256": uids,
            "basis": "extractor_source_line"}


def _pinned(citation: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in citation.items() if key != "verified_at"}


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
def state(tmp_path) -> tuple[Path, dict[tuple[str, str], int]]:
    db = tmp_path / "state.db"
    return db, _build_state_db(db)


@pytest.fixture(scope="module", autouse=True)
def _receipt() -> Iterator[None]:
    RECEIPT["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    yield
    target = os.environ.get("SESSION_ORIGIN_C3_RECEIPT", "").strip()
    if target:
        Path(target).write_text(json.dumps(RECEIPT, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _provider(live: Live, db: Path, home: Path, session_id: str, mode: str = "recorded"):
    from __init__ import QdrantMemoryProvider
    from qdrant_memory.writer import ConversationWriter

    provider = QdrantMemoryProvider()
    provider._active = True
    provider._hermes_home = str(home)
    provider._profile_id = "c3-oracle"
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
                                          collection_name=live.memory, profile_id="c3-oracle", platform="pytest",
                                          session_id=session_id)
    return provider


def _call(provider, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    return json.loads(provider.handle_tool_call(tool, args))


def _payload(live: Live, collection: str, point_id: str) -> dict[str, Any] | None:
    points = live.qdrant.retrieve(collection, [point_id], with_payload=True)
    return points[0]["payload"] if points else None


def _approve_learning(provider, candidate) -> dict[str, Any]:
    dry = json.loads(provider.approve_learning_candidate(candidate))
    assert dry.get("dry_run") is True, dry
    live_result = json.loads(provider.approve_learning_candidate(candidate, dry_run=False))
    assert live_result.get("saved") is True, live_result
    return {"dry": dry, "live": live_result}


def _approve_extraction(provider, candidate) -> dict[str, Any]:
    dry = json.loads(provider.approve_extraction_candidate(candidate))
    assert dry.get("dry_run") is True, dry
    live_result = json.loads(provider.approve_extraction_candidate(candidate, dry_run=False, approve=True))
    assert live_result.get("saved") is True, live_result
    return {"dry": dry, "live": live_result}


# --- the oracle -------------------------------------------------------------


def test_hook_in_a_approved_after_compression_switch_records_origin_a(live, state, tmp_path):
    db, ids = state
    provider = _provider(live, db, tmp_path, SID_A)
    learning_candidate = captured_learning_candidate(provider, CORRECTION, **LESSON)
    extraction_candidate = captured_extraction_candidate(provider, DECISION + "\nI will note it.", DECISION)
    provider.on_session_switch(SID_A2, parent_session_id=SID_A, reset=False, reason="compression")
    assert provider._session_id == SID_A2  # the approver is A', the capture was made in A

    learning = _approve_learning(provider, learning_candidate)
    extraction = _approve_extraction(provider, extraction_candidate)

    learning_origin = _payload(live, live.learnings, learning["live"]["id"])["session_origin"]
    extraction_origin = _payload(live, live.memory, extraction["live"]["id"])["session_origin"]
    # The correction is new in A (its A' copy is outside A's chain); the decision line was
    # carried into A from A0, so the citation is A0's original row, not A's copy.
    expected_learning = _expected_citation(SID_A, "a-u1", ids)
    expected_extraction = _expected_citation(SID_A0, "a0-a2", ids)
    for origin, producer, expected in ((learning_origin, "learning_approval", expected_learning),
                                       (extraction_origin, "source_extraction", expected_extraction)):
        assert origin["producer"] == producer
        assert origin["captured_session_id"] == SID_A, origin
        assert origin["approved_session_id"] == SID_A2, origin
        assert origin["citation_status"] == "verified", origin
        assert [_pinned(c) for c in origin["citations"]] == [expected], origin
    # The dry-run preview showed the same capture the approval stored.
    assert learning["dry"]["session_origin"]["captured_session_id"] == SID_A
    assert extraction["dry"]["session_origin"]["captured_session_id"] == SID_A
    inspected = _call(provider, "qdrant_memory_inspect", {"point_id": extraction["live"]["id"]})
    assert inspected["session_origin"]["current_status_counts"] == {"verified": 1}, inspected
    RECEIPT["cases"]["compression_switch_origin_a"] = {
        "hook_session": SID_A, "approver_session": SID_A2,
        "learning": {"captured_session_id": learning_origin["captured_session_id"],
                     "approved_session_id": learning_origin["approved_session_id"],
                     "citation": _pinned(learning_origin["citations"][0]), "expected": expected_learning},
        "extraction": {"captured_session_id": extraction_origin["captured_session_id"],
                       "approved_session_id": extraction_origin["approved_session_id"],
                       "citation": _pinned(extraction_origin["citations"][0]), "expected": expected_extraction},
        "inspect_status_counts": inspected["session_origin"]["current_status_counts"],
        "pass": True,
    }


def test_a_source_line_found_twice_is_ambiguous_and_still_stored(live, state, tmp_path):
    db, ids = state
    provider = _provider(live, db, tmp_path, SID_D)
    candidate = captured_extraction_candidate(provider, REPEATED, REPEATED)

    extraction = _approve_extraction(provider, candidate)

    origin = _payload(live, live.memory, extraction["live"]["id"])["session_origin"]
    assert origin["captured_session_id"] == SID_D
    assert "approved_session_id" not in origin  # approved in the capture session
    assert origin["citation_status"] == "unresolved:ambiguous_match", origin
    assert origin["citations"] == [], origin
    assert extraction["dry"]["session_origin"]["citation_status"] == "unresolved:ambiguous_match"
    RECEIPT["cases"]["duplicated_line_ambiguous"] = {
        "line_rows": [ids[(SID_D, "d-a1")], ids[(SID_D, "d-a3")]],
        "citation_status": origin["citation_status"], "citations": origin["citations"],
        "stored": True, "pass": True,
    }


def test_restoring_a_learning_from_session_b_merges_its_citations(live, state, tmp_path):
    db, ids = state
    provider = _provider(live, db, tmp_path, SID_A)
    candidate = captured_learning_candidate(provider, CORRECTION, **LESSON)
    provider.on_session_switch(SID_A2, parent_session_id=SID_A, reset=False, reason="compression")
    approved = _approve_learning(provider, candidate)
    point_id = approved["live"]["id"]
    first = _payload(live, live.learnings, point_id)["session_origin"]

    time.sleep(1.1)  # a replaced mapping would carry a later captured_at
    provider.on_session_switch(SID_B, parent_session_id="", reset=True)
    restored = _call(provider, "qdrant_learning_store", approved["dry"]["learning_args"])
    assert restored.get("saved") is True and restored["id"] == point_id, restored

    origin = _payload(live, live.learnings, point_id)["session_origin"]
    assert [(c["session_id"], c["basis"]) for c in origin["citations"]] == [
        (SID_A, "extractor_source_line"), (SID_B, "tool_position")], origin
    assert _pinned(origin["citations"][0]) == _expected_citation(SID_A, "a-u1", ids)
    assert origin["citations"][1]["after_msg_id"] == ids[(SID_B, "b-u1")]
    assert (origin["producer"], origin["captured_session_id"], origin["captured_at"]) == (
        "learning_approval", SID_A, first["captured_at"])
    assert live.qdrant.count(live.learnings) == 1
    RECEIPT["cases"]["learning_restore_merges"] = {
        "citations": [(c["session_id"], c["basis"]) for c in origin["citations"]],
        "first_captured_at": first["captured_at"], "stored_captured_at": origin["captured_at"],
        "captured_session_id": origin["captured_session_id"], "producer": origin["producer"],
        "point_count": 1, "pass": True,
    }


def test_the_learning_upsert_keeps_an_existing_origin(live, state, tmp_path):
    db, ids = state
    provider = _provider(live, db, tmp_path, SID_B)
    args = {"lesson": "Use the blue proxy pool for staging traffic.", "learning_type": "workflow_lesson",
            "trigger": "Routing staging traffic", "correction": "Use the blue proxy pool.", "importance": 7}
    stored = _call(provider, "qdrant_learning_store", args)
    assert stored.get("saved") is True, stored
    before = _payload(live, live.learnings, stored["id"])

    time.sleep(1.1)
    again = _call(provider, "qdrant_learning_store", {**args, "importance": 9})
    assert again.get("saved") is True and again["id"] == stored["id"], again

    after = _payload(live, live.learnings, stored["id"])
    assert after["importance"] == 9  # the upsert itself went through
    assert after["session_origin"] == before["session_origin"], (before["session_origin"], after["session_origin"])
    assert [(c["basis"], c["session_id"], c["after_msg_id"]) for c in before["session_origin"]["citations"]] == [
        ("tool_position", SID_B, ids[(SID_B, "b-u1")])]
    assert (before["session_origin"]["producer"], before["session_origin"]["captured_session_id"]) == (
        "learning_store", SID_B)
    RECEIPT["cases"]["learning_upsert_keeps_origin"] = {
        "importance_before": before["importance"], "importance_after": after["importance"],
        "origin_unchanged": True, "producer": after["session_origin"]["producer"],
        "citation_basis": after["session_origin"]["citations"][0]["basis"], "pass": True,
    }
