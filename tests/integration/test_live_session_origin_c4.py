"""C'4 oracle: read-only origin-check, trace disclosure and the improve producer.

Real path: the provider's tools (``qdrant_memory_store`` with ``cite``,
``qdrant_memory_origin_check``, ``qdrant_memory_trace``, ``qdrant_memory_improve_preview``
and ``qdrant_memory_improve_apply``) and the ``origin-check`` CLI command, against a real
Qdrant in disposable collections that must start with ``tmp_c4_``, and a synthetic
``state.db`` built from the dumped host schema (``tests/fixtures/state_db_schema.sql``) that
the product opens read-only. Vectors come from the deterministic integration embedder.
Expected digests are computed here from the literal rows and the design 5.2 / amendment G1
byte formats, never through the module under test.

Cases (design 5.9, C'4) and the failure each one names:
  1 one pruned, one drifted and one intact session give exactly
    {expired: 1, drifted: 1, verified: 1}; point count, payload and vector digests,
    payload indexes, collections and the state.db files are identical before and after
    the tool, the CLI and trace ran                    -- the check writes / counts wrong
  2 improve over a legacy conversation-turn point_id records no state.db citation, while
    improve over the same line as source_text records the captured session, and an apply
    that would overwrite a point with an origin is refused -- legacy turns get fake citations
  3 session_origin_mode off is byte-identical to main except the new tool's schema entry
    (and the improve point_ids read, which never worked on a real client before)
                                                       -- off mode leaking

Run: RUN_QDRANT_INTEGRATION=1 QDRANT_TEST_URL=http://127.0.0.1:6333 pytest tests/integration/test_live_session_origin_c4.py
Set SESSION_ORIGIN_C4_RECEIPT=<path> to write the JSON receipt of the run there.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
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
DRIVER = Path(__file__).resolve().parent / "session_origin_c4_offmode_driver.py"
HERMES_AGENT = Path.home() / ".hermes" / "hermes-agent"
PREFIX = "tmp_c4_"
# main at the branch point of C'4: the tree off mode must be byte-identical to.
C4_BASE = "757f7a2"
_FORBIDDEN = {"hermes_memory", "hermes_learnings", "hermes_session_windows"}
VECTOR_SIZE = 64
PROFILE = "c4-oracle"

# ``local-*`` is a host session id shape. Timestamp ids (YYYYMMDD_HHMMSS_hex) would also do
# for origin-check, but improve's identity filter reads their digit run in the default
# ``session://<sid>/improve-source`` URI as a phone number and drops every graph candidate.
SID_P = "local-c4-pruned"     # pruned after the store
SID_D = "local-c4-drifted"    # one cited row rewritten after the store
SID_I = "local-c4-intact"     # intact
SID_I2 = "local-c4-intact2"   # compression continuation of I
T0 = 1_794_000_000.5

MARKERS = "Please note this.\nGraph entity: project: Bluefin\nGraph entity: service: Kestrel"
SESSIONS = [(SID_P, None), (SID_D, None), (SID_I, None), (SID_I2, SID_I)]
# (session, role, content, message_uid); ids come from AUTOINCREMENT in this order.
ROWS = [
    (SID_P, "user", "We agreed the archive job runs at 02:00.", "p-u1"),
    (SID_P, "assistant", "Noted: the archive job runs at 02:00.", "p-a2"),
    (SID_D, "user", "The staging database lives on host db-7.", "d-u1"),
    (SID_D, "assistant", "Recorded: staging database on db-7.", "d-a2"),
    (SID_I, "user", "The release train leaves every Tuesday.", "i-u1"),
    (SID_I, "assistant", "Understood, releases ship on Tuesdays.", "i-a2"),
    (SID_I, "user", MARKERS, "i-u3"),
    (SID_I, "assistant", "Noted both.", "i-a4"),
    (SID_I2, "assistant", "Continuing after compaction.", "i2-a1"),
]
FACTS = {
    "expired": ("The archive job runs at 02:00.", SID_P, "p-u1", "p-a2"),
    "drifted": ("The staging database lives on host db-7.", SID_D, "d-u1", "d-a2"),
    "verified": ("The release train leaves every Tuesday.", SID_I, "i-u1", "i-a2"),
}

RECEIPT: dict[str, Any] = {"oracle": "C'4 origin-check, trace disclosure and improve producer", "cases": {}}


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


def _expected_line_citation(sid: str, uid: str, ids: dict[tuple[str, str], int]) -> dict[str, Any]:
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
    readsrc: str


@pytest.fixture
def live() -> Iterator[Live]:
    qdrant_url = os.environ.get("QDRANT_TEST_URL", "").strip()
    if not qdrant_url:
        pytest.skip("set QDRANT_TEST_URL to run live Qdrant integration tests")
    qdrant = QdrantClient(qdrant_url, timeout=10.0, **{"api_key": os.environ.get("QDRANT_TEST_API_KEY", "")})
    suffix = uuid.uuid4().hex[:12]
    names = (f"{PREFIX}{suffix}_memory", f"{PREFIX}{suffix}_learnings", f"{PREFIX}{suffix}_readsrc")
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
        yield Live(qdrant=qdrant, qdrant_url=qdrant_url, memory=names[0], learnings=names[1], readsrc=names[2])
    finally:
        _delete_and_verify(qdrant, created, PREFIX)


@pytest.fixture
def state(tmp_path) -> tuple[Path, dict[tuple[str, str], int]]:
    db = tmp_path / "statedb" / "state.db"
    db.parent.mkdir()
    return db, _build_state_db(db)


@pytest.fixture(scope="module", autouse=True)
def _receipt() -> Iterator[None]:
    RECEIPT["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    yield
    target = os.environ.get("SESSION_ORIGIN_C4_RECEIPT", "").strip()
    if target:
        Path(target).write_text(json.dumps(RECEIPT, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _provider(live: Live, db: Path, home: Path, session_id: str, mode: str = "recorded"):
    from __init__ import QdrantMemoryProvider
    from qdrant_memory.writer import ConversationWriter

    provider = QdrantMemoryProvider()
    provider._active = True
    provider._hermes_home = str(home)
    provider._profile_id = PROFILE
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
        # A collection the provider only reads: origin-check must never scan it.
        "read_sources": [{"collection": live.readsrc, "profile_id": PROFILE, "kind": "memory"}],
    }})
    for key in ("collection_name", "learning_collection_name"):
        assert str(provider._config[key]).startswith(PREFIX), f"{key} resolved to {provider._config[key]!r}"
    assert [s["collection"] for s in provider._config["read_sources"]] == [live.readsrc]
    provider._qdrant = live.qdrant
    provider._embeddings = DeterministicEmbedding(VECTOR_SIZE)
    provider._writer = ConversationWriter(qdrant=live.qdrant, embeddings=provider._embeddings,
                                          collection_name=live.memory, profile_id=PROFILE, platform="pytest",
                                          session_id=session_id)
    return provider


def _call(provider, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    return json.loads(provider.handle_tool_call(tool, args))


def _payload(live: Live, collection: str, point_id: str) -> dict[str, Any] | None:
    points = live.qdrant.retrieve(collection, [point_id], with_payload=True)
    return points[0]["payload"] if points else None


def _store_with_cite(provider, text: str, sid: str, first: int, last: int) -> str:
    cite = {"session_id": sid, "first_msg_id": first, "last_msg_id": last}
    preview = _call(provider, "qdrant_memory_store", {"text": text, "cite": cite})
    assert preview.get("session_origin", {}).get("citation_status") == "verified", preview
    stored = _call(provider, "qdrant_memory_store", {"text": text, "cite": cite, "dry_run": False, "approve": True})
    assert stored.get("saved") is True, stored
    return str(stored["id"])


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _snapshot(live: Live, db: Path) -> dict[str, Any]:
    """Everything a read-only check must leave untouched, as digests."""
    out: dict[str, Any] = {"collections": {}, "state_db_dir": {}}
    for name in (live.memory, live.learnings, live.readsrc):
        points, offset = [], None
        while True:
            page, offset = live.qdrant.scroll_page(name, {}, limit=256, offset=offset, with_payload=True,
                                                   with_vector=True)
            points.extend(page)
            if offset is None or not page:
                break
        info = live.qdrant.collection_info(name)
        out["collections"][name] = {
            "count": live.qdrant.count(name),
            "points": sorted((str(p["id"]), _digest(p.get("payload")), _digest(p.get("vector"))) for p in points),
            "payload_schema": info.get("payload_schema"),
        }
    out["our_collections_present"] = sorted(
        name for name in live.qdrant.get_collections() if name.startswith(live.memory.rsplit("_", 1)[0]))
    for path in sorted(db.parent.iterdir()):
        out["state_db_dir"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _cli():
    spec = importlib.util.spec_from_file_location("_c4_oracle_cli", REPO / "cli.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_cli(provider, argv: list[str]) -> tuple[int, str, str]:
    cli = _cli()
    parser = argparse.ArgumentParser()
    cli.register_cli(parser)
    args = parser.parse_args(argv)
    from qdrant_memory.cli_core import execute_command

    out, err = io.StringIO(), io.StringIO()
    code = execute_command(args, provider_factory=lambda: provider, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


# --- the oracle -------------------------------------------------------------


def test_origin_check_counts_pruned_drifted_intact_and_changes_nothing(live, state, tmp_path):
    db, ids = state
    provider = _provider(live, db, tmp_path / "home", SID_I)
    point_ids = {status: _store_with_cite(provider, text, sid, ids[(sid, first)], ids[(sid, last)])
                 for status, (text, sid, first, last) in FACTS.items()}
    # Decoys that carry no session_origin, or live where the check must not look: a legacy
    # conversation turn in the provider's own collection, and a copy of a cited point in a
    # read-only source collection.
    legacy_id = provider._writer.store_turn(MARKERS, "Noted both.")
    assert legacy_id and "session_origin" not in _payload(live, live.memory, legacy_id)
    copied = live.qdrant.retrieve(live.memory, [point_ids["drifted"]], with_payload=True, with_vector=True)[0]
    live.qdrant.upsert(live.readsrc, [{"id": copied["id"], "vector": copied["vector"], "payload": copied["payload"]}])

    # The host's retention prunes P; an in-place rewrite changes one cited row of D.
    conn = sqlite3.connect(db)
    conn.execute("DELETE FROM messages WHERE session_id = ?", (SID_P,))
    conn.execute("DELETE FROM sessions WHERE id = ?", (SID_P,))
    conn.execute("UPDATE messages SET content = ? WHERE message_uid = ?",
                 ("The staging database lives on host db-9.", "d-u1"))
    conn.commit()
    conn.close()

    before = _snapshot(live, db)
    checked = _call(provider, "qdrant_memory_origin_check", {"collection": "memory"})
    code, cli_out, cli_err = _run_cli(provider, ["origin-check", "--collection", "memory", "--json"])
    learning_checked = _call(provider, "qdrant_memory_origin_check", {"collection": "learning"})
    traced = {status: _call(provider, "qdrant_memory_trace", {"point_id": pid}) for status, pid in point_ids.items()}
    after = _snapshot(live, db)

    assert checked.get("counts") == {"expired": 1, "drifted": 1, "verified": 1}, checked
    assert checked["point_ids"] == {status: [pid] for status, pid in point_ids.items()}, checked
    assert (checked["points_checked"], checked["complete"], checked["next_offset"]) == (3, True, None), checked
    assert checked["read_only"] is True
    assert code == 0, cli_err
    cli_payload = json.loads(cli_out)
    assert (cli_payload["counts"], cli_payload["point_ids"]) == (checked["counts"], checked["point_ids"])
    assert (learning_checked["counts"], learning_checked["points_checked"]) == ({}, 0), learning_checked
    for status, trace in traced.items():
        origin = trace.get("session_origin")
        assert origin is not None, trace
        assert [c["current_status"] for c in origin["citations"]] == [status], origin
        assert origin["current_status_counts"] == {status: 1}, origin
        assert origin["captured_session_id"] == SID_I and origin["producer"] == "manual_store", origin
    assert before == after, "origin-check / trace changed Qdrant or state.db"

    RECEIPT["cases"]["origin_check_counts_and_zero_mutation"] = {
        "fixture": {"pruned_session": SID_P, "drifted_session": SID_D, "intact_session": SID_I,
                    "decoys": ["legacy conversation turn without session_origin (own collection)",
                               "copy of the drifted point in a read_sources collection"]},
        "counts": checked["counts"],
        "point_ids_match": True,
        "cli_counts": cli_payload["counts"],
        "learning_collection_counts": learning_checked["counts"],
        "trace_current_status": {s: [c["current_status"] for c in t["session_origin"]["citations"]]
                                 for s, t in traced.items()},
        "zero_mutation": {
            "point_counts": {k.replace(live.memory.rsplit("_", 1)[0], "<run>"): v["count"]
                             for k, v in after["collections"].items()},
            "payload_and_vector_digests_equal": True,
            "payload_schema_after": {k.replace(live.memory.rsplit("_", 1)[0], "<run>"): v["payload_schema"]
                                     for k, v in after["collections"].items()},
            "state_db_dir_listing": sorted(after["state_db_dir"]),
            "state_db_sha256_equal": before["state_db_dir"] == after["state_db_dir"],
            # Collection names carry a per-run suffix, so these differ between runs but
            # must be equal to each other within one.
            "snapshot_sha256_before": _digest(before["collections"] | {"state_db": before["state_db_dir"]}),
            "snapshot_sha256_after": _digest(after["collections"] | {"state_db": after["state_db_dir"]}),
        },
        "pass": True,
    }


def test_improve_over_a_legacy_turn_point_records_no_citation(live, state, tmp_path):
    db, ids = state
    provider = _provider(live, db, tmp_path / "home", SID_I)
    # A legacy turn whose session is still in state.db and whose marker lines exist there
    # verbatim (row i-u3): a line resolver would "verify" a citation for it.
    legacy_id = provider._writer.store_turn(MARKERS, "Noted both.")
    assert legacy_id

    # Control: the same line given as source_text in I, approved after a compression switch.
    text_report = _call(provider, "qdrant_memory_improve_preview", {"source_text": "Graph entity: project: Bluefin"})
    bluefin = next(c for c in text_report["candidates"] if c["candidate_type"] == "graph_entity_candidate")
    provider.on_session_switch(SID_I2, parent_session_id=SID_I, reset=False, reason="compression")
    dry = _call(provider, "qdrant_memory_improve_apply",
                {"report_id": text_report["report_id"], "candidate_id": bluefin["candidate_id"]})
    applied = _call(provider, "qdrant_memory_improve_apply", {"report_id": text_report["report_id"],
                                                              "candidate_id": bluefin["candidate_id"],
                                                              "dry_run": False, "approve": True})
    assert applied.get("saved") is True, applied
    bluefin_payload = _payload(live, live.memory, applied["id"])
    origin = bluefin_payload["session_origin"]
    expected = _expected_line_citation(SID_I, "i-u3", ids)
    assert (origin["producer"], origin["captured_session_id"], origin["approved_session_id"]) == (
        "improve", SID_I, SID_I2), origin
    assert origin["citation_status"] == "verified", origin
    assert [_pinned(c) for c in origin["citations"]] == [expected], origin
    assert dry["session_origin"]["captured_session_id"] == SID_I, dry

    # The legacy turn point as improve input: candidates exist, no citation is recorded.
    point_report = _call(provider, "qdrant_memory_improve_preview", {"point_ids": [legacy_id]})
    by_label = {c["proposed_payload"].get("label"): c for c in point_report["candidates"]
                if c["candidate_type"] == "graph_entity_candidate"}
    assert set(by_label) >= {"Bluefin", "Kestrel"}, point_report
    kestrel = by_label["Kestrel"]
    kestrel_dry = _call(provider, "qdrant_memory_improve_apply",
                        {"report_id": point_report["report_id"], "candidate_id": kestrel["candidate_id"]})
    kestrel_live = _call(provider, "qdrant_memory_improve_apply", {"report_id": point_report["report_id"],
                                                                   "candidate_id": kestrel["candidate_id"],
                                                                   "dry_run": False, "approve": True})
    assert kestrel_live.get("saved") is True, kestrel_live
    assert kestrel_dry.get("session_origin", "absent") is None, kestrel_dry
    assert kestrel_live.get("session_origin", "absent") is None, kestrel_live
    kestrel_payload = _payload(live, live.memory, kestrel_live["id"])
    assert "session_origin" not in kestrel_payload, kestrel_payload

    # The same entity from the legacy report targets the point that already carries an
    # origin: refused, and the origin stays byte-equal.
    _call(provider, "qdrant_memory_improve_apply",
          {"report_id": point_report["report_id"], "candidate_id": by_label["Bluefin"]["candidate_id"]})
    overwrite = _call(provider, "qdrant_memory_improve_apply", {"report_id": point_report["report_id"],
                                                                "candidate_id": by_label["Bluefin"]["candidate_id"],
                                                                "dry_run": False, "approve": True})
    assert "refusing to overwrite" in overwrite.get("error", ""), overwrite
    assert _payload(live, live.memory, applied["id"]) == bluefin_payload

    RECEIPT["cases"]["improve_legacy_turn_no_citation"] = {
        "legacy_turn": {"session_present_in_state_db": True, "marker_row_id": ids[(SID_I, "i-u3")]},
        "control_source_text": {"producer": origin["producer"], "captured_session_id": origin["captured_session_id"],
                                "approved_session_id": origin["approved_session_id"],
                                "citation": _pinned(origin["citations"][0]), "expected": expected},
        "legacy_point_ids_candidates": sorted(by_label),
        "legacy_apply_session_origin": None,
        "legacy_payload_has_session_origin": False,
        "overwrite_refused_origin_unchanged": True,
        "pass": True,
    }


def test_off_mode_is_byte_identical_to_base_except_the_new_tool(live, state, tmp_path):
    db, _ids = state
    base = tmp_path / "base"
    base.mkdir()
    # A scratch copy made with `git archive` (the bite harness) is not a git checkout; it
    # names the checkout that holds C4_BASE through SESSION_ORIGIN_BASE_REPO.
    git_dir = os.environ.get("SESSION_ORIGIN_BASE_REPO", "").strip() or str(REPO)
    archive = subprocess.run(["git", "-C", git_dir, "archive", C4_BASE], capture_output=True)
    if archive.returncode != 0:
        pytest.skip(f"base commit {C4_BASE} is not available here: {archive.stderr.decode()[:200]}")
    subprocess.run(["tar", "-x", "-C", str(base)], input=archive.stdout, check=True)
    outputs = {}
    for label, tree in (("base", base), ("head", REPO)):
        home = tmp_path / f"home-{label}"
        home.mkdir()
        out = tmp_path / f"{label}.json"
        run = subprocess.run(
            [sys.executable, str(DRIVER), str(tree), str(HERMES_AGENT), live.qdrant_url, live.memory,
             live.learnings, str(db), str(home), str(out)],
            capture_output=True, text=True, cwd=str(tmp_path), timeout=120,
        )
        assert run.returncode == 0, f"{label} driver failed: {run.stderr[-2000:]}"
        outputs[label] = json.loads(out.read_text(encoding="utf-8"))
        for collection in (live.memory, live.learnings):  # each tree starts from empty collections
            point_list = [p["id"] for p in live.qdrant.scroll_page(collection, {}, limit=100, with_payload=False)[0]]
            if point_list:
                live.qdrant.delete_ids(collection, point_list)
            assert live.qdrant.count(collection) == 0

    base_out, head_out = outputs["base"], outputs["head"]
    # The one schema delta: the new read-only tool, on the full surface only.
    head_schemas = json.loads(head_out.pop("schemas_full"))
    base_schemas = json.loads(base_out.pop("schemas_full"))
    added = [s for s in head_schemas if s["name"] not in {b["name"] for b in base_schemas}]
    assert [s["name"] for s in added] == ["qdrant_memory_origin_check"], added
    assert [s for s in head_schemas if s["name"] != "qdrant_memory_origin_check"] == base_schemas
    # The improve point_ids read never reached Qdrant before (it called a method the real
    # client does not have and swallowed the error), so main's report is empty.
    base_point_report = json.loads(base_out.pop("improve_point_ids_preview"))
    head_point_report = json.loads(head_out.pop("improve_point_ids_preview"))
    assert base_point_report["counts"]["total"] == 0, base_point_report
    assert head_point_report["counts"]["total"] >= 2, head_point_report
    head_origin_check = json.loads(head_out.pop("origin_check"))
    assert (head_origin_check["counts"], head_origin_check["points_checked"]) == ({}, 0), head_origin_check

    differing = sorted(key for key in base_out if base_out[key] != head_out.get(key))
    assert set(base_out) == set(head_out)
    assert not differing, {key: (base_out[key], head_out[key]) for key in differing}
    assert "session_origin" not in head_out["improve_payload"]
    assert json.loads(head_out["improve_apply"]).get("saved") is True
    RECEIPT["cases"]["off_mode_byte_identical"] = {
        "base": C4_BASE,
        "schema_delta": [s["name"] for s in added],
        "improve_point_ids_candidates": {"base": base_point_report["counts"]["total"],
                                         "head": head_point_report["counts"]["total"]},
        "observables": {
            key: hashlib.sha256(value.replace(live.memory, "<memory>").replace(live.learnings, "<learnings>")
                                .encode("utf-8")).hexdigest()
            for key, value in sorted(head_out.items())
        },
        "differing": differing, "pass": True,
    }
