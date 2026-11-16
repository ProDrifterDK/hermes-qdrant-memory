"""Capture CP5: mirror of the agent's built-in ``memory`` tool writes (lane C).

Failure modes, each pinned by a test below:
- the parser misses an argument shape (``operations`` batches with or without a top-level
  target, ``action+content``, ``action+old_text``, ``old_text+new_text``, ``content+old_text``,
  ``action+content+new_text+old_text``), so the op count is short;
- a ``target=user`` entry (identity) is stored without review;
- a remove or replace deletes or overwrites the old entry instead of archiving it as a
  superseded capture cited to the turn that removed it; a replace must leave two points;
- a patch is stored as the bare replaced span instead of the whole entry;
- a secret in memory text is copied into Qdrant or the queue;
- a call the tool refused (``success: false``, a persisted failure, no result) is mirrored;
- the citation is not ``agent_proposed`` from the last human row before the call to the
  calling turn, or does not verify;
- the mirror runs while ``capture_mirror_builtin_memory`` is off;
- a rerun creates new points or queue items.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from capture_support import MemQdrant, add_message, add_session, make_db, make_provider, memory_call
from qdrant_memory import capture_mirror as cm
from qdrant_memory import capture_queue as cq
from qdrant_memory import capture_run as crun
from qdrant_memory.capture_reader import cursor_path, save_cursor
from qdrant_memory.session_origin import verify_citation

NOW = 1_794_100_000.0
ENTRY_A = "The backup server lives on host alpha-01 and runs restic nightly."
ENTRY_A2 = "The backup server lives on host beta-02 and runs restic nightly."
ENTRY_B = "Owner wants status reports as short bullet lists."
ENTRY_C = "Project gamma uses port 8123 for its local dashboard."
USER_ENTRY = "Owner's preferred language for replies is Spanish."
OK = {"success": True, "done": True, "target": "memory"}

# Every argument shape the host has accepted, with how many ops each one carries.
SHAPES = [
    ({"target": "memory", "operations": [{"action": "add", "content": "x one"}, {"action": "add", "content": "x two"}]},
     "operations+target", 2),
    ({"operations": [{"action": "add", "content": "y one", "target": "memory"}]}, "operations", 1),
    ({"action": "add", "content": "z one", "target": "memory"}, "action+content+target", 1),
    ({"action": "replace", "content": "new", "old_text": "old", "target": "memory"},
     "action+content+old_text+target", 1),
    ({"action": "remove", "old_text": "gone", "target": "memory"}, "action+old_text+target", 1),
    ({"action": "patch", "content": "a", "new_text": "a", "old_text": "b", "target": "memory"},
     "action+content+new_text+old_text+target", 1),
    ({"new_text": "c", "old_text": "d", "target": "memory"}, "new_text+old_text+target", 1),
    ({"content": "e", "old_text": "f", "target": "memory"}, "content+old_text+target", 1),
]


@pytest.mark.parametrize("arguments,shape,ops", SHAPES)
def test_every_argument_shape_is_parsed_into_its_ops(arguments, shape, ops):
    calls = cm.parse_tool_calls(json.dumps([memory_call("c1", arguments),
                                            {"id": "c2", "function": {"name": "terminal", "arguments": "{}"}}]))
    assert [c.shape for c in calls] == [shape]
    assert len(calls[0].ops) == ops
    assert {op.target for op in calls[0].ops} == {"memory"}


def test_unparseable_arguments_still_count_as_one_op():
    calls = cm.parse_tool_calls(json.dumps([{"id": "c1", "function": {"name": "memory", "arguments": "{oops"}}]))
    assert [c.shape for c in calls] == ["<unparseable>"] and len(calls[0].ops) == 1


@pytest.fixture
def world(tmp_path):
    db = tmp_path / "state" / "state.db"
    conn = make_db(db)
    ids = {}
    add_session(conn, "s1")
    ids["human1"] = add_message(conn, "s1", "user", "Please remember where the backup server lives.", 1)
    ids["note1"] = add_message(conn, "s1", "assistant", "Saving that.", 2)
    ids["call_add"] = _call(conn, "s1", "c-add", {"target": "memory", "operations": [
        {"action": "add", "content": ENTRY_A}, {"action": "add", "content": ENTRY_B},
        {"action": "add", "content": USER_ENTRY, "target": "user"}]}, OK, 3)
    ids["human2"] = add_message(conn, "s1", "user", "The backup server moved to beta-02.", 5)
    ids["call_replace"] = _call(conn, "s1", "c-rep", {"target": "memory", "action": "replace", "old_text": "alpha-01",
                                                       "content": ENTRY_A2}, {**OK, "replaced_entry": ENTRY_A}, 6)
    ids["answer2"] = add_message(conn, "s1", "assistant", "Updated.", 8)
    ids["call_remove"] = _call(conn, "s1", "c-rm", {"target": "memory", "operations": [
        {"action": "remove", "old_text": "bullet lists"}]}, {**OK, "removed_entries": {"1": ENTRY_B}}, 9)
    ids["call_failed"] = _call(conn, "s1", "c-fail", {"target": "memory", "action": "add", "content": ENTRY_C},
                               {"success": False, "error": "Memory at 100,017/100,000 chars."}, 11)
    ids["call_persisted"] = _call(conn, "s1", "c-big", {"target": "memory", "action": "add", "content": ENTRY_C},
                                  None, 13, raw_result='<persisted-output>\nThis tool result was too large.\n'
                                                       'Preview (first 1500 chars):\n{"success": false, "error": "x"}')
    ids["call_patch"] = _call(conn, "s1", "c-patch", {"target": "memory", "operations": [
        {"action": "patch", "old_text": "beta-02", "content": "gamma-03"}]},
        {**OK, "patched_entries": {"1": ENTRY_A2}, "patched_entries_after": {"1": ENTRY_A2.replace("beta-02", "gamma-03")}},
        15)
    fake_key = "sk-" + "abcdefghij" * 3 + "0123456789"
    ids["call_secret"] = _call(conn, "s1", "c-sec", {"target": "memory", "action": "add",
                                                      "content": "Deploy token for the staging box is " + fake_key},
                               OK, 17)
    conn.commit()
    conn.close()
    home = tmp_path / "home"
    home.mkdir()
    return {"db": db, "home": home, "ids": ids, "qdrant": MemQdrant()}


def _call(conn, sid, call_id, arguments, result, ts, raw_result=None):
    msg = add_message(conn, sid, "assistant", "", ts, tool_calls=[memory_call(call_id, arguments)])
    if raw_result is not None or result is not None:
        add_message(conn, sid, "tool", raw_result if raw_result is not None else json.dumps(result), ts + 0.5,
                    tool_call_id=call_id)
    return msg


def _provider(world, **config):
    return make_provider(world["home"], world["db"], world["qdrant"], session_id="cli",
                         capture_mirror_builtin_memory=True, **config)


def _run(world, *, apply=False, provider=None, now=NOW):
    return crun.run_capture(provider or _provider(world), apply=apply, now=now)


def _mirror(result):
    return [c for c in result["candidates"] if c["lane"] == "mirror"]


def test_the_mirror_is_off_unless_enabled(world):
    off = crun.run_capture(make_provider(world["home"], world["db"], world["qdrant"]), now=NOW)
    assert _mirror(off) == [] and "mirror" not in off["report"]


def test_ops_are_counted_and_failed_calls_are_dropped_with_reasons(world):
    report = _run(world)["report"]
    stats = report["mirror"]
    assert stats["calls"] == 7 and stats["ops"] == 9
    assert stats["shapes"] == {"operations+target": 3, "action+content+old_text+target": 1,
                               "action+content+target": 3}
    assert report["counts"]["drops"]["tool_failed"] == 2
    assert report["counts"]["drops"]["possible_secret"] == 1


def test_routes_target_memory_stores_target_user_reviews_and_old_text_is_superseded(world):
    ids = world["ids"]
    got = {(c["msg_id"], c["kind"], c["fact"]): c for c in _mirror(_run(world))}

    assert got[(ids["call_add"], "mirror_add", ENTRY_A)]["route"] == "mirror"
    assert got[(ids["call_add"], "mirror_add", USER_ENTRY)]["route"] == "review"
    assert got[(ids["call_replace"], "mirror_replace", ENTRY_A2)]["route"] == "mirror"
    assert got[(ids["call_replace"], "mirror_superseded", ENTRY_A)]["route"] == "mirror"
    assert got[(ids["call_remove"], "mirror_superseded", ENTRY_B)]["route"] == "mirror"
    patched = ENTRY_A2.replace("beta-02", "gamma-03")
    assert got[(ids["call_patch"], "mirror_replace", patched)]["route"] == "mirror"  # whole entry, not the span
    assert got[(ids["call_patch"], "mirror_superseded", ENTRY_A2)]["route"] == "mirror"
    assert not any("gamma-03" == fact for _, _, fact in got)
    assert not any("sk-" in fact for _, _, fact in got)


def test_citations_are_agent_proposed_spans_from_the_last_human_row_to_the_calling_turn(world):
    ids = world["ids"]
    for c in _mirror(_run(world)):
        citation = c["citation"]
        assert citation["basis"] == "agent_proposed"
        assert verify_citation(citation, state_db=str(world["db"]))["status"] == "verified"
        if c["msg_id"] == ids["call_add"]:
            assert (citation["first_msg_id"], citation["last_msg_id"]) == (ids["human1"], ids["note1"])
        if c["msg_id"] == ids["call_remove"]:
            assert (citation["first_msg_id"], citation["last_msg_id"]) == (ids["human2"], ids["answer2"])


def test_apply_stores_memory_entries_archives_old_ones_and_never_deletes(world):
    result = _run(world, apply=True)
    points = world["qdrant"].collection("memory")
    by_text = {p["payload"]["text"]: p["payload"] for p in points.values()}

    assert not any(op == "delete" for op, _, _ in world["qdrant"].writes)
    assert by_text[ENTRY_A]["fact_status"] == "superseded"           # replaced
    assert by_text[ENTRY_B]["fact_status"] == "superseded"           # removed, kept
    assert by_text[ENTRY_A2]["fact_status"] == "superseded"          # patched later
    assert "fact_status" not in by_text[ENTRY_A2.replace("beta-02", "gamma-03")]
    assert USER_ENTRY not in by_text                                  # identity: review only
    for payload in by_text.values():
        origin = payload["session_origin"]
        assert origin["producer"] == "manual_store" and origin["captured_session_id"] == "s1"
        assert "approved_session_id" not in origin
        assert all(c["basis"] == "agent_proposed" for c in origin["citations"])
    # the removed entry is cited to its add turn and to the turn that removed it
    spans = [(c["first_msg_id"], c["last_msg_id"]) for c in by_text[ENTRY_B]["session_origin"]["citations"]]
    assert spans == [(world["ids"]["human1"], world["ids"]["note1"]), (world["ids"]["human2"], world["ids"]["answer2"])]
    queue = cq.CaptureQueue(str(world["home"]))
    assert [i.fact for i in queue.pending(NOW) if i.queued["lane"] == "mirror"] == [USER_ENTRY]
    applied = [e for e in queue.events() if e["event"] == "applied"]
    assert {e["point_id"] for e in applied} == set(points)
    assert len(applied) == len([c for c in result["candidates"] if c["route"] == "mirror"]) == 7


def test_a_replace_leaves_two_points(world):
    _run(world, apply=True)
    texts = {p["payload"]["text"]: p["payload"].get("fact_status", "active")
             for p in world["qdrant"].collection("memory").values()}
    assert texts[ENTRY_A] == "superseded" and ENTRY_A2 in texts


def test_a_rerun_creates_no_new_points_or_items(world):
    _run(world, apply=True)
    points = dict(world["qdrant"].collection("memory"))
    queued = [e for e in cq.CaptureQueue(str(world["home"])).events() if e["event"] == "queued"]
    save_cursor(cursor_path(str(world["home"])), str(world["db"]), 0)
    rerun = _run(world, apply=True)
    assert _mirror(rerun) == []
    assert "copied_call" not in rerun["report"]["counts"]["drops"]  # the same rows: duplicates, not copies
    assert rerun["report"]["counts"]["drops"]["duplicate_id"] == 9  # 8 mirror items and 1 heuristic item
    assert world["qdrant"].collection("memory").keys() == points.keys()
    assert [e for e in cq.CaptureQueue(str(world["home"])).events() if e["event"] == "queued"] == queued


def test_a_write_in_a_continuation_session_is_mirrored_once_and_cited_inside_its_session(world):
    """Compaction moves a conversation into a child session whose first row is the host's
    summary, and can copy the calling row again (new uid, same call) into the next child."""
    summary = "[CONTEXT COMPACTION] Earlier turns were summarized." + " filler" * 40
    args = {"target": "memory", "action": "add", "content": ENTRY_C}
    conn = sqlite3.connect(world["db"])
    add_session(conn, "s1-cont", parent="s1")
    first = add_message(conn, "s1-cont", "user", summary, 40)
    note = add_message(conn, "s1-cont", "assistant", "Continuing.", 41)
    call = _call(conn, "s1-cont", "c-cont", args, OK, 42)
    add_session(conn, "s1-cont2", parent="s1-cont")
    add_message(conn, "s1-cont2", "user", summary, 44)
    copy = _call(conn, "s1-cont2", "c-cont", args, OK, 45)
    conn.commit()
    conn.close()

    result = _run(world)

    got = [c for c in _mirror(result) if c["fact"] == ENTRY_C]
    assert [c["msg_id"] for c in got] == [call]
    assert (got[0]["citation"]["first_msg_id"], got[0]["citation"]["last_msg_id"]) == (first, note)
    assert result["report"]["counts"]["drops"]["copied_call"] == 1
    assert copy not in {c["msg_id"] for c in _mirror(result)}


def test_a_patch_without_full_entries_uses_a_known_entry_or_drops(world):
    conn = sqlite3.connect(world["db"])
    known = _call(conn, "s1", "c-p2", {"target": "memory", "action": "patch", "old_text": "port 8123",
                                       "content": "port 9000"}, OK, 30)
    add_session(conn, "s2")
    add_message(conn, "s2", "user", "Remember the dashboard port.", 31)
    _call(conn, "s2", "c-add2", {"target": "memory", "action": "add", "content": ENTRY_C}, OK, 32)
    unknown = _call(conn, "s2", "c-p3", {"target": "memory", "action": "patch", "old_text": "no such text",
                                         "content": "whatever"}, OK, 33)
    rebuilt = _call(conn, "s2", "c-p4", {"target": "memory", "action": "patch", "old_text": "port 8123",
                                         "content": "port 9000"}, OK, 34)
    conn.commit()
    conn.close()
    result = _run(world)
    facts = {(c["msg_id"], c["kind"]): c["fact"] for c in _mirror(result)}
    assert (known, "mirror_replace") not in facts  # ENTRY_C was not known yet at that row
    assert (unknown, "mirror_replace") not in facts
    assert result["report"]["counts"]["drops"]["entry_text_unknown"] == 2
    assert facts[(rebuilt, "mirror_replace")] == ENTRY_C.replace("port 8123", "port 9000")
    assert facts[(rebuilt, "mirror_superseded")] == ENTRY_C
