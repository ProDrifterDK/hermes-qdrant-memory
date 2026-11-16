"""Capture CP4 (part 1): the persisted review queue and its approval routes.

Failure modes, each pinned by a test below:
- the queue lives only in provider memory, so a restart (or the CLI's one process per
  command) loses it, and a fresh provider cannot approve what another process queued;
- the queue file is rewritten instead of appended to, or is readable by other users;
- approval stamps the approving session as the fact's origin instead of the captured one;
- approval writes although the cited row changed or is gone since capture;
- a live approval runs without a dry run of the same id first, or without approve=true;
- a cron session approves or rejects (decision P7); listing stays allowed;
- the capture tools show on the recall surface, or on installs that never turned capture on;
- an item is approved twice, approved after it was rejected, or approved after it expired;
- a short or mistyped id approves a different item;
- audit events (approved, applied, rejected) carry message text;
- an identity item (built-in memory USER target) is written through anything but the manual
  store path with an ``agent_proposed`` citation.
"""
from __future__ import annotations

import json
import os
import stat
from argparse import Namespace

import pytest

from capture_support import MemQdrant, add_message, add_session, call, make_db, make_provider
from qdrant_memory import capture_queue as cq
from qdrant_memory.session_origin import resolve_span

FACT = "Keep the nightly backups on the second disk, never on the system disk."


@pytest.fixture
def world(tmp_path):
    db = tmp_path / "state" / "state.db"
    conn = make_db(db)
    add_session(conn, "captured-session")
    ids = {"user": add_message(conn, "captured-session", "user", FACT, 1)}
    ids["answer"] = add_message(conn, "captured-session", "assistant", "Noted.", 2)
    add_session(conn, "approver-session")
    add_message(conn, "approver-session", "user", "approve the capture", 3)
    conn.commit()
    conn.close()
    home = tmp_path / "home"
    home.mkdir()
    return {"db": db, "home": home, "ids": ids, "qdrant": MemQdrant()}


def _queue_heuristic(world, *, fact=FACT, kind="invariant", expiry_days=21, now=None):
    msg_id = world["ids"]["user"]
    citation = resolve_span(str(world["db"]), "captured-session", msg_id, msg_id, basis="extractor_source_line")
    assert isinstance(citation, dict)
    event = cq.queued_event(
        run_id="run-test", lane="heuristic", kind=kind, route="review", session_id="captured-session",
        msg_id=msg_id, fact=fact, citation=citation, rule_ids=["never_always"],
        write=cq.heuristic_write(session_id="captured-session", msg_id=msg_id, kind=kind, fact=fact,
                                 rule_ids=["never_always"]),
        expiry_days=expiry_days, now=now)
    cq.CaptureQueue(str(world["home"])).append(event)
    return event


def _provider(world, **kw):
    return make_provider(world["home"], world["db"], world["qdrant"], **kw)


def test_a_fresh_provider_approves_what_another_process_queued_with_the_captured_session_as_origin(world):
    event = _queue_heuristic(world)
    cid = event["candidate_id"]
    provider = _provider(world)  # nothing in memory: only the queue file

    preview = call(provider, "qdrant_capture_approve", id=cid)
    assert preview["dry_run"] is True and preview["citation_check"]["status"] == "verified"
    assert world["qdrant"].collection("memory") == {}
    stored = call(provider, "qdrant_capture_approve", id=cid, dry_run=False, approve=True)

    assert stored["saved"] is True, stored
    point = world["qdrant"].collection("memory")[stored["point_id"]]
    origin = point["payload"]["session_origin"]
    assert origin["captured_session_id"] == "captured-session"
    assert origin["approved_session_id"] == "approver-session"
    assert point["payload"]["session_id"] == "captured-session"
    assert origin["producer"] == "source_extraction"
    (citation,) = origin["citations"]
    assert citation["basis"] == "extractor_source_line"
    for key in ("session_id", "first_msg_id", "last_msg_id", "span_sha256", "content_sha256", "uids_sha256"):
        assert citation[key] == event["citation"][key]
    assert point["payload"]["text"] == FACT
    item = cq.CaptureQueue(str(world["home"])).find(cid)
    assert item.status == "applied"
    assert [e["event"] for e in item.history] == ["queued", "approved", "applied"]


def test_the_queue_is_append_only_and_private(world):
    _queue_heuristic(world)
    path = cq.CaptureQueue(str(world["home"])).path
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    before = path.read_bytes()
    _queue_heuristic(world, fact="A second synthetic fact that is long enough to keep.")
    after = path.read_bytes()
    assert after.startswith(before) and len(after) > len(before)
    assert all(json.loads(line)["schema"] == cq.QUEUE_SCHEMA for line in after.decode().splitlines())


def test_live_approval_needs_a_dry_run_of_the_same_id_and_approve_true(world):
    cid = _queue_heuristic(world)["candidate_id"]
    provider = _provider(world)
    refused = call(provider, "qdrant_capture_approve", id=cid, dry_run=False, approve=True)
    assert "dry-run" in refused["error"]
    call(provider, "qdrant_capture_approve", id=cid)
    assert "approve=true" in call(provider, "qdrant_capture_approve", id=cid, dry_run=False)["error"]
    assert world["qdrant"].writes == []


@pytest.mark.parametrize("change", ["edit", "delete_row", "delete_session"])
def test_approval_refuses_when_the_cited_row_changed_or_is_gone(world, change):
    import sqlite3

    cid = _queue_heuristic(world)["candidate_id"]
    conn = sqlite3.connect(world["db"])
    if change == "edit":
        conn.execute("UPDATE messages SET content = ? WHERE id = ?", (FACT + " (edited)", world["ids"]["user"]))
    elif change == "delete_row":
        conn.execute("DELETE FROM messages WHERE id = ?", (world["ids"]["user"],))
    else:
        conn.execute("DELETE FROM messages WHERE session_id = 'captured-session'")
        conn.execute("DELETE FROM sessions WHERE id = 'captured-session'")
    conn.commit()
    conn.close()
    provider = _provider(world)

    preview = call(provider, "qdrant_capture_approve", id=cid)
    live = call(provider, "qdrant_capture_approve", id=cid, dry_run=False, approve=True)

    assert "error" in preview and "cited" in preview["error"]
    assert "error" in live
    assert world["qdrant"].writes == []
    assert cq.CaptureQueue(str(world["home"])).find(cid).status == "pending"


def test_a_cron_session_lists_but_never_approves_or_rejects(world):
    cid = _queue_heuristic(world)["candidate_id"]
    cron = _provider(world, platform="cron", agent_context="cron")

    listed = call(cron, "qdrant_capture_review")
    assert [i["id"] for i in listed["items"]] == [cid]
    for tool in ("qdrant_capture_approve", "qdrant_capture_reject"):
        for args in ({}, {"dry_run": False, "approve": True}):
            result = call(cron, tool, id=cid, **args)
            assert "cron" in result["error"], (tool, args, result)
    assert world["qdrant"].writes == []
    assert [e["event"] for e in cq.CaptureQueue(str(world["home"])).events()] == ["queued"]


def test_a_cron_session_marked_only_by_the_session_env_cannot_approve(world, monkeypatch):
    cid = _queue_heuristic(world)["candidate_id"]
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    provider = _provider(world)
    assert "cron" in call(provider, "qdrant_capture_approve", id=cid)["error"]


def test_capture_tools_are_full_surface_only_and_absent_while_capture_is_off(world):
    names = lambda p: {s["name"] for s in p.get_tool_schemas()}  # noqa: E731
    capture_tools = {"qdrant_capture_review", "qdrant_capture_approve", "qdrant_capture_reject"}
    assert capture_tools <= names(_provider(world))
    assert not capture_tools & names(_provider(world, tool_surface="recall"))
    assert not capture_tools & names(_provider(world, capture_mode="off"))
    assert "not available" in call(_provider(world, tool_surface="recall"), "qdrant_capture_review")["error"]
    assert "capture_mode" in call(_provider(world, capture_mode="off"), "qdrant_capture_review")["error"]


def test_an_item_is_written_once_and_never_after_rejection_or_expiry(world):
    provider = _provider(world)
    applied = _queue_heuristic(world)["candidate_id"]
    call(provider, "qdrant_capture_approve", id=applied)
    assert call(provider, "qdrant_capture_approve", id=applied, dry_run=False, approve=True)["saved"]
    again = call(provider, "qdrant_capture_approve", id=applied)
    assert "applied" in again["error"]

    rejected = _queue_heuristic(world, fact="Rejected synthetic fact, long enough to keep around.")["candidate_id"]
    assert call(provider, "qdrant_capture_reject", id=rejected)["dry_run"] is True
    assert call(provider, "qdrant_capture_reject", id=rejected, dry_run=False, approve=True)["rejected"] is True
    assert "rejected" in call(provider, "qdrant_capture_approve", id=rejected)["error"]

    old = _queue_heuristic(world, fact="Expired synthetic fact, also long enough to keep.", expiry_days=21,
                           now=1_000_000_000.0)["candidate_id"]
    assert "expired" in call(provider, "qdrant_capture_approve", id=old)["error"]
    assert len(world["qdrant"].collection("memory")) == 1


def test_ids_resolve_exactly(world):
    cid = _queue_heuristic(world)["candidate_id"]
    queue = cq.CaptureQueue(str(world["home"]))
    assert queue.find(cid).candidate_id == cid
    assert queue.find(cq.short_id(cid)).candidate_id == cid
    for wrong in (cid[:6], cid[:11], "f" * 12, cid[:12] + "x"):
        with pytest.raises(cq.CaptureQueueError):
            queue.find(wrong)


def test_audit_events_hold_ids_and_digests_never_text(world):
    provider = _provider(world)
    cid = _queue_heuristic(world)["candidate_id"]
    call(provider, "qdrant_capture_approve", id=cid)
    call(provider, "qdrant_capture_approve", id=cid, dry_run=False, approve=True)
    other = _queue_heuristic(world, fact="Another synthetic statement that the owner rejects.")["candidate_id"]
    call(provider, "qdrant_capture_reject", id=other)
    call(provider, "qdrant_capture_reject", id=other, dry_run=False, approve=True)

    for event in cq.CaptureQueue(str(world["home"])).events():
        if event["event"] == "queued":
            continue
        blob = json.dumps(event)
        assert "backups" not in blob and "rejects" not in blob, event
        assert set(event) <= cq.AUDIT_KEYS, set(event) - cq.AUDIT_KEYS


def test_a_user_target_mirror_item_is_approved_through_the_manual_store_with_an_agent_proposed_citation(world):
    first, last = world["ids"]["user"], world["ids"]["answer"]
    citation = resolve_span(str(world["db"]), "captured-session", first, last, basis="agent_proposed")
    text = "Owner prefers short status updates in Spanish."
    event = cq.queued_event(
        run_id="run-test", lane="mirror", kind="mirror_add", route="review", session_id="captured-session",
        msg_id=last, fact=text, citation=citation, rule_ids=["mirror:target_user"],
        write=cq.manual_write(text=text, tags=["capture", "builtin_memory", "target:user"]), target="user")
    cq.CaptureQueue(str(world["home"])).append(event)
    provider = _provider(world)

    call(provider, "qdrant_capture_approve", id=event["candidate_id"])
    stored = call(provider, "qdrant_capture_approve", id=event["candidate_id"], dry_run=False, approve=True)

    point = world["qdrant"].collection("memory")[stored["point_id"]]
    origin = point["payload"]["session_origin"]
    assert origin["producer"] == "manual_store" and origin["captured_session_id"] == "captured-session"
    assert origin["approved_session_id"] == "approver-session"
    assert origin["citations"][0]["basis"] == "agent_proposed"
    assert origin["citations"][0]["first_msg_id"] == first and origin["citations"][0]["last_msg_id"] == last


PROTECTED = {"source_type": "conversation", "requires_review": True, "fact_status": "review_required",
             "lineage_review_event_ids": ["event-1"]}


def test_capture_approval_refuses_to_overwrite_a_protected_point(world):
    """Heuristic lane: the extraction approval path's lineage overwrite refusal holds."""
    cid = _queue_heuristic(world)["candidate_id"]
    target = cq.point_id_for(cid)
    world["qdrant"].upsert("memory", [{"id": target, "vector": [1.0], "payload": {"text": "flagged", **PROTECTED}}])
    world["qdrant"].writes.clear()
    provider = _provider(world)
    call(provider, "qdrant_capture_approve", id=cid)
    result = call(provider, "qdrant_capture_approve", id=cid, dry_run=False, approve=True)
    assert "error" in result
    assert world["qdrant"].writes == []
    assert world["qdrant"].collection("memory")[target]["payload"]["text"] == "flagged"
    assert cq.CaptureQueue(str(world["home"])).find(cid).status == "pending"


def test_capture_approval_of_a_mirror_item_refuses_to_overwrite_a_protected_point(world):
    """Manual store lane: the writer's lineage overwrite refusal holds."""
    from qdrant_memory.capture_apply import MIRROR_SOURCE
    from qdrant_memory.schema import make_point_id

    msg = world["ids"]["answer"]
    citation = resolve_span(str(world["db"]), "captured-session", msg, msg, basis="agent_proposed")
    text = "Owner prefers short status updates in Spanish."
    event = cq.queued_event(run_id="r", lane="mirror", kind="mirror_add", route="review", session_id="captured-session",
                            msg_id=msg, fact=text, citation=citation, rule_ids=["mirror:target_user"],
                            write=cq.manual_write(text=text, tags=["capture"]), target="user")
    cq.CaptureQueue(str(world["home"])).append(event)
    target = make_point_id(MIRROR_SOURCE, text)
    world["qdrant"].upsert("memory", [{"id": target, "vector": [1.0], "payload": {"text": text, **PROTECTED}}])
    world["qdrant"].writes.clear()
    provider = _provider(world)
    call(provider, "qdrant_capture_approve", id=event["candidate_id"])
    result = call(provider, "qdrant_capture_approve", id=event["candidate_id"], dry_run=False, approve=True)
    assert "error" in result
    assert world["qdrant"].writes == []


def _cli(provider, capsys, **fields):
    from qdrant_memory.cli_core import execute_command
    import io

    out = io.StringIO()
    args = Namespace(qdrant_subcommand="capture", json=True, **fields)
    code = execute_command(args, provider_factory=lambda: provider, stdout=out)
    return code, out.getvalue(), capsys.readouterr().err


def test_cli_review_approve_reject(world, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(world["home"]))
    cid = _queue_heuristic(world)["candidate_id"]
    provider = _provider(world)

    code, out, _ = _cli(provider, capsys, capture_subcommand="review", limit=10, offset=0)
    assert code == 0 and json.loads(out)["items"][0]["id"] == cid
    code, out, _ = _cli(provider, capsys, capture_subcommand="approve", id=cq.short_id(cid), dry_run=True,
                        approve=False)
    assert code == 0 and json.loads(out)["dry_run"] is True and world["qdrant"].writes == []
    code, out, err = _cli(_provider(world), capsys, capture_subcommand="approve", id=cq.short_id(cid),
                          dry_run=False, approve=True)
    assert code == 0, err
    assert json.loads(out)["saved"] is True

    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    other = _queue_heuristic(world, fact="Another synthetic fact for the cron refusal check.")["candidate_id"]
    code, _, err = _cli(_provider(world), capsys, capture_subcommand="reject", id=other, dry_run=False, approve=True)
    assert code != 0 and "cron" in err
