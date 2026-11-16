"""Approving a capture item with a distilled ``fact`` (the owner's or the agent's one sentence).

Failure modes, each pinned by a test below:
- the queued message is stored although a distilled fact was given (heuristic and mirror lanes);
- the citation no longer points at the exact queued source row, or is not re-verified: an
  edited/vanished row must still refuse the write when a fact is given;
- the write gate runs on the queued message instead of the new text, so a secret in the new
  text is stored;
- the dry run previews one text and the live call stores another (the preview binds the text);
- an empty, whitespace-only or oversized fact is accepted;
- the audit does not say the text was edited, or carries text (original or new) instead of
  digests;
- the stored point does not say its text was edited;
- the CLI drops ``--fact``;
- the tool schema does not offer ``fact`` or does not tell the agent to propose a sentence;
- the digest's chat hint does not mention the distilled sentence.
"""
from __future__ import annotations

import io
import json
import sqlite3
from argparse import Namespace

import pytest

from capture_support import MemQdrant, add_message, add_session, call, make_db, make_provider
from qdrant_memory import capture_queue as cq
from qdrant_memory.session_origin import resolve_span

RAW = ("Nice work on the report. Yes, that was the layout I had in mind. Remember to put the summary "
       "table first whenever I ask for the weekly report please")
DISTILLED = "Weekly reports start with the summary table."


@pytest.fixture
def world(tmp_path):
    db = tmp_path / "state" / "state.db"
    conn = make_db(db)
    add_session(conn, "captured-session")
    ids = {"user": add_message(conn, "captured-session", "user", RAW, 1)}
    ids["answer"] = add_message(conn, "captured-session", "assistant", "Saved.", 2)
    add_session(conn, "approver-session")
    conn.commit()
    conn.close()
    home = tmp_path / "home"
    home.mkdir()
    return {"db": db, "home": home, "ids": ids, "qdrant": MemQdrant()}


def _queue_heuristic(world, *, fact=RAW, kind="preference"):
    msg_id = world["ids"]["user"]
    citation = resolve_span(str(world["db"]), "captured-session", msg_id, msg_id, basis="extractor_source_line")
    event = cq.queued_event(
        run_id="run-test", lane="heuristic", kind=kind, route="review", session_id="captured-session",
        msg_id=msg_id, fact=fact, citation=citation, rule_ids=["record_request"],
        write=cq.heuristic_write(session_id="captured-session", msg_id=msg_id, kind=kind, fact=fact,
                                 rule_ids=["record_request"]))
    cq.CaptureQueue(str(world["home"])).append(event)
    return event


def _queue_mirror(world, text="Owner prefers status updates in Spanish, short, with the commands at the end."):
    first, last = world["ids"]["user"], world["ids"]["answer"]
    citation = resolve_span(str(world["db"]), "captured-session", first, last, basis="agent_proposed")
    event = cq.queued_event(
        run_id="run-test", lane="mirror", kind="mirror_add", route="review", session_id="captured-session",
        msg_id=last, fact=text, citation=citation, rule_ids=["mirror:target_user"],
        write=cq.manual_write(text=text, tags=["capture", "builtin_memory", "target:user"]), target="user")
    cq.CaptureQueue(str(world["home"])).append(event)
    return event


def _provider(world, **kw):
    return make_provider(world["home"], world["db"], world["qdrant"], **kw)


def _approve(provider, cid, fact):
    preview = call(provider, "qdrant_capture_approve", id=cid, fact=fact)
    if "error" in preview:
        return preview, preview
    return preview, call(provider, "qdrant_capture_approve", id=cid, fact=fact, dry_run=False, approve=True)


def test_a_heuristic_item_approved_with_a_fact_stores_the_fact_with_the_exact_verified_citation(world):
    event = _queue_heuristic(world)
    provider = _provider(world)

    preview, stored = _approve(provider, event["candidate_id"], DISTILLED)

    assert preview["dry_run"] is True and preview["fact"] == DISTILLED and preview["fact_edited"] is True
    assert stored["saved"] is True, stored
    point = world["qdrant"].collection("memory")[stored["point_id"]]
    payload = point["payload"]
    assert payload["text"] == DISTILLED
    assert RAW not in json.dumps(payload)
    assert payload["capture_fact_edited"] is True
    assert payload["capture_original_fact_sha256"] == cq.sha256_text(RAW)
    (citation,) = payload["session_origin"]["citations"]
    for key in ("session_id", "first_msg_id", "last_msg_id", "span_sha256", "content_sha256", "uids_sha256"):
        assert citation[key] == event["citation"][key]
    assert citation["basis"] == "extractor_source_line" and citation.get("verified_at")
    # The stored vector is the new text's, not the message's.
    assert point["vector"] == provider._embeddings.embed_document(DISTILLED)


def test_the_audit_records_the_edit_as_digests_only(world):
    cid = _queue_heuristic(world)["candidate_id"]
    _approve(_provider(world), cid, DISTILLED)

    events = [e for e in cq.CaptureQueue(str(world["home"])).events() if e["candidate_id"] == cid]
    approved, applied = events[1], events[2]
    assert [e["event"] for e in events] == ["queued", "approved", "applied"]
    for event in (approved, applied):
        assert set(event) <= cq.AUDIT_KEYS
        assert event["fact_edited"] is True
        assert event["original_fact_sha256"] == cq.sha256_text(RAW)
        blob = json.dumps(event)
        assert "summary" not in blob and "layout" not in blob
    assert applied["fact_sha256"] == cq.sha256_text(DISTILLED)


def test_without_a_fact_the_queued_text_is_stored_and_nothing_says_edited(world):
    cid = _queue_heuristic(world)["candidate_id"]
    preview, stored = _approve(_provider(world), cid, None)
    assert preview["fact_edited"] is False
    payload = world["qdrant"].collection("memory")[stored["point_id"]]["payload"]
    assert payload["text"] == RAW and "capture_fact_edited" not in payload
    applied = cq.CaptureQueue(str(world["home"])).events()[-1]
    assert "fact_edited" not in applied and applied["fact_sha256"] == cq.sha256_text(RAW)


def test_a_mirror_item_approved_with_a_fact_stores_the_fact_through_the_manual_store(world):
    event = _queue_mirror(world)
    preview, stored = _approve(_provider(world), event["candidate_id"], "Owner wants short Spanish status updates.")
    assert stored["saved"] is True, stored
    payload = world["qdrant"].collection("memory")[stored["point_id"]]["payload"]
    assert payload["text"] == "Owner wants short Spanish status updates."
    assert payload["capture_fact_edited"] is True
    assert payload["capture_original_fact_sha256"] == event["fact_sha256"]
    origin = payload["session_origin"]
    assert origin["producer"] == "manual_store" and origin["citations"][0]["basis"] == "agent_proposed"
    assert origin["citations"][0]["span_sha256"] == event["citation"]["span_sha256"]


def test_an_edited_cited_row_still_refuses_a_distilled_fact(world):
    cid = _queue_heuristic(world)["candidate_id"]
    conn = sqlite3.connect(world["db"])
    conn.execute("UPDATE messages SET content = ? WHERE id = ?", (RAW + " (edited)", world["ids"]["user"]))
    conn.commit()
    conn.close()
    preview, live = _approve(_provider(world), cid, DISTILLED)
    assert "cited" in preview["error"] and "error" in live
    assert world["qdrant"].writes == []
    assert cq.CaptureQueue(str(world["home"])).find(cid).status == "pending"


@pytest.mark.parametrize("lane", ["heuristic", "mirror"])
def test_the_write_gate_runs_on_the_new_text(world, lane):
    cid = (_queue_heuristic(world) if lane == "heuristic" else _queue_mirror(world))["candidate_id"]
    secret = "Use the deploy token ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8" + " for the weekly report."
    preview, live = _approve(_provider(world), cid, secret)
    assert "error" in live, live
    assert all(op != "upsert" for op, _, _ in world["qdrant"].writes)
    assert cq.CaptureQueue(str(world["home"])).find(cid).status == "pending"


def test_the_live_text_must_be_the_previewed_text(world):
    cid = _queue_heuristic(world)["candidate_id"]
    provider = _provider(world)
    call(provider, "qdrant_capture_approve", id=cid, fact=DISTILLED)
    swapped = call(provider, "qdrant_capture_approve", id=cid, fact="A different sentence the owner never saw.",
                   dry_run=False, approve=True)
    assert "dry-run" in swapped["error"]
    raw_live = call(provider, "qdrant_capture_approve", id=cid, dry_run=False, approve=True)
    assert "dry-run" in raw_live["error"]
    assert world["qdrant"].writes == []


@pytest.mark.parametrize("bad", ["", "   \n\t ", "x" * 501])
def test_an_unusable_fact_is_refused(world, bad):
    cid = _queue_heuristic(world)["candidate_id"]
    result = call(_provider(world), "qdrant_capture_approve", id=cid, fact=bad)
    assert "fact" in result["error"]
    assert world["qdrant"].writes == []


def test_whitespace_in_a_fact_is_collapsed_to_one_line(world):
    cid = _queue_heuristic(world)["candidate_id"]
    _, stored = _approve(_provider(world), cid, "  Weekly reports start\n with the   summary table. ")
    payload = world["qdrant"].collection("memory")[stored["point_id"]]["payload"]
    assert payload["text"] == "Weekly reports start with the summary table."


def test_cli_fact_flag(world, monkeypatch, capsys):
    from qdrant_memory.cli_core import execute_command

    monkeypatch.setenv("HERMES_HOME", str(world["home"]))
    cid = _queue_heuristic(world)["candidate_id"]
    out = io.StringIO()
    args = Namespace(qdrant_subcommand="capture", capture_subcommand="approve", id=cq.short_id(cid), dry_run=False,
                     approve=True, json=True, fact=DISTILLED)
    code = execute_command(args, provider_factory=lambda: _provider(world), stdout=out)
    assert code == 0, capsys.readouterr().err
    stored = json.loads(out.getvalue())
    assert world["qdrant"].collection("memory")[stored["point_id"]]["payload"]["text"] == DISTILLED


def test_cli_parser_accepts_fact():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("qdrant_plugin_cli_fact", Path(__file__).resolve().parents[1] / "cli.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import argparse

    parser = argparse.ArgumentParser()
    module.register_cli(parser)
    args = parser.parse_args(["capture", "approve", "abcdefabcdef", "--fact", DISTILLED])
    assert args.fact == DISTILLED
    assert parser.parse_args(["capture", "approve", "abcdefabcdef"]).fact is None


def test_tool_schema_offers_fact_and_tells_the_agent_to_distil():
    from qdrant_memory.tools import CAPTURE_APPROVE_SCHEMA, CAPTURE_REJECT_SCHEMA

    props = CAPTURE_APPROVE_SCHEMA["parameters"]["properties"]
    assert props["fact"]["type"] == "string"
    assert "fact" not in CAPTURE_APPROVE_SCHEMA["parameters"]["required"]
    assert "one-sentence" in CAPTURE_APPROVE_SCHEMA["description"]
    assert "fact" not in CAPTURE_REJECT_SCHEMA["parameters"]["properties"]


@pytest.mark.parametrize("language,needle", [("en", "one sentence"), ("es", "una frase")])
def test_digest_chat_hint_mentions_the_distilled_sentence(world, language, needle):
    from qdrant_memory.capture_digest import digest

    _queue_heuristic(world)
    text = digest(str(world["home"]), language=language)
    assert needle in text
    assert "--fact" in text
