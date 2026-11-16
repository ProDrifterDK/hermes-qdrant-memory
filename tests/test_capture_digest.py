"""Capture digest: the daily plain-text summary a no-agent cron job delivers.

Failure modes, each pinned by a test below:
- it prints something when nothing happened since the last digest (the cron would deliver
  an empty or repeated message);
- it repeats the same news every day (its marker does not advance);
- it mutates anything besides its own marker (queue, cursor, Qdrant, run reports);
- it lists more than the requested items (decision P5: at most 10), or lists items that are
  no longer pending (approved, rejected, expired by time);
- a multi-line fact breaks the one-line format;
- it omits the ids and the exact commands needed to approve or reject;
- the Spanish variant is not Spanish.
"""
from __future__ import annotations

import io
import json
import os
from argparse import Namespace

import pytest

from qdrant_memory import capture_digest as cd
from qdrant_memory import capture_queue as cq

NOW = 1_794_100_000.0
CITATION = {"store": "hermes-state.db", "session_id": "s", "first_msg_id": 1, "last_msg_id": 1, "msg_count": 1,
            "uids_sha256": "a" * 64, "span_sha256": "b" * 64, "content_sha256": "c" * 64,
            "basis": "extractor_source_line", "verified_at": "2026-11-01T00:00:00Z"}


def _queue(home, n, *, route="review", now=NOW, fact="Synthetic durable fact number {n} about the backups."):
    events = []
    for i in range(n):
        text = fact.format(n=i)
        events.append(cq.queued_event(run_id="r", lane="heuristic", kind="invariant", route=route, session_id="s",
                                      msg_id=100 + i, fact=text, citation=CITATION, rule_ids=["remember"],
                                      write=cq.heuristic_write(session_id="s", msg_id=100 + i, kind="invariant",
                                                               fact=text, rule_ids=["remember"]), now=now))
    cq.CaptureQueue(str(home)).append(*events)
    return events


def _snapshot(home):
    return {str(p.relative_to(home)): p.read_bytes() for p in home.rglob("*") if p.is_file()
            and p.name != "digest.json"}


def test_nothing_new_prints_nothing(tmp_path):
    assert cd.digest(str(tmp_path), now=NOW) == ""          # no queue at all
    _queue(tmp_path, 2)
    assert cd.digest(str(tmp_path), now=NOW) != ""
    assert cd.digest(str(tmp_path), now=NOW + 60) == ""     # the same news is not repeated


def test_the_digest_changes_nothing_but_its_marker(tmp_path):
    _queue(tmp_path, 3)
    (tmp_path / "qdrant_memory" / "capture" / "cursor.json").write_text("{}")
    before = _snapshot(tmp_path)
    cd.digest(str(tmp_path), now=NOW)
    assert _snapshot(tmp_path) == before
    assert (tmp_path / "qdrant_memory" / "capture" / "digest.json").exists()


def test_counts_items_ids_and_commands(tmp_path):
    events = _queue(tmp_path, 14)
    _queue(tmp_path, 2, route="mirror", fact="Mirrored synthetic entry {n} about the dashboards.")
    queue = cq.CaptureQueue(str(tmp_path))
    queue.append(cq.audit_event("rejected", events[0]["candidate_id"], now=NOW),
                 cq.audit_event("approved", events[1]["candidate_id"], now=NOW),
                 cq.audit_event("applied", events[1]["candidate_id"], now=NOW, point_id="p1", route="review",
                                rule="review:approved"))
    text = cd.digest(str(tmp_path), max_items=10, now=NOW)
    lines = text.splitlines()

    item_lines = [line for line in lines if line.startswith("- [")]
    assert len(item_lines) == 10
    assert events[0]["candidate_id"][:12] not in text and events[1]["candidate_id"][:12] not in text
    assert all(len(line.split("] ", 1)[0]) == len("- [") + 12 for line in item_lines)
    assert "14 queued for review" in text and "2 mirrored" in text and "1 rejected" in text and "1 approved" in text
    assert "Pending review: 12" in text
    assert "hermes qdrant capture approve <id> --no-dry-run --approve" in text
    assert "hermes qdrant capture reject <id> --no-dry-run --approve" in text
    assert "(+2 more" in text


def test_items_are_one_line_and_expired_items_are_not_pending(tmp_path):
    _queue(tmp_path, 1, fact="First line of a fact {n}\nsecond line\n\nthird line with enough words.")
    _queue(tmp_path, 1, now=NOW - 30 * 86400, fact="An old fact {n} that already expired by time.")
    text = cd.digest(str(tmp_path), now=NOW)
    item_lines = [line for line in text.splitlines() if line.startswith("- [")]
    assert len(item_lines) == 1 and "second line third line" in item_lines[0]
    assert "already expired" not in text


def test_items_expiring_soon_are_flagged(tmp_path):
    _queue(tmp_path, 1, now=NOW - 19 * 86400)
    text = cd.digest(str(tmp_path), now=NOW)
    assert "Expiring within 3 days" in text


def test_spanish(tmp_path):
    _queue(tmp_path, 1)
    text = cd.digest(str(tmp_path), language="es", now=NOW)
    assert "por revisar" in text and "Aprobar" in text and "Rechazar" in text


def test_cli_prints_nothing_when_nothing_is_new(tmp_path, monkeypatch):
    from qdrant_memory.cli_core import execute_command

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    args = Namespace(qdrant_subcommand="capture", capture_subcommand="digest", max_items=10, language=None, json=False)
    out = io.StringIO()
    assert execute_command(args, stdout=out) == 0 and out.getvalue() == ""
    _queue(tmp_path, 1)
    out = io.StringIO()
    assert execute_command(args, stdout=out) == 0 and "1 queued for review" in out.getvalue()
    out = io.StringIO()
    assert execute_command(args, stdout=out) == 0 and out.getvalue() == ""
