"""Capture CP4 (part 2): ``capture run`` with the heuristic detector (lane A).

Failure modes, each pinned by a test below:
- host-injected or machine-written rows become "the owner said" candidates;
- in a shared channel, another person's message becomes a candidate (``capture_owner_senders``
  empty means no shared-channel row qualifies);
- a candidate is queued whose citation does not resolve to its exact row;
- a message holding a secret reaches the queue (or a digest);
- a dry run writes anything (queue, Qdrant, cursor) or the state.db directory changes;
- ``--apply`` runs outside ``capture_mode: on``, or a run happens at all with capture off;
- the cursor advances without ``--apply`` in mode ``on``; a rerun of the same rows creates new
  points or new queue items;
- a repeated statement becomes a second item or a second point instead of a merged citation;
- the auto route fires without a second signal, ignores the daily cap, or applies an identity,
  money or security item, or a kind outside preference/invariant/correction/decision;
- a run report carries message text;
- an overdue review item is written instead of expiring with an event;
- code, quoted lines and links fire the detector.
"""
from __future__ import annotations

import json
import os
import sqlite3

import pytest

from capture_support import MemQdrant, add_message, add_session, make_db, make_provider
from qdrant_memory import capture_queue as cq
from qdrant_memory import capture_run as crun
from qdrant_memory.capture_detect import detect
from qdrant_memory.capture_reader import cursor_path, load_cursor, save_cursor

NOW = 1_794_100_000.0  # one day after the synthetic rows

OWNER_RULE = "From now on keep the nightly backups on the second disk only."
OWNER_PREF = "Prefiero que los resúmenes de estado sean cortos y en español."
OWNER_SCOPE = "Local models stay out of scope for the benchmark for now."
OWNER_MONEY = "Never invest more than the monthly budget in a single payment."
HOST_TEXT = "[IMPORTANT: Background process proc_1 completed normally] never mind this one"
CRON_TEXT = "Remember to always run the report"
SHARED_OTHER = "[Someone Else] from now on we always deploy on Fridays"
SHARED_OWNER = "[Owner Name] from now on we always deploy on Mondays"


@pytest.fixture
def world(tmp_path):
    db = tmp_path / "state" / "state.db"
    conn = make_db(db)
    ids = {}
    add_session(conn, "dm")
    ids["rule"] = add_message(conn, "dm", "user", OWNER_RULE, 1)
    ids["rule_answer"] = add_message(conn, "dm", "assistant", "Noted, I will keep that rule.", 2)
    ids["pref"] = add_message(conn, "dm", "user", OWNER_PREF, 3)
    ids["plain"] = add_message(conn, "dm", "user", "what is the weather like today", 4)
    ids["host"] = add_message(conn, "dm", "user", HOST_TEXT, 5)
    fake_key = "sk-" + "abcdefghij" * 3 + "0123456789"  # built at runtime: scanner-shaped, not a literal
    ids["secret"] = add_message(conn, "dm", "user", "Remember my api key is " + fake_key, 6)
    ids["code"] = add_message(conn, "dm", "user", "Look at this:\n```\n# never edit this file\n```", 7)
    ids["scope"] = add_message(conn, "dm", "user", OWNER_SCOPE, 8)
    ids["money"] = add_message(conn, "dm", "user", OWNER_MONEY, 9)
    ids["assistant_never"] = add_message(conn, "dm", "assistant", "I will never do that again.", 10)
    add_session(conn, "cron-run", "cron")
    ids["cron"] = add_message(conn, "cron-run", "user", CRON_TEXT, 11)
    add_session(conn, "group", "discord")
    conn.execute("UPDATE sessions SET chat_type = 'group' WHERE id = 'group'")
    ids["shared_other"] = add_message(conn, "group", "user", SHARED_OTHER, 12)
    ids["shared_owner"] = add_message(conn, "group", "user", SHARED_OWNER, 13)
    conn.commit()
    conn.close()
    home = tmp_path / "home"
    home.mkdir()
    return {"db": db, "home": home, "ids": ids, "qdrant": MemQdrant()}


def _provider(world, **config):
    return make_provider(world["home"], world["db"], world["qdrant"], session_id="cli", **config)


def _run(world, *, apply=False, provider=None, **kw):
    provider = provider or _provider(world)
    return crun.run_capture(provider, apply=apply, now=kw.pop("now", NOW), **kw)


def _by_msg(result):
    return {c["msg_id"]: c for c in result["candidates"]}


def test_only_the_owners_own_rows_become_candidates(world):
    ids = world["ids"]
    result = _run(world)
    got = _by_msg(result)

    assert set(got) == {ids["rule"], ids["pref"], ids["scope"], ids["money"]}
    assert result["report"]["counts"]["drops"]["possible_secret"] == 1
    assert result["report"]["counts"]["drops"]["not_owner"] == 2

    with_owner = _run(world, provider=_provider(world, capture_owner_senders=["owner name"]))
    assert ids["shared_owner"] in _by_msg(with_owner) and ids["shared_other"] not in _by_msg(with_owner)


def test_every_candidate_cites_its_exact_row_verified(world):
    from qdrant_memory.session_origin import verify_citation

    for candidate in _run(world)["candidates"]:
        citation = candidate["citation"]
        assert citation["first_msg_id"] == citation["last_msg_id"] == candidate["msg_id"]
        assert citation["basis"] == "extractor_source_line"
        assert verify_citation(citation, state_db=str(world["db"]))["status"] == "verified"


def test_an_unresolvable_citation_drops_the_candidate_with_a_reason(world, monkeypatch):
    monkeypatch.setattr(crun, "resolve_span", lambda *a, **k: "unresolved:no_match")
    result = _run(world)
    assert result["candidates"] == []
    assert result["report"]["counts"]["drops"]["citation_unresolved:no_match"] == 4


def test_a_dry_run_writes_nothing_but_its_report(world):
    listing = sorted(os.listdir(world["db"].parent))
    result = _run(world)

    assert result["report"]["apply"] is False
    assert not cq.CaptureQueue(str(world["home"])).path.exists()
    assert world["qdrant"].writes == []
    assert not cursor_path(str(world["home"])).exists()
    assert sorted(os.listdir(world["db"].parent)) == listing
    assert cq.run_report_path(str(world["home"]), result["report"]["run_id"]).exists()


def test_modes_gate_the_run(world):
    with pytest.raises(crun.CaptureRunRefused, match="off"):
        _run(world, provider=_provider(world, capture_mode="off"))
    with pytest.raises(crun.CaptureRunRefused, match="capture_mode: on"):
        _run(world, apply=True, provider=_provider(world, capture_mode="dry_run"))
    assert _run(world, provider=_provider(world, capture_mode="dry_run"))["report"]["apply"] is False


def test_apply_queues_review_items_advances_the_cursor_and_a_rerun_adds_nothing(world):
    result = _run(world, apply=True)
    queue = cq.CaptureQueue(str(world["home"]))
    pending = queue.pending(NOW)

    assert {item.queued["msg_id"] for item in pending} == set(_by_msg(result))
    assert all(item.route == "review" for item in pending)
    assert world["qdrant"].writes == []  # review-only: nothing stored without approval
    assert load_cursor(cursor_path(str(world["home"])), str(world["db"]))["last_msg_id"] == world["ids"]["shared_owner"]

    events_before = queue.path.read_bytes()
    save_cursor(cursor_path(str(world["home"])), str(world["db"]), 0)  # the same night again
    rerun = _run(world, apply=True)
    assert all(c["route"] == "drop" for c in rerun["candidates"]) or rerun["candidates"] == []
    assert rerun["report"]["counts"]["drops"]["duplicate_id"] == 4
    assert [line for line in queue.path.read_bytes().splitlines() if b'"queued"' in line] == \
        [line for line in events_before.splitlines() if b'"queued"' in line]
    assert world["qdrant"].writes == []


def test_a_repeated_statement_merges_its_citation(world):
    _run(world, apply=True)
    conn = sqlite3.connect(world["db"])
    repeat = add_message(conn, "dm", "user", OWNER_RULE.lower(), 20)
    conn.commit()
    conn.close()

    second = _run(world, apply=True)
    queue = cq.CaptureQueue(str(world["home"]))
    (merged,) = [c for c in second["candidates"] if c["msg_id"] == repeat]
    assert merged["route"] == "merge_queued"
    (target,) = [i for i in queue.pending(NOW) if i.queued["msg_id"] == world["ids"]["rule"]]
    assert [c["first_msg_id"] for c in target.extra_citations] == [repeat]
    assert len(queue.pending(NOW)) == 4


def test_a_compaction_clone_of_a_row_is_neither_a_new_item_nor_a_merged_citation(world):
    conn = sqlite3.connect(world["db"])
    uid = conn.execute("SELECT message_uid FROM messages WHERE id = ?", (world["ids"]["rule"],)).fetchone()[0]
    clone = conn.execute("INSERT INTO messages (session_id, role, content, timestamp, message_uid) VALUES (?, ?, ?, ?, ?)",
                         ("dm", "user", OWNER_RULE, 1_794_000_001.0, uid)).lastrowid
    conn.commit()
    conn.close()
    result = _run(world, apply=True)
    assert clone not in _by_msg(result)
    assert result["report"]["counts"]["drops"]["compaction_clone"] == 1
    (item,) = [i for i in cq.CaptureQueue(str(world["home"])).pending(NOW) if i.queued["msg_id"] == world["ids"]["rule"]]
    assert item.extra_citations == []


def test_a_near_match_of_a_stored_capture_merges_a_citation_into_that_point(world):
    provider = _provider(world)
    _run(world, apply=True, provider=provider)
    queue = cq.CaptureQueue(str(world["home"]))
    (rule,) = [i for i in queue.pending(NOW) if i.queued["msg_id"] == world["ids"]["rule"]]
    provider.capture_approve(rule.candidate_id)
    assert json.loads(provider.capture_approve(rule.candidate_id, dry_run=False, approve=True))["saved"]
    conn = sqlite3.connect(world["db"])
    near = add_message(conn, "dm", "user", "From now on keep the nightly backups on the second disk only, please.", 21)
    conn.commit()
    conn.close()

    third = _run(world, apply=True, provider=provider)

    (candidate,) = [c for c in third["candidates"] if c["msg_id"] == near]
    stored_id = cq.point_id_for(rule.candidate_id)
    assert candidate["route"] == "merge_point" and candidate["into"] == stored_id
    points = world["qdrant"].collection("memory")
    assert len(points) == 1
    cited = [c["first_msg_id"] for c in points[stored_id]["payload"]["session_origin"]["citations"]]
    assert cited == [world["ids"]["rule"], near]


class _Opinion:
    def __init__(self, confidence=0.95, scope="global", reversal=False):
        self.calls = 0
        self.value = {"confidence": confidence, "scope": scope, "reversal": reversal}

    def __call__(self, record, kind):
        self.calls += 1
        return dict(self.value)


def test_auto_needs_the_switch_and_a_second_signal(world):
    plain = _run(world, apply=True, provider=_provider(world, capture_auto_apply=True))
    assert {c["route"] for c in plain["candidates"]} == {"review"}
    assert world["qdrant"].writes == []


def _many_rules_db(world, count):
    conn = sqlite3.connect(world["db"])
    salt = os.urandom(3).hex()  # distinct words per row: no dense near match between them
    ids = [add_message(conn, "dm", "user",
                       f"From now on alpha{n}{salt} beta{n}{salt} gamma{n}{salt} delta{n}{salt} stays the default.",
                       30 + n) for n in range(count)]
    conn.commit()
    conn.close()
    return ids


def test_the_auto_route_holds_its_daily_cap_and_never_applies_sensitive_or_other_kinds(world):
    _many_rules_db(world, 7)
    provider = _provider(world, capture_auto_apply=True, capture_daily_auto_cap=5)
    result = _run(world, apply=True, provider=provider, second_opinion=_Opinion())

    routes = {c["msg_id"]: c["route"] for c in result["candidates"]}
    auto = [m for m, r in routes.items() if r == "auto"]
    assert len(auto) == 5
    assert routes[world["ids"]["money"]] == "review"  # money: never auto
    # In row order the first five eligible are the rule, the preference, the scope decision and two rules.
    assert routes[world["ids"]["rule"]] == routes[world["ids"]["pref"]] == routes[world["ids"]["scope"]] == "auto"
    assert len(world["qdrant"].collection("memory")) == 5
    queue = cq.CaptureQueue(str(world["home"]))
    applied = [e for e in queue.events() if e["event"] == "applied"]
    assert len(applied) == 5 and all(e["rule"] == "auto:hybrid" for e in applied)

    later = _many_rules_db(world, 2)
    again = _run(world, apply=True, provider=provider, second_opinion=_Opinion(), now=NOW + 60)
    assert {c["route"] for c in again["candidates"] if c["msg_id"] in later} == {"review"}  # cap of the same day
    next_day = _many_rules_db(world, 1)
    tomorrow = _run(world, apply=True, provider=provider, second_opinion=_Opinion(), now=NOW + 86400)
    assert {c["route"] for c in tomorrow["candidates"] if c["msg_id"] in next_day} == {"auto"}


@pytest.mark.parametrize("opinion", [_Opinion(confidence=0.8), _Opinion(scope="task"), _Opinion(reversal=True)])
def test_the_auto_rule_needs_a_confident_global_unreversed_second_signal(world, opinion):
    provider = _provider(world, capture_auto_apply=True)
    result = _run(world, apply=True, provider=provider, second_opinion=opinion)
    assert {c["route"] for c in result["candidates"]} == {"review"}


PROTECTED = {"source_type": "conversation", "requires_review": True, "fact_status": "review_required",
             "lineage_review_event_ids": ["event-1"]}


def test_the_run_never_overwrites_a_protected_point(world):
    """Exact target taken (auto route) and dense near match (merge): both leave the point alone."""
    from capture_support import BagEmbedding

    ids = world["ids"]
    taken = cq.point_id_for(cq.capture_id("dm", ids["rule"], "invariant", OWNER_RULE))
    near = "Prefiero que los resúmenes de estado sean cortos y en español, siempre."
    world["qdrant"].upsert("memory", [
        {"id": taken, "vector": [1.0], "payload": {"text": "flagged", **PROTECTED}},
        {"id": "00000000-0000-0000-0000-00000000beef", "vector": BagEmbedding().embed_document(near),
         "payload": {"text": near, "capture_lane": "heuristic", **PROTECTED}},
    ])
    world["qdrant"].writes.clear()
    before = {pid: dict(p["payload"]) for pid, p in world["qdrant"].collection("memory").items()}

    result = _run(world, apply=True, provider=_provider(world, capture_auto_apply=True), second_opinion=_Opinion())

    routes = {c["msg_id"]: c for c in result["candidates"]}
    assert ids["rule"] not in routes  # its exact point exists: dropped as already stored, never written
    assert result["report"]["counts"]["drops"]["already_stored"] == 1
    assert routes[ids["pref"]]["route"] == "merge_point" and routes[ids["pref"]]["outcome"] == "merge_refused"
    assert {pid: p["payload"] for pid, p in world["qdrant"].collection("memory").items() if pid in before} == before
    assert not any(op == "upsert" and set(pids) & set(before) for op, _, pids in world["qdrant"].writes)


def test_run_reports_hold_no_message_text(world):
    result = _run(world, apply=True)
    blob = cq.run_report_path(str(world["home"]), result["report"]["run_id"]).read_text()
    for text in (OWNER_RULE, OWNER_PREF, OWNER_SCOPE, "backups", "resúmenes"):
        assert text not in blob


def test_overdue_review_items_expire_with_an_event_and_no_write(world):
    _run(world, apply=True)
    later = NOW + 22 * 86400
    result = _run(world, apply=True, now=later)
    queue = cq.CaptureQueue(str(world["home"]))
    assert len(result["report"]["expired"]) == 4
    assert {e["event"] for e in queue.events()} == {"queued", "expired"}
    assert queue.pending(later) == []
    assert world["qdrant"].writes == []


@pytest.mark.parametrize("text,fires", [
    ("Look:\n```\nnever edit this\n```", False),
    ("> from now on we deploy daily\nwhat do you think?", False),
    ("see https://example.com/never-again", False),
    ("`always_on: true` is the flag name", False),
    ("No. The service must not restart on its own.", True),
    ("Recuerda que el backup va en el disco grande.", True),
])
def test_code_quotes_and_links_do_not_fire(text, fires):
    assert (detect(text) is not None) is fires
