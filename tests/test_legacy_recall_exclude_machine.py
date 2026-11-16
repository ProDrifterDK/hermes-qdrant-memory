"""``auto_recall_exclude_machine``: stored turns of test harnesses and probes in automatic recall.

Failure modes, written before the code:

1. A conversation-turn point of a one-shot run (its session's source is ``oneshot`` in
   ``state.db``) reaches the legacy block, whatever its text says.
2. A turn of a delegated or subagent session is dropped (their findings are asked about
   later), also when its text looks like a scripted check.
3. A turn whose session is gone from ``state.db`` is dropped although a person wrote it, or
   an unreadable/missing ``state.db`` empties the block or raises.
4. A turn whose session is gone and whose text proves a scripted check (the prompt demands an
   exact reply and the answer is that literal, copied from the prompt) is kept.
5. A turn whose user side is host text that the always-on scaffolding filter does not know
   (``capture_reader`` host shapes) is kept.
6. A person's interactive session is judged by its text: a probe typed in a ``cli`` session
   whose row still exists is dropped (known sessions are judged by structure).
7. A document or a manual fact is judged at all (only conversation turns are).
8. A dropped point shrinks the block although more memories ranked below it (no backfill).
9. With the option off (the default) the block differs from the plain legacy path.
10. The hybrid mode's dense lane still injects such a turn.
11. The filter writes to Qdrant or creates files next to ``state.db``.

Only the network boundaries are fake (the Qdrant transport and the embedder, shared with
``test_session_prefetch``). ``state.db`` is a real SQLite file built from the production
schema; the provider is wired through its real ``initialize`` and read through ``prefetch``.
"""
from __future__ import annotations

import importlib
import re
import sqlite3
from pathlib import Path

from qdrant_memory.retriever import format_for_prompt

tsp = importlib.import_module("test_session_prefetch")

QUERY = tsp.QUERY
MEMORY = tsp.MEMORY
Q = "relay gateway port shards cache queue depth deploy freeze"

# marker -> (session id, user side, assistant side). Sessions absent from SESSIONS are gone.
TURNS = {
    # 1: a one-shot run asking a real-looking question: dropped by structure, not text.
    "ONESHOTPROBE": ("probe", f"{Q} answer? ONESHOTPROBE", f"{Q} ONESHOTPROBE answer"),
    # 2: a delegated subagent session whose text looks like a scripted check: kept.
    "CHILDOK": ("child", f"Reply with exactly: CHILDOK {Q}", "CHILDOK"),
    # 3: a person's turn whose session is gone (it even says "exactly"): kept.
    "GONEHUMAN": ("gone-human", f"which {Q} did we pick exactly? GONEHUMAN",
                  f"{Q} uses port 7781 GONEHUMAN"),
    # 4: a scripted check whose session is gone: dropped by its own text.
    "GONEPROBE": ("gone-probe", f"Reply with exactly: GONEPROBE {Q} answer", "GONEPROBE"),
    # 5: host text the scaffolding filter does not know (capture_reader.EXTRA_HOST_SHAPES).
    "HOSTLOOP": ("gone-host", f"You've reached the maximum number of tool-calling iterations allowed. {Q} HOSTLOOP",
                 f"{Q} HOSTLOOP summary"),
    # 6: a probe typed in a person's interactive session that still exists: kept.
    "CLIPROBE": ("human", f"Reply with exactly: CLIPROBE {Q}", "CLIPROBE"),
}
# 7: a manual fact shaped like a scripted check: never judged.
FACT = ("FACTOK", f"User: Reply with exactly: FACTOK {Q}\nAssistant: FACTOK")
SESSIONS = (
    ("probe", "oneshot", None),
    ("child", "subagent", "root"),
    ("human", "cli", None),
)
DROPPED = {"ONESHOTPROBE", "GONEPROBE", "HOSTLOOP"}
KEPT = {"CHILDOK", "GONEHUMAN", "CLIPROBE", "FACTOK"}


def _state_db(path: Path) -> Path:
    tsp._state_db(path)  # sessions 'root' and 'tip' (the caller) with their rows
    conn = sqlite3.connect(path)
    for sid, source, parent in SESSIONS:
        conn.execute("INSERT INTO sessions (id, source, started_at, parent_session_id) VALUES (?, ?, 1.7e9, ?)",
                     (sid, source, parent))
        conn.execute("INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, 'user', 'x', 1.7e9)",
                     (sid,))
    conn.commit()
    conn.close()
    return path


def _seed(store):
    emb = tsp.TokenEmbeddings()
    tsp._seed_memory(store)
    for marker, (sid, user, assistant) in TURNS.items():
        text = f"User: {user}\nAssistant: {assistant}"
        store[MEMORY][marker] = {"id": marker, "vector": emb.embed_document(text), "payload": {
            "text": text, "profile_id": "default", "source_type": "conversation", "chunk_type": "turn",
            "memory_kind": "conversation_turn", "source": "hermes_conversation_turn", "session_id": sid,
            "importance": 5, "created_at": "2099-01-01T00:00:00+00:00"}}
    marker, text = FACT
    store[MEMORY][marker] = {"id": marker, "vector": emb.embed_document(text), "payload": {
        "text": text, "profile_id": "default", "source_type": "manual", "chunk_type": "fact",
        "memory_kind": "manual_fact", "importance": 5, "created_at": "2099-01-01T00:00:00+00:00"}}


def _provider(monkeypatch, tmp_path, *, state_db=True, **overrides):
    tmp_path.mkdir(parents=True, exist_ok=True)
    db = tmp_path / "state.db"
    _state_db(db)
    if not state_db:
        overrides["session_recall_state_db"] = str(tmp_path / "missing" / "state.db")
    store, log = {}, []
    settings = {"auto_recall_top_k": 10, "display_tokens": 4000, **overrides}
    provider = tsp._provider(monkeypatch, tmp_path, store, log, **settings)
    _seed(store)
    log.clear()
    return provider, store, log


def _plain_legacy(provider):
    chunks = (provider._memory_recall() or provider._retriever).search(
        QUERY, top_k=int(provider._config["auto_recall_top_k"]), update_access=False)
    return format_for_prompt(chunks, int(provider._config["display_tokens"]))


def _listing(path: Path):
    return sorted(p.name for p in path.iterdir())


def test_off_by_default_and_then_the_legacy_block_is_unchanged(monkeypatch, tmp_path):
    provider, _, log = _provider(monkeypatch, tmp_path)
    try:
        assert provider._config["auto_recall_exclude_machine"] is False
        out = provider.prefetch(QUERY, session_id="tip")
        assert out == _plain_legacy(provider)
        for marker in DROPPED | KEPT:
            assert marker in out, marker  # precondition: every point ranks into the block
        assert tsp._mutations(log) == []
    finally:
        provider.shutdown()


def test_on_drops_probe_turns_and_keeps_delegated_gone_and_interactive_ones(monkeypatch, tmp_path):
    provider, store, log = _provider(monkeypatch, tmp_path, auto_recall_exclude_machine=True)
    before = _listing(tmp_path)
    try:
        out = provider.prefetch(QUERY, session_id="tip")
        assert out.startswith("# Relevant Long-Term Memory")
        assert {m for m in DROPPED | KEPT if m in out} == KEPT
        assert "relay-gateway.md" in out  # the ordinary memory is still recalled
        assert tsp._mutations(log) == []
        assert _listing(tmp_path) == before  # state.db read without sidecars or other files
    finally:
        provider.shutdown()


def test_a_dropped_turn_frees_its_slot_for_the_next_memory(monkeypatch, tmp_path):
    top_k = 4
    off, _, _ = _provider(monkeypatch, tmp_path / "off", auto_recall_top_k=top_k)
    on, _, _ = _provider(monkeypatch, tmp_path / "on", auto_recall_top_k=top_k, auto_recall_exclude_machine=True)
    try:
        shown_off = off.prefetch(QUERY, session_id="tip")
        shown_on = on.prefetch(QUERY, session_id="tip")
        assert any(m in shown_off for m in DROPPED)  # precondition: a dropped turn ranks in the top k
        assert len(re.findall(r"(?m)^\d+\. \[", shown_off)) == top_k
        assert len(re.findall(r"(?m)^\d+\. \[", shown_on)) == top_k
        assert not any(m in shown_on for m in DROPPED)
    finally:
        off.shutdown()
        on.shutdown()


def test_without_a_readable_state_db_only_the_text_decides(monkeypatch, tmp_path):
    provider, _, log = _provider(monkeypatch, tmp_path, state_db=False, auto_recall_exclude_machine=True)
    try:
        out = provider.prefetch(QUERY, session_id="tip")
        # No session row anywhere: a one-shot run's ordinary-looking turn is kept, a scripted
        # check (also the subagent's and the cli probe) and host text are dropped by their text.
        assert {m for m in DROPPED | KEPT if m in out} == {"ONESHOTPROBE", "GONEHUMAN", "FACTOK"}
        assert tsp._mutations(log) == []
    finally:
        provider.shutdown()


def test_hybrid_mode_dense_lane_drops_the_same_turns(monkeypatch, tmp_path):
    for exclude in (False, True):
        provider, _, log = _provider(monkeypatch, tmp_path / str(exclude), auto_recall_mode="hybrid",
                                     auto_recall_exclude_machine=exclude)
        try:
            out = provider.prefetch(QUERY, session_id="tip")
            assert out.startswith("# Relevant Long-Term Memory (Hybrid)")
            expected = KEPT if exclude else KEPT | DROPPED
            assert {m for m in DROPPED | KEPT if m in out} == expected, exclude
            assert tsp._mutations(log) == []
        finally:
            provider.shutdown()
