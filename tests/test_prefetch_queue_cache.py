"""The block ``queue_prefetch`` computes is injected only into the message it was computed for.

Hermes calls ``queue_prefetch(text)`` after a completed turn with that turn's message, and
``prefetch(next_text)`` at the start of the next turn. Failure modes, written before the code:

1. ``prefetch`` injects the block queued for the previous message into a different message
   (legacy and hybrid mode), instead of searching with its own words.
2. A turn that recalls nothing (a short confirmation, host text) leaves the queued block in
   place, and a later turn injects it: after its memory was quarantined, or into a message
   it was never computed for.
3. The queued search finishes after the next turn already started, and the block it stores
   then is injected into a later turn.
4. The fix throws away the queued block also when the next message is the same one, so the
   cache never saves the search it exists for.

Only the network boundaries are fake (the Qdrant transport and the embedder); the provider
is wired through its real ``initialize`` (see ``test_session_prefetch``).
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from test_session_prefetch import MEMORY, InlineExecutor as _Inline, TokenEmbeddings, _provider


class InlineExecutor(_Inline):
    """Runs ``queue_prefetch``'s background job at once; ``provider.shutdown`` may close it."""

    def shutdown(self, *args, **kwargs):
        pass


HARBOR_Q = "harbor crane lifting schedule tide window"
ORCHARD_Q = "orchard irrigation valve pressure setting"
HARBOR = "HARBORFACT"
ORCHARD = "ORCHARDFACT"


def _seed(store):
    emb = TokenEmbeddings()
    for pid, text in (
        ("fact-harbor", f"Harbor crane lifting schedule follows the tide window {HARBOR}"),
        ("fact-orchard", f"Orchard irrigation valve pressure setting is three bar {ORCHARD}"),
    ):
        store[MEMORY][pid] = {
            "id": pid,
            "vector": emb.embed_document(text),
            "payload": {"text": text, "profile_id": "default", "source_type": "manual", "importance": 7,
                        "created_at": "2099-01-01T00:00:00+00:00"},
        }


def _setup(monkeypatch, tmp_path, **overrides):
    store, log = {}, []
    # The floor keeps the zero-similarity memory out of each fresh block, so the two
    # messages' blocks are disjoint.
    provider = _provider(monkeypatch, tmp_path, store, log, auto_recall_min_final_score=0.05, **overrides)
    _seed(store)
    return provider, store, log


def _searches(log):
    return sum(1 for method, path in log if method == "POST" and path.rstrip("/").endswith("points/search"))


def _quarantine(store, pid):
    store[MEMORY][pid]["payload"]["consolidation_quarantined"] = True


@pytest.mark.parametrize("mode", ["legacy", "hybrid"])
def test_block_queued_for_another_message_is_never_injected(monkeypatch, tmp_path, mode):
    provider, store, log = _setup(monkeypatch, tmp_path, auto_recall_mode=mode)
    provider._executor = InlineExecutor()
    try:
        fresh = provider.prefetch(ORCHARD_Q, session_id="tip")
        # Precondition: the two messages get different blocks (the hybrid formatter lists both
        # memories, in another order; the legacy floor keeps only the matching one).
        assert ORCHARD in fresh and fresh != provider.prefetch(HARBOR_Q, session_id="tip")
        before = _searches(log)
        provider.queue_prefetch(HARBOR_Q, session_id="tip")
        assert _searches(log) > before  # precondition: the queued search ran

        assert provider.prefetch(ORCHARD_Q, session_id="tip") == fresh
    finally:
        provider.shutdown()


def test_same_message_reuses_the_queued_block(monkeypatch, tmp_path):
    provider, store, log = _setup(monkeypatch, tmp_path)
    provider._executor = InlineExecutor()
    try:
        fresh = provider.prefetch(HARBOR_Q, session_id="tip")
        assert HARBOR in fresh
        provider.queue_prefetch(HARBOR_Q, session_id="tip")
        before = _searches(log)
        # Same words, other spacing: the host's text and the turn's text differ only in whitespace.
        assert provider.prefetch(f"  {HARBOR_Q.replace(' ', '  ')} ", session_id="tip") == fresh
        assert _searches(log) == before  # served from the queued block, no second search
        # The queued block is used once.
        provider.prefetch(HARBOR_Q, session_id="tip")
        assert _searches(log) > before
    finally:
        provider.shutdown()


@pytest.mark.parametrize("skipped", ["ok, dale", "[IMPORTANT: Background process proc_1a2b3c4d5e6f completed normally]"])
def test_block_left_by_a_turn_that_recalls_nothing_is_not_injected_later(monkeypatch, tmp_path, skipped):
    provider, store, log = _setup(monkeypatch, tmp_path, auto_recall_min_word_chars=12)
    provider._executor = InlineExecutor()
    try:
        provider.queue_prefetch(HARBOR_Q, session_id="tip")
        assert provider.prefetch(skipped, session_id="tip") == ""  # precondition: this turn recalls nothing
        # Another message two turns later gets its own block.
        assert HARBOR not in provider.prefetch(ORCHARD_Q, session_id="tip")

        provider.queue_prefetch(HARBOR_Q, session_id="tip")
        assert provider.prefetch(skipped, session_id="tip") == ""
        _quarantine(store, "fact-harbor")
        # The same message again, after its memory was quarantined: a fresh search.
        assert HARBOR not in provider.prefetch(HARBOR_Q, session_id="tip")
    finally:
        provider.shutdown()


def test_queued_block_that_lands_after_the_next_turn_started_is_discarded(monkeypatch, tmp_path):
    provider, store, log = _setup(monkeypatch, tmp_path)
    release, entered = threading.Event(), threading.Event()
    original = TokenEmbeddings._embed
    main = threading.main_thread()

    def gated(self, text):
        if threading.current_thread() is not main:  # only the queued background search waits
            entered.set()
            release.wait(10.0)
        return original(self, text)

    for name in ("embed", "embed_query", "embed_document"):
        monkeypatch.setattr(TokenEmbeddings, name, gated)
    provider._executor.shutdown(wait=True)
    provider._executor = ThreadPoolExecutor(max_workers=1)
    try:
        provider.queue_prefetch(HARBOR_Q, session_id="tip")
        assert entered.wait(5.0)  # precondition: the queued search is running
        assert ORCHARD in provider.prefetch(ORCHARD_Q, session_id="tip")
        release.set()
        provider._executor.shutdown(wait=True)  # the late block has been stored by now
        provider._executor = InlineExecutor()
        _quarantine(store, "fact-harbor")
        assert HARBOR not in provider.prefetch(HARBOR_Q, session_id="tip")
    finally:
        release.set()
        provider.shutdown()
