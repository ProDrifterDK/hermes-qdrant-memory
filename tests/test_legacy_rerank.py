"""Opt-in rerank of the legacy automatic-recall block (``auto_recall_rerank``).

Failure modes, written before the code:

1. With the option at its default (off), the legacy block or the reranker traffic changes:
   the reranker is asked from the legacy path, or the block differs from the plain search.
2. With the option on, the block is not in rerank-score order, admits a chunk below
   ``auto_recall_min_rerank_score``, or leaves a bare header when nothing clears it. The
   relative final-score floor still cuts a reranked chunk the reranker found relevant.
   ``off`` fails to parse (dropping the whole block in the fail-open prefetch), or 0 is
   read as off although rerank scores can be negative.
3. The reranker is down, answers an error or an invalid body, or is slower than the
   timeout: the block is not the mode-off block byte for byte, or the turn waits longer
   than the timeout.
4. The widened pool changes the dense request (a larger Qdrant limit shifts the min-max
   normalization), so even the fallback block differs from mode off.
5. The pool lets a quarantined point or a machine turn (``auto_recall_exclude_machine``)
   reach the reranker or the block.
6. The session lane repeats a message the reranked legacy block already shows; or the lane
   changes although the legacy rerank did not apply (fail-open must be today's output).
7. ``queue_prefetch`` asks the shared reranker for a block that only an identical next
   message could use.
8. One stored chunk longer than the reranker's context (conversation turns reach 12,000
   characters) makes the whole call fail, so every message whose pool holds one silently
   keeps the dense order.

Only the network boundaries are fake: the Qdrant transport, the embedder, and a local
llama.cpp-style ``/v1/rerank`` server that scores documents by marker words.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from qdrant_memory.retriever import above_auto_recall_floor, format_for_prompt
from test_session_prefetch import MEMORY, TokenEmbeddings, _index_windows, _provider

QUERY = "relay gateway port listen address"
# Marker -> score. Dense order (bag-of-tokens cosine) puts DENSETOP and the fillers ahead of
# the answer; the reranker reverses that. Leaked points score highest so a leak shows.
SCORES = {"LONGANSWER": 7.0, "ANSWERFACT": 6.0, "DENSETOP": -4.0, "FILLER": -6.0, "QUARANTINED": 9.0, "Reply with exactly": 9.0}


class ScriptedReranker:
    """``POST /v1/rerank``: scores by marker words; ``mode`` = ok | error | invalid | slow."""

    def __init__(self, mode: str = "ok", slow_s: float = 3.0, max_doc_chars: int = 0):
        self.mode, self.slow_s = mode, slow_s
        # Like llama.cpp: a document longer than the context fails the whole request (0 = no limit).
        self.max_doc_chars = max_doc_chars
        self.requests: list[list[str]] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                docs = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["documents"]
                stub.requests.append(docs)
                if stub.mode == "slow":
                    time.sleep(stub.slow_s)
                too_long = stub.max_doc_chars and any(len(d) > stub.max_doc_chars for d in docs)
                if stub.mode == "error" or too_long:
                    self.send_response(500)
                    self.end_headers()
                    return
                results = [{"index": i, "relevance_score": stub.score(d)} for i, d in enumerate(docs)]
                body = json.dumps({"results": results[:-1] if stub.mode == "invalid" else results})
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body.encode())
                except OSError:
                    pass

            def log_message(self, format, *args):  # noqa: A002 - the base class's name
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @staticmethod
    def score(doc: str) -> float:
        return next((value for marker, value in SCORES.items() if marker in doc), 0.0)

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _seed(store, *, quarantined=True):
    emb = TokenEmbeddings()
    points = {
        "dense-top": ("manual", "relay gateway port listen address relay gateway port listen DENSETOP notes"),
        "answer": ("manual", "ANSWERFACT the relay binds 7781 after the load test"),
        "quarantined": ("manual", "relay gateway port listen address QUARANTINED copy"),
        "probe": ("turn", "User: Reply with exactly: OK relay gateway port listen address\nAssistant: OK"),
    }
    for n in range(6):
        points[f"filler-{n}"] = ("manual", f"relay gateway port FILLER{n} note")
    for pid, (kind, text) in points.items():
        payload = {"text": text, "profile_id": "default", "importance": 7, "created_at": "2099-01-01T00:00:00+00:00"}
        if kind == "turn":
            payload.update(source_type="conversation", memory_kind="conversation_turn", session_id="gone-session")
        else:
            payload["source_type"] = "manual"
        if pid == "quarantined" and quarantined:
            payload["consolidation_quarantined"] = True
        store[MEMORY][pid] = {"id": pid, "vector": emb.embed_document(text), "payload": payload}


def _plain_legacy(provider, query=QUERY):
    """The mode-off legacy block, built by hand from the search and the floors."""
    recall = provider._noise_filtered(provider._memory_recall() or provider._retriever)
    chunks = recall.search(query, top_k=int(provider._config["auto_recall_top_k"]), update_access=False)
    chunks = above_auto_recall_floor(chunks, provider._config.get("auto_recall_min_final_score"),
                                     provider._config.get("auto_recall_min_raw_score"))
    return format_for_prompt(chunks, int(provider._config["display_tokens"]))


@pytest.fixture
def reranker():
    stub = ScriptedReranker()
    yield stub
    stub.close()


def _setup(monkeypatch, tmp_path, url, **overrides):
    store, log = {}, []
    overrides.setdefault("auto_recall_exclude_machine", True)  # the probe turn stays out unless a test lets it in
    provider = _provider(monkeypatch, tmp_path, store, log, session_recall_rerank_url=url, **overrides)
    _seed(store)
    return provider, store


def test_default_off_never_asks_the_reranker_and_keeps_the_block(monkeypatch, tmp_path, reranker):
    provider, _ = _setup(monkeypatch, tmp_path, reranker.url)
    try:
        assert provider._config["auto_recall_rerank"] is False
        out = provider.prefetch(QUERY, session_id="tip")
        assert out == _plain_legacy(provider) and out.startswith("# Relevant Long-Term Memory")
        assert "ANSWERFACT" not in out  # precondition: the answer is outside the dense top 5
        assert reranker.requests == []
    finally:
        provider.shutdown()


@pytest.mark.parametrize("floor", [0, "0", 0.0])
def test_rerank_orders_by_score_and_admits_by_the_absolute_floor(monkeypatch, tmp_path, reranker, floor):
    provider, _ = _setup(monkeypatch, tmp_path, reranker.url, auto_recall_rerank=True,
                         auto_recall_min_rerank_score=floor)
    try:
        out = provider.prefetch(QUERY, session_id="tip")
        assert len(reranker.requests) == 1 and len(reranker.requests[0]) > 5  # the widened pool
        # Only the answer scores >= 0: DENSETOP (-4) and the fillers (-6) are left out.
        assert out.startswith("# Relevant Long-Term Memory") and "ANSWERFACT" in out
        assert "DENSETOP" not in out and "FILLER" not in out
        # A relative final-score floor does not cut a reranked chunk.
        provider._config["auto_recall_min_final_score"] = 0.99
        assert provider.prefetch(QUERY, session_id="tip") == out
        # Nothing clears the floor: no block at all, not a bare header.
        provider._config["auto_recall_min_rerank_score"] = 1000.0
        assert provider.prefetch(QUERY, session_id="tip") == ""
    finally:
        provider.shutdown()


@pytest.mark.parametrize("floor", ["off", "", None])
def test_floor_off_admits_the_pool_in_rerank_order(monkeypatch, tmp_path, reranker, floor):
    provider, _ = _setup(monkeypatch, tmp_path, reranker.url, auto_recall_rerank=True,
                         auto_recall_min_rerank_score=floor)
    try:
        out = provider.prefetch(QUERY, session_id="tip")
        entries = out.split("\n\n")[2:]
        assert len(entries) == 5
        assert "ANSWERFACT" in entries[0] and "DENSETOP" in entries[1]
        assert all("FILLER" in e for e in entries[2:])
    finally:
        provider.shutdown()


@pytest.mark.parametrize("mode", ["down", "error", "invalid", "slow"])
def test_reranker_failure_gives_the_mode_off_block_byte_for_byte(monkeypatch, tmp_path, mode):
    stub = ScriptedReranker(mode=mode if mode != "down" else "ok", slow_s=3.0)
    url = stub.url
    if mode == "down":
        stub.close()  # nothing listens on the port any more
    timeout = 0.5
    # search_candidates below the widened pool: a pool that enlarged the Qdrant request would
    # rescale the min-max scores, and the printed scores of the fallback block would differ.
    provider, _ = _setup(monkeypatch, tmp_path, url, search_candidates=6, auto_recall_rerank_candidates=20,
                         auto_recall_rerank_timeout_s=timeout, auto_recall_min_rerank_score=0.0)
    try:
        expected = provider.prefetch(QUERY, session_id="tip")
        assert expected == _plain_legacy(provider) and expected
        provider._config["auto_recall_rerank"] = True
        started = time.monotonic()
        out = provider.prefetch(QUERY, session_id="tip")
        elapsed = time.monotonic() - started
        assert out == expected
        assert elapsed < timeout + 0.4, elapsed
        if mode != "down":
            assert len(stub.requests) == 1  # it was asked
    finally:
        provider.shutdown()
        if mode != "down":
            stub.close()


def test_quarantined_points_and_machine_turns_never_reach_the_reranker(monkeypatch, tmp_path, reranker):
    provider, store = _setup(monkeypatch, tmp_path, reranker.url, auto_recall_rerank=True,
                             auto_recall_min_rerank_score="off", auto_recall_exclude_machine=False)
    try:
        # Preconditions: without the filters both would be in the pool (and win the rerank).
        store[MEMORY]["quarantined"]["payload"]["consolidation_quarantined"] = False
        provider.prefetch(QUERY, session_id="tip")
        unfiltered = " ".join(reranker.requests[-1])
        assert "QUARANTINED" in unfiltered and "Reply with exactly" in unfiltered

        store[MEMORY]["quarantined"]["payload"]["consolidation_quarantined"] = True
        provider._config["auto_recall_exclude_machine"] = True
        out = provider.prefetch(QUERY, session_id="tip")
        sent = " ".join(reranker.requests[-1])
        for leaked in ("QUARANTINED", "Reply with exactly"):
            assert leaked not in sent and leaked not in out, leaked
        assert "ANSWERFACT" in out.split("\n\n")[2]
    finally:
        provider.shutdown()


def test_queue_prefetch_does_not_ask_the_reranker(monkeypatch, tmp_path, reranker):
    provider, _ = _setup(monkeypatch, tmp_path, reranker.url, auto_recall_rerank=True,
                         auto_recall_min_rerank_score=0.0)
    try:
        provider.queue_prefetch(QUERY, session_id="tip")
        provider._executor.shutdown(wait=True)
        provider._executor = None
        assert reranker.requests == []
        out = provider.prefetch(QUERY, session_id="tip")  # the same message: computed now, reranked
        assert "ANSWERFACT" in out and len(reranker.requests) == 1
    finally:
        provider.shutdown()


def test_a_chunk_longer_than_the_reranker_context_does_not_fail_the_call(monkeypatch, tmp_path):
    stub = ScriptedReranker(max_doc_chars=1300)
    # display_tokens 300: the block shows at most 1,200 characters, so the reranker is sent at most that.
    provider, store = _setup(monkeypatch, tmp_path, stub.url, auto_recall_rerank=True, display_tokens=300,
                             auto_recall_min_rerank_score=0.0)
    try:
        emb = TokenEmbeddings()
        text = "User: relay gateway port listen address?\nAssistant: LONGANSWER " + "long   reply " * 400
        store[MEMORY]["long-turn"] = {"id": "long-turn", "vector": emb.embed_document(text), "payload": {
            "text": text, "profile_id": "default", "source_type": "conversation", "memory_kind": "conversation_turn",
            "session_id": "root", "importance": 5, "created_at": "2099-01-01T00:00:00+00:00"}}
        out = provider.prefetch(QUERY, session_id="tip")
        sent = stub.requests[-1]
        assert any(d.startswith("User: relay gateway port listen address? Assistant: LONGANSWER long reply") for d in sent)
        assert max(len(d) for d in sent) <= 1200
        assert "LONGANSWER" in out.split("\n\n")[2]  # the long turn is reranked to the top of the block
    finally:
        provider.shutdown()
        stub.close()


SHARED = "relay gateway listens on port 7781 after the load test"  # a root-session row of the lane fixture


def test_lane_does_not_repeat_a_message_the_reranked_block_shows(monkeypatch, tmp_path, reranker):
    provider, store = _setup(
        monkeypatch, tmp_path, reranker.url, session_prefetch_enabled=True, session_prefetch_scope="all",
        session_prefetch_max_chars=6000, session_prefetch_total_chars=12000, session_prefetch_top_k=10,
        auto_recall_top_k=10, display_tokens=1000,
    )
    emb = TokenEmbeddings()
    # Query words in the user side put the turn in the dense top 5 too (see the fail-open check).
    turn = f"User: relay gateway port listen address: which one did we pick ANSWERFACT\nAssistant: {SHARED}"
    store[MEMORY]["turn-shared"] = {"id": "turn-shared", "vector": emb.embed_document(turn), "payload": {
        "text": turn, "profile_id": "default", "source_type": "conversation", "memory_kind": "conversation_turn",
        "session_id": "root", "importance": 5, "created_at": "2099-01-01T00:00:00+00:00"}}
    _index_windows(provider)
    try:
        def lane_part(out):
            head = "# Recalled Earlier Conversation"
            assert head in out
            return out[out.index(head):]

        # Precondition (mode off): the lane repeats the message the legacy block may show.
        assert SHARED in lane_part(provider.prefetch(QUERY, session_id="tip"))

        provider._config.update(auto_recall_rerank=True, auto_recall_min_rerank_score=0.0)
        out = provider.prefetch(QUERY, session_id="tip")
        legacy = out[: out.index("# Recalled Earlier Conversation")]
        assert SHARED in legacy  # the reranked legacy block shows it
        assert SHARED not in lane_part(out)
        assert "shard count 37" in lane_part(out)  # the lane still brings what legacy lacks

        # The reranker fails: legacy and lane are what the mode-off plugin gives with the same
        # failing reranker (the lane then keeps its dense order), byte for byte.
        reranker.mode = "error"
        provider._config.update(auto_recall_rerank=False)
        today = provider.prefetch(QUERY, session_id="tip")
        provider._config.update(auto_recall_rerank=True)
        assert provider.prefetch(QUERY, session_id="tip") == today
        # Both blocks show the message: the lane's dedup is off again, although it could apply.
        assert SHARED in lane_part(today) and SHARED in today[: today.index("# Recalled Earlier Conversation")]
    finally:
        provider.shutdown()
