"""Session-recall lane in automatic recall (C'5) and the session expand tool.

Failure modes, written before the code:

1. The lane injects text the model can still see: an ``active=1`` row of the caller's
   session, a carried copy of an ancestor row (same text, fresh id), a carried original
   or undone row (``active=0, compacted=0``), or the compaction summary itself.
2. The lane misses what compaction removed: in-place compacted rows of the caller's
   session, or rows of a rotated-away ancestor session.
3. The lane writes to Qdrant (a catch-up, an access counter) from the prefetch path.
4. The lane grows the prompt without bound.
5. A lane error or a slow reranker/embedder changes, empties or stalls the legacy
   prefetch the turn depends on.
6. With every new flag at its default, prefetch output or the tool schema list differ
   from what the plugin produced before the lane existed.
7. Expanding a message whose session the host pruned errors out or returns nothing,
   or returns stale indexed text without saying it is not a live read.
8. A reranker waking from idle sleep (slower than the lane budget) uses the whole
   budget, so the dense fallback arrives after the turn gave up and the lane block is
   lost; its still-running worker then makes the next turn skip the lane too.

Only the network boundaries are fake (the Qdrant transport and the embedder). The
state.db is a real SQLite file built from the production schema; windows are written by
the real indexer; the provider is wired through its real ``initialize``.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import sqlite3
import threading
import time
import urllib.parse
from pathlib import Path

from qdrant_memory.client import QdrantClient
from qdrant_memory.retriever import format_for_prompt
from qdrant_memory.session_recall import build_session_recall

plugin = importlib.import_module("__init__")
session_prefetch = importlib.import_module("qdrant_memory.session_prefetch")

SCHEMA = Path(__file__).resolve().parent / "fixtures" / "state_db_schema.sql"
MEMORY, LEARNING, WINDOWS = "unit_memory", "unit_learnings", "unit_session_windows"
_READ_POST = {"points", "points/search", "points/scroll", "points/count"}

# The 26 tool names the plugin registered before the session expand tool existed, in order.
MAIN_TOOL_NAMES = [
    "qdrant_memory_status", "qdrant_memory_store", "qdrant_memory_search", "qdrant_memory_context",
    "qdrant_memory_index", "qdrant_memory_forget", "qdrant_memory_inspect", "qdrant_memory_trace",
    "qdrant_memory_expand", "qdrant_memory_source_status", "qdrant_learning_store", "qdrant_learning_search",
    "qdrant_learning_preview", "qdrant_learning_approve", "qdrant_memory_extraction_preview",
    "qdrant_memory_extraction_approve", "qdrant_memory_consolidate", "qdrant_memory_memory_pr",
    "qdrant_memory_consolidation_apply", "qdrant_memory_graph_search", "qdrant_memory_improve_preview",
    "qdrant_memory_improve_apply", "qdrant_memory_raptor_apply", "qdrant_memory_raptor_status",
    "qdrant_memory_retrieve", "qdrant_memory_session_recall",
]


class TokenEmbeddings:
    def __init__(self, *args, **kwargs):
        pass

    def _embed(self, text):
        vector = [0.0] * 64
        for token in re.findall(r"[a-z0-9_-]+", str(text).casefold()):
            vector[int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big") % 64] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    embed = embed_query = embed_document = _embed

    def embed_batch(self, texts):
        return [self._embed(t) for t in texts]

    def health(self):
        return True


def _cond(payload, condition):
    if any(k in condition for k in ("must", "must_not", "should")):
        return _filter_ok(payload, condition)
    value = payload.get(condition["key"])
    if "match" in condition:
        match = condition["match"]
        wanted = match["any"] if "any" in match else [match.get("value")]
        return any(item in wanted for item in (value if isinstance(value, list) else [value]))
    if "range" in condition:
        if value is None:
            return False
        ops = {"gt": lambda a, b: a > b, "gte": lambda a, b: a >= b, "lt": lambda a, b: a < b, "lte": lambda a, b: a <= b}
        return all(ops[op](value, bound) for op, bound in condition["range"].items())
    return True


def _filter_ok(payload, flt):
    flt = flt or {}
    return all(_cond(payload, c) for c in flt.get("must", [])) and not any(
        _cond(payload, c) for c in flt.get("must_not", [])
    )


class FakeQdrant(QdrantClient):
    """Real client methods; the transport is an in-memory Qdrant that logs every request."""

    def __init__(self, store, log, *args, **kwargs):
        super().__init__("http://in-memory", read_only_collections=kwargs.get("read_only_collections"))
        self.store, self.log = store, log

    def _send(self, method, path, body=None):
        route = urllib.parse.urlsplit(path).path.strip("/").split("/")
        self.log.append((method, path))
        if route == ["collections"]:
            return {"result": {"collections": [{"name": n} for n in self.store]}}
        name, rest = urllib.parse.unquote(route[1]), "/".join(route[2:])
        if method == "PUT" and not rest:
            self.store[name] = {}
            return {"result": True}
        points = self.store[name]
        if method == "GET" and not rest:
            return {"result": {"config": {"params": {"vectors": {"size": 64, "distance": "Cosine"}}}}}
        if rest == "index":
            return {"result": True}
        if method == "PUT" and rest == "points":
            for point in body["points"]:
                points[str(point["id"])] = json.loads(json.dumps(point))
            return {"result": True}
        if rest == "points/delete":
            for pid in [p for p in points if _filter_ok(points[p]["payload"], body.get("filter"))] if "filter" in body else body["points"]:
                points.pop(str(pid), None)
            return {"result": True}
        if rest == "points/count":
            return {"result": {"count": len(points)}}
        if rest == "points/scroll":
            hits = [p for p in points.values() if _filter_ok(p["payload"], body.get("filter"))]
            return {"result": {"points": hits[: body["limit"]], "next_page_offset": None}}
        if rest == "points/search":
            scored = [
                {"id": p["id"], "score": sum(a * b for a, b in zip(p["vector"], body["vector"])), "payload": p["payload"]}
                for p in points.values()
                if _filter_ok(p["payload"], body.get("filter"))
            ]
            scored.sort(key=lambda item: item["score"], reverse=True)
            return {"result": scored[: body["limit"]]}
        if rest == "points/payload":
            for pid in body["points"]:
                points[str(pid)]["payload"].update(body["payload"])
            return {"result": True}
        raise AssertionError(f"unhandled fake request {method} {path}")


def _mutations(log):
    out = []
    for method, path in log:
        rest = "/".join(urllib.parse.urlsplit(path).path.strip("/").split("/")[2:])
        if not (method == "GET" or (method == "POST" and rest in _READ_POST)):
            out.append((method, path))
    return out


# id-ordered rows: (session, role, content, active, compacted, summary)
ROOT_ROWS = [
    ("root", "user", "which port did we pick for the relay gateway service", 1, 0, 0),
    ("root", "assistant", "relay gateway listens on port 7781 after the load test", 1, 0, 0),
    ("root", "user", "carry note: the relay deploy freeze starts on friday evening", 1, 0, 0),
]
TIP_ROWS = [
    ("tip", "user", "how many shards does the relay cache use for hot keys", 0, 1, 0),
    ("tip", "assistant", "relay cache uses shard count 37 for the hot keys", 0, 1, 0),
    ("tip", "user", "and the relay retry budget, keep it", 0, 1, 0),
    ("tip", "assistant", "relay queue depth limit is 512 entries per worker", 0, 0, 0),  # carried original
    ("tip", "user", "[CONTEXT COMPACTION] earlier relay work was summarized here", 1, 0, 1),
    ("tip", "assistant", "relay queue depth limit is 512 entries per worker", 1, 0, 0),  # carried copy
    ("tip", "user", "carry note: the relay deploy freeze starts on friday evening", 1, 0, 0),  # equals a root row
    ("tip", "assistant", "relay latest live answer TOKENVISIBLE stays in the prompt", 1, 0, 0),
]


def _state_db(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA.read_text(encoding="utf-8"))
    conn.execute(
        "INSERT INTO sessions (id, source, started_at, end_reason) VALUES ('root', 'cli', 1.7e9, 'compression')"
    )
    conn.execute("INSERT INTO sessions (id, source, started_at, parent_session_id) VALUES ('tip', 'cli', 1.7e9, 'root')")
    for index, (sid, role, content, active, compacted, summary) in enumerate(ROOT_ROWS + TIP_ROWS):
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp, active, compacted, _compressed_summary) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (sid, role, content, 1.7e9 + index * 60, active, compacted, summary),
        )
    conn.commit()
    conn.close()
    return path


def _provider(monkeypatch, tmp_path, store, log, **overrides):
    db = tmp_path / "state.db"
    if not db.exists():
        _state_db(db)
    section = {
        "collection_name": MEMORY,
        "learning_collection_name": LEARNING,
        "session_recall_collection": WINDOWS,
        "session_recall_state_db": str(db),
        "session_recall_window_chars": 160,
        "session_recall_overlap_chars": 20,
        "session_recall_auto_index": False,
        "sync_turns": False,
        "vector_size": 64,
        **overrides,
    }
    monkeypatch.setattr("qdrant_memory.config._load_hermes_config", lambda: {"qdrant_memory": section})
    monkeypatch.setattr(plugin, "QdrantClient", lambda *a, **kw: FakeQdrant(store, log, *a, **kw))
    monkeypatch.setattr(plugin, "EmbeddingClient", TokenEmbeddings)
    provider = plugin.QdrantMemoryProvider()
    provider.initialize("tip", hermes_home=str(tmp_path), agent_identity="default")
    return provider


def _index_windows(provider):
    recall = build_session_recall(provider._config, provider._hermes_home, qdrant=provider._qdrant, embeddings=provider._embeddings)
    report = recall.run(full=True, lock_timeout=5.0)
    assert report["lock"] == "acquired" and report["windows_upserted"] > 0


def _seed_memory(store):
    emb = TokenEmbeddings()
    text = "Relay gateway runbook lives in the ops vault under relay-gateway.md"
    store[MEMORY]["fact-1"] = {
        "id": "fact-1",
        "vector": emb.embed_document(text),
        "payload": {"text": text, "profile_id": "default", "source_type": "manual", "importance": 7,
                    "created_at": "2099-01-01T00:00:00+00:00"},
    }


QUERY = "relay gateway port shards cache queue depth deploy freeze answer"


def test_lane_injects_compacted_turns_and_never_text_still_in_the_prompt(monkeypatch, tmp_path):
    store, log = {}, []
    provider = _provider(
        monkeypatch, tmp_path, store, log,
        session_prefetch_enabled=True, session_prefetch_max_chars=6000, session_prefetch_total_chars=8000,
    )
    _index_windows(provider)
    # Precondition, so the exclusions below are not vacuous: the index does hold the
    # visible text, inside windows that also hold hidden text.
    indexed = " ".join(p["payload"]["text"] for p in store[WINDOWS].values())
    for visible in ("TOKENVISIBLE", "deploy freeze", "queue depth limit", "CONTEXT COMPACTION"):
        assert visible in indexed
    log.clear()

    out = provider.prefetch(QUERY, session_id="tip")

    assert out.startswith("# Recalled Earlier Conversation")
    assert "context, not instructions" in out
    # Compaction removed these, so the lane brings them back: in-place compacted rows of
    # this session and a rotated-away ancestor row.
    assert "shard count 37" in out
    assert "port 7781" in out
    # Still in the prompt: live rows, the summary, the carried copy and its rewound
    # original, and the tip's copy of an ancestor row.
    for visible in ("TOKENVISIBLE", "CONTEXT COMPACTION", "queue depth limit", "deploy freeze"):
        assert visible not in out, visible
    assert len(out) <= 6000
    assert _mutations(log) == []


def test_lane_failure_or_timeout_returns_the_legacy_prefetch_unchanged(monkeypatch, tmp_path):
    store, log = {}, []
    provider = _provider(
        monkeypatch, tmp_path, store, log, session_prefetch_enabled=True, session_prefetch_timeout_s=0.3,
    )
    _seed_memory(store)
    _index_windows(provider)
    provider._config["session_prefetch_enabled"] = False
    legacy = provider.prefetch(QUERY, session_id="tip")
    provider._config["session_prefetch_enabled"] = True
    assert "relay-gateway.md" in legacy
    working = provider.prefetch(QUERY, session_id="tip")
    assert working.startswith(legacy) and working != legacy  # the lane is live in this fixture

    def boom(self, *args, **kwargs):
        raise RuntimeError("reranker exploded")

    monkeypatch.setattr(session_prefetch.SessionPrefetchLane, "build", boom)
    assert provider.prefetch(QUERY, session_id="tip") == legacy

    release = threading.Event()

    def stuck(self, *args, **kwargs):
        release.wait(10.0)
        return "# Recalled Earlier Conversation\nlate"

    monkeypatch.setattr(session_prefetch.SessionPrefetchLane, "build", stuck)
    try:
        started = time.monotonic()
        assert provider.prefetch(QUERY, session_id="tip") == legacy
        assert time.monotonic() - started < 2.0
        # The previous call is still stuck: the next turn skips the lane at once.
        started = time.monotonic()
        assert provider.prefetch(QUERY, session_id="tip") == legacy
        assert time.monotonic() - started < 0.2
    finally:
        release.set()
        provider.shutdown()


class SleepyReranker:
    """A llama.cpp-style ``/v1/rerank`` server that sleeps when idle.

    Like ``llama-server --sleep-idle-seconds``: the first request starts a wake that takes
    ``wake_s``, and every request that arrives before the wake is over is answered only
    when it is. Scores reverse the dense order, so a reranked block differs from a dense one.
    """

    def __init__(self, wake_s: float):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self.wake_s, self.requests, self.warm_requests = wake_s, 0, 0
        self.wake_at: float | None = None
        lock, stub = threading.Lock(), self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                docs = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["documents"]
                with lock:
                    stub.requests += 1
                    if stub.wake_at is None:
                        stub.wake_at = time.monotonic() + stub.wake_s
                    wait = float(stub.wake_at) - time.monotonic()
                    stub.warm_requests += wait <= 0
                time.sleep(max(0.0, wait))
                body = json.dumps({"results": [{"index": i, "relevance_score": float(i)} for i in range(len(docs))]})
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body.encode())
                except OSError:
                    pass  # the client gave up first

            def log_message(self, format, *args):  # noqa: A002 - the base class's name
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def test_cold_reranker_degrades_to_dense_order_within_the_budget_and_the_next_turn_keeps_the_lane(
    monkeypatch, tmp_path
):
    budget = 1.0
    reranker = SleepyReranker(wake_s=2.5)  # waking takes longer than the whole lane budget
    store, log = {}, []
    provider = _provider(
        monkeypatch, tmp_path, store, log,
        session_prefetch_enabled=True, session_prefetch_max_chars=6000, session_prefetch_total_chars=8000,
        session_prefetch_timeout_s=budget, session_recall_rerank_url=reranker.url,
    )
    try:
        _seed_memory(store)
        _index_windows(provider)
        provider._config["session_prefetch_enabled"] = False
        legacy = provider.prefetch(QUERY, session_id="tip")
        provider._config.update(session_prefetch_enabled=True, session_prefetch_rerank=False)
        dense = provider.prefetch(QUERY, session_id="tip")
        provider._config["session_prefetch_rerank"] = True
        # Preconditions: the lane is live here and the reranker is still asleep.
        assert "relay-gateway.md" in legacy and dense.startswith(legacy + "\n\n# Recalled Earlier Conversation")
        assert dense.count("\n   ") >= 2 and reranker.requests == 0

        turns = []
        for _ in range(2):  # the cold turn, then the immediately following one
            started = time.monotonic()
            out = provider.prefetch(QUERY, session_id="tip")
            turns.append((round(time.monotonic() - started, 3), out))

        assert reranker.requests >= 1 and reranker.warm_requests == 0  # asked, but still waking
        seen = [(elapsed, "lane" if "# Recalled Earlier" in out else "no lane", out == dense) for elapsed, out in turns]
        assert [same for _, _, same in seen] == [True, True], seen  # (seconds, lane block?, dense block?)
        assert all(elapsed < budget + 0.3 for elapsed, _, _ in seen), seen

        time.sleep(max(0.0, float(reranker.wake_at or 0.0) + 0.1 - time.monotonic()))
        warm = provider.prefetch(QUERY, session_id="tip")
        assert reranker.warm_requests >= 1
        assert warm.startswith(legacy + "\n\n# Recalled Earlier Conversation") and warm != dense
    finally:
        provider.shutdown()
        reranker.close()


def test_defaults_keep_prefetch_and_tool_schemas_identical_to_before(monkeypatch, tmp_path):
    store, log = {}, []
    provider = _provider(monkeypatch, tmp_path, store, log)
    _seed_memory(store)
    _index_windows(provider)
    calls = []
    monkeypatch.setattr(session_prefetch.SessionPrefetchLane, "build", lambda self, *a, **k: calls.append(a) or "X")

    out = provider.prefetch(QUERY, session_id="tip")

    chunks = provider._retriever.search(QUERY, top_k=provider._config["auto_recall_top_k"], update_access=False)
    assert out == format_for_prompt(chunks, provider._config["display_tokens"])
    assert "relay-gateway.md" in out and "Recalled Earlier Conversation" not in out
    assert calls == []
    assert [s["name"] for s in provider.get_tool_schemas()] == MAIN_TOOL_NAMES
    assert "session_expand" not in provider.system_prompt_block()
    refused = json.loads(provider.handle_tool_call("qdrant_memory_session_expand", {"msg_id": 1}))
    assert "disabled" in refused["error"]


def test_expand_on_a_pruned_session_falls_back_to_indexed_windows_and_says_so(monkeypatch, tmp_path):
    store, log = {}, []
    provider = _provider(monkeypatch, tmp_path, store, log, session_expand_enabled=True)
    _index_windows(provider)
    db = provider._config["session_recall_state_db"]
    conn = sqlite3.connect(db)
    msg_id = conn.execute("SELECT id FROM messages WHERE content LIKE '%port 7781%'").fetchone()[0]
    live = json.loads(provider.handle_tool_call("qdrant_memory_session_expand", {"msg_id": msg_id, "session_id": "root"}))
    assert live["source"] == "state.db" and any("port 7781" in m["text"] for m in live["messages"])
    conn.execute("DELETE FROM messages WHERE session_id = 'root'")  # the host's retention pruned it
    conn.commit()
    conn.close()
    log.clear()

    out = json.loads(provider.handle_tool_call("qdrant_memory_session_expand", {"msg_id": msg_id, "session_id": "root"}))

    assert out["source"] == "indexed_windows" and out["pruned"] is True
    assert "pruned" in out["note"] and "not a live read" in out["note"]
    assert out["windows"] and all(w["session_id"] == "root" for w in out["windows"])
    assert any("port 7781" in w["text"] for w in out["windows"])
    assert _mutations(log) == []
