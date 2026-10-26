"""Search-time dedup contracts for session recall, with fake services.

Only the two network boundaries are faked (Qdrant, reranker). The dedup, pool,
provenance and scope wiring are the real code under test. The fake Qdrant plays
Qdrant's role for the filter subset the lane uses (``session_id``/``session_source``
``match any``, ``start_ts``/``end_ts`` ranges), so scope semantics are exercised
end to end: if the lane ever stopped pushing the scope filter into the dense
search, the chain sessions planted here would flood the results and ``also_in``.

Contracts and the failure each one names:

(a) exact duplicates: identical windows in several sessions collapse before the
    reranker; the top-k holds distinct texts only and the representative carries
    ``duplicate_count`` and a correct, newest-first ``also_in``. Fails if grouping
    happens after rerank, uses the wrong key, or loses the member trail.
(a2) offset-shifted near-duplicates collapse the same way, while windows that only
    share vocabulary (other rows of the same table) never merge.
(b) pool fullness: with heavy duplication the reranker receives exactly
    ``candidates`` distinct texts. Fails without the over-fetch.
(c) scope: with ``scope=past``, neither results nor ``also_in`` ever name a session
    from the caller's chain.
(d) config: ``session_recall_dedup=false`` restores the raw pool (fetch equals
    ``candidates``, ``candidates == candidates_raw``, duplicates reach the reranker).
"""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from qdrant_memory.session_recall import (
    DEDUP_FETCH_CAP,
    DEDUP_OVERFETCH,
    NEAR_DUP_TOKEN_RUN,
    RecallSettings,
    SessionRecall,
    _near_duplicate,
)

LORE = (
    "The lighthouse keeper's ledger notes that the foghorn maintenance window opens "
    "at dawn every third Tuesday and that the lamp oil reserve must never fall below "
    "forty barrels before the winter storms arrive, because the supply ship only "
    "docks twice a season and the harbour master refuses to sign off on any delivery "
    "that arrives after the fog season has started."
)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class FakeQdrant:
    """Dense search over planted points; implements the lane's filter subset."""

    def __init__(self) -> None:
        self.points: list[dict] = []
        self.search_calls: list[tuple[int, dict | None]] = []

    def get_collections(self):
        return ["windows"]

    def add(self, *, session_id: str, text: str, score: float, ts: float, msg_id: int = 1,
            source: str = "cli") -> None:
        self.points.append({
            "payload": {
                "session_id": session_id,
                "session_source": source,
                "first_msg_id": msg_id,
                "last_msg_id": msg_id,
                "start_ts": ts,
                "end_ts": ts,
                "char_start": 0,
                "text": text,
                "text_sha256": sha(text),
            },
            "score": score,
        })

    @staticmethod
    def _cond_match(payload: dict, cond: dict) -> bool:
        key = cond.get("key")
        value = payload.get(key)
        if value is None:
            return False
        if "match" in cond:
            options = cond["match"].get("any") or [cond["match"].get("value")]
            return value in options
        rng = cond.get("range") or {}
        if "gte" in rng and not float(value) >= float(rng["gte"]):
            return False
        if "lt" in rng and not float(value) < float(rng["lt"]):
            return False
        return True

    def _match(self, point: dict, filter_: dict | None) -> bool:
        if not filter_:
            return True
        for cond in filter_.get("must") or []:
            if not self._cond_match(point["payload"], cond):
                return False
        for cond in filter_.get("must_not") or []:
            if self._cond_match(point["payload"], cond):
                return False
        return True

    def search(self, name, vector, limit, filter=None, with_payload=True, with_vector=False):
        self.search_calls.append((int(limit), filter))
        selected = sorted(self.points, key=lambda p: -p["score"])
        return [dict(p) for p in selected if self._match(p, filter)][:limit]


class FakeEmbeddings:
    def embed(self, text):
        return [1.0]


class RecordingReranker:
    """A real local HTTP reranker that records the documents it was asked to score."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        reranker = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                reranker.requests.append(body)
                scores = []
                for index, doc in enumerate(body["documents"]):
                    scores.append(1.0 if "foghorn" in doc else round(0.5 - index * 0.001, 6))
                payload = json.dumps({
                    "results": [{"index": i, "relevance_score": s} for i, s in enumerate(scores)]
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5.0)

    @property
    def documents(self) -> list[list[str]]:
        return [request["documents"] for request in self.requests]


@pytest.fixture
def reranker():
    server = RecordingReranker()
    yield server
    server.stop()


def make_recall(qdrant: FakeQdrant, rerank_url: str = "", **overrides) -> SessionRecall:
    settings = RecallSettings(
        collection="windows",
        state_db="/unused/state.db",
        state_dir="/unused",
        lock_dir="",
        rerank_url=rerank_url,
        **overrides,
    )
    return SessionRecall(settings, qdrant=qdrant, embeddings=FakeEmbeddings())


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


T0 = datetime(2026, 9, 22, 10, 0).timestamp()
DAY = 86400.0
FILLER = (
    "Harbour chatter unrelated to the ledger: crates of rope, two spare rudders and a "
    "crate of tea arrived on the morning tide while the gulls argued about the mast."
)


def unique_text(index: int) -> str:
    """A distinct window-shaped text: no shared run with any other unique_text."""
    return " ".join(f"u{index}t{word}" for word in range(120))


# --- (a) exact duplicates collapse with provenance ------------------------------


def test_a_exact_duplicates_collapse_before_rerank_with_provenance(reranker):
    qdrant = FakeQdrant()
    replay_scores = [0.93, 0.95, 0.91, 0.90, 0.89, 0.88]
    replay_sessions = [f"discord-{i:02d}" for i in range(6)]
    for index, (session, score) in enumerate(zip(replay_sessions, replay_scores)):
        qdrant.add(session_id=session, text=LORE, score=score, ts=T0 + index * DAY, msg_id=100 + index)
    trunc_sessions = [f"bridge-{i:02d}" for i in range(15)]
    for index, session in enumerate(trunc_sessions):
        qdrant.add(session_id=session, text=FILLER, score=0.94 - index * 0.001,
                   ts=T0 + 10 * DAY + index * 3600, msg_id=200 + index)
    for index in range(15):
        qdrant.add(session_id=f"distinct-{index:02d}", text=unique_text(index),
                   score=0.5 + index * 0.01, ts=T0 + 30 * DAY + index * 60, msg_id=300 + index)

    recall = make_recall(qdrant, reranker.url, candidates=16, top_k=8)
    result = recall.search("ledger foghorn maintenance", scope="all")

    assert result["rerank"] == "applied"
    assert result["candidates_raw"] == 36, "every planted window fits in one over-fetched pool"
    assert result["candidates"] == 16, "the rerank pool stays full of distinct texts"
    assert result["duplicates_collapsed"] == 20
    texts = [item["text"] for item in result["results"]]
    assert len(texts) == 8 and len(set(texts)) == 8, "the top-k holds distinct texts only"

    docs = reranker.documents[-1]
    assert len(docs) == 16 and len(set(docs)) == 16, "the reranker scored distinct texts only"
    assert LORE in docs and docs.count(LORE) == 1

    top = result["results"][0]
    assert top["text"] == LORE
    assert top["session_id"] == "discord-01", "the representative is the highest-scoring copy"
    assert top["duplicate_count"] == 5
    expected_also_in = [
        {"session_id": replay_sessions[i], "first_msg_id": 100 + i, "start_time": iso(T0 + i * DAY)}
        for i in (5, 4, 3, 2, 0)  # every copy except the representative, newest first
    ]
    assert top["also_in"] == expected_also_in
    assert "also_in_truncated" not in top

    collapsed = next(item for item in result["results"] if item["text"] == FILLER)
    assert collapsed["duplicate_count"] == 14
    assert len(collapsed["also_in"]) == 10, "also_in caps at 10 entries"
    assert collapsed["also_in_truncated"] is True
    assert collapsed["also_in"][0]["session_id"] == "bridge-14", "newest first"
    times = [entry["start_time"] for entry in collapsed["also_in"]]
    assert times == sorted(times, reverse=True)


def test_a2_offset_shifted_near_duplicates_collapse_but_table_rows_do_not(reranker):
    qdrant = FakeQdrant()
    long_lore = LORE + " " + LORE  # >1000 chars, replays cut it at different offsets
    cuts = [long_lore[0:1000], long_lore[97:1097], long_lore[211:1211]]
    for index, text in enumerate(cuts):
        qdrant.add(session_id=f"cut-{index}", text=text, score=0.9 - index * 0.01,
                   ts=T0 + index * DAY, msg_id=10 + index)
    header = "model adapter status score notes"
    rows = [
        f"{header} candidate-02 gpt-6-astra valid completed score=100 plan-alpha",
        f"{header} candidate-03 gpt-6-sol valid completed score=0 plan-beta",
        f"{header} candidate-04 gpt-6-luna valid completed score=75 plan-gamma",
    ]
    for index, row in enumerate(rows):
        qdrant.add(session_id=f"report-{index}", text=row, score=0.6 - index * 0.01,
                   ts=T0 + 5 * DAY + index * 60, msg_id=50 + index)

    recall = make_recall(qdrant, reranker.url, candidates=8, top_k=6)
    result = recall.search("ledger foghorn", scope="all")

    cut = next(item for item in result["results"] if item["text"] == cuts[0])
    assert cut["duplicate_count"] == 2, "the offset-shifted replays collapse into one group"
    assert {entry["session_id"] for entry in cut["also_in"]} == {"cut-1", "cut-2"}
    row_texts = [item["text"] for item in result["results"] if item["text"] in rows]
    assert len(row_texts) == 3, "table rows that share only vocabulary never merge"
    for item in result["results"]:
        if item["text"] in rows:
            assert item["duplicate_count"] == 0 and item["also_in"] == []


# --- (b) pool fullness ----------------------------------------------------------


def test_b_heavy_duplication_still_fills_the_rerank_pool(reranker):
    qdrant = FakeQdrant()
    # Eight texts, three exact copies each, all scoring above the singles: the raw
    # top-24 is nothing but copies unless the lane over-fetches.
    for text_index in range(8):
        for copy in range(3):
            qdrant.add(session_id=f"replay-{text_index}-{copy}",
                       text=f"{LORE} variant-{text_index}",
                       score=0.95 - text_index * 0.005 - copy * 0.001,
                       ts=T0 + text_index * DAY + copy * 3600, msg_id=10 * text_index + copy)
    for index in range(30):
        qdrant.add(session_id=f"single-{index:02d}", text=unique_text(index),
                   score=0.5 + index * 0.002, ts=T0 + 20 * DAY + index * 60, msg_id=500 + index)

    recall = make_recall(qdrant, reranker.url, candidates=24, top_k=10)
    result = recall.search("ledger foghorn", scope="all")

    assert result["candidates"] == 24, "the pool holds candidates distinct texts, not fewer"
    assert result["candidates_raw"] == 54
    assert result["duplicates_collapsed"] == 30
    docs = reranker.documents[-1]
    assert len(docs) == 24, "the reranker received exactly candidates documents"
    assert len(set(docs)) == 24, "every rerank document is a distinct text"
    fetch_limits = [limit for limit, _ in qdrant.search_calls]
    assert fetch_limits == [min(24 * DEDUP_OVERFETCH, DEDUP_FETCH_CAP)]


# --- (c) scope ------------------------------------------------------------------


def test_c_scope_past_also_in_never_names_the_caller_chain(reranker):
    qdrant = FakeQdrant()
    # The chain copies score highest: if the scope filter ever stops reaching the
    # dense search, they take over the results and the provenance.
    for session, score, ts in (("caller", 0.99, T0 + DAY), ("ancestor", 0.98, T0)):
        qdrant.add(session_id=session, text=LORE, score=score, ts=ts, msg_id=7)
    past_sessions = ["telegram-1", "telegram-2", "telegram-3"]
    for index, session in enumerate(past_sessions):
        qdrant.add(session_id=session, text=LORE, score=0.9 - index * 0.01,
                   ts=T0 - (3 - index) * DAY, msg_id=20 + index)
    for index in range(6):
        qdrant.add(session_id=f"other-{index}", text=unique_text(index),
                   score=0.3 + index * 0.01, ts=T0 - 10 * DAY, msg_id=60 + index)

    recall = make_recall(qdrant, reranker.url, candidates=8, top_k=5)
    chain = ["caller", "ancestor"]
    past = recall.search("ledger foghorn", scope="past", chain=chain)
    every = recall.search("ledger foghorn", scope="all", chain=chain)

    lore = next(item for item in past["results"] if item["text"] == LORE)
    assert lore["session_id"] == "telegram-1", "the past representative is the best past copy"
    assert lore["duplicate_count"] == 2
    assert {entry["session_id"] for entry in lore["also_in"]} == {"telegram-2", "telegram-3"}
    chain_leak = {item["session_id"] for item in past["results"]} & set(chain)
    provenance_leak = {
        entry["session_id"] for item in past["results"] for entry in item["also_in"]
    } & set(chain)
    assert not chain_leak, f"scope=past results named chain sessions: {chain_leak}"
    assert not provenance_leak, f"also_in named chain sessions: {provenance_leak}"

    # Control: with scope=all the chain copies do collapse into the same group,
    # so the fixture can leak and the past-scope assertions above are meaningful.
    lore_all = next(item for item in every["results"] if item["text"] == LORE)
    assert lore_all["session_id"] == "caller"
    assert lore_all["duplicate_count"] == 4
    assert {entry["session_id"] for entry in lore_all["also_in"]} == {
        "ancestor", "telegram-1", "telegram-2", "telegram-3",
    }


# --- (d) config -----------------------------------------------------------------


def test_dedup_disabled_restores_the_raw_pool(reranker):
    qdrant = FakeQdrant()
    for copy in range(3):
        qdrant.add(session_id=f"replay-{copy}", text=LORE, score=0.95 - copy * 0.01,
                   ts=T0 + copy * DAY, msg_id=1 + copy)
    for index in range(9):
        qdrant.add(session_id=f"single-{index:02d}", text=unique_text(index),
                   score=0.5 + index * 0.01, ts=T0 + 10 * DAY + index * 60, msg_id=40 + index)

    settings = RecallSettings.from_config(
        {"session_recall_collection": "windows", "session_recall_dedup": "false"},
        "/unused/home",
    )
    assert settings.dedup is False
    assert RecallSettings.from_config(
        {"session_recall_collection": "windows"}, "/unused/home"
    ).dedup is True, "dedup defaults to on"

    settings = RecallSettings(
        collection="windows", state_db="/unused/state.db", state_dir="/unused", lock_dir="",
        rerank_url=reranker.url, candidates=settings.candidates, dedup=False,
    )
    recall = SessionRecall(settings, qdrant=qdrant, embeddings=FakeEmbeddings())
    result = recall.search("ledger foghorn", scope="all", top_k=6)

    assert [limit for limit, _ in qdrant.search_calls] == [settings.candidates], \
        "without dedup the fetch is exactly candidates, no over-fetch"
    assert result["candidates"] == result["candidates_raw"] == 12
    assert result["duplicates_collapsed"] == 0
    docs = reranker.documents[-1]
    assert len(set(docs)) < len(docs), "raw duplicates reach the reranker again"
    texts = [item["text"] for item in result["results"]]
    assert len(set(texts)) < len(texts), "the top-k holds repeated texts again"
    for item in result["results"]:
        if item["text"] == LORE:
            assert item["duplicate_count"] == 0 and item["also_in"] == []


# --- the near-duplicate rule itself ----------------------------------------------


def test_near_duplicate_rule_bounds():
    # A window fully inside a longer one (a cut of the same text) always matches.
    tokens = [f"t{i}" for i in range(100)]
    assert _near_duplicate(" ".join(tokens), " ".join(tokens[25:])) is True
    # Equal-length windows around the 0.75 boundary: a 76-token shared run merges,
    # a 74-token one (measured false-merge territory) does not.
    tail = tokens[24:]
    shifted_76 = tail + [f"x{i}" for i in range(24)]
    assert _near_duplicate(" ".join(tokens), " ".join(shifted_76)) is True
    tail = tokens[26:]
    shifted_74 = tail + [f"x{i}" for i in range(26)]
    assert _near_duplicate(" ".join(tokens), " ".join(shifted_74)) is False
    # Same vocabulary, no shared run: other rows of one table.
    rows = [
        "model candidate-02 astra valid completed score=100 plan alpha",
        "model candidate-03 sol valid completed score=0 plan beta",
    ]
    assert _near_duplicate(*rows) is False
    # Adjacent windows of one session share only the 128-char overlap.
    window_a = [f"w{i}" for i in range(170)]
    window_b = [f"w{i}" for i in range(150, 320)]
    assert _near_duplicate(" ".join(window_a), " ".join(window_b)) is False
    assert _near_duplicate("", "abc") is False
    assert NEAR_DUP_TOKEN_RUN == 0.75
