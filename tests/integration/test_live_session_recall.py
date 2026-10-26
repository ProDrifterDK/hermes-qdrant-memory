"""Live contracts for session recall: raw conversation windows from state.db.

Every test runs the real path: a fixture state.db built from the production schema
(``tests/fixtures/state_db_schema.sql``) and opened read-only by the product, real
Qdrant, real bge-m3, and the real cross-encoder where a contract is about reranking.
Collections are disposable and prefixed (``session_recall_collections``).

Contracts and the failure each one names:

(a) incremental == full: a tail re-window that misses the extended last window, or a
    cursor that skips or repeats windows, leaves a point set no full rebuild produces.
(b) a paraphrased question finds a planted past fact: rank 1 with the reranker, top 5
    dense-only. Fails if windows are cut wrongly, the vector is not the window's, or
    the rerank order is not applied.
(c) current-session recall: a fact in the compressed ancestor of the caller is found
    with scope=current and absent with scope=past, and a message written after the
    last index pass is found through the inline catch-up.
(d) a planted credential reaches no stored payload, no tool output, and the stored
    vector is the embedding of the redacted text.
(e) an unreachable reranker gives dense order and says so.
(f) index_lag is exact after a partial (time-budgeted) run.

Run with RUN_QDRANT_INTEGRATION=1 QDRANT_TEST_URL=http://127.0.0.1:6333 and a
llama.cpp reranker at QDRANT_TEST_RERANK_URL (default http://127.0.0.1:18111).
Set SESSION_RECALL_ARTIFACT to also copy the run artifact (ranks, latency) there.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from conftest import SessionRecallCollections
from qdrant_memory.config import DEFAULTS
from qdrant_memory.session_recall import build_session_recall

TRUTHY = {"1", "true", "yes", "on"}
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_QDRANT_INTEGRATION", "").strip().lower() not in TRUTHY,
    reason="set RUN_QDRANT_INTEGRATION=1 to run live Qdrant integration tests",
)

SCHEMA = Path(__file__).resolve().parents[1] / "fixtures" / "state_db_schema.sql"
RERANK_URL = os.environ.get("QDRANT_TEST_RERANK_URL", "http://127.0.0.1:18111")
UNREACHABLE_RERANK_URL = "http://127.0.0.1:9"
T0 = datetime(2026, 8, 3, 9, 0).timestamp()
DAY = 86400.0
ARTIFACT: dict[str, Any] = {"contracts": {}, "latency": {}}

# --- fixture database -------------------------------------------------------


class FixtureDB:
    """A writable state.db built from the production schema; the product opens it read-only."""

    def __init__(self, path: Path):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA.read_text(encoding="utf-8"))
        self.rows: list[tuple[int, str, str, Any, float]] = []

    def session(self, sid: str, source: str, started_at: float, parent: str | None = None) -> None:
        self.conn.execute(
            "INSERT INTO sessions (id, source, started_at, parent_session_id) VALUES (?, ?, ?, ?)",
            (sid, source, started_at, parent),
        )
        self.conn.commit()

    def say(self, sid: str, role: str, content: Any, ts: float) -> int:
        cur = self.conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (sid, role, content, ts),
        )
        self.conn.commit()
        self.rows.append((int(cur.lastrowid), sid, role, content, ts))
        return int(cur.lastrowid)

    def eligible(self) -> list[tuple[int, str, str, Any, float]]:
        """The contract's reading: user/assistant rows with non-blank content."""
        return [r for r in self.rows if r[2] in ("user", "assistant") and isinstance(r[3], str) and r[3].strip()]


_SENTENCES = [
    "I restarted the {svc} on {host} and the {metric} went back to normal after {n} minutes.",
    "Can you check why the {svc} logs on {host} show {n} retries per hour since the last deploy?",
    "The {svc} migration finished; {n} rows were rewritten and the old index on {host} can go.",
    "Let's keep the {svc} canary at {n} percent until the {metric} graph stays flat for a day.",
    "I tried the {food} recipe again, this time with {n} grams of butter, and it came out better.",
    "For the trip to {city} we booked {n} nights near the station; the hotel has decent wifi.",
    "The {svc} dashboard now has a panel for {metric}, grouped by host, starting with {host}.",
    "Remember to renew the {svc} certificate on {host} before it expires in {n} days.",
    "The book club picked a novel set in {city}; we meet in {n} weeks to talk about it.",
    "I benchmarked the {svc} on {host}: p95 latency is {n} ms with the new connection pool.",
    "The {svc} on {host} ran out of file descriptors again, so I raised the limit to {n}.",
    "We agreed the {svc} alerts should page only when {metric} stays bad for {n} minutes.",
]
_FILL = {
    "svc": ["postgres", "redis", "nginx", "telegram bridge", "discord bot", "embedding server",
            "backup job", "cron runner", "billing worker", "search indexer"],
    "host": ["the NAS", "the mini PC", "the Hetzner storage box", "the laptop", "the desktop",
             "the Pi in the garage", "the Frankfurt VPS"],
    "metric": ["memory usage", "queue depth", "error rate", "disk IO", "CPU steal", "p99 latency"],
    "food": ["lasagna", "sourdough", "ramen", "empanadas", "curry"],
    "city": ["Valparaiso", "Lisbon", "Kyoto", "Mendoza", "Berlin"],
}


def filler(rng: random.Random, sentences: int) -> str:
    out = []
    for _ in range(sentences):
        template = rng.choice(_SENTENCES)
        values = {key: rng.choice(options) for key, options in _FILL.items()}
        out.append(template.format(n=rng.randint(2, 900), **values))
    return " ".join(out)


# Near misses: same entities or same shape as the planted facts, different answers.
NEAR_MISSES = [
    "We moved the nightly Postgres backups to the Hetzner storage box in Falkenstein; the restore test passed in 14 minutes.",
    "The API gateway on the Frankfurt VPS now terminates TLS on port 8443 for the billing worker.",
    "For Kael's voice notes we transcribe with Whisper large-v3 on the desktop GPU, not on a server.",
    "The production feature flag is ORION_RED_RAMP and it stays at 40 percent for the whole quarter.",
]
PLANTED_PAST = (
    "Final call on the Kael voice gateway: it runs on the Hetzner AX52 in Falkenstein on port 7443, "
    "because the OVH box in Frankfurt kept dropping UDP packets during the call tests."
)
PAST_QUERY = "Where did we end up hosting the speech gateway for Kael, and what was wrong with the other machine?"
PLANTED_ANCESTOR = (
    "For the staging rollout the flag is ORION_BLUE_RAMP; keep it at 15 percent until Thursday's "
    "review, then we decide whether to widen it."
)
ANCESTOR_QUERY = "What percentage did we agree to hold the staging rollout flag at?"
FRESH_FACT = "Heads up: the rollback drill moved to 22:40 UTC on the 14th, and Marta runs it this time."
FRESH_QUERY = "When is the rollback drill now and who is running it?"

_SEED = hashlib.sha256(b"session-recall-secret-fixture").hexdigest()
# Built from fragments so this file never holds a contiguous credential-shaped literal.
SECRETS = {
    "api_key": "s" + "k-" + "live" + _SEED[:32],
    "bearer": _SEED[8:48],
    "password": "Zr8" + _SEED[40:56],
    "pem_line_1": hashlib.sha256(b"pem-1").hexdigest() + "AQAB",
    "pem_line_2": hashlib.sha256(b"pem-2").hexdigest() + "Zx9w",
}
_PEM = "\n".join([
    "-----BEGIN " + "RSA PRIV" + "ATE KEY-----",
    SECRETS["pem_line_1"],
    SECRETS["pem_line_2"],
    "-----END " + "RSA PRIV" + "ATE KEY-----",
])
SECRET_QUERY = "what key did I paste for the billing service?"


def recall_config(ctx: SessionRecallCollections, collection: str, db: Path, home: Path, rerank_url: str,
                  **extra: Any) -> dict[str, Any]:
    config = dict(DEFAULTS)
    config.update({
        "qdrant_url": ctx.qdrant_url,
        "embedding_url": ctx.embedding_url,
        "embedding_model": ctx.embedding_model,
        "vector_size": ctx.vector_size,
        "distance": ctx.distance,
        "collection_name": f"{ctx.prefix}_never_written_memory",
        "learning_collection_name": f"{ctx.prefix}_never_written_learnings",
        "lineage_lock_dir": str(home / "locks"),
        "session_recall_collection": collection,
        "session_recall_state_db": str(db),
        "session_recall_rerank_url": rerank_url,
        "session_recall_inline_catchup_seconds": 120.0,
    })
    config.update(extra)
    return config


def provider_for(monkeypatch, config: dict[str, Any], home: Path, session_id: str):
    monkeypatch.setenv("HERMES_HOME", str(home))
    from __init__ import QdrantMemoryProvider

    from qdrant_memory.client import QdrantClient
    from qdrant_memory.embeddings import EmbeddingClient

    provider = QdrantMemoryProvider()
    provider._active = True
    provider._hermes_home = str(home)
    provider._session_id = session_id
    provider._config.update(config)
    # The clients `initialize` builds, against the test services.
    provider._qdrant = QdrantClient(config["qdrant_url"])
    provider._embeddings = EmbeddingClient(config["embedding_url"], config["embedding_model"])
    return provider


def recall_tool(provider, **args: Any) -> dict[str, Any]:
    raw = provider.handle_tool_call("qdrant_memory_session_recall", args)
    payload = json.loads(raw)
    assert "error" not in payload, payload
    return payload


def all_payloads(ctx: SessionRecallCollections, collection: str, *, with_vector: bool = False) -> list[dict[str, Any]]:
    return ctx.qdrant.scroll_by_filter(collection, {"must": []}, limit=256, with_vector=with_vector)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


class CountingClock:
    """Deterministic clock: one second per read, so a budget stops at a fixed batch."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 1.0
        return self.now


# --- the shared corpus (b, c, d, e) -----------------------------------------


@pytest.fixture(scope="module")
def corpus(session_recall_collections: SessionRecallCollections, tmp_path_factory):
    ctx = session_recall_collections
    home = tmp_path_factory.mktemp("srecall_home")
    db = FixtureDB(home / "state.db")
    rng = random.Random(20261026)
    sources = ["cli", "telegram", "discord"]
    near = list(NEAR_MISSES)
    for index in range(12):
        sid = f"distractor-{index:02d}"
        start = T0 + index * 3 * DAY
        db.session(sid, sources[index % 3], start)
        for turn in range(18):
            text = filler(rng, rng.randint(5, 8))
            if near and turn == 6 and index % 3 == 0:
                text = f"{text} {near.pop(0)} {filler(rng, 2)}"
            db.say(sid, "user" if turn % 2 == 0 else "assistant", text, start + turn * 90)
        db.say(sid, "tool", filler(rng, 3), start + 2000)

    past = "past-gateway"
    db.session(past, "telegram", T0 + 40 * DAY)
    for turn in range(8):
        ts = T0 + 40 * DAY + turn * 120
        if turn == 4:
            db.say(past, "user", f"{filler(rng, 2)} {PLANTED_PAST}", ts)
        elif turn == 5:
            db.say(past, "assistant", "Got it. I'll note that the voice gateway lives on the AX52 in Falkenstein, "
                                      "port 7443, and that the Frankfurt OVH machine was dropped for UDP packet loss.", ts)
        else:
            db.say(past, "user" if turn % 2 == 0 else "assistant", filler(rng, 5), ts)

    leaky = "past-leaky"
    db.session(leaky, "discord", T0 + 44 * DAY)
    base = T0 + 44 * DAY
    db.say(leaky, "user", f"{filler(rng, 2)} Here is the key for the billing service, don't lose it: {SECRETS['api_key']}", base)
    db.say(leaky, "user", "and the billing webhook signer sends Authoriz" + "ation: Bear" + f"er {SECRETS['bearer']}", base + 30)
    db.say(leaky, "user", "reporting replica creds: pass" + "word=" + SECRETS["password"], base + 60)
    db.say(leaky, "user", f"the deploy key for the billing box:\n{_PEM}", base + 90)
    db.say(leaky, "assistant", "Stored. I won't repeat the billing service key back in chat.", base + 120)

    ancestor, caller = "chain-ancestor", "chain-caller"
    db.session(ancestor, "cli", T0 + 50 * DAY)
    db.session(caller, "cli", T0 + 50 * DAY + 7200, parent=ancestor)
    for turn in range(10):
        ts = T0 + 50 * DAY + turn * 300
        text = f"{filler(rng, 2)} {PLANTED_ANCESTOR}" if turn == 3 else filler(rng, 5)
        db.say(ancestor, "user" if turn % 2 == 0 else "assistant", text, ts)
    db.say(ancestor, "session_meta", "compressed into chain-caller", T0 + 50 * DAY + 7100)
    for turn in range(4):
        db.say(caller, "user" if turn % 2 == 0 else "assistant", filler(rng, 4), T0 + 50 * DAY + 7200 + turn * 60)

    collection = ctx.new_collection("corpus")
    config = recall_config(ctx, collection, db.path, home, RERANK_URL)
    recall = build_session_recall(config, str(home))
    started = time.monotonic()
    report = recall.run(lock_timeout=10.0)
    ARTIFACT["corpus"] = {
        "collection": collection,
        "windows": ctx.qdrant.count(collection),
        "index_seconds": round(time.monotonic() - started, 2),
        "report": report,
    }
    assert report["lock"] == "acquired" and not report["stopped"], report
    yield {"db": db, "home": home, "collection": collection, "config": config, "recall": recall,
           "past": past, "leaky": leaky, "ancestor": ancestor, "caller": caller}
    db.conn.close()


@pytest.fixture(scope="module", autouse=True)
def write_artifact(tmp_path_factory):
    yield
    if not ARTIFACT["contracts"]:
        return
    ARTIFACT["written_at"] = datetime.now().isoformat(timespec="seconds")
    text = json.dumps(ARTIFACT, indent=2, sort_keys=True, default=str)
    path = tmp_path_factory.mktemp("srecall_artifact") / "session_recall_live.json"
    path.write_text(text, encoding="utf-8")
    print(f"\nsession recall artifact: {path}")
    if os.environ.get("SESSION_RECALL_ARTIFACT"):
        Path(os.environ["SESSION_RECALL_ARTIFACT"]).write_text(text, encoding="utf-8")


def _rank_of(results: list[dict[str, Any]], needle: str) -> int | None:
    return next((item["rank"] for item in results if needle in item["text"]), None)


def test_b_paraphrased_question_finds_the_planted_past_fact(corpus, monkeypatch):
    with_rerank = provider_for(monkeypatch, corpus["config"], corpus["home"], corpus["caller"])
    reranked = recall_tool(with_rerank, query=PAST_QUERY, scope="past")
    dense_config = dict(corpus["config"], session_recall_rerank_url="")
    dense = recall_tool(provider_for(monkeypatch, dense_config, corpus["home"], corpus["caller"]),
                        query=PAST_QUERY, scope="past")

    assert reranked["rerank"] == "applied", reranked["rerank"]
    assert reranked["context_not_instruction"] is True and "not instruction authority" in reranked["authority"]
    assert reranked["candidates"] == 128, "the latency line must measure a full 128-candidate rerank"
    rerank_rank = _rank_of(reranked["results"], "AX52")
    dense_rank = _rank_of(dense["results"], "AX52")
    top = reranked["results"][0]
    ARTIFACT["contracts"]["b"] = {"rerank_rank": rerank_rank, "dense_rank": dense_rank,
                                  "dense_rerank_status": dense["rerank"]}
    ARTIFACT["latency"]["past_query_timings"] = reranked["timings"]
    assert rerank_rank == 1, [item["text"][:120] for item in reranked["results"][:5]]
    assert top["session_id"] == corpus["past"] and top["session_source"] == "telegram"
    assert top["first_msg_id"] and top["start_time"].startswith("2026-09-12")
    assert dense["rerank"].startswith("unavailable:")
    assert dense_rank is not None and dense_rank <= 5, [item["text"][:120] for item in dense["results"][:5]]


def test_c_current_scope_reaches_the_compressed_ancestor_and_past_does_not(corpus, monkeypatch):
    provider = provider_for(monkeypatch, corpus["config"], corpus["home"], corpus["caller"])
    db: FixtureDB = corpus["db"]
    # Written after the last index pass: only the inline catch-up can make it findable.
    db.say(corpus["caller"], "user", FRESH_FACT, T0 + 50 * DAY + 9000)

    fresh = recall_tool(provider, query=FRESH_QUERY, scope="current")
    current = recall_tool(provider, query=ANCESTOR_QUERY, scope="current")
    past = recall_tool(provider, query=ANCESTOR_QUERY, scope="past")
    chain = {corpus["ancestor"], corpus["caller"]}
    ARTIFACT["latency"]["inline_catchup"] = fresh["catchup"]
    ARTIFACT["contracts"]["c"] = {
        "fresh_rank": _rank_of(fresh["results"], "22:40"),
        "ancestor_rank_current": _rank_of(current["results"], "ORION_BLUE_RAMP"),
        "ancestor_in_past": _rank_of(past["results"], "ORION_BLUE_RAMP"),
        "lag_after_catchup": fresh["index_lag"],
    }

    assert fresh["catchup"]["windows_upserted"] >= 1 and not fresh["catchup"]["stopped"], fresh["catchup"]
    assert fresh["index_lag"]["current_chain_messages"] == 0
    assert _rank_of(fresh["results"], "22:40") == 1
    assert {item["session_id"] for item in current["results"]} <= chain
    assert _rank_of(current["results"], "ORION_BLUE_RAMP") == 1
    assert any(item["session_id"] == corpus["ancestor"] for item in current["results"])
    assert past["results"], "scope=past must still answer from other sessions"
    assert not {item["session_id"] for item in past["results"]} & chain
    assert _rank_of(past["results"], "ORION_BLUE_RAMP") is None


def test_d_planted_credentials_reach_no_payload_vector_text_or_output(corpus, monkeypatch, session_recall_collections):
    ctx = session_recall_collections
    points = all_payloads(ctx, corpus["collection"], with_vector=True)
    stored = json.dumps([point["payload"] for point in points])
    provider = provider_for(monkeypatch, corpus["config"], corpus["home"], corpus["caller"])
    output = provider.handle_tool_call("qdrant_memory_session_recall", {"query": SECRET_QUERY, "scope": "past"})

    leaked_stored = sorted(name for name, value in SECRETS.items() if value in stored)
    leaked_output = sorted(name for name, value in SECRETS.items() if value in output)
    secret_windows = [p for p in points if p["payload"]["session_id"] == corpus["leaky"]]
    assert secret_windows, "the secret session must be indexed, not dropped"
    window = next(p for p in secret_windows if "billing service" in p["payload"]["text"])
    # The vector must be the embedding of the stored (redacted) text, not of the raw text.
    from qdrant_memory.embeddings import EmbeddingClient

    reembedded = EmbeddingClient(ctx.embedding_url, ctx.embedding_model, timeout=120.0).embed_batch(
        [window["payload"]["text"]]
    )[0]
    similarity = cosine(reembedded, window["vector"])
    ARTIFACT["contracts"]["d"] = {"leaked_stored": leaked_stored, "leaked_output": leaked_output,
                                  "vector_vs_redacted_text_cosine": similarity}

    assert leaked_stored == [] and leaked_output == []
    assert "[redacted: possible secret-bearing value]" in window["payload"]["text"]
    assert "billing service" in output and "[redacted" in output
    assert similarity >= 0.9995, similarity


def test_e_unreachable_reranker_falls_back_to_dense_order_and_says_so(corpus, monkeypatch, session_recall_collections):
    ctx = session_recall_collections
    config = dict(corpus["config"], session_recall_rerank_url=UNREACHABLE_RERANK_URL)
    result = recall_tool(provider_for(monkeypatch, config, corpus["home"], corpus["caller"]),
                         query=PAST_QUERY, scope="all", top_k=8)
    from qdrant_memory.embeddings import EmbeddingClient

    vector = EmbeddingClient(ctx.embedding_url, ctx.embedding_model, timeout=120.0).embed(PAST_QUERY)
    dense = ctx.qdrant.search(corpus["collection"], vector, 8)
    ARTIFACT["contracts"]["e"] = {"rerank": result["rerank"]}

    assert result["rerank"].startswith("unavailable:"), result["rerank"]
    assert all(item["rerank_score"] is None for item in result["results"])
    assert [item["text"] for item in result["results"]] == [hit["payload"]["text"] for hit in dense]


# --- small databases for the indexing contracts (a, f) ----------------------


def _small_db(path: Path, rng: random.Random) -> FixtureDB:
    db = FixtureDB(path)
    db.session("s1", "cli", T0)
    db.session("s2", "telegram", T0 + DAY)
    for turn in range(10):
        db.say("s1", "user" if turn % 2 == 0 else "assistant", f"(s1m{turn}) {filler(rng, rng.randint(1, 3))}", T0 + turn * 60)
    db.say("s1", "tool", "tool output never indexed", T0 + 700)
    db.say("s1", "assistant", "", T0 + 710)
    db.say("s1", "assistant", "  \n ", T0 + 720)
    db.say("s1", "user", None, T0 + 730)
    db.say("s1", "session_meta", "meta row never indexed", T0 + 740)
    for turn in range(5):
        db.say("s2", "user" if turn % 2 == 0 else "assistant", f"(s2m{turn}) {filler(rng, 2)}", T0 + DAY + turn * 60)
    return db


def _point_set(ctx: SessionRecallCollections, collection: str) -> dict[str, str]:
    points = all_payloads(ctx, collection)
    for point in points:
        assert hashlib.sha256(point["payload"]["text"].encode()).hexdigest() == point["payload"]["text_sha256"]
    return {str(point["id"]): point["payload"]["text_sha256"] for point in points}


def test_a_indexing_in_increments_equals_one_full_rebuild(session_recall_collections, tmp_path):
    ctx = session_recall_collections
    rng = random.Random(11)
    db = _small_db(tmp_path / "state.db", rng)
    small = {"session_recall_window_chars": 300, "session_recall_overlap_chars": 40}
    inc_config = recall_config(ctx, ctx.new_collection("inc"), db.path, tmp_path / "inc", "", **small)
    full_config = recall_config(ctx, ctx.new_collection("full"), db.path, tmp_path / "full", "", **small)
    inc = build_session_recall(inc_config, str(tmp_path / "inc"))
    inc.clock = CountingClock()

    partial = inc.run(max_seconds=7, batch=2)
    after_partial = _point_set(ctx, inc_config["session_recall_collection"])
    inc.clock = time.monotonic
    rest = inc.run(batch=2)
    for turn in range(6):
        db.say("s1", "user" if turn % 2 == 0 else "assistant", f"(s1x{turn}) {filler(rng, 2)}", T0 + 800 + turn * 60)
    db.session("s3", "cli", T0 + 2 * DAY, parent="s1")
    for turn in range(3):
        db.say("s3", "user" if turn % 2 == 0 else "assistant", f"(s3m{turn}) {filler(rng, 2)}", T0 + 2 * DAY + turn)
    incremental = inc.run(batch=3)
    inc_set = _point_set(ctx, inc_config["session_recall_collection"])

    full = build_session_recall(full_config, str(tmp_path / "full"))
    full.run(full=True, batch=8)
    full_set = _point_set(ctx, full_config["session_recall_collection"])
    again = full.run(full=True, batch=8)
    noop = inc.run(batch=3)
    ARTIFACT["contracts"]["a"] = {"partial": partial, "rest": rest, "incremental": incremental,
                                  "points": len(full_set)}

    assert partial["stopped"] == "time budget" and 0 < len(after_partial) < len(full_set), partial
    assert incremental["windows_upserted"] < len(full_set), "the append pass must not re-embed everything"
    assert inc_set == full_set
    assert _point_set(ctx, full_config["session_recall_collection"]) == full_set
    assert again["windows_upserted"] == len(full_set)
    assert noop["sessions_selected"] == 0 and noop["windows_upserted"] == 0


def _oracle_lag(ctx: SessionRecallCollections, collection: str, db: FixtureDB) -> dict[str, Any]:
    """Lag from the stored windows and the inserted rows only: no cursor, no product code."""
    covered: dict[str, str] = {}
    for point in sorted(all_payloads(ctx, collection), key=lambda p: p["payload"]["char_start"]):
        payload = point["payload"]
        text = covered.get(payload["session_id"], "")
        assert payload["char_start"] <= len(text), "stored windows must be a contiguous prefix"
        covered[payload["session_id"]] = text[: payload["char_start"]] + payload["text"]
    eligible = db.eligible()
    unindexed = [row for row in eligible if f"] {row[3].strip()}\n" not in covered.get(row[1], "")]
    newest = max(row[4] for row in eligible)
    seconds = 0.0
    if unindexed:
        oldest = min(unindexed, key=lambda row: row[0])
        before = [row[4] for row in eligible if row[0] < oldest[0]]
        seconds = newest - (max(before) if before else oldest[4])
    return {"messages": len(unindexed), "seconds": round(seconds, 3)}


def test_f_index_lag_is_exact_after_a_partial_run(session_recall_collections, tmp_path, monkeypatch, capsys):
    ctx = session_recall_collections
    db = _small_db(tmp_path / "state.db", random.Random(5))
    home = tmp_path / "home"
    config = recall_config(ctx, ctx.new_collection("lag"), db.path, home, "",
                           session_recall_window_chars=300, session_recall_overlap_chars=40)
    recall = build_session_recall(config, str(home))
    recall.clock = CountingClock()
    report = recall.run(max_seconds=9, batch=1)
    lag = recall.lag()
    oracle = _oracle_lag(ctx, config["session_recall_collection"], db)

    # The operator's view of the same index: `session-recall status --json`.
    home.mkdir(exist_ok=True)
    (home / "qdrant_memory.json").write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    import argparse
    import importlib.util

    from qdrant_memory.cli_core import execute_command

    # By path: `import cli` resolves to Hermes' own cli.py, which is first on sys.path.
    spec = importlib.util.spec_from_file_location("srecall_plugin_cli", Path(__file__).resolve().parents[2] / "cli.py")
    assert spec and spec.loader
    plugin_cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin_cli)
    parser = argparse.ArgumentParser()
    plugin_cli.register_cli(parser)
    capsys.readouterr()
    exit_code = execute_command(parser.parse_args(["session-recall", "status", "--json"]))
    status = json.loads(capsys.readouterr().out)
    ARTIFACT["contracts"]["f"] = {"report": report, "lag": lag, "oracle": oracle, "cli_lag": status["index_lag"]}

    assert report["stopped"] == "time budget" and report["windows_upserted"] > 0, report
    assert 0 < oracle["messages"] < len(db.eligible()), oracle
    assert {"messages": lag["messages"], "seconds": lag["seconds"]} == oracle
    assert exit_code == 0
    assert {"messages": status["index_lag"]["messages"], "seconds": status["index_lag"]["seconds"]} == oracle
    assert status["windows"] == report["windows_upserted"]
