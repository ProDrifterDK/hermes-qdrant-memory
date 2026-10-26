"""The sync_turn background catch-up for session recall, end to end with fake services.

Only the two network boundaries are faked (Qdrant, embeddings). The state.db is a real
SQLite file built from the production schema; the hook, debounce thread, lock,
windowing and cursor are the real code.

Escaped in development: the catch-up built a Qdrant client from the default config
instead of using the provider's, so a non-live test calling ``sync_turn`` created an
(empty) ``hermes_session_windows`` collection on the developer's real Qdrant.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

SCHEMA = Path(__file__).resolve().parent / "fixtures" / "state_db_schema.sql"
COLLECTION = "unit_session_windows"


class RecordingQdrant:
    def __init__(self) -> None:
        self.points: dict[str, dict] = {}
        self.collections: set[str] = set()

    def get_collections(self):
        return sorted(self.collections)

    def ensure_collection(self, name, vector_size, distance):
        if name in self.collections:
            return {"exists": True}
        self.collections.add(name)
        return {"created": True}

    def create_payload_index(self, name, field_name, field_schema):
        return {}

    def collection_vector_size(self, name):
        return 8

    def count(self, name):
        return len(self.points)

    def upsert(self, name, points):
        assert name == COLLECTION, f"session recall wrote outside its collection: {name}"
        for point in points:
            self.points[point["id"]] = point["payload"]

    def delete_filter(self, name, filter):
        assert name == COLLECTION
        return {}


class UnitEmbedding:
    def embed_batch(self, texts):
        return [[1.0] + [0.0] * 7 for _ in texts]


def _provider(tmp_path: Path, monkeypatch):
    from __init__ import QdrantMemoryProvider
    from qdrant_memory.config import load_config

    db = tmp_path / "state.db"
    conn = sqlite3.connect(db)
    conn.executescript(SCHEMA.read_text(encoding="utf-8"))
    conn.execute("INSERT INTO sessions (id, source, started_at) VALUES ('s1', 'cli', 1.7e9)")
    for index in range(6):
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES ('s1', ?, ?, ?)",
            ("user" if index % 2 == 0 else "assistant", f"turn {index} " + "words " * 60, 1.7e9 + index),
        )
    conn.commit()
    conn.close()

    # Any client the lane builds for itself is a bug here: it would reach real services.
    def refuse(*args, **kwargs):
        raise AssertionError("session recall built its own client instead of the provider's")

    monkeypatch.setattr("qdrant_memory.client.QdrantClient.__init__", refuse)
    monkeypatch.setattr("qdrant_memory.embeddings.EmbeddingClient.__init__", refuse)

    provider = QdrantMemoryProvider()
    provider._config = load_config(hermes_home=str(tmp_path), hermes_config={"qdrant_memory": {
        "session_recall_collection": COLLECTION,
        "session_recall_state_db": str(db),
        "vector_size": 8,
        "lineage_lock_dir": str(tmp_path / "locks"),
        "sync_turns": False,
    }})
    provider._active = True
    provider._write_enabled = True
    provider._hermes_home = str(tmp_path)
    provider._qdrant = RecordingQdrant()
    provider._embeddings = UnitEmbedding()
    return provider


def test_sync_turn_catch_up_indexes_through_the_providers_own_clients(tmp_path, monkeypatch):
    provider = _provider(tmp_path, monkeypatch)

    provider.sync_turn("a user turn", "an assistant turn")

    thread = provider._session_recall_thread
    assert thread is not None
    thread.join(timeout=10.0)
    assert provider._qdrant.points, "the background pass wrote nothing"
    assert {payload["session_id"] for payload in provider._qdrant.points.values()} == {"s1"}
    assert (tmp_path / "qdrant_memory" / "session_recall" / f"{COLLECTION}.state.json").is_file()


def test_sync_turn_never_waits_for_a_held_writer_lock(tmp_path, monkeypatch):
    from qdrant_memory.lineage import collection_write_lock

    provider = _provider(tmp_path, monkeypatch)
    # The lock the CLI indexer would hold, in the directory the provider resolved
    # (tests/conftest.py pins it for the session through the environment override).
    with collection_write_lock(
        collection_name=f"session-recall:{COLLECTION}", timeout=1.0, lock_dir=provider._config["lineage_lock_dir"]
    ):
        started = time.monotonic()
        provider.sync_turn("a user turn", "an assistant turn")
        returned_after = time.monotonic() - started
        thread = provider._session_recall_thread
        assert thread is not None
        thread.join(timeout=10.0)
        assert not thread.is_alive(), "the background pass must give up on a held lock, not wait"

    assert returned_after < 0.5
    assert provider._qdrant.points == {}
