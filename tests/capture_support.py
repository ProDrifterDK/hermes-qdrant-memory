"""Shared fakes for the capture pipeline tests: a stateful Qdrant, a bag-of-words embedder,
a synthetic ``state.db`` built from the host's dumped schema, and a provider wired to them.

Everything here is synthetic text.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA = Path(__file__).resolve().parent / "fixtures" / "state_db_schema.sql"
T0 = 1_794_000_000.0


class MemQdrant:
    """Points kept per collection; retrieve, search (cosine, simple filters), delete, count."""

    def __init__(self) -> None:
        self.points: dict[tuple[str, str], dict[str, Any]] = {}
        self.writes: list[tuple[str, str, list[str]]] = []

    def health(self) -> bool:
        return True

    def get_collections(self) -> list[str]:
        return sorted({name for name, _ in self.points})

    def ensure_collection(self, name, vector_size, distance):
        return {"exists": True, "name": name}

    def upsert(self, name, points):
        for point in points:
            self.points[(name, str(point["id"]))] = copy.deepcopy(point)
        self.writes.append(("upsert", name, [str(p["id"]) for p in points]))
        return {"status": "ok"}

    def retrieve(self, name, ids, with_payload=True, with_vector=False):
        out = []
        for pid in ids:
            point = self.points.get((name, str(pid)))
            if point is None:
                continue
            item = {"id": point["id"]}
            if with_payload:
                item["payload"] = copy.deepcopy(point.get("payload") or {})
            if with_vector:
                item["vector"] = list(point.get("vector") or [])
            out.append(item)
        return out

    @staticmethod
    def _cond(payload: dict[str, Any], cond: dict[str, Any]) -> bool:
        value = payload.get(cond["key"])
        match = cond.get("match") or {}
        if "value" in match:
            return value == match["value"]
        if "any" in match:
            return value in match["any"]
        return False

    def _matches(self, payload: dict[str, Any], flt: dict[str, Any] | None) -> bool:
        if not flt:
            return True
        if not all(self._cond(payload, c) for c in flt.get("must", [])):
            return False
        return not any(self._cond(payload, c) for c in flt.get("must_not", []))

    def search(self, name, vector, limit, filter=None, with_payload=True, with_vector=False):
        hits = []
        for (coll, _pid), point in self.points.items():
            if coll != name or not self._matches(point.get("payload") or {}, filter):
                continue
            hits.append({"id": point["id"], "score": cosine(vector, point.get("vector") or []),
                         "payload": copy.deepcopy(point.get("payload") or {})})
        hits.sort(key=lambda h: -h["score"])
        return hits[: int(limit)]

    def delete_ids(self, name, ids):
        for pid in ids:
            self.points.pop((name, str(pid)), None)
        self.writes.append(("delete", name, [str(i) for i in ids]))
        return {"status": "ok"}

    def scroll_by_filter(self, name, filter, *, limit=256, with_payload=True, with_vector=False, max_total=None):
        return [copy.deepcopy(p) for (coll, _), p in self.points.items() if coll == name
                and self._matches(p.get("payload") or {}, filter)]

    def count(self, name):
        return sum(1 for coll, _ in self.points if coll == name)

    def collection(self, name) -> dict[str, dict[str, Any]]:
        return {pid: p for (coll, pid), p in self.points.items() if coll == name}


def cosine(a, b) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class BagEmbedding:
    """Deterministic bag-of-words vectors: shared words give a high cosine."""

    dims = 64

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for word in re.findall(r"\w+", text.lower()):
            vec[int(hashlib.sha256(word.encode()).hexdigest(), 16) % self.dims] += 1.0
        return vec

    def embed_document(self, text):
        return self._vec(text)

    def embed_query(self, text):
        return self._vec(text)

    def health(self):
        return True


# --- synthetic state.db ---------------------------------------------------------------------


def make_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA.read_text(encoding="utf-8"))
    return conn


def add_session(conn, sid, source="local", *, identity=True, title="", parent=None):
    title = f"title {sid}" if title == "" else title
    conn.execute("INSERT INTO sessions (id, source, user_id, chat_type, origin_json, title, parent_session_id, "
                 "started_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                 (sid, source, "uid:1" if identity else None, "dm" if identity else None,
                  '{"platform": "x"}' if identity else None, title, parent, T0))


def add_message(conn, sid, role, content, ts_offset=0.0, *, tool_calls=None, tool_call_id=None):
    cur = conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp, message_uid, tool_calls, tool_call_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (sid, role, content, T0 + ts_offset, os.urandom(6).hex(),
         json.dumps(tool_calls) if tool_calls is not None else None, tool_call_id))
    return int(cur.lastrowid)


def memory_call(call_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": "memory", "arguments": json.dumps(arguments)}}


# --- provider ------------------------------------------------------------------------------


def make_provider(home: Path, state_db: Path, qdrant: MemQdrant | None = None, *, session_id="approver-session",
                  platform="cli", agent_context="primary", **config):
    from __init__ import QdrantMemoryProvider
    from qdrant_memory.writer import ConversationWriter

    provider = QdrantMemoryProvider()
    provider._hermes_home = str(home)
    provider._session_id = session_id
    provider._platform = platform
    provider._agent_context = agent_context
    provider._qdrant = qdrant if qdrant is not None else MemQdrant()
    provider._embeddings = BagEmbedding()
    provider._active = True
    provider._config.update({
        "collection_name": "memory",
        "learning_collection_name": "learnings",
        "embedding_model": "bag",
        "session_recall_state_db": str(state_db),
        "capture_mode": "on",
        **config,
    })
    provider._writer = ConversationWriter(qdrant=provider._qdrant, embeddings=provider._embeddings,
                                          collection_name="memory", profile_id="default", platform=platform,
                                          session_id=session_id, model="bag")
    return provider


def call(provider, tool: str, **args) -> dict[str, Any]:
    return json.loads(provider.handle_tool_call(tool, args))
