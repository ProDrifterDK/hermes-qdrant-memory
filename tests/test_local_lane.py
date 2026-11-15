"""Local/sovereign profile lane: read-only ``read_sources`` and the ``recall`` tool surface.

The fake below replaces only the HTTP transport (``QdrantClient._send``), so every
request still passes through the real client methods and the real read-only guard in
``QdrantClient._request``. The provider is wired through its real ``initialize``.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import math
import re
import urllib.parse

import pytest

from qdrant_memory.client import QdrantClient, ReadOnlyCollectionError
from qdrant_memory.learning import LearningStore
from qdrant_memory.writer import ConversationWriter

plugin = importlib.import_module("__init__")

OWN_MEMORY = "local_memory"
OWN_LEARNING = "local_learnings"
OWN_SESSIONS = "local_session_windows"
SHARED_MEMORY = "shared_memory"
SHARED_LEARNING = "shared_learnings"
MISSING_SOURCE = "shared_missing"
READ_ONLY = {SHARED_MEMORY, SHARED_LEARNING, MISSING_SOURCE}
RECALL_SIX = {
    "qdrant_memory_context",
    "qdrant_memory_search",
    "qdrant_memory_expand",
    "qdrant_memory_session_recall",
    "qdrant_learning_search",
    "qdrant_memory_status",
}
EXPAND = "qdrant_memory_session_expand"
_READ_POST = {"points", "points/search", "points/scroll", "points/count"}


class TokenEmbeddings:
    def __init__(self, *args, **kwargs):
        pass

    def _embed(self, text):
        vector = [0.0] * 64
        for token in re.findall(r"[a-z0-9_-]+", str(text).casefold()):
            digest = hashlib.sha256(token.encode()).digest()
            vector[int.from_bytes(digest[:4], "big") % 64] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_query(self, text):
        return self._embed(text)

    def embed_document(self, text):
        return self._embed(text)

    def health(self):
        return True


def _matches(payload, condition):
    if "match" not in condition:
        return True  # ranges are not needed by these scenarios
    value = payload.get(condition["key"])
    values = value if isinstance(value, list) else [value]
    match = condition["match"]
    wanted = match["any"] if "any" in match else [match.get("value")]
    return any(item in wanted for item in values)


def _filter_ok(payload, flt):
    flt = flt or {}
    return all(_matches(payload, c) for c in flt.get("must", [])) and not any(
        _matches(payload, c) for c in flt.get("must_not", [])
    )


class InMemoryQdrant(QdrantClient):
    """Real client methods and guard; the transport is an in-memory Qdrant that records requests."""

    def __init__(self, store, log, *args, **kwargs):
        super().__init__("http://in-memory", read_only_collections=kwargs.get("read_only_collections"))
        self.store = store
        self.log = log

    def _send(self, method, path, body=None):
        route = urllib.parse.urlsplit(path).path.strip("/").split("/")
        self.log.append((method, path))
        if route == ["collections"]:
            return {"result": {"collections": [{"name": n} for n in self.store]}}
        name, rest = urllib.parse.unquote(route[1]), "/".join(route[2:])
        if method == "PUT" and not rest:
            self.store[name] = {}
            return {"result": True}
        if name not in self.store:
            raise RuntimeError(f"Qdrant HTTP 404: collection {name} not found")
        points = self.store[name]
        if method == "GET" and not rest:
            return {"result": {"config": {"params": {"vectors": {"size": 64, "distance": "Cosine"}}}}}
        if rest == "index":
            return {"result": True}
        if method == "PUT" and rest == "points":
            for point in body["points"]:
                points[str(point["id"])] = json.loads(json.dumps(point))
            return {"result": True}
        if rest == "points/payload":
            for pid in body["points"]:
                points[str(pid)]["payload"].update(body["payload"])
            return {"result": True}
        if rest == "points/delete":
            for pid in body.get("points", []):
                points.pop(str(pid), None)
            return {"result": True}
        if rest == "points/count":
            return {"result": {"count": len(points)}}
        if rest == "points":
            return {"result": [points[str(i)] for i in body["ids"] if str(i) in points]}
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
        raise AssertionError(f"unhandled fake request {method} {path}")


def _is_mutation(method, path):
    route = urllib.parse.urlsplit(path).path.strip("/").split("/")
    rest = "/".join(route[2:])
    return not (method == "GET" or (method == "POST" and rest in _READ_POST))


def _target(path):
    route = urllib.parse.urlsplit(path).path.strip("/").split("/")
    return urllib.parse.unquote(route[1]) if len(route) > 1 else ""


def _local_provider(monkeypatch, tmp_path, store, log, **overrides):
    section = {
        "collection_name": OWN_MEMORY,
        "learning_collection_name": OWN_LEARNING,
        "session_recall_collection": OWN_SESSIONS,
        "sync_turns": False,
        "vector_size": 64,
        "read_sources": [
            {"collection": SHARED_MEMORY, "profile_id": "default", "kind": "memory"},
            {"collection": MISSING_SOURCE, "profile_id": None, "kind": "memory"},
            {"collection": SHARED_LEARNING, "profile_id": "default", "kind": "learning"},
        ],
        **overrides,
    }
    monkeypatch.setattr("qdrant_memory.config._load_hermes_config", lambda: {"qdrant_memory": section})
    monkeypatch.setattr(plugin, "QdrantClient", lambda *a, **kw: InMemoryQdrant(store, log, *a, **kw))
    monkeypatch.setattr(plugin, "EmbeddingClient", TokenEmbeddings)
    provider = plugin.QdrantMemoryProvider()
    provider.initialize("session-local", hermes_home=str(tmp_path), agent_identity="local27b")
    return provider


def _seed(store, log):
    """Seed through the plugin's own writers on an unguarded client (main's view of the shared store)."""
    admin = InMemoryQdrant(store, log)
    for name in (SHARED_MEMORY, SHARED_LEARNING):
        admin.ensure_collection(name, 64, "Cosine")
    main = ConversationWriter(qdrant=admin, embeddings=TokenEmbeddings(), collection_name=SHARED_MEMORY, profile_id="default")
    main.store_text("The desk band lives in ~/.hermes/deskband and is started by deskband.sh")
    other = ConversationWriter(qdrant=admin, embeddings=TokenEmbeddings(), collection_name=SHARED_MEMORY, profile_id="other")
    other.store_text("Other profile secret note: desk band moved to /srv/elsewhere")
    LearningStore(qdrant=admin, embeddings=TokenEmbeddings(), collection_name=SHARED_LEARNING, profile_id="default").store(
        lesson="Restart the desk band with systemctl --user restart deskband before debugging it",
        learning_type="workflow_lesson",
    )


def test_read_sources_are_never_mutated_by_recall_paths(monkeypatch, tmp_path):
    store, log = {}, []
    _seed(store, log)
    shared_before = json.dumps({n: store[n] for n in (SHARED_MEMORY, SHARED_LEARNING)}, sort_keys=True)
    provider = _local_provider(monkeypatch, tmp_path, store, log)
    own = ConversationWriter(qdrant=provider._qdrant, embeddings=TokenEmbeddings(), collection_name=OWN_MEMORY, profile_id="local27b")
    own_id = own.store_text("Local note: the desk band config file is deskband.toml")
    log.clear()

    legacy = provider.prefetch("where does the desk band live")
    provider._config["auto_recall_mode"] = "hybrid"
    hybrid = provider.prefetch("where does the desk band live")
    context = json.loads(provider.handle_tool_call("qdrant_memory_context", {"topic": "desk band", "template": "source_backed_answer"}))
    search = json.loads(provider.handle_tool_call("qdrant_memory_search", {"query": "desk band"}))
    learning = json.loads(provider.handle_tool_call("qdrant_learning_search", {"query": "restart desk band"}))
    shared_id = next(r["id"] for r in search["results"] if r["source_collection"] == SHARED_MEMORY)
    expanded = json.loads(provider.handle_tool_call("qdrant_memory_expand", {"point_id": shared_id}))

    # Recall reads main's facts, merged and labelled, with main's profile filter applied.
    assert "~/.hermes/deskband" in legacy and "~/.hermes/deskband" in hybrid
    assert "Hybrid" in hybrid
    assert {r["source_collection"] for r in search["results"]} == {OWN_MEMORY, SHARED_MEMORY}
    assert all("elsewhere" not in r["text"] for r in search["results"])
    assert [r["score"] for r in search["results"]] == sorted((r["score"] for r in search["results"]), reverse=True)
    assert {r["source_collection"] for r in learning["results"]} == {SHARED_LEARNING}
    context_labels = {e["source_collection"] for section in context["sections"].values() for e in section}
    assert SHARED_MEMORY in context_labels
    assert expanded["collection_name"] == SHARED_MEMORY and "deskband" in expanded["text"]

    # Zero mutating requests reached any read source; the missing one was never created.
    assert [(m, p) for m, p in log if _target(p) in READ_ONLY and _is_mutation(m, p)] == []
    assert MISSING_SOURCE not in store
    assert json.dumps({n: store[n] for n in (SHARED_MEMORY, SHARED_LEARNING)}, sort_keys=True) == shared_before
    # Own-collection behaviour is unchanged: the search tool still bumps own access counters.
    assert store[OWN_MEMORY][own_id]["payload"].get("access_count") == 1

    # The guard holds for any call site, not just the recall paths above.
    attempts = [
        lambda: provider._qdrant.upsert(SHARED_MEMORY, [{"id": shared_id, "vector": [0.0] * 64, "payload": {}}]),
        lambda: provider._qdrant.update_payload(SHARED_MEMORY, shared_id, {"access_count": 99}),
        lambda: provider._qdrant.delete_ids(SHARED_MEMORY, [shared_id]),
        lambda: provider._qdrant.delete_filter(SHARED_LEARNING, {"must": []}),
        lambda: provider._qdrant.create_payload_index(SHARED_MEMORY, "profile_id", "keyword"),
        lambda: provider._qdrant.ensure_collection(MISSING_SOURCE, 64, "Cosine"),
    ]
    for attempt in attempts:
        with pytest.raises(ReadOnlyCollectionError):
            attempt()
    assert [(m, p) for m, p in log if _target(p) in READ_ONLY and _is_mutation(m, p)] == []
    assert MISSING_SOURCE not in store
    assert json.dumps({n: store[n] for n in (SHARED_MEMORY, SHARED_LEARNING)}, sort_keys=True) == shared_before


def test_merged_recall_ranks_own_and_read_source_hits_on_one_scale(monkeypatch, tmp_path):
    """A stronger hit outranks a weaker one whichever collection holds it.

    Equal importance (10) and a future created_at (recency exactly 1.0) make each
    score equal to the hit's similarity normalized over the merged pool. Per-collection
    normalization gives every collection's best hit 1.0, so a weak own hit would tie
    the strong shared one and beat the shared runner-up, and survive min_final_score.
    """
    query = "alpha beta gamma delta"
    strong = {"best": "alpha beta gamma delta", "hit": "alpha beta gamma zeta", "filler": "kappa lambda"}  # raw 1.0 / 0.75 / 0
    weak = {"hit": "alpha beta zeta eta", "filler": "sigma tau"}  # raw 0.5 / 0
    cases = {
        "memory": (OWN_MEMORY, SHARED_MEMORY, "qdrant_memory_search", {"source_type": "manual"}),
        "learning": (OWN_LEARNING, SHARED_LEARNING, "qdrant_learning_search", {"source_type": "learning"}),
    }
    for read_source_strong in (True, False):  # the reverse case: the own collection holds the strong hits
        for kind, (own_name, shared_name, tool, extra) in cases.items():
            store, log = {}, []
            emb = TokenEmbeddings()
            strong_name, weak_name = (shared_name, own_name) if read_source_strong else (own_name, shared_name)
            for name, texts in ((strong_name, strong), (weak_name, weak)):
                store[name] = {
                    f"{name}-{key}": {
                        "id": f"{name}-{key}",
                        "vector": emb.embed_document(text),
                        "payload": {
                            "text": text,
                            "importance": 10,
                            "created_at": "2099-01-01T00:00:00+00:00",
                            "profile_id": "default" if name == shared_name else "local27b",
                            **extra,
                        },
                    }
                    for key, text in texts.items()
                }
            provider = _local_provider(monkeypatch, tmp_path, store, log)
            out = json.loads(provider.handle_tool_call(tool, {"query": query, "top_k": 3}))["results"]
            assert [r["id"] for r in out] == [f"{strong_name}-best", f"{strong_name}-hit", f"{weak_name}-hit"], (kind, out)
            assert [r["source_collection"] for r in out] == [strong_name, strong_name, weak_name]
            assert out[0]["score"] == 1.0
            # Runner-up and the other collection's best hit: below the pool's best, ordered by similarity.
            assert out[1]["score"] == pytest.approx(0.75) and out[2]["score"] == pytest.approx(0.5)

            # min_final_score cuts on the merged scale: the weak collection's best hit (0.5) is dropped.
            cut = _local_provider(monkeypatch, tmp_path, store, log, min_final_score=0.6)
            out = json.loads(cut.handle_tool_call(tool, {"query": query, "top_k": 3}))["results"]
            assert [r["id"] for r in out] == [f"{strong_name}-best", f"{strong_name}-hit"], (kind, out)


def test_recall_surface_exposes_six_tools_and_refuses_hidden_writes(monkeypatch, tmp_path, caplog):
    store, log = {}, []
    # The opt-in session expand tool is off by default: every surface is exactly what it was.
    default_schemas = [s for s in plugin.TOOL_SCHEMAS if s["name"] != EXPAND]
    full = _local_provider(monkeypatch, tmp_path, store, log, read_sources=[])
    assert full.get_tool_schemas() == default_schemas and len(full.get_tool_schemas()) == 27

    with caplog.at_level(logging.WARNING, logger="qdrant_memory.config"):
        typo = _local_provider(monkeypatch, tmp_path, store, log, read_sources=[], tool_surface="recal")
    assert typo.get_tool_schemas() == default_schemas
    assert "tool_surface" in caplog.text

    recall = _local_provider(monkeypatch, tmp_path, store, log, read_sources=[], tool_surface="recall")
    names = [schema["name"] for schema in recall.get_tool_schemas()]
    assert sorted(names) == sorted(RECALL_SIX)
    assert set(re.findall(r"qdrant_[a-z_]+", recall.system_prompt_block())) <= RECALL_SIX

    # With session_expand_enabled the expand tool joins both surfaces and the prompt block.
    full_x = _local_provider(monkeypatch, tmp_path, store, log, read_sources=[], session_expand_enabled=True)
    assert full_x.get_tool_schemas() == plugin.TOOL_SCHEMAS and len(full_x.get_tool_schemas()) == 28
    recall_x = _local_provider(
        monkeypatch, tmp_path, store, log, read_sources=[], tool_surface="recall", session_expand_enabled=True
    )
    assert sorted(s["name"] for s in recall_x.get_tool_schemas()) == sorted(RECALL_SIX | {EXPAND})
    assert set(re.findall(r"qdrant_[a-z_]+", recall_x.system_prompt_block())) == RECALL_SIX | {EXPAND}

    store_args = {"text": "Local profile decided to cache the desk band path", "dry_run": False, "approve": True}
    log.clear()
    refused = json.loads(recall.handle_tool_call("qdrant_memory_store", store_args))
    assert "tool_surface" in refused["error"] and "recall" in refused["error"]
    assert [(m, p) for m, p in log if _is_mutation(m, p)] == []

    # Control: the same call on the full surface does write, so the refusal above is what stopped it.
    saved = json.loads(full.handle_tool_call("qdrant_memory_store", store_args))
    assert saved["saved"] is True
    assert any(m == "PUT" and _target(p) == OWN_MEMORY for m, p in log)
