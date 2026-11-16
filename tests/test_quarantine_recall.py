"""Quarantined points (``consolidation_quarantined: true``) are hidden from recall.

The watcher's guarded-auto quarantines heading-noise chunks and duplicate copies instead of
deleting them. Quarantine only means something if recall stops showing them.

Failure modes, written before the code:

1. The legacy automatic-recall block (``auto_recall_mode: legacy``, production) shows a
   quarantined point whose text matches the message best.
2. The point is dropped only after the top-k cut, so it eats a slot and the block shrinks
   instead of showing the next memory.
3. A transport that ignores the server-side filter lets the point through (no client check).
4. ``qdrant_memory_search`` (and ``qdrant_memory_context``, same retriever) returns it.
5. The hybrid automatic-recall block shows it.
6. A memory read source (another profile's collection merged into recall) shows its
   quarantined points.
7. ``qdrant_learning_search`` returns a quarantined learning.
8. ``qdrant_memory_graph_search`` resolves an entity to a quarantined memory.
9. ``qdrant_memory_inspect`` can no longer reach the point, or the marker is filtered so
   broadly that a point with ``consolidation_quarantined: false`` disappears.
10. Restoring the point (marker gone, as the snapshot restore writes it back) leaves it hidden.
11. Recall writes to Qdrant while filtering.
"""
from __future__ import annotations

import importlib
import json

tsp = importlib.import_module("test_session_prefetch")
tgr = importlib.import_module("test_graph_retriever")

MEMORY, LEARNING = tsp.MEMORY, tsp.LEARNING
QUERY = "harbour crane schedule berth seven tide window"
HIDDEN = "Harbour crane schedule berth seven tide window QUARANTINEDNOTE"
NEAR = "Harbour crane schedule for berth seven follows the tide window VISIBLEONE"
FAR = "Harbour crane berth seven maintenance log VISIBLETWO"
UNMARKED = "Harbour crane schedule berth seven tide window note FALSEMARKER"


def _point(pid, text, **extra):
    emb = tsp.TokenEmbeddings()
    return {"id": pid, "vector": emb.embed_document(text), "payload": {
        "text": text, "profile_id": "default", "source_type": "manual", "importance": 7,
        "created_at": "2099-01-01T00:00:00+00:00", **extra}}


QUARANTINE = {"consolidation_quarantined": True, "consolidation_quarantined_at": "2099-01-02T00:00:00Z",
              "consolidation_quarantine_reason": "heading_noise"}


class RecallQdrant(tsp.FakeQdrant):
    """The shared in-memory transport plus point retrieval by id (inspect)."""

    def _send(self, method, path, body=None):
        route = path.split("?")[0].strip("/").split("/")
        if method == "POST" and len(route) == 3 and route[2] == "points":
            self.log.append((method, path))
            points = self.store[route[1]]
            return {"result": [points[str(pid)] for pid in body["ids"] if str(pid) in points]}
        return super()._send(method, path, body)


class FilterIgnoringQdrant(RecallQdrant):
    """A transport that drops every ``must_not`` the client sends (search and scroll)."""

    def _send(self, method, path, body=None):
        if isinstance(body, dict) and isinstance(body.get("filter"), dict):
            body = {**body, "filter": {k: v for k, v in body["filter"].items() if k != "must_not"}}
        return super()._send(method, path, body)


def _provider(monkeypatch, tmp_path, *, client=RecallQdrant, **overrides):
    tmp_path.mkdir(parents=True, exist_ok=True)
    store, log = {}, []
    settings = {"auto_recall_top_k": 2, "display_tokens": 4000, **overrides}
    monkeypatch.setattr(tsp, "FakeQdrant", client)  # tsp._provider builds its client from this name
    provider = tsp._provider(monkeypatch, tmp_path, store, log, **settings)
    store.setdefault(MEMORY, {})
    store.setdefault(LEARNING, {})
    store[MEMORY]["hidden"] = _point("hidden", HIDDEN, **QUARANTINE)
    store[MEMORY]["near"] = _point("near", NEAR)
    store[MEMORY]["far"] = _point("far", FAR)
    log.clear()
    return provider, store, log


def _search_tool(provider, query=QUERY, **args):
    return json.loads(provider.handle_tool_call("qdrant_memory_search", {"query": query, "top_k": 5, **args}))


def test_legacy_block_hides_a_quarantined_point_and_backfills_its_slot(monkeypatch, tmp_path):
    # search_candidates equal to top_k: a point filtered only after the search would cost a slot.
    provider, store, log = _provider(monkeypatch, tmp_path, search_candidates=2)
    try:
        assert provider._config["auto_recall_mode"] == "legacy"
        out = provider.prefetch(QUERY, session_id="tip")
        assert "QUARANTINEDNOTE" not in out
        assert "VISIBLEONE" in out and "VISIBLETWO" in out  # top_k=2: both slots still filled
        assert tsp._mutations(log) == []
    finally:
        provider.shutdown()


def test_precondition_without_the_marker_the_point_ranks_first(monkeypatch, tmp_path):
    """Control (and failure mode 10): the same point without the marker is recalled first."""
    provider, store, _ = _provider(monkeypatch, tmp_path)
    try:
        payload = store[MEMORY]["hidden"]["payload"]
        for key in QUARANTINE:
            payload.pop(key)
        out = provider.prefetch(QUERY, session_id="tip")
        assert out.index("QUARANTINEDNOTE") < out.index("VISIBLEONE")
    finally:
        provider.shutdown()


def test_the_client_drops_it_even_when_the_server_ignores_the_filter(monkeypatch, tmp_path):
    provider, _, _ = _provider(monkeypatch, tmp_path, client=FilterIgnoringQdrant, auto_recall_top_k=3)
    try:
        assert isinstance(provider._qdrant, FilterIgnoringQdrant)
        assert "QUARANTINEDNOTE" not in provider.prefetch(QUERY, session_id="tip")
        assert all(h["id"] != "hidden" for h in _search_tool(provider)["results"])
    finally:
        provider.shutdown()


def test_search_tool_hides_it_and_inspect_still_reaches_it(monkeypatch, tmp_path):
    provider, store, log = _provider(monkeypatch, tmp_path)
    store[MEMORY]["unmarked"] = _point("unmarked", UNMARKED, consolidation_quarantined=False)
    try:
        ids = [hit["id"] for hit in _search_tool(provider)["results"]]
        assert "hidden" not in ids and "near" in ids
        assert "unmarked" in ids  # an explicit false marker is not quarantine
        inspected = json.loads(provider.handle_tool_call("qdrant_memory_inspect", {"point_id": "hidden"}))
        assert inspected["found"] is True, inspected
        assert "QUARANTINEDNOTE" in inspected["snippet"]
    finally:
        provider.shutdown()


def test_hybrid_block_hides_it(monkeypatch, tmp_path):
    provider, _, log = _provider(monkeypatch, tmp_path, auto_recall_mode="hybrid", auto_recall_top_k=3)
    try:
        out = provider.prefetch(QUERY, session_id="tip")
        assert out.startswith("# Relevant Long-Term Memory (Hybrid)")
        assert "QUARANTINEDNOTE" not in out and "VISIBLEONE" in out
        assert tsp._mutations(log) == []
    finally:
        provider.shutdown()


def test_a_read_source_quarantined_point_is_hidden(monkeypatch, tmp_path):
    other = "unit_other_memory"
    provider, store, _ = _provider(monkeypatch, tmp_path, auto_recall_top_k=3,
                                   read_sources=[{"collection": other, "kind": "memory"}])
    store[other] = {"other-hidden": _point("other-hidden", HIDDEN.replace("QUARANTINEDNOTE", "OTHERHIDDEN"),
                                           **QUARANTINE)}
    store[MEMORY].pop("hidden")
    try:
        assert provider._memory_recall() is not None  # precondition: the read source is merged
        out = provider.prefetch(QUERY, session_id="tip")
        assert "OTHERHIDDEN" not in out and "VISIBLEONE" in out
    finally:
        provider.shutdown()


def test_learning_search_hides_a_quarantined_learning(monkeypatch, tmp_path):
    provider, store, _ = _provider(monkeypatch, tmp_path)
    learning = {"source_type": "learning", "learning_type": "lesson", "importance": 8}
    store[LEARNING]["l-hidden"] = _point("l-hidden", HIDDEN, **learning, **QUARANTINE)
    store[LEARNING]["l-near"] = _point("l-near", NEAR, **learning)
    try:
        result = json.loads(provider.handle_tool_call("qdrant_learning_search", {"query": QUERY, "top_k": 5}))
        ids = [hit["id"] for hit in result["results"]]
        assert "l-near" in ids and "l-hidden" not in ids
    finally:
        provider.shutdown()


def test_graph_entity_does_not_resolve_to_a_quarantined_memory():
    qdrant = tgr.FakeGraphQdrant(search_results=[])
    qdrant.add_point("entity-1", {
        "memory_kind": "graph_entity", "entity_id": "entity-aaaaaaaaaaaaaaaa", "entity_type": "tool",
        "label": "crane", "source_point_ids": ["mem-1", "mem-q"], "confidence": 0.9, "canonical": False,
        "requires_review": True, "fact_status": "active", "profile_id": "coder"})
    qdrant.add_point("mem-1", {"text": "crane notes", "source_type": "project_doc", "importance": 8,
                               "fact_status": "active", "profile_id": "coder"})
    qdrant.add_point("mem-q", {"text": "## Notes", "source_type": "vault_note", "importance": 3,
                               "fact_status": "active", "profile_id": "coder", **QUARANTINE})
    retriever = tgr.GraphMemoryRetriever(qdrant=qdrant, embeddings=tgr.FakeEmbedding(), collection_name="memory",
                                         scope={"profile_id": "coder"})
    ids = [pid for pid, _ in retriever._resolve_entity_memories("entity-aaaaaaaaaaaaaaaa", None,
                                                                include_fact_history=False)]
    assert ids == ["mem-1"]
