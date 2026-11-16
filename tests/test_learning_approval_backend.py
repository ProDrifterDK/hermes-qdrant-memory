"""The learning approval backend (``approve_learning_candidate``) kept after capture CP1.

The heuristic that produced learning candidates in the session hooks, and the in-memory queue
the preview/approve tools read, were removed; the gated write path stays for the CP4 queue.
"""
from __future__ import annotations

import json

from qdrant_memory.lesson_extractor import LearningCandidate


class FakeEmbedding:
    def __init__(self):
        self.documents = []
        self.queries = []

    def embed_document(self, text):
        self.documents.append(text)
        return [0.7, 0.8]

    def embed_query(self, text):
        self.queries.append(text)
        return [0.1, 0.2]

    def health(self):
        return True


class FakeQdrant:
    def __init__(self):
        self.upserts = []
        self.searches = []
        self.payload_updates = []
        self.search_results = []
        self.raise_on_search = False
        self.counts = {"memory": 3, "learnings": 2}
        self.collections = ["memory", "learnings"]
        self.health_ok = True

    def upsert(self, name, points):
        self.upserts.append((name, points))
        return {"status": "ok"}

    def retrieve(self, name, ids, with_payload=True, with_vector=False):
        return []  # no stored points: every store is a create

    def search(self, name, vector, limit, filter=None, with_payload=True, with_vector=False):
        self.searches.append((name, vector, limit, filter, with_payload, with_vector))
        if self.raise_on_search:
            raise RuntimeError("search unavailable")
        return self.search_results

    def update_payload(self, name, point_id, payload):
        self.payload_updates.append((name, point_id, payload))
        return {"status": "ok"}

    def health(self):
        return self.health_ok

    def get_collections(self):
        return self.collections

    def count(self, name):
        return self.counts[name]


def _provider():
    from __init__ import QdrantMemoryProvider

    provider = QdrantMemoryProvider()
    provider._qdrant = FakeQdrant()
    provider._embeddings = FakeEmbedding()
    provider._active = True
    provider._config.update(
        {
            "collection_name": "memory",
            "learning_collection_name": "learnings",
            "qdrant_url": "http://qdrant",
            "embedding_url": "http://embed/v1",
            "embedding_model": "bge-m3",
            "vector_size": 1024,
            "auto_recall": True,
            "sync_turns": True,
            "search_candidates": 20,
            "decay_rate": 0.001,
            "min_raw_score": 0.0,
            "min_final_score": 0.0,
            "learning_enabled": True,
        }
    )
    return provider


def _candidate(**overrides):
    fields = {
        "lesson": "Use the Hermes venv to run pytest.",
        "learning_type": "environment_quirk",
        "trigger": "pytest: command not found",
        "correction": "run venv/bin/python -m pytest",
        "evidence": "Synthetic candidate for the approval backend.",
        "confidence": 0.9,
        "importance": 8,
    }
    fields.update(overrides)
    return LearningCandidate(**fields)


def test_learning_approve_dry_run_default_does_not_upsert():
    provider = _provider()

    approved = json.loads(provider.approve_learning_candidate(_candidate()))

    assert approved["dry_run"] is True
    assert approved["saved"] is False
    assert approved["write_decision"]["decision"] == "learning_candidate"
    assert provider._qdrant.upserts == []


def test_learning_approve_live_stores_to_learning_collection_only():
    provider = _provider()

    approved = json.loads(provider.approve_learning_candidate(_candidate(), dry_run=False))

    assert approved["dry_run"] is False
    assert approved["saved"] is True
    assert approved["write_decision"]["decision"] == "learning_candidate"
    assert [name for name, _ in provider._qdrant.upserts] == ["learnings"]


def test_learning_approve_rejects_secret_in_persisted_non_lesson_field_before_upsert():
    provider = _provider()
    secret_value = "".join(["abc", "def", "ghi", "jkl", "mnop"])
    candidate = _candidate(learning_type="user_correction", trigger="Authorization: " + "Bearer " + secret_value)

    approved = json.loads(provider.approve_learning_candidate(candidate, dry_run=False))

    assert "error" in approved
    assert "rejected" in approved["error"]
    assert provider._qdrant.upserts == []


def test_learning_approve_respects_learning_enabled_false():
    provider = _provider()
    provider._config["learning_enabled"] = False

    approved = json.loads(provider.approve_learning_candidate(_candidate(), dry_run=False))

    assert "disabled" in approved["error"]
    assert provider._qdrant.upserts == []
