from __future__ import annotations

import json


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


def _provider_with_auto_extract():
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
            "learning_auto_extract_enabled": True,
            "learning_auto_extract_mode": "preview",
            "learning_auto_extract_min_confidence": 0.8,
            "learning_auto_extract_max_candidates_per_session": 3,
            "learning_auto_extract_require_evidence": True,
            "learning_auto_extract_semantic_dedupe_enabled": True,
            "learning_auto_extract_semantic_dedupe_threshold": 0.9,
            "learning_auto_extract_semantic_dedupe_top_k": 3,
        }
    )
    return provider


def test_on_pre_compress_preview_collects_candidates_without_upsert():
    provider = _provider_with_auto_extract()
    messages = [
        {"role": "tool", "name": "terminal", "content": "pytest: command not found"},
        {"role": "assistant", "content": "Correction: use <hermes-venv>/bin/python -m pytest tests -q."},
    ]

    block = provider.on_pre_compress(messages)

    assert "Qdrant Learning Candidates" in block
    assert "not stored" in block
    assert provider._qdrant.upserts == []
    preview = json.loads(provider.handle_tool_call("qdrant_learning_preview", {}))
    assert preview["count"] == 1
    assert preview["candidates"][0]["learning_type"] == "tool_failure_lesson"


def test_on_session_end_preview_collects_candidates_without_upsert():
    provider = _provider_with_auto_extract()
    messages = [
        {"role": "user", "content": "Actually, my surname is Gárate, not Garate."},
        {"role": "assistant", "content": "I will use Gárate."},
    ]

    provider.on_session_end(messages)

    assert provider._qdrant.upserts == []
    preview = json.loads(provider.handle_tool_call("qdrant_learning_preview", {"include_metadata": True}))
    assert preview["count"] == 1
    assert preview["candidates"][0]["learning_type"] == "user_correction"
    assert preview["candidates"][0]["metadata"]["source_hook"] == "on_session_end"


def test_auto_extract_disabled_is_noop():
    provider = _provider_with_auto_extract()
    provider._config["learning_auto_extract_enabled"] = False

    block = provider.on_pre_compress([
        {"role": "user", "content": "Actually, use OpenAI Codex provider, not OpenRouter."},
        {"role": "assistant", "content": "Corrected."},
    ])

    assert block == ""
    preview = json.loads(provider.handle_tool_call("qdrant_learning_preview", {}))
    assert preview["count"] == 0


def test_learning_approve_dry_run_default_does_not_upsert():
    provider = _provider_with_auto_extract()
    provider.on_session_end([
        {"role": "user", "content": "Actually, my surname is Gárate, not Garate."},
        {"role": "assistant", "content": "I will use Gárate."},
    ])
    candidate_id = json.loads(provider.handle_tool_call("qdrant_learning_preview", {}))["candidates"][0]["candidate_id"]

    approved = json.loads(provider.handle_tool_call("qdrant_learning_approve", {"candidate_id": candidate_id}))

    assert approved["dry_run"] is True
    assert approved["saved"] is False
    assert provider._qdrant.upserts == []


def test_learning_approve_live_stores_to_learning_collection_only():
    provider = _provider_with_auto_extract()
    provider.on_session_end([
        {"role": "user", "content": "Actually, my surname is Gárate, not Garate."},
        {"role": "assistant", "content": "I will use Gárate."},
    ])
    candidate_id = json.loads(provider.handle_tool_call("qdrant_learning_preview", {}))["candidates"][0]["candidate_id"]

    approved = json.loads(provider.handle_tool_call("qdrant_learning_approve", {"candidate_id": candidate_id, "dry_run": False}))

    assert approved["dry_run"] is False
    assert approved["saved"] is True
    assert provider._qdrant.upserts[0][0] == "learnings"


def test_on_session_switch_reset_clears_pending_learning_candidates():
    provider = _provider_with_auto_extract()
    provider.on_session_end([
        {"role": "user", "content": "Actually, use OpenAI Codex provider, not OpenRouter."},
        {"role": "assistant", "content": "Corrected."},
    ])

    provider.on_session_switch("new-session", reset=True)

    preview = json.loads(provider.handle_tool_call("qdrant_learning_preview", {}))
    assert preview["count"] == 0


def test_pending_candidate_cap_is_enforced_across_multiple_hooks():
    provider = _provider_with_auto_extract()
    provider._config["learning_auto_extract_max_candidates_per_session"] = 2

    provider.on_session_end([
        {"role": "user", "content": "Actually, use OpenAI Codex provider, not OpenRouter."},
        {"role": "assistant", "content": "Corrected."},
    ])
    provider.on_pre_compress([
        {"role": "user", "content": "Actually, use Gárate, not Garate."},
        {"role": "assistant", "content": "Corrected."},
        {"role": "tool", "name": "terminal", "content": "pytest: command not found"},
        {"role": "assistant", "content": "Correction: use venv/bin/python -m pytest."},
    ])

    preview = json.loads(provider.handle_tool_call("qdrant_learning_preview", {}))
    assert preview["count"] == 2


def test_semantic_duplicate_existing_learning_is_not_added_to_pending_preview():
    provider = _provider_with_auto_extract()
    provider._qdrant.search_results = [
        {
            "id": "existing-learning",
            "score": 0.95,
            "payload": {
                "text": "User correction: use Gárate, not Garate.",
                "source_type": "learning",
                "learning_type": "user_correction",
            },
        }
    ]

    provider.on_session_end([
        {"role": "user", "content": "Actually, spell my surname Gárate, not Garate."},
        {"role": "assistant", "content": "Corrected."},
    ])

    preview = json.loads(provider.handle_tool_call("qdrant_learning_preview", {}))
    assert preview["count"] == 0
    assert provider._qdrant.upserts == []
    assert provider._qdrant.searches[0][0] == "learnings"
    filter_payload = provider._qdrant.searches[0][3]
    assert {"key": "source_type", "match": {"value": "learning"}} in filter_payload["must"]
    assert {"key": "learning_type", "match": {"value": "user_correction"}} in filter_payload["must"]


def test_semantic_dedupe_failure_fails_open_and_keeps_candidate_pending():
    provider = _provider_with_auto_extract()
    provider._qdrant.raise_on_search = True

    provider.on_session_end([
        {"role": "user", "content": "Actually, spell my surname Gárate, not Garate."},
        {"role": "assistant", "content": "Corrected."},
    ])

    preview = json.loads(provider.handle_tool_call("qdrant_learning_preview", {}))
    assert preview["count"] == 1
    assert provider._qdrant.upserts == []


def test_semantic_dedupe_disabled_keeps_candidate_pending_without_search():
    provider = _provider_with_auto_extract()
    provider._config["learning_auto_extract_semantic_dedupe_enabled"] = False
    provider._qdrant.search_results = [{"id": "existing-learning", "score": 0.99, "payload": {"source_type": "learning"}}]

    provider.on_session_end([
        {"role": "user", "content": "Actually, spell my surname Gárate, not Garate."},
        {"role": "assistant", "content": "Corrected."},
    ])

    preview = json.loads(provider.handle_tool_call("qdrant_learning_preview", {}))
    assert preview["count"] == 1
    assert provider._qdrant.searches == []


def test_on_pre_compress_does_not_return_secret_bearing_candidate_text():
    provider = _provider_with_auto_extract()

    secret_like = " ".join([
        "Authorization:",
        "Bearer",
        "".join(["ghp", "_", "1234567890abcdef", "1234567890abcdef", "123456"]),
    ])
    block = provider.on_pre_compress([
        {"role": "user", "content": f"Actually, use {secret_like} instead."},
        {"role": "assistant", "content": "Corrected."},
    ])

    assert block == ""
    assert "ghp_" not in block
