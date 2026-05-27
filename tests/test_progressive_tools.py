from __future__ import annotations

import hashlib
import json


def _hash_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


class FakeQdrant:
    def __init__(self, points):
        self.points = {str(point["id"]): point for point in points}
        self.retrieve_calls = []
        self.search_calls = []
        self.upsert_calls = []
        self.delete_ids_calls = []
        self.delete_filter_calls = []
        self.update_payload_calls = []
        self.scroll_by_filter_calls = []

    def retrieve(self, name, ids, *, with_payload=True, with_vector=False):
        self.retrieve_calls.append(
            {"name": name, "ids": list(ids), "with_payload": with_payload, "with_vector": with_vector}
        )
        return [self.points[str(point_id)] for point_id in ids if str(point_id) in self.points]

    def search(self, *args, **kwargs):  # pragma: no cover - progressive disclosure must not use semantic search
        self.search_calls.append((args, kwargs))
        return []

    def upsert(self, *args, **kwargs):  # pragma: no cover - read-only tools must not mutate
        self.upsert_calls.append((args, kwargs))

    def delete_ids(self, *args, **kwargs):  # pragma: no cover - read-only tools must not mutate
        self.delete_ids_calls.append((args, kwargs))

    def delete_filter(self, *args, **kwargs):  # pragma: no cover - read-only tools must not mutate
        self.delete_filter_calls.append((args, kwargs))

    def update_payload(self, *args, **kwargs):  # pragma: no cover - read-only tools must not mutate
        self.update_payload_calls.append((args, kwargs))

    def scroll_by_filter(self, *args, **kwargs):  # pragma: no cover - downstream is explicitly unsupported for now
        self.scroll_by_filter_calls.append((args, kwargs))
        return []


def _provider_with_points(points):
    from __init__ import QdrantMemoryProvider

    provider = QdrantMemoryProvider()
    provider._qdrant = FakeQdrant(points)
    provider._config.update({"collection_name": "memory_test", "learning_collection_name": "learning_test"})
    return provider


def test_progressive_inspect_trace_expand_and_source_status_are_exact_read_only(tmp_path):
    source = tmp_path / "source.md"
    source.write_text("title\nalpha source\nbeta source\ngamma\n", encoding="utf-8")
    excerpt_text = "alpha source\nbeta source"
    child = {
        "id": "child",
        "payload": {
            "text": "compact child chunk",
            "source_type": "indexed_file",
            "source_uri": source.as_uri(),
            "locator": {"line_start": 2, "line_end": 3},
            "content_hash": _hash_text(excerpt_text),
            "derivation_type": "indexed_chunk",
            "derived_from": [
                {"source_uri": "memory://point/root", "derivation_type": "summary"},
                {"source_uri": "session://example-session", "locator": {"message_id": "m1"}},
            ],
        },
        "vector": [0.1, 0.2],
    }
    root = {"id": "root", "payload": {"text": "root memory text", "source_type": "manual"}}
    provider = _provider_with_points([child, root])

    inspected = json.loads(provider.handle_tool_call("qdrant_memory_inspect", {"point_id": "child"}))
    traced = json.loads(provider.handle_tool_call("qdrant_memory_trace", {"point_id": "child", "direction": "both"}))
    expanded = json.loads(provider.handle_tool_call("qdrant_memory_expand", {"point_id": "child", "max_chars": 100}))
    status = json.loads(provider.handle_tool_call("qdrant_memory_source_status", {"point_id": "child"}))

    assert inspected["found"] is True
    assert inspected["point_id"] == "child"
    assert "payload" not in inspected
    assert inspected["source"]["source_uri"] == source.as_uri()
    assert inspected["source"]["derivation_type"] == "indexed_chunk"
    assert traced["direction"] == "both"
    assert [edge["source_uri"] for edge in traced["upstream"]] == ["memory://point/root", "session://example-session"]
    assert traced["upstream"][0]["status"] == "exists"
    assert traced["upstream"][0]["point_id"] == "root"
    assert traced["upstream"][1]["status"] == "unsupported"
    assert traced["downstream"]["supported"] is False
    assert expanded["status"] == "exists"
    assert expanded["text"] == "alpha source\nbeta source\n"
    assert status["status"] == "exists"
    assert status["changed"] is False
    assert provider._qdrant.search_calls == []
    assert provider._qdrant.upsert_calls == []
    assert provider._qdrant.delete_ids_calls == []
    assert provider._qdrant.delete_filter_calls == []
    assert provider._qdrant.update_payload_calls == []

    source.write_text("title\nalpha source\nchanged\ngamma\n", encoding="utf-8")
    changed = json.loads(provider.handle_tool_call("qdrant_memory_source_status", {"point_id": "child"}))
    assert changed["status"] == "changed"
    assert changed["changed"] is True


def test_progressive_tools_handle_legacy_point_without_source_uri():
    provider = _provider_with_points([{"id": "legacy", "payload": {"text": "legacy point text", "source_type": "manual"}}])

    inspected = json.loads(provider.handle_tool_call("qdrant_memory_inspect", {"point_id": "legacy"}))
    expanded = json.loads(provider.handle_tool_call("qdrant_memory_expand", {"point_id": "legacy", "max_chars": 6}))
    status = json.loads(provider.handle_tool_call("qdrant_memory_source_status", {"point_id": "legacy"}))

    assert inspected["found"] is True
    assert "payload" not in inspected
    assert inspected["snippet"] == "legacy point text"
    assert expanded["status"] == "unknown"
    assert expanded["fallback"] == "point_text"
    assert expanded["text"] == "legacy"
    assert expanded["truncated"] is True
    assert status["status"] == "unknown"
    assert status["reason"] == "missing_source_uri"


def test_qdrant_memory_expand_omits_untrusted_nested_derived_from_metadata():
    text_sentinel = "DERIVED_FROM_TEXT_SHOULD_NOT_LEAK"
    content_sentinel = "DERIVED_FROM_CONTENT_SHOULD_NOT_LEAK"
    lesson_sentinel = "DERIVED_FROM_LESSON_SHOULD_NOT_LEAK"
    secret_sentinel = "DERIVED_FROM_SECRET_SHOULD_NOT_LEAK"
    provider = _provider_with_points(
        [
            {
                "id": "leaky",
                "payload": {
                    "text": "abcdef",
                    "source_type": "manual",
                    "source_uri": "memory://point/leaky",
                    "derived_from": [
                        {
                            "source_uri": "memory://point/parent",
                            "derivation_type": "summary",
                            "text": text_sentinel * 50,
                            "content": content_sentinel * 50,
                            "lesson": lesson_sentinel * 50,
                            "secrets": {"token": secret_sentinel},
                        }
                    ],
                },
            }
        ]
    )

    expanded = json.loads(provider.handle_tool_call("qdrant_memory_expand", {"point_id": "leaky", "max_chars": 2}))
    serialized = json.dumps(expanded, sort_keys=True)

    assert expanded["found"] is True
    assert expanded["text"] == "ab"
    assert expanded["chars"] == 2
    assert expanded["truncated"] is True
    assert "derived_from" not in expanded.get("source", {})
    assert text_sentinel not in serialized
    assert content_sentinel not in serialized
    assert lesson_sentinel not in serialized
    assert secret_sentinel not in serialized
    assert "\"content\"" not in serialized
    assert "\"lesson\"" not in serialized
    assert len(serialized) < 700


def test_progressive_inspect_missing_point_returns_tool_error():
    provider = _provider_with_points([])

    missing = json.loads(provider.handle_tool_call("qdrant_memory_inspect", {"point_id": "missing"}))

    assert "error" in missing
    assert "not found" in missing["error"].lower()


def test_progressive_missing_point_errors_do_not_echo_long_point_id():
    provider = _provider_with_points([])
    long_point_id = "p" + "x" * 4999

    for tool_name in (
        "qdrant_memory_inspect",
        "qdrant_memory_trace",
        "qdrant_memory_source_status",
        "qdrant_memory_expand",
    ):
        payload = json.loads(provider.handle_tool_call(tool_name, {"point_id": long_point_id}))
        serialized = json.dumps(payload, sort_keys=True)

        assert "error" in payload
        assert "not found" in payload["error"].lower()
        assert long_point_id not in serialized
        assert len(serialized) < 1000
