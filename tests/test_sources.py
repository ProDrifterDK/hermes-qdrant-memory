from __future__ import annotations

import hashlib
import json

from qdrant_memory.sources import (
    FileSourceResolver,
    MemoryPointResolver,
    SourceResolverRegistry,
    expand_point,
    expand_source_metadata,
    inspect_point,
    retrieve_point,
    source_status_for_point,
    trace_point,
)


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_registry_returns_structured_unsupported_for_unknown_scheme():
    registry = SourceResolverRegistry([FileSourceResolver()])

    expansion = registry.expand("session://2026-05-25/example", max_chars=100)
    status = registry.stat("skill://example/readme")

    assert expansion["supported"] is False
    assert expansion["status"] == "unsupported"
    assert expansion["reason"] == "unsupported_scheme"
    assert expansion["scheme"] == "session"
    assert "session" in expansion["message"]
    assert status["supported"] is False
    assert status["status"] == "unsupported"
    assert status["scheme"] == "skill"


def test_registry_expand_compacts_long_unsupported_source_uri():
    registry = SourceResolverRegistry([FileSourceResolver()])
    source_uri = "session://" + "x" * 5000

    expansion = registry.expand(source_uri, max_chars=1)
    serialized = json.dumps(expansion, sort_keys=True)

    assert expansion["supported"] is False
    assert expansion["status"] == "unsupported"
    assert expansion["reason"] == "unsupported_scheme"
    assert expansion["text"] == ""
    assert len(expansion["source_uri"]) <= 512
    assert len(serialized) < 2000


def test_file_resolver_expands_line_range_and_enforces_max_chars(tmp_path):
    source = tmp_path / "note.md"
    source.write_text("one\ntwo\nthree\nfour\nfive\n", encoding="utf-8")
    resolver = FileSourceResolver()

    expanded = resolver.expand(source.as_uri(), {"line_start": 2, "line_end": 4}, max_chars=100)

    assert expanded["supported"] is True
    assert expanded["status"] == "exists"
    assert expanded["text"] == "two\nthree\nfour\n"
    assert expanded["locator"]["line_start"] == 2
    assert expanded["locator"]["line_end"] == 4
    assert expanded["truncated"] is False

    bounded = resolver.expand(source.as_uri(), {"line_start": 2, "line_end": 4}, max_chars=7)

    assert bounded["text"] == "two\nthr"
    assert bounded["truncated"] is True
    assert bounded["max_chars"] == 7


def test_file_resolver_expand_sanitizes_untrusted_locator_metadata(tmp_path):
    sentinel = "EXPAND_LOCATOR_SENTINEL_SHOULD_NOT_LEAK"
    source = tmp_path / "note.md"
    source.write_text("abcdef\n", encoding="utf-8")
    resolver = FileSourceResolver()
    locator = {
        "line_start": 1,
        "line_end": 1,
        "heading": "h" * 250 + sentinel,
        "arbitrary": "x" * 1000 + sentinel,
        "nested": {"secret": sentinel},
        "items": [sentinel],
    }

    expanded = resolver.expand(source.as_uri(), locator, max_chars=2)
    serialized = json.dumps(expanded, sort_keys=True)

    assert expanded["text"] == "ab"
    assert expanded["chars"] == 2
    assert expanded["truncated"] is True
    assert len(expanded["text"]) <= expanded["max_chars"]
    assert expanded["locator"]["line_start"] == 1
    assert expanded["locator"]["line_end"] == 1
    assert "path" not in expanded
    assert sentinel not in serialized
    assert "arbitrary" not in serialized
    assert "nested" not in serialized
    assert "items" not in serialized
    assert len(serialized) < 2000


def test_file_resolver_rejects_malformed_relative_and_dot_segment_file_uris():
    resolver = FileSourceResolver()
    cases = [
        ("file://[bad", "malformed_file_uri"),
        ("file:relative.txt", "relative_file_uri"),
        ("file://localhost/../../etc/passwd", "unsafe_file_uri"),
    ]

    for source_uri, reason in cases:
        expanded = resolver.expand(source_uri, max_chars=10)
        status = resolver.stat(source_uri)

        assert expanded["supported"] is False
        assert expanded["status"] == "unsupported"
        assert expanded["reason"] == reason
        assert expanded["text"] == ""
        assert expanded["chars"] == 0
        assert status["supported"] is False
        assert status["status"] == "unsupported"
        assert status["reason"] == reason


def test_file_resolver_expand_compacts_long_unsupported_source_uri():
    resolver = FileSourceResolver()
    source_uri = "file://example.com/" + "x" * 5000

    expanded = resolver.expand(source_uri, max_chars=1)
    serialized = json.dumps(expanded, sort_keys=True)

    assert expanded["supported"] is False
    assert expanded["status"] == "unsupported"
    assert expanded["reason"] == "non_local_file_uri"
    assert expanded["text"] == ""
    assert len(expanded["source_uri"]) <= 512
    assert len(serialized) < 2000


def test_file_resolver_keeps_absolute_localhost_file_uri_working(tmp_path):
    source = tmp_path / "note.txt"
    source.write_text("localhost file uri", encoding="utf-8")
    resolver = FileSourceResolver()
    source_uri = f"file://localhost{source.as_posix()}"

    expanded = resolver.expand(source_uri, max_chars=100)

    assert expanded["supported"] is True
    assert expanded["status"] == "exists"
    assert expanded["text"] == "localhost file uri"


def test_file_resolver_refuses_too_large_locator_excerpt_without_reading_huge_line(tmp_path):
    source = tmp_path / "huge.txt"
    source.write_text("x" * 20_000, encoding="utf-8")
    resolver = FileSourceResolver(max_file_bytes=128)

    expanded = resolver.expand(source.as_uri(), {"line_start": 1, "line_end": 1}, max_chars=10)

    assert expanded["supported"] is False
    assert expanded["status"] == "unsupported"
    assert expanded["reason"] == "source_too_large"
    assert expanded["text"] == ""
    assert expanded["chars"] == 0
    assert expanded["max_chars"] == 10
    assert expanded["file_size"] == source.stat().st_size


def test_file_resolver_legacy_file_source_without_verification_metadata_is_unknown(tmp_path):
    source = tmp_path / "legacy.txt"
    source.write_text("legacy file text", encoding="utf-8")
    resolver = FileSourceResolver()

    status = resolver.stat(source.as_uri())

    assert status["supported"] is True
    assert status["exists"] is True
    assert status["status"] == "unknown"
    assert status["changed"] is None
    assert status["reason"] == "missing_verification_metadata"


def test_file_resolver_reports_missing_existing_and_changed_sources(tmp_path):
    source = tmp_path / "note.txt"
    source.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    resolver = FileSourceResolver()
    locator = {"line_start": 2, "line_end": 2}
    expected_hash = _sha256_text("beta")

    current = resolver.stat(source.as_uri(), locator, content_hash=expected_hash)

    assert current["supported"] is True
    assert current["status"] == "exists"
    assert current["exists"] is True
    assert current["changed"] is False
    assert current["actual_hash"] == expected_hash

    source.write_text("alpha\nchanged\ngamma\n", encoding="utf-8")
    changed = resolver.stat(source.as_uri(), locator, content_hash=expected_hash)

    assert changed["status"] == "changed"
    assert changed["exists"] is True
    assert changed["changed"] is True
    assert changed["actual_hash"] != expected_hash

    missing = resolver.stat((tmp_path / "missing.txt").as_uri(), locator)
    assert missing["supported"] is True
    assert missing["status"] == "missing"
    assert missing["exists"] is False


def test_file_resolver_refuses_binary_sources(tmp_path):
    source = tmp_path / "blob.bin"
    source.write_bytes(b"abc\x00def")
    resolver = FileSourceResolver()

    expanded = resolver.expand(source.as_uri(), max_chars=100)
    status = resolver.stat(source.as_uri())

    assert expanded["supported"] is False
    assert expanded["status"] == "unsupported"
    assert expanded["reason"] == "binary_source"
    assert expanded["text"] == ""
    assert status["supported"] is False
    assert status["status"] == "unsupported"
    assert status["exists"] is True


def test_memory_resolver_uses_exact_lookup_and_does_not_recurse():
    calls: list[str] = []
    points = {
        "point-1": {
            "id": "point-1",
            "payload": {
                "text": "alpha memory text",
                "source_uri": "memory://point/point-1",
                "derived_from": [{"source_uri": "memory://point/other"}],
            },
        }
    }

    def lookup(point_id: str):
        calls.append(point_id)
        return points.get(point_id)

    resolver = MemoryPointResolver(lookup)

    expanded = resolver.expand("memory://point/point-1", max_chars=5)
    status = resolver.stat("memory://point/missing")

    assert calls == ["point-1", "missing"]
    assert expanded["supported"] is True
    assert expanded["status"] == "exists"
    assert expanded["point_id"] == "point-1"
    assert expanded["text"] == "alpha"
    assert expanded["truncated"] is True
    assert expanded["source"]["source_uri"] == "memory://point/point-1"
    assert "payload" not in expanded
    assert status["status"] == "missing"
    assert status["exists"] is False


def test_memory_resolver_expand_does_not_return_unbounded_payload_text():
    secret_tail = "UNBOUNDED_PAYLOAD_TEXT_SHOULD_NOT_LEAK"
    full_text = "0123456789" + secret_tail
    resolver = MemoryPointResolver(
        lambda point_id: {
            "id": point_id,
            "payload": {
                "text": full_text,
                "lesson": full_text,
                "content": full_text,
                "source_uri": f"memory://point/{point_id}",
                "source_type": "manual",
            },
        }
    )

    expanded = resolver.expand("memory://point/point-1", max_chars=10)
    serialized = json.dumps(expanded, sort_keys=True)

    assert expanded["text"] == "0123456789"
    assert expanded["chars"] == 10
    assert expanded["truncated"] is True
    assert "payload" not in expanded
    assert secret_tail not in serialized
    assert "lesson" not in serialized
    assert "content" not in serialized


def test_memory_resolver_expand_sanitizes_untrusted_source_metadata():
    sentinel = "EXPAND_METADATA_SENTINEL_SHOULD_NOT_LEAK"
    huge_with_sentinel = "x" * 700 + sentinel
    resolver = MemoryPointResolver(
        lambda point_id: {
            "id": point_id,
            "payload": {
                "text": "abcdef",
                "source_uri": f"memory://point/{point_id}",
                "source_type": huge_with_sentinel,
                "content_hash": huge_with_sentinel,
                "file_path": huge_with_sentinel,
                "file_sha256": huge_with_sentinel,
                "chunk_hash": huge_with_sentinel,
                "locator": {
                    "line_start": 1,
                    "line_end": 1,
                    "heading": "h" * 250 + sentinel,
                    "arbitrary": huge_with_sentinel,
                    "nested": {"secret": sentinel},
                },
                "derived_from": [{"source_uri": "memory://point/parent", "text": huge_with_sentinel}],
            },
        }
    )

    expanded = resolver.expand("memory://point/point-1", max_chars=2)
    serialized = json.dumps(expanded, sort_keys=True)

    assert expanded["text"] == "ab"
    assert expanded["chars"] == 2
    assert expanded["truncated"] is True
    assert sentinel not in serialized
    assert "file_path" not in serialized
    assert "file_sha256" not in serialized
    assert "chunk_hash" not in serialized
    assert "derived_from" not in serialized
    assert "arbitrary" not in serialized
    assert "nested" not in serialized
    assert len(serialized) < 2000


def test_memory_resolver_expand_omits_untrusted_nested_derived_from_metadata():
    text_sentinel = "DERIVED_FROM_TEXT_SHOULD_NOT_LEAK"
    content_sentinel = "DERIVED_FROM_CONTENT_SHOULD_NOT_LEAK"
    lesson_sentinel = "DERIVED_FROM_LESSON_SHOULD_NOT_LEAK"
    secret_sentinel = "DERIVED_FROM_SECRET_SHOULD_NOT_LEAK"
    resolver = MemoryPointResolver(
        lambda point_id: {
            "id": point_id,
            "payload": {
                "text": "abcdef",
                "source_uri": f"memory://point/{point_id}",
                "source_type": "manual",
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
    )

    expanded = resolver.expand("memory://point/point-1", max_chars=2)
    serialized = json.dumps(expanded, sort_keys=True)

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
    assert len(serialized) < 600


def test_expand_source_metadata_omits_absent_optional_values():
    metadata = expand_source_metadata({"source_type": "manual"})

    assert metadata == {"source_type": "manual"}
    assert "source_uri" not in metadata
    assert "source_modified_at" not in metadata


class _MismatchedQdrant:
    def retrieve(self, collection_name, ids, *, with_payload=True, with_vector=False):
        return [
            {
                "id": "other-point",
                "payload": {
                    "text": "wrong point text",
                    "source_uri": "memory://point/other-point",
                },
            }
        ]


def test_retrieve_point_rejects_mismatched_returned_ids():
    point = retrieve_point(_MismatchedQdrant(), "memory", "requested-point")

    assert point is None


def test_expand_point_treats_mismatched_retrieve_result_as_not_found():
    expanded = expand_point(_MismatchedQdrant(), "memory", "requested-point", max_chars=100)

    assert expanded == {
        "found": False,
        "point_id": "requested-point",
        "collection": "memory",
        "collection_name": "memory",
    }


class _FakeQdrant:
    def __init__(self, point):
        self.point = point

    def retrieve(self, collection_name, ids, *, with_payload=True, with_vector=False):
        if ids == [self.point["id"]]:
            return [self.point]
        return []


def test_expand_point_preserves_unsupported_source_uri_instead_of_point_text_fallback():
    point = {
        "id": "manual-1",
        "payload": {
            "text": "manual point text should not be used when source_uri is unsupported",
            "source_type": "manual",
            "source_uri": "session://abc",
        },
    }
    qdrant = _FakeQdrant(point)

    expanded = expand_point(qdrant, "memory", "manual-1", max_chars=10)
    serialized = json.dumps(expanded, sort_keys=True)

    assert expanded["found"] is True
    assert expanded["supported"] is False
    assert expanded["status"] == "unsupported"
    assert expanded["reason"] == "unsupported_scheme"
    assert expanded["scheme"] == "session"
    assert expanded["text"] == ""
    assert "fallback" not in expanded
    assert "manual point text" not in serialized


def test_expand_point_compacts_long_unsupported_source_uri_payload():
    source_uri = "session://" + "x" * 5000
    point = {
        "id": "manual-1",
        "payload": {
            "text": "manual point text should not be used when source_uri is unsupported",
            "source_type": "manual",
            "source_uri": source_uri,
        },
    }
    qdrant = _FakeQdrant(point)

    expanded = expand_point(qdrant, "memory", "manual-1", max_chars=1)
    serialized = json.dumps(expanded, sort_keys=True)

    assert expanded["found"] is True
    assert expanded["supported"] is False
    assert expanded["status"] == "unsupported"
    assert expanded["reason"] == "unsupported_scheme"
    assert expanded["text"] == ""
    assert len(expanded["source_uri"]) <= 512
    assert len(serialized) < 2000
    assert "manual point text" not in serialized


def test_expand_point_compacts_long_point_id_in_public_expand_paths():
    long_point_id = "p" + "x" * 4999
    assert len(long_point_id) == 5000
    fallback_point = {
        "id": long_point_id,
        "payload": {
            "text": "fallback point text should remain bounded",
            "source_type": "manual",
        },
    }

    fallback = expand_point(_FakeQdrant(fallback_point), "memory", long_point_id, max_chars=5)
    neighbors = expand_point(_FakeQdrant(fallback_point), "memory", long_point_id, mode="neighbors", max_chars=5)
    missing = expand_point(_FakeQdrant({"id": "other-point", "payload": {}}), "memory", long_point_id, max_chars=5)

    for expanded in (fallback, neighbors, missing):
        serialized = json.dumps(expanded, sort_keys=True)
        assert long_point_id not in serialized
        assert len(serialized) < 2000
        assert len(expanded["point_id"]) <= 512

    assert fallback["status"] == "unknown"
    assert fallback["reason"] == "missing_source_uri"
    assert fallback["fallback"] == "point_text"
    assert fallback["text"] == "fallb"

    assert neighbors["status"] == "unsupported"
    assert neighbors["reason"] == "neighbors_unsupported"
    assert neighbors["neighbors"] == []
    assert neighbors["text"] == "fallb"

    assert missing["found"] is False
    assert missing["collection"] == "memory"
    assert missing["collection_name"] == "memory"


def test_memory_resolver_expand_compacts_long_point_id_in_error_paths():
    long_point_id = "x" * 5000
    registry = SourceResolverRegistry([MemoryPointResolver(lambda point_id: None)])

    expanded = registry.expand(f"memory://point/{long_point_id}", max_chars=1)
    serialized = json.dumps(expanded, sort_keys=True)

    assert expanded["status"] == "missing"
    assert expanded["text"] == ""
    assert len(expanded["source_uri"]) <= 512
    assert len(expanded["point_id"]) <= 512
    assert len(serialized) < 2000


def test_inspect_point_sanitizes_payload_and_source_metadata():
    sentinel = "INSPECT_SENTINEL_SHOULD_NOT_LEAK"
    huge_with_sentinel = "x" * 700 + sentinel
    point = {
        "id": "inspect-1",
        "payload": {
            "text": "visible snippet " + huge_with_sentinel,
            "source_type": "manual",
            "source_uri": "session://" + "x" * 5000 + sentinel,
            "locator": {
                "line_start": 1,
                "line_end": 2,
                "heading": "h" * 250 + sentinel,
                "arbitrary": huge_with_sentinel,
                "nested": {"secret": sentinel},
            },
            "derived_from": [{"source_uri": "memory://point/parent", "text": huge_with_sentinel}],
            "content_hash": huge_with_sentinel,
            "file_path": huge_with_sentinel,
            "api" + "_key": huge_with_sentinel,
        },
    }

    inspected = inspect_point(_FakeQdrant(point), "memory", "inspect-1")
    serialized = json.dumps(inspected, sort_keys=True)

    assert inspected["found"] is True
    assert inspected["point_id"] == "inspect-1"
    assert "payload" not in inspected
    assert inspected["source"]["source_type"] == "manual"
    assert len(inspected["source"]["source_uri"]) <= 512
    assert inspected["source"]["locator"]["line_start"] == 1
    assert inspected["source"]["locator"]["line_end"] == 2
    assert len(inspected["source"]["locator"]["heading"]) <= 200
    assert "snippet" in inspected
    assert len(inspected["snippet"]) <= 240
    assert sentinel not in serialized
    assert "derived_from" not in serialized
    assert "api_key" not in serialized
    assert "file_path" not in serialized
    assert "arbitrary" not in serialized
    assert "nested" not in serialized
    assert len(serialized) < 2000


def test_trace_point_sanitizes_untrusted_derived_edges():
    sentinel = "TRACE_SENTINEL_SHOULD_NOT_LEAK"
    huge_with_sentinel = "x" * 700 + sentinel
    long_parent_id = "p" + "x" * 4999
    child = {
        "id": "child",
        "payload": {
            "text": "child text",
            "derived_from": [
                {
                    "point_id": long_parent_id,
                    "derivation_type": "summary",
                    "text": huge_with_sentinel,
                    "content": huge_with_sentinel,
                    "lesson": huge_with_sentinel,
                    "locator": {"heading": huge_with_sentinel, "nested": {"secret": sentinel}},
                    "secrets": {"token": sentinel},
                    "arbitrary": huge_with_sentinel,
                },
                sentinel,
            ],
        },
    }
    parent = {"id": long_parent_id, "payload": {"text": "parent text"}}

    class _MultiPointQdrant:
        def retrieve(self, collection_name, ids, *, with_payload=True, with_vector=False):
            points = {"child": child, long_parent_id: parent}
            return [points[point_id] for point_id in ids if point_id in points]

    traced = trace_point(_MultiPointQdrant(), "memory", "child")
    serialized = json.dumps(traced, sort_keys=True)

    assert traced["found"] is True
    assert traced["upstream"][0]["status"] == "exists"
    assert traced["upstream"][0]["source_uri"].startswith("memory://point/")
    assert len(traced["upstream"][0]["source_uri"]) <= 512
    assert len(traced["upstream"][0]["point_id"]) <= 512
    assert traced["upstream"][1] == {"status": "unknown", "reason": "invalid_edge"}
    assert sentinel not in serialized
    assert long_parent_id not in serialized
    for forbidden in ("text", "content", "lesson", "secrets", "arbitrary", "nested"):
        assert forbidden not in serialized
    assert len(serialized) < 2500


def test_inspect_point_surfaces_valid_memory_kind_and_omits_invalid_legacy_kind():
    valid = {
        "id": "valid-kind",
        "payload": {"text": "stable API decision", "source_type": "manual", "memory_kind": "decision"},
    }
    invalid = {
        "id": "invalid-kind",
        "payload": {"text": "legacy bad kind", "source_type": "manual", "memory_kind": "unknown_kind"},
    }

    valid_inspected = inspect_point(_FakeQdrant(valid), "memory", "valid-kind")
    invalid_inspected = inspect_point(_FakeQdrant(invalid), "memory", "invalid-kind")

    assert valid_inspected["source"]["memory_kind"] == "decision"
    assert "memory_kind" not in invalid_inspected.get("source", {})


def test_trace_point_surfaces_valid_relation_type_and_omits_invalid_legacy_relation_type():
    child = {
        "id": "child",
        "payload": {
            "text": "child text",
            "derived_from": [
                {"source_uri": "session://valid", "relation_type": "SUPPORTS"},
                {"source_uri": "session://invalid", "relation_type": "NOPE"},
            ],
        },
    }

    traced = trace_point(_FakeQdrant(child), "memory", "child")

    assert traced["upstream"][0]["relation_type"] == "SUPPORTS"
    assert "relation_type" not in traced["upstream"][1]


def test_source_status_for_point_sanitizes_status_metadata(tmp_path):
    sentinel = "STATUS_SENTINEL_SHOULD_NOT_LEAK"
    source = tmp_path / "source.md"
    source.write_text("alpha\nbeta\n", encoding="utf-8")
    point = {
        "id": "status-1",
        "payload": {
            "text": "status text",
            "source_uri": source.as_uri(),
            "source_type": "indexed_file",
            "locator": {
                "line_start": 1,
                "line_end": 1,
                "heading": "h" * 250 + sentinel,
                "arbitrary": "x" * 700 + sentinel,
                "nested": {"secret": sentinel},
            },
            "content_hash": _sha256_text("different"),
        },
    }

    status = source_status_for_point(_FakeQdrant(point), "memory", "status-1")
    serialized = json.dumps(status, sort_keys=True)

    assert status["found"] is True
    assert status["status"] == "changed"
    assert status["changed"] is True
    assert len(status["source_uri"]) <= 512
    assert len(status["point_id"]) <= 512
    assert status["locator"]["line_start"] == 1
    assert status["locator"]["line_end"] == 1
    assert len(status["locator"]["heading"]) <= 200
    assert sentinel not in serialized
    assert "arbitrary" not in serialized
    assert "nested" not in serialized
    assert len(serialized) < 2500


def test_source_registry_stat_sanitizes_resolver_errors():
    sentinel = "REGISTRY_ERROR_SENTINEL_SHOULD_NOT_LEAK"

    class _ExplodingResolver:
        schemes = {"boom"}

        def stat(self, source_uri, locator=None, *, content_hash=None, source_modified_at=None):
            raise RuntimeError("exploded " + sentinel)

        def expand(self, source_uri, locator=None, *, mode="excerpt", max_chars=8000):
            raise RuntimeError("unused")

    registry = SourceResolverRegistry([_ExplodingResolver()])
    status = registry.stat("boom://" + "x" * 5000 + sentinel)
    serialized = json.dumps(status, sort_keys=True)

    assert status["status"] == "unknown"
    assert status["reason"] == "resolver_error"
    assert len(status["source_uri"]) <= 512
    assert sentinel not in serialized
    assert len(serialized) < 2000
