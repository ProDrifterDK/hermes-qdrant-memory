from __future__ import annotations

from qdrant_memory.client import QdrantClient


class RecordingQdrantClient(QdrantClient):
    def __init__(self):
        super().__init__("http://unused")
        self.calls = []

    def _request(self, method, path, body=None):
        self.calls.append((method, path, body))
        if len(self.calls) == 1:
            return {"result": {"points": [{"id": 0, "payload": {"file_path": "a"}}], "next_page_offset": 0}}
        if len(self.calls) == 2:
            return {"result": {"points": [{"id": 1, "payload": {"file_path": "a"}}], "next_page_offset": None}}
        raise AssertionError("unexpected extra scroll call")


def test_scroll_by_filter_continues_when_next_page_offset_is_zero():
    client = RecordingQdrantClient()

    points = client.scroll_by_filter("memory", {"must": []}, limit=1)

    assert [point["id"] for point in points] == [0, 1]
    assert client.calls[0][2]["limit"] == 1
    assert client.calls[1][2]["offset"] == 0


def test_extract_vector_size_from_single_vector_config():
    info = {"config": {"params": {"vectors": {"size": 1024, "distance": "Cosine"}}}}

    assert QdrantClient._extract_vector_size(info) == 1024


def test_extract_vector_size_from_named_vectors_when_sizes_match():
    info = {"config": {"params": {"vectors": {"dense": {"size": 1024}, "other": {"size": "1024"}}}}}

    assert QdrantClient._extract_vector_size(info) == 1024


def test_extract_vector_size_returns_none_for_mixed_named_vector_sizes():
    info = {"config": {"params": {"vectors": {"dense": {"size": 1024}, "other": {"size": 768}}}}}

    assert QdrantClient._extract_vector_size(info) is None


def test_extract_vector_size_returns_none_for_missing_or_invalid_sizes():
    assert QdrantClient._extract_vector_size({"config": {"params": {}}}) is None
    assert QdrantClient._extract_vector_size({"config": {"params": {"vectors": {"size": "not-int"}}}}) is None
    assert QdrantClient._extract_vector_size({"config": {"params": {"vectors": {"dense": {"size": "not-int"}}}}}) is None
