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
