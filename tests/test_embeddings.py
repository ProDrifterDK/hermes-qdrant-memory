from __future__ import annotations

from qdrant_memory.config import load_config
from qdrant_memory.embeddings import EmbeddingClient


class RecordingEmbeddingClient(EmbeddingClient):
    def __init__(self, **kwargs):
        super().__init__("http://embedding.test/v1", "bge-m3", **kwargs)
        self.inputs: list[str] = []

    def _request(self, method: str, path: str, body=None):  # noqa: D401 - test double
        assert method == "POST"
        assert path == "/embeddings"
        text = body["input"]
        self.inputs.append(text)
        return {"data": [{"embedding": [1.0, 0.0]}]}


class TooLargeThenChunkClient(EmbeddingClient):
    def __init__(self, *, limit: int, **kwargs):
        super().__init__("http://embedding.test/v1", "bge-m3", **kwargs)
        self.limit = limit
        self.inputs: list[str] = []

    def _request(self, method: str, path: str, body=None):  # noqa: D401 - test double
        text = body["input"]
        self.inputs.append(text)
        if len(text) > self.limit:
            raise RuntimeError(
                f"Embedding HTTP 500: input is too large to process. "
                f"increase the physical batch size (current batch size: {self.limit})"
            )
        return {"data": [{"embedding": [1.0, 0.0]}]}


def test_embed_document_splits_long_text_before_requesting_embedding_server():
    client = RecordingEmbeddingClient(document_prefix="", max_input_chars=10, max_chunks=10)

    vector = client.embed_document("abcdefghijklmnopqrstuvwxyz")

    assert vector == [1.0, 0.0]
    assert len(client.inputs) == 3
    assert all(len(text) <= 10 for text in client.inputs)
    assert "".join(client.inputs) == "abcdefghijklmnopqrstuvwxyz"


def test_embed_recursively_chunks_when_server_reports_input_too_large():
    client = TooLargeThenChunkClient(limit=10, max_input_chars=0, max_chunks=10)

    vector = client.embed("abcdefghijklmnopqrstuvwxyz")

    assert vector == [1.0, 0.0]
    assert client.inputs[0] == "abcdefghijklmnopqrstuvwxyz"
    assert len(client.inputs) > 1
    assert all(len(text) <= 10 for text in client.inputs[1:])


def test_config_defaults_include_embedding_chunk_limits(tmp_path):
    cfg = load_config(hermes_home=str(tmp_path), hermes_config={})

    assert cfg["embedding_max_input_chars"] == 12000
    assert cfg["embedding_max_chunks"] == 16


def test_config_allows_overriding_embedding_chunk_limits(tmp_path):
    cfg = load_config(
        hermes_home=str(tmp_path),
        hermes_config={"qdrant_memory": {"embedding_max_input_chars": "8000", "embedding_max_chunks": "4"}},
    )

    assert cfg["embedding_max_input_chars"] == 8000
    assert cfg["embedding_max_chunks"] == 4
