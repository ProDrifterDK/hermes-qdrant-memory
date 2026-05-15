from __future__ import annotations

import json

from qdrant_memory.config import load_config
from qdrant_memory.indexer import FileIndexer, chunk_markdown, chunk_text, classify_source_type
from qdrant_memory.tools import FORGET_SCHEMA, INDEX_SCHEMA


class FakeEmbedding:
    def __init__(self):
        self.documents = []

    def embed_document(self, text):
        self.documents.append(text)
        return [0.3, 0.4]


class FakeQdrant:
    def __init__(self):
        self.upserts = []
        self.deleted = []

    def upsert(self, name, points):
        self.upserts.append((name, points))
        return {"status": "ok"}

    def delete_ids(self, name, ids):
        self.deleted.append((name, ids))
        return {"status": "ok"}


def test_config_index_defaults_and_list_coercion(tmp_path):
    cfg = load_config(
        hermes_home=str(tmp_path),
        hermes_config={"qdrant_memory": {"index_dirs": "~/notes,/tmp/docs", "index_extensions": "md,txt"}},
    )
    assert cfg["index_max_files"] == 500
    assert cfg["index_dry_run_default"] is True
    assert cfg["index_dirs"] == ["~/notes", "/tmp/docs"]
    assert cfg["index_extensions"] == ["md", "txt"]


def test_tool_schemas_include_index_and_forget():
    assert INDEX_SCHEMA["name"] == "qdrant_memory_index"
    assert INDEX_SCHEMA["parameters"]["properties"]["dry_run"]["type"] == "boolean"
    assert FORGET_SCHEMA["name"] == "qdrant_memory_forget"
    assert FORGET_SCHEMA["parameters"]["required"] == ["ids"]


def test_markdown_chunking_uses_headings_and_bounds():
    text = "---\ntags: [alpha, beta]\n---\n# Intro\nhello\n\n## Details\n" + ("word " * 300)
    chunks = chunk_markdown(text, max_chars=180)
    headings = [heading for heading, _ in chunks]
    assert "Intro" in headings
    assert "Details" in headings
    assert all(len(body) <= 180 for _, body in chunks)


def test_text_chunking_groups_paragraphs_and_bounds():
    text = "para one\n\n" + ("x" * 500) + "\n\npara three"
    chunks = chunk_text(text, max_chars=120)
    assert len(chunks) > 2
    assert chunks[0][1] == "para one"
    assert all(len(body) <= 120 for _, body in chunks)


def test_walker_excludes_dirs_and_binaryish_files(tmp_path):
    (tmp_path / "keep.md").write_text("# Keep\nhello", encoding="utf-8")
    (tmp_path / "skip.bin").write_bytes(b"\x00\x01")
    git = tmp_path / ".git"
    git.mkdir()
    (git / "ignored.md").write_text("# Ignored", encoding="utf-8")
    node_modules = tmp_path / "node_modules"
    node_modules.mkdir()
    (node_modules / "ignored.txt").write_text("ignored", encoding="utf-8")

    indexer = FileIndexer(config={"index_extensions": [".md", ".txt"], "index_max_files": 10})
    files, skipped = indexer.iter_files([tmp_path])
    assert [p.name for p in files] == ["keep.md"]
    skipped_paths = "\n".join(item["path"] for item in skipped)
    assert "skip.bin" in skipped_paths
    assert "ignored.md" not in skipped_paths  # directory pruned safely


def test_prepare_file_metadata_classification_tags_and_idempotent_ids(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    path = docs / "plan.md"
    path.write_text("---\ntags:\n  - roadmap\n---\n# Plan\nRemember #decision here", encoding="utf-8")
    indexer = FileIndexer(config={"max_chunk_tokens": 128})

    first = indexer.prepare_file(path)
    second = indexer.prepare_file(path)

    assert first[0].id == second[0].id
    assert first[0].source_type == "project_doc"
    assert first[0].heading == "Plan"
    assert "roadmap" in first[0].tags
    assert "decision" in first[0].tags
    assert first[0].payload()["file_path"] == str(path.resolve())


def test_classify_source_type_skill_and_vault_paths(tmp_path):
    assert classify_source_type(tmp_path / "skills" / "agent" / "README.md") == "skill_doc"
    assert classify_source_type(tmp_path / "vault" / "Daily.md") == "vault_note"
    assert classify_source_type(tmp_path / "Example Vault" / "Daily.md") == "vault_note"
    assert classify_source_type(tmp_path / "Example Vault" / "docs" / "Project.md") == "vault_note"
    assert classify_source_type(tmp_path / "misc" / "note.txt") == "indexed_file"


def test_dry_run_prepares_without_embedding_or_upsert(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# Note\nhello world", encoding="utf-8")
    emb = FakeEmbedding()
    qdrant = FakeQdrant()
    indexer = FileIndexer(qdrant=qdrant, embeddings=emb, collection_name="c", config={"index_max_files": 10})

    summary = indexer.index([path], dry_run=True)

    assert summary["dry_run"] is True
    assert summary["chunks_prepared"] == 1
    assert summary["chunks_upserted"] == 0
    assert emb.documents == []
    assert qdrant.upserts == []


def test_index_execution_with_fake_embedding_and_qdrant(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("alpha\n\nbeta", encoding="utf-8")
    emb = FakeEmbedding()
    qdrant = FakeQdrant()
    indexer = FileIndexer(
        qdrant=qdrant,
        embeddings=emb,
        collection_name="c",
        config={"index_max_files": 10, "max_chunk_tokens": 128},
        profile_id="coder",
        platform="cli",
        session_id="s1",
        model="fake-model",
    )

    summary = indexer.index([path], dry_run=False)

    assert summary["chunks_upserted"] == summary["chunks_prepared"] == 1
    assert emb.documents == ["alpha\n\nbeta"]
    name, points = qdrant.upserts[0]
    assert name == "c"
    assert points[0]["vector"] == [0.3, 0.4]
    payload = points[0]["payload"]
    assert payload["source_type"] == "indexed_file"
    assert payload["chunk_type"] == "file_chunk"
    assert payload["profile_id"] == "coder"
    assert payload["model"] == "fake-model"


def test_provider_index_tool_requires_paths_when_unconfigured():
    from __init__ import QdrantMemoryProvider

    provider = QdrantMemoryProvider()
    provider._config["index_dirs"] = []
    result = json.loads(provider.handle_tool_call("qdrant_memory_index", {}))
    assert "paths are required" in result["error"]


def test_provider_forget_dry_run_and_execute():
    from __init__ import QdrantMemoryProvider

    provider = QdrantMemoryProvider()
    provider._qdrant = FakeQdrant()
    provider._config["collection_name"] = "c"
    dry = json.loads(provider.handle_tool_call("qdrant_memory_forget", {"ids": ["a"]}))
    live = json.loads(provider.handle_tool_call("qdrant_memory_forget", {"ids": ["a"], "dry_run": False}))
    assert dry == {"dry_run": True, "ids": ["a"], "deleted": 0}
    assert live["deleted"] == 1
    assert provider._qdrant.deleted == [("c", ["a"])]
