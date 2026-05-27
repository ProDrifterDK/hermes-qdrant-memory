from __future__ import annotations

import json
from datetime import datetime

from qdrant_memory.config import load_config
from qdrant_memory.indexer import FileChunk, FileIndexer, chunk_markdown, chunk_text, classify_source_type, make_file_chunk_id
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
        self.delete_filters = []
        self.points = []

    def upsert(self, name, points):
        self.upserts.append((name, points))
        return {"status": "ok"}

    def delete_ids(self, name, ids):
        self.deleted.append((name, ids))
        return {"status": "ok"}

    def delete_filter(self, name, filter):
        self.delete_filters.append((name, filter))
        return {"status": "ok"}

    def scroll_by_filter(self, name, filter, limit=256, with_payload=True, with_vector=False):
        must = filter.get("must", [])

        def matches(point):
            payload = point.get("payload", {})
            for clause in must:
                key = clause["key"]
                value = clause["match"]["value"]
                if payload.get(key) != value:
                    return False
            return True

        return [point for point in self.points if matches(point)]


class FakeLegacyQdrant(FakeQdrant):
    def __getattribute__(self, name):
        if name == "scroll_by_filter":
            raise AttributeError(name)
        return super().__getattribute__(name)


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
    assert "markdown/text files or source folders" in INDEX_SCHEMA["description"]
    assert "vault" not in INDEX_SCHEMA["description"].lower()
    assert FORGET_SCHEMA["name"] == "qdrant_memory_forget"
    assert FORGET_SCHEMA["parameters"]["required"] == ["ids"]


def test_markdown_chunking_uses_headings_and_bounds():
    text = "---\ntags: [alpha, beta]\n---\n# Intro\nhello\n\n## Details\n" + ("word " * 300)
    chunks = chunk_markdown(text, max_chars=180)
    headings = [heading for heading, _ in chunks]
    assert "Intro" in headings
    assert "Details" in headings
    assert all(len(body) <= 180 for _, body in chunks)


def test_markdown_chunking_skips_heading_only_sections():
    text = "# Daily\n\n## Tareas\n\n## Notas\nActual note body.\n\n### Contribution\n\n## Reflexión\n- Real reflection."

    chunks = chunk_markdown(text, max_chars=180)

    bodies = [body for _, body in chunks]
    assert "## Tareas" not in bodies
    assert "### Contribution" not in bodies
    assert any("Actual note body" in body for body in bodies)
    assert any("Real reflection" in body for body in bodies)


def test_markdown_chunking_skips_all_heading_only_documents():
    text = "# Daily\n\n## Tareas\n\n### Contribution\n\n## Reflexión"

    assert chunk_markdown(text, max_chars=180) == []


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


def test_prepare_file_records_source_derivation_metadata_for_markdown_chunks(tmp_path):
    path = tmp_path / "plan.md"
    path.write_text("# Intro\nhello\n\n## Details\nline one\nline two\n", encoding="utf-8")
    indexer = FileIndexer(config={"max_chunk_tokens": 128})

    chunks = indexer.prepare_file(path)
    chunks_by_heading = {chunk.heading: chunk for chunk in chunks}

    intro_payload = chunks_by_heading["Intro"].payload()
    details_payload = chunks_by_heading["Details"].payload()
    assert intro_payload["source_uri"] == path.resolve().as_uri()
    assert intro_payload["locator"] == {"line_start": 1, "line_end": 2, "heading": "Intro"}
    assert details_payload["locator"] == {"line_start": 4, "line_end": 6, "heading": "Details"}
    assert intro_payload["content_hash"] == f"sha256:{chunks_by_heading['Intro'].chunk_hash}"
    datetime.fromisoformat(intro_payload["source_modified_at"].replace("Z", "+00:00"))
    assert intro_payload["memory_kind"] == "source_chunk"
    assert intro_payload["derivation_type"] == "indexed_chunk"
    assert intro_payload["canonical"] is True
    assert intro_payload["stale"] is False
    assert intro_payload["requires_review"] is False


def test_prepare_file_locates_text_chunks_with_whitespace_only_blank_lines(tmp_path):
    para_one = "A" * 180
    para_two = "B" * 180
    para_three = "C" * 180
    path = tmp_path / "notes.txt"
    path.write_text(f"{para_one}\n   \n{para_two}\n\t \n{para_three}", encoding="utf-8")
    indexer = FileIndexer(config={"max_chunk_tokens": 1})

    chunks = indexer.prepare_file(path)

    assert len(chunks) == 2
    assert chunks[0].text == f"{para_one}\n\n{para_two}"
    assert chunks[0].line_start == 1
    assert chunks[0].line_end == 3
    assert chunks[0].payload()["locator"] == {"line_start": 1, "line_end": 3}
    assert chunks[1].line_start == 5
    assert chunks[1].line_end == 5


def test_file_chunk_payload_adds_fact_metadata_from_explicit_fact_tag():
    chunk = FileChunk(
        id="chunk-1",
        text="The production API endpoint is /api/v2.",
        source="test.md",
        source_type="indexed_file",
        file_path="/vault/test.md",
        file_mtime=1.0,
        file_size=12,
        file_sha256="abc",
        chunk_index=0,
        chunk_count=1,
        chunk_hash="hash",
        heading="API",
        tags=["fact:production.api.endpoint"],
    )

    payload = chunk.payload()

    assert payload["fact_key"] == "production.api.endpoint"
    assert payload["reconsolidation_key"] == "production.api.endpoint"
    assert payload["topic"] == "API"


def test_file_chunk_payload_uses_heading_as_topic_not_fact_key_when_weak():
    chunk = FileChunk(
        id="chunk-1",
        text="General notes and loose ideas.",
        source="notes.md",
        source_type="indexed_file",
        file_path="/vault/notes.md",
        file_mtime=1.0,
        file_size=12,
        file_sha256="abc",
        chunk_index=0,
        chunk_count=1,
        chunk_hash="hash",
        heading="Notes",
    )

    payload = chunk.payload()

    assert payload["topic"] == "Notes"
    assert "fact_key" not in payload
    assert "reconsolidation_key" not in payload


def test_file_chunk_payload_extracts_fact_key_from_clear_chunk_statement():
    chunk = FileChunk(
        id="chunk-1",
        text="Production API endpoint is /api/v2.",
        source="api.md",
        source_type="project_doc",
        file_path="/project/docs/api.md",
        file_mtime=1.0,
        file_size=12,
        file_sha256="abc",
        chunk_index=0,
        chunk_count=1,
        chunk_hash="hash",
        heading="API",
    )

    payload = chunk.payload()

    assert payload["subject"] == "Production API endpoint"
    assert payload["fact_key"] == "production.api.endpoint"
    assert payload["reconsolidation_key"] == "production.api.endpoint"


def test_classify_source_type_uses_generic_file_types_for_vault_named_paths(tmp_path):
    assert classify_source_type(tmp_path / "skills" / "agent" / "README.md") == "skill_doc"
    assert classify_source_type(tmp_path / "vault" / "Daily.md") == "indexed_file"
    assert classify_source_type(tmp_path / "Example Vault" / "Daily.md") == "indexed_file"
    assert classify_source_type(tmp_path / "Example Vault" / "docs" / "Project.md") == "project_doc"
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


def test_file_chunk_payload_includes_manifest_fields(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("alpha\n\nbeta", encoding="utf-8")
    emb = FakeEmbedding()
    qdrant = FakeQdrant()
    indexer = FileIndexer(qdrant=qdrant, embeddings=emb, collection_name="c", config={"max_chunk_tokens": 128})

    summary = indexer.index([path], dry_run=False)

    assert summary["chunks_upserted"] == 1
    point = qdrant.upserts[0][1][0]
    payload = point["payload"]
    assert payload["manifest_version"] == 1
    assert payload["chunk_id"] == point["id"]
    assert payload["file_path"] == str(path.resolve())
    assert payload["file_size"] == path.stat().st_size
    assert len(payload["file_sha256"]) == 64
    assert len(payload["chunk_hash"]) == 64
    assert payload["chunk_index"] == 0
    assert payload["chunk_count"] == 1


def test_dry_run_reports_stale_chunk_ids_when_file_shrinks(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("alpha", encoding="utf-8")
    file_path = str(path.resolve())
    stale_ids = [make_file_chunk_id(file_path, 1), make_file_chunk_id(file_path, 2)]
    qdrant = FakeQdrant()
    qdrant.points = [
        {"id": make_file_chunk_id(file_path, 0), "payload": {"file_path": file_path}},
        {"id": stale_ids[0], "payload": {"file_path": file_path}},
        {"id": stale_ids[1], "payload": {"file_path": file_path}},
    ]
    emb = FakeEmbedding()
    indexer = FileIndexer(qdrant=qdrant, embeddings=emb, collection_name="c", config={"max_chunk_tokens": 128})

    summary = indexer.index([path], dry_run=True)

    assert summary["manifest_checked"] is True
    assert summary["stale_ids"] == stale_ids
    assert summary["stale_count"] == 2
    assert summary["chunks_deleted"] == 0
    assert summary["chunks_upserted"] == 0
    assert emb.documents == []
    assert qdrant.upserts == []
    assert qdrant.deleted == []


def test_live_index_deletes_only_stale_ids_before_upsert(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("alpha", encoding="utf-8")
    file_path = str(path.resolve())
    stale_id = make_file_chunk_id(file_path, 1)
    qdrant = FakeQdrant()
    qdrant.points = [
        {"id": make_file_chunk_id(file_path, 0), "payload": {"file_path": file_path}},
        {"id": stale_id, "payload": {"file_path": file_path}},
    ]
    emb = FakeEmbedding()
    indexer = FileIndexer(qdrant=qdrant, embeddings=emb, collection_name="c", config={"max_chunk_tokens": 128})

    summary = indexer.index([path], dry_run=False)

    assert summary["delete_mode"] == "ids"
    assert summary["chunks_deleted"] == 1
    assert qdrant.deleted == [("c", [stale_id])]
    assert qdrant.delete_filters == []
    assert summary["chunks_upserted"] == summary["chunks_prepared"] == 1


def test_live_index_prefers_id_sync_over_filter_delete_even_with_force(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("alpha", encoding="utf-8")
    file_path = str(path.resolve())
    stale_id = make_file_chunk_id(file_path, 1)
    qdrant = FakeQdrant()
    qdrant.points = [{"id": stale_id, "payload": {"file_path": file_path}}]
    indexer = FileIndexer(qdrant=qdrant, embeddings=FakeEmbedding(), collection_name="c", config={"max_chunk_tokens": 128})

    summary = indexer.index([path], dry_run=False, force=True)

    assert summary["delete_mode"] == "ids"
    assert qdrant.deleted == [("c", [stale_id])]
    assert qdrant.delete_filters == []


def test_live_index_falls_back_to_delete_filter_when_scroll_unavailable_and_force_true(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("alpha", encoding="utf-8")
    qdrant = FakeLegacyQdrant()
    indexer = FileIndexer(qdrant=qdrant, embeddings=FakeEmbedding(), collection_name="c", config={"max_chunk_tokens": 128})

    summary = indexer.index([path], dry_run=False, force=True)

    assert summary["manifest_checked"] is False
    assert summary["delete_mode"] == "filter"
    assert qdrant.delete_filters == [("c", {"must": [{"key": "file_path", "match": {"value": str(path.resolve())}}]})]


def test_empty_file_with_existing_chunks_reports_all_existing_as_stale(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("", encoding="utf-8")
    file_path = str(path.resolve())
    old_ids = [make_file_chunk_id(file_path, 0), make_file_chunk_id(file_path, 1)]
    qdrant = FakeQdrant()
    qdrant.points = [{"id": point_id, "payload": {"file_path": file_path}} for point_id in old_ids]
    indexer = FileIndexer(qdrant=qdrant, embeddings=FakeEmbedding(), collection_name="c", config={"max_chunk_tokens": 128})

    dry = indexer.index([path], dry_run=True)
    live = indexer.index([path], dry_run=False)

    assert dry["chunks_prepared"] == 0
    assert dry["stale_ids"] == old_ids
    assert live["chunks_deleted"] == 2
    assert qdrant.deleted == [("c", old_ids)]


def test_dry_run_directory_sync_reports_deleted_files_without_mutation(tmp_path):
    keep = tmp_path / "keep.md"
    keep.write_text("# Keep\ncurrent", encoding="utf-8")
    removed_path = str((tmp_path / "removed.md").resolve())
    removed_id = make_file_chunk_id(removed_path, 0)
    qdrant = FakeQdrant()
    qdrant.points = [
        {"id": make_file_chunk_id(str(keep.resolve()), 0), "payload": {"file_path": str(keep.resolve()), "chunk_type": "file_chunk"}},
        {"id": removed_id, "payload": {"file_path": removed_path, "chunk_type": "file_chunk"}},
    ]
    emb = FakeEmbedding()
    indexer = FileIndexer(qdrant=qdrant, embeddings=emb, collection_name="c", config={"max_chunk_tokens": 128})

    summary = indexer.index([tmp_path], dry_run=True)

    assert summary["directory_manifest_checked"] is True
    assert summary["deleted_file_paths"] == [removed_path]
    assert summary["deleted_file_ids"] == [removed_id]
    assert summary["stale_ids"] == [removed_id]
    assert summary["chunks_deleted"] == 0
    assert qdrant.deleted == []
    assert qdrant.upserts == []
    assert emb.documents == []


def test_live_directory_sync_deletes_chunks_for_removed_files(tmp_path):
    keep = tmp_path / "keep.md"
    keep.write_text("# Keep\ncurrent", encoding="utf-8")
    removed_path = str((tmp_path / "removed.md").resolve())
    removed_id = make_file_chunk_id(removed_path, 0)
    qdrant = FakeQdrant()
    qdrant.points = [
        {"id": removed_id, "payload": {"file_path": removed_path, "chunk_type": "file_chunk"}},
    ]
    indexer = FileIndexer(qdrant=qdrant, embeddings=FakeEmbedding(), collection_name="c", config={"max_chunk_tokens": 128})

    summary = indexer.index([tmp_path], dry_run=False)

    assert summary["directory_manifest_checked"] is True
    assert summary["deleted_file_paths"] == [removed_path]
    assert summary["chunks_deleted"] == 1
    assert qdrant.deleted == [("c", [removed_id])]


def test_directory_sync_does_not_delete_existing_but_excluded_file(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("still here", encoding="utf-8")
    file_path = str(path.resolve())
    point_id = make_file_chunk_id(file_path, 0)
    qdrant = FakeQdrant()
    qdrant.points = [{"id": point_id, "payload": {"file_path": file_path, "chunk_type": "file_chunk"}}]
    indexer = FileIndexer(
        qdrant=qdrant,
        embeddings=FakeEmbedding(),
        collection_name="c",
        config={"index_extensions": [".md"], "max_chunk_tokens": 128},
    )

    summary = indexer.index([tmp_path], dry_run=False)

    assert summary["directory_manifest_checked"] is True
    assert summary["deleted_file_paths"] == []
    assert summary["deleted_file_ids"] == []
    assert summary["chunks_deleted"] == 0
    assert qdrant.deleted == []


def test_directory_sync_is_skipped_when_max_files_truncates_walk_after_many_skips(tmp_path):
    for i in range(60):
        (tmp_path / f"a{i:02d}.bin").write_bytes(b"binary-ish")
    keep = tmp_path / "z_keep.md"
    extra = tmp_path / "z_extra.md"
    keep.write_text("# Keep", encoding="utf-8")
    extra.write_text("# Extra", encoding="utf-8")
    removed_path = str((tmp_path / "z_removed.md").resolve())
    removed_id = make_file_chunk_id(removed_path, 0)
    qdrant = FakeQdrant()
    qdrant.points = [{"id": removed_id, "payload": {"file_path": removed_path, "chunk_type": "file_chunk"}}]
    indexer = FileIndexer(
        qdrant=qdrant,
        embeddings=FakeEmbedding(),
        collection_name="c",
        config={"index_max_files": 1, "max_chunk_tokens": 128},
    )

    summary = indexer.index([tmp_path], dry_run=False)

    assert summary["directory_manifest_checked"] is False
    assert summary["deleted_file_paths"] == []
    assert summary["chunks_deleted"] == 0
    assert qdrant.deleted == []


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
