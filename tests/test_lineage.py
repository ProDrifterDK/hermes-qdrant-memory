from __future__ import annotations

import builtins
import copy
import hashlib
import json
import multiprocessing
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import qdrant_memory.lineage as lineage
from qdrant_memory.indexer import FileIndexer
from qdrant_memory.lineage import collection_write_lock, plan_file_lineage


class FakeEmbedding:
    def __init__(self):
        self.documents = []

    def embed_document(self, text):
        self.documents.append(text)
        return [0.3, 0.4]


class FakeQdrant:
    def __init__(self):
        self.points = []
        self.upserts = []
        self.updates = []
        self.deleted = []
        self.delete_filters = []
        self.call_log = []

    def upsert(self, name, points):
        self.call_log.append(("upsert", name, [str(point["id"]) for point in points]))
        self.upserts.append((name, points))
        by_id = {str(point["id"]): dict(point) for point in self.points}
        for point in points:
            by_id[str(point["id"])] = dict(point)
        self.points = list(by_id.values())
        return {"status": "ok"}

    def update_payload(self, name, point_id, payload):
        self.call_log.append(("update_payload", name, str(point_id)))
        self.updates.append((name, str(point_id), dict(payload)))
        for point in self.points:
            if str(point.get("id")) == str(point_id):
                point["payload"] = {**(point.get("payload") or {}), **payload}
                return {"status": "ok"}
        raise KeyError(point_id)

    def retrieve(self, name, ids, with_payload=True, with_vector=False):
        wanted = {str(point_id) for point_id in ids}
        result = []
        for point in self.points:
            if str(point.get("id")) not in wanted:
                continue
            item = {"id": point["id"]}
            if with_payload:
                item["payload"] = dict(point.get("payload") or {})
            if with_vector and "vector" in point:
                item["vector"] = point["vector"]
            result.append(item)
        return result

    def scroll_by_filter(self, name, filter, limit=256, with_payload=True, with_vector=False):
        must = filter.get("must", [])
        return [point for point in self.points if all(
            (point.get("payload") or {}).get(clause["key"]) == clause["match"]["value"]
            for clause in must
        )]

    def delete_ids(self, name, ids):
        self.call_log.append(("delete_ids", name, list(ids)))
        self.deleted.append((name, ids))
        wanted = {str(point_id) for point_id in ids}
        self.points = [point for point in self.points if str(point.get("id")) not in wanted]
        return {"status": "ok"}

    def delete_filter(self, name, filter):
        self.call_log.append(("delete_filter", name, filter))
        self.delete_filters.append((name, filter))
        return {"status": "ok"}


def _indexer(qdrant, embeddings, *, mode="capture", lock_dir="", max_chunk_tokens=128):
    return FileIndexer(
        qdrant=qdrant,
        embeddings=embeddings,
        collection_name="lineage_test",
        config={
            "lineage_mode": mode,
            "qdrant_url": "http://disposable.invalid:6333",
            "max_chunk_tokens": max_chunk_tokens,
            "lineage_lock_dir": str(lock_dir),
        },
    )


def _hold_lock(lock_dir, ready, release):
    with collection_write_lock(collection_name="lineage_test", lock_dir=lock_dir):
        ready.set()
        release.wait(5)


def _points(qdrant, *, kind=None, relation=None):
    result = []
    for point in qdrant.points:
        payload = point.get("payload") or {}
        if kind is not None and payload.get("memory_kind") != kind:
            continue
        if relation is not None and payload.get("relation_type") != relation:
            continue
        result.append(point)
    return result


def _assert_no_event_fields(qdrant, point_ids):
    touched = [point for point in qdrant.points if str(point.get("id")) in set(point_ids)]
    assert {str(point["id"]) for point in touched} == set(point_ids)
    for point in touched:
        payload = point.get("payload") or {}
        assert all(key not in payload for key in ("head_event_id", "pending_event_id", "lineage_event_id"))
        assert not any(str(key).startswith("event_") for key in payload)


def _seed_legacy(path: Path, *, mutate=None):
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    indexer = _indexer(qdrant, embeddings, mode="off")
    chunk = indexer.prepare_file(path)[0]
    payload = chunk.payload(profile_id="default")
    if mutate:
        mutate(payload)
    qdrant.points = [{"id": chunk.id, "vector": [0.1, 0.2], "payload": payload}]
    return qdrant, embeddings, indexer


def test_new_index_has_version_and_derivation_and_empty_file_head(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha\n\n## More\nbeta", encoding="utf-8")
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    event_control = {"id": "22222222-2222-2222-2222-222222222222", "vector": {}, "payload": {
        "memory_kind": "graph_entity", "lineage_role": "change_event",
        "file_path": str(path.resolve()), "profile_id": "default", "created_at": "2026-01-01T00:00:00Z",
    }}
    qdrant.points.append(event_control)

    result = _indexer(qdrant, embeddings).index([path], dry_run=False)

    sources = [p for p in _points(qdrant, kind="graph_entity") if p["payload"].get("lineage_role") == "file_source"]
    versions = [p for p in _points(qdrant, kind="graph_entity") if p["payload"].get("lineage_role") == "file_version"]
    chunks = _points(qdrant, kind="source_chunk")
    part_of = _points(qdrant, kind="graph_edge", relation="PART_OF")
    derived = _points(qdrant, kind="graph_edge", relation="DERIVED_FROM")
    assert len(sources) == len(versions) == len(part_of) == 1
    assert len(derived) == len(chunks) == result["chunks_prepared"]
    assert {edge["payload"]["source_point_id"] for edge in derived} == {str(chunk["id"]) for chunk in chunks}
    assert {edge["payload"]["target_point_id"] for edge in derived} == {str(versions[0]["id"])}
    assert sources[0]["payload"]["current_version_id"] == str(versions[0]["id"])
    assert sources[0]["payload"]["source_deleted"] is False
    assert all(point_id in {str(point["id"]) for point in qdrant.points} for point_id in result["lineage_read_back_ids"])
    _assert_no_event_fields(qdrant, result["lineage_read_back_ids"])
    assert sum(point["payload"].get("lineage_role") == "change_event" for point in qdrant.points) == 1
    assert all("head_event_id" not in point["payload"] and "pending_event_id" not in point["payload"] for point in qdrant.points)
    assert next(point for point in qdrant.points if point["id"] == event_control["id"]) == event_control
    assert result["lineage_repair_ids"] == []

    empty = tmp_path / "empty.txt"
    empty.write_bytes(b"")
    empty_result = _indexer(qdrant, embeddings).index([empty], dry_run=False)
    empty_source = next(p for p in _points(qdrant, kind="graph_entity")
                        if p["payload"].get("lineage_role") == "file_source" and p["payload"].get("file_path") == str(empty.resolve()))
    empty_version = next(p for p in _points(qdrant, kind="graph_entity")
                         if p["payload"].get("lineage_role") == "file_version" and p["payload"].get("file_path") == str(empty.resolve()))
    assert empty_source["payload"]["current_version_id"] == str(empty_version["id"])
    assert empty_source["payload"]["source_deleted"] is False
    assert empty_result["chunks_prepared"] == 0


def test_identical_rerun_is_idempotent_and_does_not_embed_or_write(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    indexer = _indexer(qdrant, embeddings)
    first = indexer.index([path], dry_run=False)
    before = {str(p["id"]): (copy.deepcopy(p.get("payload") or {}), copy.deepcopy(p.get("vector"))) for p in qdrant.points}
    qdrant.call_log.clear()
    embeddings.documents.clear()

    second = indexer.index([path], dry_run=False)

    after = {str(p["id"]): (copy.deepcopy(p.get("payload") or {}), copy.deepcopy(p.get("vector"))) for p in qdrant.points}
    assert before == after
    assert first["lineage_point_ids"] == second["lineage_point_ids"]
    assert first["lineage_edge_ids"] == second["lineage_edge_ids"]
    assert embeddings.documents == []
    assert qdrant.call_log == []
    _assert_no_event_fields(qdrant, second["lineage_read_back_ids"])
    assert not any(p["payload"].get("lineage_role") == "change_event" for p in qdrant.points)


def test_idempotence_snapshot_detects_in_place_vector_mutation(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    _indexer(qdrant, embeddings).index([path], dry_run=False)
    before = {
        str(point["id"]): copy.deepcopy(point.get("vector"))
        for point in qdrant.points
    }

    chunk = _points(qdrant, kind="source_chunk")[0]
    chunk["vector"][0] = 9.9
    after = {
        str(point["id"]): copy.deepcopy(point.get("vector"))
        for point in qdrant.points
    }

    assert before != after


def test_unchanged_reindex_repairs_exactly_missing_edge(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    indexer = _indexer(qdrant, embeddings)
    indexer.index([path], dry_run=False)
    edge = _points(qdrant, kind="graph_edge", relation="DERIVED_FROM")[0]
    edge_id = str(edge["id"])
    qdrant.points = [point for point in qdrant.points if str(point["id"]) != edge_id]
    before = {str(p["id"]): dict(p.get("payload") or {}) for p in qdrant.points}
    qdrant.call_log.clear()
    embeddings.documents.clear()

    result = indexer.index([path], dry_run=False)

    mutation_ids = [point_id for call in qdrant.call_log if call[0] == "upsert" for point_id in call[2]]
    assert result["lineage_repair_ids"] == mutation_ids == [edge_id]
    assert embeddings.documents == []
    _assert_no_event_fields(qdrant, result["lineage_read_back_ids"])
    assert all(dict(next(p for p in qdrant.points if str(p["id"]) == point_id)["payload"]) == payload
               for point_id, payload in before.items())


def test_unchanged_reindex_reports_patched_chunk_as_repair(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    indexer = _indexer(qdrant, embeddings)
    indexer.index([path], dry_run=False)
    chunk = _points(qdrant, kind="source_chunk")[0]
    chunk_id = str(chunk["id"])
    chunk["payload"].pop("derived_from")
    qdrant.call_log.clear()
    embeddings.documents.clear()

    result = indexer.index([path], dry_run=False)

    assert result["lineage_repair_ids"] == [chunk_id]
    assert qdrant.call_log == [("update_payload", "lineage_test", chunk_id)]
    assert (chunk["payload"].get("derived_from") or [])[0]["relation_type"] == "DERIVED_FROM"
    assert embeddings.documents == []


def test_concurrent_completed_repair_is_not_reported_as_our_write(tmp_path, monkeypatch):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    indexer = _indexer(qdrant, embeddings)
    indexer.index([path], dry_run=False)
    edge = copy.deepcopy(_points(qdrant, kind="graph_edge", relation="DERIVED_FROM")[0])
    qdrant.points = [point for point in qdrant.points if str(point["id"]) != str(edge["id"])]
    real_lock = lineage.collection_write_lock

    @contextmanager
    def finish_repair_before_our_lock_body(**kwargs):
        with real_lock(**kwargs) as held:
            qdrant.points.append(copy.deepcopy(edge))
            qdrant.call_log.clear()
            yield held

    monkeypatch.setattr(lineage, "collection_write_lock", finish_repair_before_our_lock_body)
    result = indexer.index([path], dry_run=False)

    assert result["partial_failure"] is False
    assert result["lineage_repair_ids"] == []
    assert qdrant.call_log == []


def test_dry_run_has_zero_mutation_and_embedding_calls(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()

    result = _indexer(qdrant, embeddings).index([path], dry_run=True)

    assert result["lineage_coverage"]["eligible_files"] == 1
    assert qdrant.call_log == []
    assert qdrant.upserts == qdrant.updates == qdrant.deleted == qdrant.delete_filters == []
    assert embeddings.documents == []


def test_changed_capture_is_blocked_without_touching_old_point(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    indexer = _indexer(qdrant, embeddings)
    indexer.index([path], dry_run=False)
    old_chunk = _points(qdrant, kind="source_chunk")[0]
    old = (dict(old_chunk["payload"]), list(old_chunk["vector"]))
    qdrant.call_log.clear()
    embeddings.documents.clear()
    path.write_text("# Note\nbeta", encoding="utf-8")

    result = indexer.index([path], dry_run=False)

    blocked = result["lineage_blocked_files"][0]
    assert blocked["lineage_blocked_reason"] == "retirement_requires_reconcile"
    current = next(point for point in qdrant.points if str(point["id"]) == str(old_chunk["id"]))
    assert current["payload"] == old[0] and current["vector"] == old[1]
    assert qdrant.call_log == [] and embeddings.documents == []


def test_deleted_capture_is_blocked_without_deleting_chunks(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    indexer = _indexer(qdrant, embeddings)
    indexer.index([tmp_path], dry_run=False)
    before = json.loads(json.dumps(qdrant.points))
    qdrant.call_log.clear()
    path.unlink()

    result = indexer.index([tmp_path], dry_run=False)

    assert result["lineage_blocked_files"][0]["lineage_blocked_reason"] == "retirement_requires_reconcile"
    assert qdrant.points == before
    assert qdrant.call_log == []


def test_no_hash_legacy_is_refused_with_zero_writes(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    qdrant, embeddings, _ = _seed_legacy(
        path,
        mutate=lambda payload: [payload.pop(key, None) for key in ("file_sha256", "source_uri", "locator", "chunk_hash")],
    )
    before = json.loads(json.dumps(qdrant.points))

    result = _indexer(qdrant, embeddings).index([path], dry_run=False)

    blocked = result["lineage_blocked_files"][0]
    assert blocked["lineage_baseline_basis"] == "missing_indexed_file_sha256"
    assert blocked["lineage_blocked_reason"] == "legacy_baseline_requires_reconcile"
    assert set(blocked["lineage_missing_fields"]) == {"file_sha256", "source_uri", "locator", "chunk_hash"}
    assert qdrant.points == before
    assert qdrant.call_log == [] and embeddings.documents == []


def test_event_managed_source_is_blocked_without_reset(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    indexer = _indexer(qdrant, embeddings)
    indexer.index([path], dry_run=False)
    source = next(point for point in _points(qdrant, kind="graph_entity")
                  if point["payload"].get("lineage_role") == "file_source")
    source["payload"]["head_event_id"] = "33333333-3333-3333-3333-333333333333"
    before = json.loads(json.dumps(qdrant.points))
    qdrant.call_log.clear()
    embeddings.documents.clear()

    result = indexer.index([path], dry_run=False)

    assert result["lineage_blocked_files"][0]["lineage_missing_fields"] == ["event_managed_source"]
    assert qdrant.points == before
    assert qdrant.call_log == [] and embeddings.documents == []


def test_hash_present_missing_source_binding_is_incomplete_not_changed(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    qdrant, embeddings, _ = _seed_legacy(path, mutate=lambda payload: payload.pop("source_uri", None))

    result = _indexer(qdrant, embeddings).index([path], dry_run=False)

    blocked = result["lineage_blocked_files"][0]
    assert blocked["lineage_baseline_basis"] == "incomplete_indexed_provenance"
    assert blocked["lineage_missing_fields"] == ["source_uri"]
    assert blocked["lineage_blocked_reason"] == "legacy_baseline_requires_reconcile"
    assert qdrant.call_log == [] and embeddings.documents == []


def test_hash_present_missing_only_memory_kind_is_eligible_despite_unrelated_turn(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    qdrant, embeddings, _ = _seed_legacy(path, mutate=lambda payload: payload.pop("memory_kind", None))
    qdrant.points.append({"id": "turn-1", "vector": [9.0], "payload": {"chunk_type": "turn", "profile_id": "default"}})

    result = _indexer(qdrant, embeddings).index([path], dry_run=False)

    assert result["partial_failure"] is False
    assert result["lineage_files"][0]["lineage_baseline_basis"] == "indexed_file_sha256"
    legacy = next(point for point in qdrant.points if point["id"] != "turn-1" and point["payload"].get("chunk_type") == "file_chunk")
    assert legacy["payload"]["memory_kind"] == "source_chunk"
    assert next(point for point in qdrant.points if point["id"] == "turn-1")["vector"] == [9.0]
    assert embeddings.documents == []


@pytest.mark.parametrize(
    ("hashes", "missing"),
    [(["BAD"], "malformed_file_sha256"), (["a" * 64, "b" * 64], "ambiguous_file_sha256")],
)
def test_malformed_or_ambiguous_hashes_block_without_zero_hash_classification(tmp_path, hashes, missing):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    qdrant, embeddings, indexer = _seed_legacy(path)
    base = qdrant.points[0]
    qdrant.points = []
    for index, value in enumerate(hashes):
        payload = dict(base["payload"])
        payload["file_sha256"] = value
        payload["chunk_index"] = index
        qdrant.points.append({"id": f"old-{index}", "vector": [index], "payload": payload})

    result = _indexer(qdrant, embeddings).index([path], dry_run=False)

    blocked = result["lineage_blocked_files"][0]
    assert blocked["lineage_baseline_basis"] == "incomplete_indexed_provenance"
    assert blocked["lineage_missing_fields"] == [missing]
    assert qdrant.call_log == [] and embeddings.documents == []


def test_collection_lock_times_out_on_contention(tmp_path):
    lock_dir = tmp_path / "locks"
    kwargs = {"collection_name": "lineage_test", "lock_dir": str(lock_dir)}
    with collection_write_lock(**kwargs):
        with pytest.raises(TimeoutError, match="timed out"):
            with collection_write_lock(**kwargs, timeout=0):
                pass
    assert lock_dir.stat().st_mode & 0o077 == 0
    assert next(lock_dir.iterdir()).stat().st_mode & 0o077 == 0


def test_failed_lock_acquisition_never_mutates(tmp_path, monkeypatch):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()

    @contextmanager
    def fail_lock(**kwargs):
        raise TimeoutError("busy")
        yield

    monkeypatch.setattr("qdrant_memory.lineage.collection_write_lock", fail_lock)
    result = _indexer(qdrant, embeddings).index([path], dry_run=False)

    assert result["partial_failure"] is True
    assert "busy" in result["errors"][0]["error"]
    assert qdrant.call_log == []


def test_missing_profile_blocks_but_other_profile_twin_is_not_adopted(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    qdrant, embeddings, _ = _seed_legacy(path)
    original = qdrant.points[0]
    missing_profile = {**original, "id": "missing-profile", "payload": dict(original["payload"])}
    missing_profile["payload"].pop("profile_id", None)
    foreign = {**original, "id": "foreign", "payload": {**original["payload"], "profile_id": "other"}}
    qdrant.points = [missing_profile, foreign]

    result = _indexer(qdrant, embeddings).index([path], dry_run=False)

    assert result["lineage_blocked_files"][0]["lineage_missing_fields"] == ["profile_id"]
    assert qdrant.call_log == []
    qdrant.points = [foreign]
    result = _indexer(qdrant, embeddings).index([path], dry_run=True)
    assert result["lineage_files"][0]["lineage_baseline_basis"] == "new_file"
    assert "foreign" not in result["lineage_existing_ids"]


def test_structural_same_path_record_is_not_stale_chunk(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    structural_id = "11111111-1111-1111-1111-111111111111"
    qdrant.points = [{"id": structural_id, "payload": {
        "file_path": str(path.resolve()), "chunk_type": "graph_record", "profile_id": "default",
        "memory_kind": "graph_entity", "lineage_role": "file_version",
    }}]

    result = _indexer(qdrant, embeddings, mode="off").index([path], dry_run=False)

    assert structural_id not in result["stale_ids"]
    assert not qdrant.deleted
    assert any(str(point["id"]) == structural_id for point in qdrant.points)


def test_snapshot_rejects_stat_change_during_single_read(tmp_path, monkeypatch):
    path = tmp_path / "note.md"
    path.write_text("alpha", encoding="utf-8")
    original = Path.read_bytes

    def changing_read(target):
        value = original(target)
        target.write_bytes(value + b"!")
        return value

    monkeypatch.setattr(Path, "read_bytes", changing_read)
    prepared = _indexer(FakeQdrant(), FakeEmbedding()).prepare([path])
    assert prepared["file_manifests"] == []
    assert prepared["errors"][0]["error"] == "file changed during snapshot read"


def test_interrupted_chunk_publish_repairs_from_durable_metadata(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")

    class FailChunkOnce(FakeQdrant):
        failed = False

        def upsert(self, name, points):
            if not self.failed and any((point.get("payload") or {}).get("chunk_type") == "file_chunk" for point in points):
                self.failed = True
                raise TimeoutError("unknown chunk outcome")
            return super().upsert(name, points)

    qdrant, embeddings = FailChunkOnce(), FakeEmbedding()
    indexer = _indexer(qdrant, embeddings)
    first = indexer.index([path], dry_run=False)
    assert first["partial_failure"] is True
    assert _points(qdrant, kind="source_chunk") == []
    assert len(_points(qdrant, kind="graph_edge", relation="DERIVED_FROM")) == 1

    second = indexer.index([path], dry_run=False)
    chunks = _points(qdrant, kind="source_chunk")
    derived = _points(qdrant, kind="graph_edge", relation="DERIVED_FROM")
    assert second["partial_failure"] is False
    assert len(chunks) == len(derived) == 1
    assert derived[0]["payload"]["source_point_id"] == str(chunks[0]["id"])


def test_unknown_chunk_write_outcome_is_not_reported_success_and_rerun_reconciles(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")

    class PersistThenTimeout(FakeQdrant):
        failed = False

        def upsert(self, name, points):
            result = super().upsert(name, points)
            if not self.failed and any((point.get("payload") or {}).get("chunk_type") == "file_chunk" for point in points):
                self.failed = True
                raise TimeoutError("unknown chunk outcome")
            return result

    qdrant, embeddings = PersistThenTimeout(), FakeEmbedding()
    indexer = _indexer(qdrant, embeddings)
    first = indexer.index([path], dry_run=False)
    assert first["partial_failure"] is True
    assert first["lineage_coverage"]["captured_files"] == 0
    chunk = _points(qdrant, kind="source_chunk")[0]
    assert chunk["payload"]["file_version_id"]
    assert _points(qdrant, kind="graph_edge", relation="DERIVED_FROM")[0]["payload"]["source_point_id"] == str(chunk["id"])

    qdrant.call_log.clear()
    embeddings.documents.clear()
    second = indexer.index([path], dry_run=False)
    assert second["partial_failure"] is False
    assert embeddings.documents == []
    assert qdrant.call_log == []


def test_prepare_reads_each_file_once(tmp_path, monkeypatch):
    path = tmp_path / "note.md"
    path.write_text("alpha", encoding="utf-8")
    original = Path.read_bytes
    calls = []

    def counted_read(target):
        calls.append(str(target))
        return original(target)

    monkeypatch.setattr(Path, "read_bytes", counted_read)
    prepared = _indexer(FakeQdrant(), FakeEmbedding()).prepare([path])
    assert prepared["chunks_prepared"] == 1
    assert calls == [str(path.resolve())]


def test_capture_inventory_matches_contract_formulas(tmp_path):
    path = tmp_path / "note.txt"
    body = b"alpha"
    path.write_bytes(body)
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()

    result = _indexer(qdrant, embeddings).index([path], dry_run=False)

    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = lambda value: hashlib.sha256(canonical(value).encode()).hexdigest()
    point_id = lambda source, text: str(uuid.UUID(hashlib.sha256(f"{source}\n{text}".encode()).hexdigest()[:32]))
    entity_id = lambda kind, label: "entity-" + hashlib.sha256(
        f"entity|default|{kind}|{label}".encode()).hexdigest()[:16]
    edge_id = lambda source, relation, target: "edge-" + hashlib.sha256(
        f"edge|default|{source}|{relation}|{target}".encode()).hexdigest()[:16]
    graph_id = lambda logical: point_id("graph-record-v1", logical)
    resolved = str(path.resolve())
    file_sha = hashlib.sha256(body).hexdigest()
    chunk_hash = hashlib.sha256(body).hexdigest()
    scope_key = digest(["memory-scope-v1", "lineage_test", "default", "", ""])
    source_key = digest(["file-source-v1", scope_key, resolved])
    version_digest = digest(["file-version-v1", source_key, file_sha])
    source_entity = entity_id("source", source_key)
    version_entity = entity_id("source", version_digest)
    chunk_id = point_id("indexed-file-v2", canonical([
        scope_key, resolved, file_sha, "text-markdown-v1:512", 0, chunk_hash,
    ]))
    chunk_endpoint = entity_id("memory_point", digest(["point-endpoint-v1", scope_key, chunk_id]))
    expected_ids = {
        graph_id(source_entity),
        graph_id(version_entity),
        graph_id(edge_id(version_entity, "PART_OF", source_entity)),
        chunk_id,
        graph_id(edge_id(chunk_endpoint, "DERIVED_FROM", version_entity)),
    }

    assert result["chunks_prepared"] == 1
    assert {str(point["id"]) for point in qdrant.points} == expected_ids


def test_event_fields_absent_on_initial_rerun_and_repair(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    indexer = _indexer(qdrant, embeddings)

    initial = indexer.index([path], dry_run=False)
    _assert_no_event_fields(qdrant, initial["lineage_read_back_ids"])
    rerun = indexer.index([path], dry_run=False)
    _assert_no_event_fields(qdrant, rerun["lineage_read_back_ids"])
    edge = _points(qdrant, kind="graph_edge", relation="DERIVED_FROM")[0]
    qdrant.points = [point for point in qdrant.points if str(point["id"]) != str(edge["id"])]
    repaired = indexer.index([path], dry_run=False)
    _assert_no_event_fields(qdrant, repaired["lineage_read_back_ids"])


def test_changed_capture_race_fails_before_stale_publish(tmp_path, monkeypatch):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    real_lock = lineage.collection_write_lock
    state = {"injected": False}

    @contextmanager
    def racing_lock(**kwargs):
        with real_lock(**kwargs) as held:
            if not state["injected"]:
                state["injected"] = True
                path.write_text("# Note\nbeta", encoding="utf-8")
                lineage.collection_write_lock = real_lock
                try:
                    state["newer"] = _indexer(
                        qdrant, embeddings, lock_dir=tmp_path / "newer-lock"
                    ).index([path], dry_run=False)
                finally:
                    lineage.collection_write_lock = racing_lock
            yield held

    monkeypatch.setattr(lineage, "collection_write_lock", racing_lock)
    stale = _indexer(qdrant, embeddings, lock_dir=tmp_path / "stale-lock").index([path], dry_run=False)

    source = next(point for point in qdrant.points if (point.get("payload") or {}).get("lineage_role") == "file_source")
    chunks = _points(qdrant, kind="source_chunk")
    version = next(point for point in qdrant.points if (point.get("payload") or {}).get("lineage_role") == "file_version")
    assert state["newer"]["partial_failure"] is False
    assert stale["partial_failure"] is True
    assert "source appeared with conflicting head" in stale["errors"][0]["error"]
    assert source["payload"]["current_version_id"] == str(version["id"])
    assert [point["payload"]["text"] for point in chunks] == ["# Note\nbeta"]
    assert stale["lineage_repair_ids"] == []


def test_source_appearance_race_fails_with_distinct_reason(tmp_path, monkeypatch):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    indexer = _indexer(qdrant, embeddings, lock_dir=tmp_path / "outer-lock")
    prepared = indexer.prepare([path])
    plan = plan_file_lineage(
        collection_name="lineage_test", profile_id="default", user_id_hash="", chat_id_hash="",
        manifest=prepared["file_manifests"][0], chunks=prepared["chunks"], existing_points=[],
    )
    source_record = next(
        record for record in plan["structural_records"]
        if record["payload"].get("lineage_role") == "file_source"
    )
    real_lock = lineage.collection_write_lock

    @contextmanager
    def racing_lock(**kwargs):
        with real_lock(**kwargs) as held:
            qdrant.upsert("lineage_test", [{"id": source_record["id"], "vector": {}, "payload": source_record["payload"]}])
            yield held

    monkeypatch.setattr(lineage, "collection_write_lock", racing_lock)
    result = indexer.index([path], dry_run=False)

    assert result["partial_failure"] is True
    assert result["errors"][0]["error"].endswith("lineage source appeared before apply")
    assert result["lineage_repair_ids"] == []
    assert _points(qdrant, kind="source_chunk") == []
    assert source_record["payload"]["current_version_id"] == next(
        point["payload"]["current_version_id"] for point in qdrant.points
        if (point.get("payload") or {}).get("lineage_role") == "file_source"
    )


def test_legacy_writer_race_fails_on_owned_inventory_change(tmp_path, monkeypatch):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    real_lock = lineage.collection_write_lock

    @contextmanager
    def racing_lock(**kwargs):
        with real_lock(**kwargs) as held:
            lineage.collection_write_lock = real_lock
            try:
                legacy = _indexer(qdrant, embeddings, mode="off", lock_dir=tmp_path / "legacy-lock")
                legacy.index([path], dry_run=False)
            finally:
                lineage.collection_write_lock = racing_lock
            yield held

    monkeypatch.setattr(lineage, "collection_write_lock", racing_lock)
    result = _indexer(qdrant, embeddings, lock_dir=tmp_path / "capture-lock").index([path], dry_run=False)

    assert result["partial_failure"] is True
    assert "chunk inventory changed before apply" in result["errors"][0]["error"]
    assert result["lineage_repair_ids"] == []
    assert len(_points(qdrant, kind="source_chunk")) == 1
    assert not any((point.get("payload") or {}).get("lineage_record") for point in qdrant.points)


def test_identical_capture_race_still_fails_on_chunk_collision(tmp_path, monkeypatch):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    real_lock = lineage.collection_write_lock
    state = {"injected": False}

    @contextmanager
    def racing_lock(**kwargs):
        with real_lock(**kwargs) as held:
            if not state["injected"]:
                state["injected"] = True
                lineage.collection_write_lock = real_lock
                try:
                    state["newer"] = _indexer(
                        qdrant, embeddings, lock_dir=tmp_path / "newer-lock"
                    ).index([path], dry_run=False)
                finally:
                    lineage.collection_write_lock = racing_lock
            yield held

    monkeypatch.setattr(lineage, "collection_write_lock", racing_lock)
    result = _indexer(qdrant, embeddings, lock_dir=tmp_path / "stale-lock").index([path], dry_run=False)

    assert state["newer"]["partial_failure"] is False
    assert result["partial_failure"] is True
    assert "lineage chunk identity collision" in result["errors"][0]["error"]
    assert len(_points(qdrant, kind="source_chunk")) == 1


def test_interrupted_second_chunk_converges_without_retirement_label(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# A\nalpha\n\n# B\nbeta", encoding="utf-8")

    class FailSecondChunk(FakeQdrant):
        chunk_writes = 0

        def upsert(self, name, points):
            if any((point.get("payload") or {}).get("chunk_type") == "file_chunk" for point in points):
                self.chunk_writes += 1
                if self.chunk_writes == 2:
                    raise TimeoutError("second chunk interrupted")
            return super().upsert(name, points)

    qdrant, embeddings = FailSecondChunk(), FakeEmbedding()
    indexer = _indexer(qdrant, embeddings)
    first = indexer.index([path], dry_run=False)
    assert first["chunks_prepared"] == 2
    assert first["partial_failure"] is True
    assert len(_points(qdrant, kind="source_chunk")) == 1
    embeddings.documents.clear()

    second = indexer.index([path], dry_run=False)

    assert second["partial_failure"] is False
    assert len(_points(qdrant, kind="source_chunk")) == 2
    assert embeddings.documents == ["# B\nbeta"]
    assert not any(
        item["lineage_blocked_reason"] == "retirement_requires_reconcile"
        for item in second["lineage_files"]
    )


@pytest.mark.parametrize("case", ["missing_binding", "foreign_id", "wrong_version"])
def test_incomplete_capture_rejects_nonconvergent_survivor(tmp_path, case):
    path = tmp_path / "note.md"
    path.write_text("# A\nalpha\n\n# B\nbeta", encoding="utf-8")

    class FailSecondChunk(FakeQdrant):
        chunk_writes = 0

        def upsert(self, name, points):
            if any((point.get("payload") or {}).get("chunk_type") == "file_chunk" for point in points):
                self.chunk_writes += 1
                if self.chunk_writes == 2:
                    raise TimeoutError("second chunk interrupted")
            return super().upsert(name, points)

    qdrant, embeddings = FailSecondChunk(), FakeEmbedding()
    indexer = _indexer(qdrant, embeddings)
    first = indexer.index([path], dry_run=False)
    assert first["partial_failure"] is True
    survivor = _points(qdrant, kind="source_chunk")[0]
    if case == "missing_binding":
        survivor["payload"].pop("source_uri")
    elif case == "foreign_id":
        survivor["id"] = "99999999-9999-9999-9999-999999999999"
    else:
        survivor["payload"]["file_version_id"] = "88888888-8888-8888-8888-888888888888"
    qdrant.call_log.clear()
    embeddings.documents.clear()

    result = indexer.index([path], dry_run=False)

    assert result["refused"] is True
    assert result["partial_failure"] is True
    assert result["lineage_blocked_files"][0]["lineage_blocked_reason"] == "incomplete_capture"
    assert len(_points(qdrant, kind="source_chunk")) == 1
    assert qdrant.call_log == []
    assert embeddings.documents == []


def test_off_mode_refuses_to_destroy_or_duplicate_captured_file(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    _indexer(qdrant, embeddings).index([path], dry_run=False)
    v2_chunks = {str(point["id"]) for point in _points(qdrant, kind="source_chunk")}
    edge_targets = {
        edge["payload"]["source_point_id"]
        for edge in _points(qdrant, kind="graph_edge", relation="DERIVED_FROM")
    }
    legacy_ids = {chunk.id for chunk in _indexer(qdrant, embeddings, mode="off").prepare_file(path)}
    qdrant.call_log.clear()
    qdrant.deleted.clear()
    embeddings.documents.clear()

    result = _indexer(qdrant, embeddings, mode="off").index([path], dry_run=False)

    assert result["lineage_blocked_files"][0]["lineage_blocked_reason"] == "lineage_managed_requires_capture_or_retirement"
    assert {str(point["id"]) for point in _points(qdrant, kind="source_chunk")} == v2_chunks
    assert edge_targets <= v2_chunks
    assert legacy_ids.isdisjoint({str(point["id"]) for point in qdrant.points})
    assert qdrant.deleted == [] and qdrant.call_log == []
    assert embeddings.documents == []


# The off-mode reindex is an overwrite route: it rewrites every chunk payload in
# place at its content-derived id, and under --force it deletes the ids it is about
# to rewrite. Its block decision therefore has to answer from the shared overwrite
# predicate. The predecessor test only covers a captured file (identity fields); the
# shapes below are what a transition writes onto an otherwise ordinary chunk, and a
# local subset decision let an ordinary reindex — cron or operator, under `off` or
# after a downgrade — re-serve a stale-pending-review fact as active.


def _demote_chunk(qdrant, **fields):
    """Apply the transition's demotion patch to the single indexed chunk."""
    chunk = _points(qdrant, kind="source_chunk")[0]
    payload = chunk["payload"]
    patch = {"requires_review": True, "fact_status": "review_required",
             "lineage_review_event_ids": ["40dfae4c-0000-4000-8000-000000000001"]}
    patch.update(fields)
    payload.update(patch)
    return chunk, patch


def _off_reindex(tmp_path, qdrant, embeddings, *, force):
    qdrant.call_log.clear()
    qdrant.deleted.clear()
    qdrant.upserts.clear()
    embeddings.documents.clear()
    return _indexer(qdrant, embeddings, mode="off").index(
        [tmp_path / "note.md"], dry_run=False, force=force,
    )


@pytest.mark.parametrize("force", [False, True], ids=("plain", "force"))
def test_off_mode_refuses_to_rewrite_a_chunk_carrying_only_review_state(tmp_path, force):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    _indexer(qdrant, embeddings, mode="off").index([path], dry_run=False)
    chunk, patch = _demote_chunk(qdrant)
    before = json.dumps(chunk["payload"], sort_keys=True, separators=(",", ":")).encode()

    result = _off_reindex(tmp_path, qdrant, embeddings, force=force)

    assert result["lineage_blocked_files"][0]["lineage_blocked_reason"] == "lineage_managed_requires_capture_or_retirement"
    assert result["partial_failure"] is True
    after = _points(qdrant, kind="source_chunk")[0]
    assert json.dumps(after["payload"], sort_keys=True, separators=(",", ":")).encode() == before
    assert after["payload"]["requires_review"] is True
    assert after["payload"]["fact_status"] == "review_required"
    assert qdrant.deleted == []
    assert qdrant.upserts == []
    assert embeddings.documents == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lineage_role", "chunk"),
        ("lineage_scope_key", "a" * 64),
        ("lineage_source_key", "b" * 64),
    ],
    ids=("role", "scope-key", "source-key"),
)
def test_off_mode_refuses_to_rewrite_a_chunk_carrying_one_identity_field(tmp_path, field, value):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    _indexer(qdrant, embeddings, mode="off").index([path], dry_run=False)
    chunk, _ = _demote_chunk(qdrant, **{field: value})
    before = json.dumps(chunk["payload"], sort_keys=True, separators=(",", ":")).encode()

    result = _off_reindex(tmp_path, qdrant, embeddings, force=True)

    assert result["lineage_blocked_files"][0]["lineage_blocked_reason"] == "lineage_managed_requires_capture_or_retirement"
    after = _points(qdrant, kind="source_chunk")[0]
    assert json.dumps(after["payload"], sort_keys=True, separators=(",", ":")).encode() == before
    assert qdrant.deleted == []
    assert qdrant.upserts == []


@pytest.mark.parametrize(
    ("off_profile", "off_user", "off_chat"),
    [("default", "", ""), ("default", "u2", "c2"), ("other", "u1", "c1")],
    ids=("unscoped", "foreign-user-chat", "foreign-profile"),
)
def test_off_mode_scope_blind_block_preserves_adopted_chunk(
    tmp_path, off_profile, off_user, off_chat
):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    legacy = _indexer(qdrant, embeddings, mode="off")
    legacy.user_id_hash, legacy.chat_id_hash = "u1", "c1"
    legacy.index([path], dry_run=False)
    capture = _indexer(qdrant, embeddings)
    capture.user_id_hash, capture.chat_id_hash = "u1", "c1"
    assert not capture.index([path], dry_run=False).get("partial_failure", False)
    chunk = _points(qdrant, kind="source_chunk")[0]
    chunk_id = str(chunk["id"])
    before = json.dumps(chunk, sort_keys=True, separators=(",", ":")).encode()
    qdrant.call_log.clear()
    embeddings.documents.clear()
    off = _indexer(qdrant, embeddings, mode="off")
    off.profile_id = off_profile
    off.user_id_hash, off.chat_id_hash = off_user, off_chat

    result = off.index([path], dry_run=False)

    after_chunk = next(point for point in qdrant.points if str(point["id"]) == chunk_id)
    assert result.get("refused") is True
    assert result.get("partial_failure") is True
    assert (result.get("lineage_blocked_files") or [{}])[0].get("lineage_blocked_reason") == "lineage_managed_requires_capture_or_retirement"
    assert json.dumps(after_chunk, sort_keys=True, separators=(",", ":")).encode() == before
    assert qdrant.call_log == []
    assert embeddings.documents == []


def test_off_mode_inventory_read_failure_refuses_without_writing(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    resolved = str(path.resolve())
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    owner = _indexer(qdrant, embeddings)
    owner.user_id_hash, owner.chat_id_hash = "u1", "c1"
    assert owner.index([path], dry_run=False)["partial_failure"] is False
    chunk = _points(qdrant, kind="source_chunk")[0]
    chunk_id = str(chunk["id"])
    before = json.dumps(chunk, sort_keys=True, separators=(",", ":")).encode()
    original_scroll = qdrant.scroll_by_filter
    raised = False

    def raise_once(name, filter, limit=256, with_payload=True, with_vector=False):
        nonlocal raised
        if not raised and resolved in json.dumps(filter):
            raised = True
            raise RuntimeError("simulated transport error on inventory read")
        return original_scroll(
            name, filter, limit=limit, with_payload=with_payload, with_vector=with_vector
        )

    qdrant.scroll_by_filter = raise_once
    qdrant.call_log.clear()
    embeddings.documents.clear()
    off = _indexer(qdrant, embeddings, mode="off")
    off.user_id_hash, off.chat_id_hash = "u1", "c1"

    result = off.index([path], dry_run=False)

    after = next(point for point in qdrant.points if str(point["id"]) == chunk_id)
    reason = "lineage_inventory_unavailable_refused_fail_closed"
    assert raised is True
    assert result.get("refused") is True
    assert result["partial_failure"] is True
    assert result["chunks_upserted"] == 0
    assert result["foreign_scope_chunks"] == []
    assert result["refusals"] == [{"file_path": resolved, "reason": reason}]
    assert result["lineage_blocked_files"][0]["lineage_blocked_reason"] == reason
    assert result["errors"] == [{
        "file_path": resolved,
        "error": "manifest sync failed: simulated transport error on inventory read",
    }]
    assert json.dumps(after, sort_keys=True, separators=(",", ":")).encode() == before
    assert qdrant.call_log == []
    assert qdrant.deleted == [] and qdrant.delete_filters == []
    assert embeddings.documents == []


def test_off_mode_force_skips_lineage_managed_filter_delete_in_dry_and_live_runs(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    _indexer(qdrant, embeddings).index([path], dry_run=False)
    v2_chunks = {str(point["id"]) for point in _points(qdrant, kind="source_chunk")}
    qdrant.call_log.clear()
    qdrant.delete_filters.clear()
    embeddings.documents.clear()
    off = _indexer(qdrant, embeddings, mode="off")

    dry = off.index([path], dry_run=True, force=True)
    live = off.index([path], dry_run=False, force=True)

    assert dry["delete_mode"] == live["delete_mode"] == "none"
    assert dry["filter_delete_paths"] == live["filter_delete_paths"] == []
    assert live["chunks_deleted"] == 0
    assert live["errors"] == []
    assert qdrant.delete_filters == []
    assert {str(point["id"]) for point in _points(qdrant, kind="source_chunk")} == v2_chunks
    assert qdrant.call_log == []
    assert embeddings.documents == []


@pytest.mark.parametrize("shape", ["captured", "review_only"], ids=("captured", "review-only"))
def test_directory_off_mode_does_not_delete_removed_lineage_managed_chunks(tmp_path, shape):
    """A removed file's chunks are the second place the off-mode block decision runs.

    It used to decide from three identity fields, so a chunk whose only lineage state
    was the review marker a transition wrote was deleted as stale — the same payload
    the store, extraction and restore routes refuse to replace.
    """
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    if shape == "captured":
        _indexer(qdrant, embeddings).index([path], dry_run=False)
    else:
        _indexer(qdrant, embeddings, mode="off").index([path], dry_run=False)
        _demote_chunk(qdrant)
    v2_chunks = {str(point["id"]) for point in _points(qdrant, kind="source_chunk")}
    path.unlink()
    qdrant.call_log.clear()
    qdrant.deleted.clear()

    result = _indexer(qdrant, embeddings, mode="off").index([tmp_path], dry_run=False)

    assert result["deleted_file_ids"] == []
    assert result["stale_ids"] == []
    assert qdrant.deleted == []
    assert {str(point["id"]) for point in _points(qdrant, kind="source_chunk")} == v2_chunks
    assert result["lineage_blocked_files"][0]["lineage_blocked_reason"] == "lineage_managed_requires_capture_or_retirement"


def test_a_removed_file_whose_siblings_are_ordinary_is_blocked_as_a_file(tmp_path):
    """The removed-file block is decided per file, not per chunk.

    A transition demotes one chunk of a multi-chunk file. That chunk carries the review
    state, its siblings carry nothing, and the file is gone from disk. Deciding per point
    here — the decision the present-file site already made per file — marked the path
    blocked, then deleted every sibling anyway and listed the path as deleted, so the
    block held exactly the one id that was protected and leaked all the others.
    """
    path = tmp_path / "note.md"
    path.write_text(
        "# Note\n\n" + "\n\n".join(
            f"## Section {index}\n" + " ".join(f"s{index}w{word}" for word in range(12))
            for index in range(6)
        ) + "\n",
        encoding="utf-8",
    )
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    _indexer(qdrant, embeddings, mode="off").index([path], dry_run=False)
    chunks = _points(qdrant, kind="source_chunk")
    assert len(chunks) > 1, "the fixture must produce siblings for this pin to mean anything"
    protected = chunks[0]
    payload = protected["payload"]
    payload.update({
        "requires_review": True,
        "fact_status": "review_required",
        "lineage_review_event_ids": ["40dfae4c-0000-4000-8000-000000000002"],
    })
    v2_chunks = {str(point["id"]) for point in chunks}
    path.unlink()
    qdrant.call_log.clear()
    qdrant.deleted.clear()
    qdrant.upserts.clear()
    embeddings.documents.clear()

    result = _indexer(qdrant, embeddings, mode="off").index([tmp_path], dry_run=False)

    assert result["lineage_blocked_files"][0]["lineage_blocked_reason"] == "lineage_managed_requires_capture_or_retirement"
    assert result["refused"] is True
    assert result["deleted_file_paths"] == []
    assert result["deleted_file_ids"] == []
    assert result["stale_ids"] == []
    assert result["files_with_stale_chunks"] == 0
    assert qdrant.deleted == []
    assert qdrant.upserts == []
    assert embeddings.documents == []
    assert {str(point["id"]) for point in _points(qdrant, kind="source_chunk")} == v2_chunks


@pytest.mark.parametrize("force", [False, True], ids=("plain", "force"))
def test_a_removed_file_with_one_protected_sibling_survives_force_too(tmp_path, force):
    path = tmp_path / "note.md"
    path.write_text(
        "# Note\n\n" + "\n\n".join(
            f"## Section {index}\n" + " ".join(f"s{index}w{word}" for word in range(12))
            for index in range(5)
        ) + "\n",
        encoding="utf-8",
    )
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    _indexer(qdrant, embeddings, mode="off").index([path], dry_run=False)
    chunks = _points(qdrant, kind="source_chunk")
    # The protected chunk is the last one, so a per-point decision deletes its sibling
    # predecessors instead of its successors. Both orders are the same defect.
    chunks[-1]["payload"].update({
        "requires_review": True,
        "fact_status": "review_required",
        "lineage_review_event_ids": ["40dfae4c-0000-4000-8000-000000000003"],
    })
    v2_chunks = {str(point["id"]) for point in chunks}
    path.unlink()
    qdrant.call_log.clear()
    qdrant.deleted.clear()

    result = _indexer(qdrant, embeddings, mode="off").index(
        [tmp_path], dry_run=False, force=force,
    )

    assert result["deleted_file_ids"] == []
    assert result["stale_ids"] == []
    assert qdrant.deleted == []
    assert {str(point["id"]) for point in _points(qdrant, kind="source_chunk")} == v2_chunks
    assert result["lineage_blocked_files"][0]["lineage_blocked_reason"] == "lineage_managed_requires_capture_or_retirement"


def _seed_foreign_protected_sibling(path, qdrant):
    """A file path holding one owned ordinary chunk and one foreign protected chunk.

    The protected point carries review state and an unclaimed profile, which is the shape
    a transition can leave behind: the review state lands on whichever point the sweep
    wrote, and `_owned_file_chunks` only claims the points whose profile, user and chat
    hashes match the running indexer.
    """
    resolved = str(path.resolve())
    owned_id = "22222222-2222-4222-8222-222222222222"
    foreign_id = "11111111-1111-4111-8111-111111111111"
    qdrant.points.extend([
        {"id": owned_id, "vector": [0.1, 0.2], "payload": {
            "file_path": resolved, "chunk_type": "file_chunk", "memory_kind": "source_chunk",
            "profile_id": "default", "user_id_hash": "u1", "chat_id_hash": "c1",
        }},
        {"id": foreign_id, "vector": [0.1, 0.2], "payload": {
            "file_path": resolved, "chunk_type": "file_chunk", "memory_kind": "source_chunk",
            "profile_id": "other", "user_id_hash": "u1", "chat_id_hash": "c1",
            "requires_review": True, "fact_status": "review_required",
            "lineage_review_event_ids": ["40dfae4c-0000-4000-8000-000000000005"],
        }},
    ])
    return owned_id, foreign_id


def test_a_removed_file_whose_only_protected_chunk_is_foreign_scope_is_blocked(tmp_path):
    """The removed-file and present-file sites answer the same question over one population.

    The present-file site reads every scope (`profile_id=None`), because the transition's
    review state can land on a point the ownership filter does not claim. The removed-file
    site read owned chunks only, so a removed path whose only protected chunk was
    foreign-scope was not blocked, and its owned chunks were deleted live while §27
    promises `off` refuses managed files. Deciding both sites over the same population is
    the difference between the two branches of one block agreeing and diverging.
    """
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    owned_id, foreign_id = _seed_foreign_protected_sibling(path, qdrant)
    indexer = _indexer(qdrant, embeddings, mode="off")
    indexer.user_id_hash, indexer.chat_id_hash = "u1", "c1"
    path.unlink()
    qdrant.call_log.clear()
    qdrant.deleted.clear()
    qdrant.upserts.clear()

    result = indexer.index([tmp_path], dry_run=False)

    assert result["refused"] is True
    assert result["lineage_blocked_files"][0]["lineage_blocked_reason"] == "lineage_managed_requires_capture_or_retirement"
    assert result["lineage_blocked_files"][0]["file_path"] == str(path.resolve())
    assert result["deleted_file_paths"] == []
    assert result["deleted_file_ids"] == []
    assert result["stale_ids"] == []
    assert result["files_with_stale_chunks"] == 0
    assert qdrant.deleted == []
    assert qdrant.upserts == []
    assert {str(point["id"]) for point in qdrant.points} == {owned_id, foreign_id}


def test_a_present_file_whose_only_protected_chunk_is_foreign_scope_is_blocked_too(tmp_path):
    """The present-file half of the same question, so the pair cannot drift apart again."""
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    owned_id, foreign_id = _seed_foreign_protected_sibling(path, qdrant)
    indexer = _indexer(qdrant, embeddings, mode="off")
    indexer.user_id_hash, indexer.chat_id_hash = "u1", "c1"

    result = indexer.index([tmp_path], dry_run=False, force=True)

    assert result["refused"] is True
    assert result["lineage_blocked_files"][0]["lineage_blocked_reason"] == "lineage_managed_requires_capture_or_retirement"
    assert qdrant.deleted == []
    assert qdrant.upserts == []
    assert {str(point["id"]) for point in qdrant.points} == {owned_id, foreign_id}


def test_foreign_scope_only_force_has_dry_live_parity_without_deletion(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    resolved = str(path.resolve())
    foreign_id = "11111111-1111-1111-1111-111111111111"
    foreign = {"id": foreign_id, "vector": [0.1, 0.2], "payload": {
        "file_path": resolved, "chunk_type": "file_chunk", "profile_id": "default",
        "user_id_hash": "u1", "chat_id_hash": "c1",
    }}
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    qdrant.points = [copy.deepcopy(foreign)]
    off = _indexer(qdrant, embeddings, mode="off")
    off.user_id_hash, off.chat_id_hash = "u2", "c2"

    dry = off.index([path], dry_run=True, force=True)
    live = off.index([path], dry_run=False, force=True)

    for result in (dry, live):
        assert result["stale_ids"] == []
        assert result["delete_mode"] == "none"
        assert result["chunks_deleted"] == 0
        assert result["filter_delete_paths"] == []
        assert result["foreign_scope_chunks"] == [{"file_path": resolved, "count": 1}]
        assert result["partial_failure"] is False
    assert qdrant.deleted == [] and qdrant.delete_filters == []
    assert next(point for point in qdrant.points if str(point["id"]) == foreign_id) == foreign


def test_directory_foreign_scope_report_excludes_owned_chunks(tmp_path):
    captured = tmp_path / "cap.md"
    captured.write_text("# Captured\nalpha", encoding="utf-8")
    foreign = tmp_path / "foreign.md"
    foreign.write_text("# Foreign\nbeta", encoding="utf-8")
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    owner = _indexer(qdrant, embeddings)
    owner.user_id_hash, owner.chat_id_hash = "u1", "c1"
    assert owner.index([captured], dry_run=False).get("partial_failure") is False
    owner_off = _indexer(qdrant, embeddings, mode="off")
    owner_off.user_id_hash, owner_off.chat_id_hash = "u1", "c1"

    assert owner_off.index([tmp_path], dry_run=True)["foreign_scope_chunks"] == []

    qdrant.points.append({
        "id": "11111111-1111-1111-1111-111111111111",
        "vector": [0.1, 0.2],
        "payload": {
            "file_path": str(foreign.resolve()),
            "chunk_type": "file_chunk",
            "profile_id": "default",
            "user_id_hash": "u2",
            "chat_id_hash": "c2",
        },
    })
    result = owner_off.index([tmp_path], dry_run=True)

    assert result["foreign_scope_chunks"] == [{
        "file_path": str(foreign.resolve()), "count": 1,
    }]


def test_dry_run_foreign_scope_report_redacts_v2_chunk_ids(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    resolved = str(path.resolve())
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    owner = _indexer(qdrant, embeddings)
    owner.user_id_hash, owner.chat_id_hash = "u1", "c1"
    assert owner.index([path], dry_run=False)["partial_failure"] is False
    owner_ids = {str(point["id"]) for point in _points(qdrant, kind="source_chunk")}
    qdrant.call_log.clear()
    foreign = _indexer(qdrant, embeddings, mode="off")
    foreign.profile_id = "other"
    foreign.user_id_hash, foreign.chat_id_hash = "u1", "c1"

    result = foreign.index([path], dry_run=True)

    report = result["foreign_scope_chunks"]
    rendered = json.dumps(report, sort_keys=True)
    assert report == [{"file_path": resolved, "count": len(owner_ids)}]
    assert all("chunk_ids" not in item for item in report)
    assert all(point_id not in rendered for point_id in owner_ids)
    assert qdrant.call_log == []


def test_off_mode_foreign_chunk_overwritten_this_run_is_not_reported_as_leftover(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    qdrant, embeddings, foreign = _seed_legacy(path)
    foreign.user_id_hash, foreign.chat_id_hash = "u1", "c1"
    target = _indexer(qdrant, embeddings, mode="off")
    target.user_id_hash, target.chat_id_hash = "u2", "c2"

    dry = target.index([path], dry_run=True)
    live = target.index([path], dry_run=False)

    assert dry["foreign_scope_chunks"] == live["foreign_scope_chunks"] == []
    assert live["chunks_upserted"] == 1
    stored = next(point for point in qdrant.points if str(point["id"]) == foreign.prepare_file(path)[0].id)
    assert stored["payload"]["user_id_hash"] == "u2"
    assert stored["payload"]["chat_id_hash"] == "c2"


def test_foreign_scope_chunks_are_reported_but_never_deleted_by_force_or_directory_scan(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    resolved = str(path.resolve())
    foreign_id = "11111111-1111-1111-1111-111111111111"
    owned_id = "22222222-2222-2222-2222-222222222222"
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    foreign = {"id": foreign_id, "vector": [0.1, 0.2], "payload": {
        "file_path": resolved, "chunk_type": "file_chunk", "profile_id": "default",
        "user_id_hash": "u1", "chat_id_hash": "c1",
    }}
    owned = {"id": owned_id, "vector": [0.1, 0.2], "payload": {
        "file_path": resolved, "chunk_type": "file_chunk", "profile_id": "default",
        "user_id_hash": "u2", "chat_id_hash": "c2",
    }}
    qdrant.points = [copy.deepcopy(foreign), copy.deepcopy(owned)]
    off = _indexer(qdrant, embeddings, mode="off")
    off.user_id_hash, off.chat_id_hash = "u2", "c2"

    dry_force = off.index([path], dry_run=True, force=True)
    force = off.index([path], dry_run=False, force=True)

    assert dry_force["stale_ids"] == force["stale_ids"] == [owned_id]
    assert dry_force["delete_mode"] == force["delete_mode"] == "ids"
    assert qdrant.deleted == [("lineage_test", [owned_id])]
    assert next(point for point in qdrant.points if str(point["id"]) == foreign_id) == foreign
    assert force["foreign_scope_chunks"] == [{"file_path": resolved, "count": 1}]

    qdrant.points = [copy.deepcopy(foreign)]
    qdrant.call_log.clear()
    qdrant.deleted.clear()
    path.unlink()
    dry = off.index([tmp_path], dry_run=True)
    live = off.index([tmp_path], dry_run=False)

    for result in (dry, live):
        assert result["stale_ids"] == []
        assert result["deleted_file_ids"] == []
        assert result["delete_mode"] == "none"
        assert result["foreign_scope_chunks"] == [{"file_path": resolved, "count": 1}]
    assert qdrant.call_log == []
    assert qdrant.deleted == []
    assert qdrant.points == [foreign]


@pytest.mark.parametrize(
    ("field", "before_delta", "after_delta"),
    [
        ("st_size", 1, 0),
        ("st_mtime_ns", 0, 1),
        ("st_ctime_ns", 0, 1),
        ("st_ino", 0, 1),
        ("st_dev", 0, 1),
    ],
)
def test_snapshot_rejects_each_identity_drift_component(tmp_path, monkeypatch, field, before_delta, after_delta):
    path = (tmp_path / "note.md").resolve()
    path.write_text("alpha", encoding="utf-8")
    real_stat = Path.stat
    baseline = real_stat(path)
    calls = 0

    def fake_stat(target, *args, **kwargs):
        nonlocal calls
        if target != path:
            return real_stat(target, *args, **kwargs)
        calls += 1
        values = {
            "st_size": baseline.st_size,
            "st_mtime": baseline.st_mtime,
            "st_mtime_ns": baseline.st_mtime_ns,
            "st_ctime_ns": baseline.st_ctime_ns,
            "st_ino": baseline.st_ino,
            "st_dev": baseline.st_dev,
        }
        values[field] += before_delta if calls == 1 else after_delta
        return SimpleNamespace(**values)

    monkeypatch.setattr(Path, "resolve", lambda target: target)
    monkeypatch.setattr(Path, "stat", fake_stat)
    with pytest.raises(RuntimeError, match="file changed during snapshot read"):
        _indexer(FakeQdrant(), FakeEmbedding())._prepare_snapshot(path)


def test_snapshot_rejects_same_size_rewrite_with_restored_mtime(tmp_path, monkeypatch):
    path = tmp_path / "note.md"
    path.write_bytes(b"alpha")
    original_read = Path.read_bytes
    original = path.stat()

    def rewrite_after_read(target):
        value = original_read(target)
        target.write_bytes(b"bravo")
        os.utime(target, ns=(original.st_atime_ns, original.st_mtime_ns))
        return value

    monkeypatch.setattr(Path, "read_bytes", rewrite_after_read)
    with pytest.raises(RuntimeError, match="file changed during snapshot read"):
        _indexer(FakeQdrant(), FakeEmbedding())._prepare_snapshot(path)


def test_access_bookkeeping_does_not_block_repair(tmp_path, monkeypatch):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    indexer = _indexer(qdrant, embeddings)
    indexer.index([path], dry_run=False)
    chunk = _points(qdrant, kind="source_chunk")[0]
    edge = _points(qdrant, kind="graph_edge", relation="DERIVED_FROM")[0]
    qdrant.points = [point for point in qdrant.points if str(point["id"]) != str(edge["id"])]
    real_lock = lineage.collection_write_lock

    @contextmanager
    def bump_access(**kwargs):
        with real_lock(**kwargs) as held:
            qdrant.update_payload("lineage_test", str(chunk["id"]), {
                "access_count": 7, "last_accessed": "2026-09-20T00:00:00Z",
            })
            qdrant.call_log.clear()
            yield held

    monkeypatch.setattr(lineage, "collection_write_lock", bump_access)
    result = indexer.index([path], dry_run=False)

    mutation_ids = [point_id for call in qdrant.call_log if call[0] == "upsert" for point_id in call[2]]
    assert result["partial_failure"] is False
    assert result["lineage_repair_ids"] == mutation_ids == [str(edge["id"])]
    assert any(str(point["id"]) == str(edge["id"]) for point in qdrant.points)


def test_invariant_change_still_blocks_repair(tmp_path, monkeypatch):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    indexer = _indexer(qdrant, embeddings)
    indexer.index([path], dry_run=False)
    chunk = _points(qdrant, kind="source_chunk")[0]
    edge = _points(qdrant, kind="graph_edge", relation="DERIVED_FROM")[0]
    qdrant.points = [point for point in qdrant.points if str(point["id"]) != str(edge["id"])]
    real_lock = lineage.collection_write_lock

    @contextmanager
    def change_hash(**kwargs):
        with real_lock(**kwargs) as held:
            qdrant.update_payload("lineage_test", str(chunk["id"]), {"chunk_hash": "0" * 64})
            qdrant.call_log.clear()
            yield held

    monkeypatch.setattr(lineage, "collection_write_lock", change_hash)
    result = indexer.index([path], dry_run=False)

    assert result["partial_failure"] is True
    assert "lineage baseline changed before apply" in result["errors"][0]["error"]
    assert not any(str(point["id"]) == str(edge["id"]) for point in qdrant.points)
    assert result["lineage_repair_ids"] == []


@pytest.mark.parametrize("field", ["file_sha256", "text"])
def test_snapshot_fingerprint_includes_file_identity_and_text(tmp_path, monkeypatch, field):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    indexer = _indexer(qdrant, embeddings)
    indexer.index([path], dry_run=False)
    chunk = _points(qdrant, kind="source_chunk")[0]
    real_lock = lineage.collection_write_lock

    @contextmanager
    def change_snapshot(**kwargs):
        with real_lock(**kwargs) as held:
            value = "0" * 64 if field == "file_sha256" else "foreign text"
            qdrant.update_payload("lineage_test", str(chunk["id"]), {field: value})
            qdrant.call_log.clear()
            yield held

    monkeypatch.setattr(lineage, "collection_write_lock", change_snapshot)
    result = indexer.index([path], dry_run=False)

    assert result["partial_failure"] is True
    assert result["errors"][0]["error"].endswith("lineage baseline changed before apply")
    assert qdrant.call_log == []


def test_expected_source_fingerprint_blocks_apply_time_source_rewrite(tmp_path, monkeypatch):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    indexer = _indexer(qdrant, embeddings)
    indexer.index([path], dry_run=False)
    source = next(point for point in _points(qdrant, kind="graph_entity")
                  if point["payload"].get("lineage_role") == "file_source")
    real_lock = lineage.collection_write_lock

    @contextmanager
    def change_source(**kwargs):
        with real_lock(**kwargs) as held:
            qdrant.update_payload("lineage_test", str(source["id"]), {"source_deleted": True})
            qdrant.call_log.clear()
            yield held

    monkeypatch.setattr(lineage, "collection_write_lock", change_source)
    result = indexer.index([path], dry_run=False)

    assert result["partial_failure"] is True
    assert result["errors"][0]["error"].endswith("lineage source changed before apply")
    assert qdrant.call_log == []


def test_lock_path_ignores_origin_spelling_but_separates_collection_and_uid(tmp_path, monkeypatch):
    real_lock = lineage.collection_write_lock
    observed = []

    @contextmanager
    def record_lock(**kwargs):
        with real_lock(**kwargs) as held:
            observed.append(held)
            yield held

    monkeypatch.setattr(lineage, "collection_write_lock", record_lock)
    origins = [
        "http://localhost", "http://127.0.0.1:6333",
        "http://localhost:6333/", "http://localhost:6333/qdrant",
    ]
    for index, origin in enumerate(origins):
        path = tmp_path / f"note-{index}.md"
        path.write_text("alpha", encoding="utf-8")
        indexer = FileIndexer(
            qdrant=FakeQdrant(), embeddings=FakeEmbedding(), collection_name="lineage_test",
            config={"lineage_mode": "capture", "qdrant_url": origin,
                    "lineage_lock_dir": str(tmp_path / "locks")},
        )
        assert indexer.index([path], dry_run=False)["partial_failure"] is False
    assert len(set(observed)) == 1
    assert lineage._collection_lock_digest("a", os.getuid()) != lineage._collection_lock_digest("b", os.getuid())
    assert lineage._collection_lock_digest("a", 1000) != lineage._collection_lock_digest("a", 1001)


def test_missing_lock_directory_and_backend_fail_closed(tmp_path, monkeypatch):
    lock_file = tmp_path / "not-a-directory"
    lock_file.write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError, match="lock directory unavailable"):
        with collection_write_lock(collection_name="c", lock_dir=str(lock_file)):
            pass

    real_import = builtins.__import__

    def no_fcntl(name, *args, **kwargs):
        if name == "fcntl":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_fcntl)
    with pytest.raises(RuntimeError, match="locking is unsupported"):
        with collection_write_lock(collection_name="c", lock_dir=str(tmp_path / "locks")):
            pass


def test_collection_lock_contends_across_processes(tmp_path):
    context = multiprocessing.get_context("fork")
    ready, release = context.Event(), context.Event()
    process = context.Process(target=_hold_lock, args=(str(tmp_path / "locks"), ready, release))
    process.start()
    try:
        assert ready.wait(5)
        with pytest.raises(TimeoutError, match="timed out"):
            with collection_write_lock(
                collection_name="lineage_test", lock_dir=str(tmp_path / "locks"), timeout=0.1
            ):
                pass
    finally:
        release.set()
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(5)
    assert process.exitcode == 0


def test_merge_loop_rejects_head_changed_after_source_reread(tmp_path, monkeypatch):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    indexer = _indexer(qdrant, embeddings)
    indexer.index([path], dry_run=False)
    source = next(point for point in qdrant.points if (point.get("payload") or {}).get("lineage_role") == "file_source")
    edge = _points(qdrant, kind="graph_edge", relation="DERIVED_FROM")[0]
    qdrant.points = [point for point in qdrant.points if str(point["id"]) != str(edge["id"])]
    source_id = str(source["id"])
    original_retrieve = qdrant.retrieve
    state = {"under_lock": False, "source_checked": False, "changed": False}
    real_lock = lineage.collection_write_lock

    @contextmanager
    def mark_lock(**kwargs):
        with real_lock(**kwargs) as held:
            state["under_lock"] = True
            yield held

    def racing_retrieve(name, ids, with_payload=True, with_vector=False):
        ids = [str(value) for value in ids]
        if state["under_lock"] and ids == [source_id]:
            state["source_checked"] = True
        elif state["source_checked"] and source_id in ids and len(ids) > 1 and not state["changed"]:
            state["changed"] = True
            source["payload"]["current_version_id"] = "11111111-1111-1111-1111-111111111111"
        return original_retrieve(name, ids, with_payload=with_payload, with_vector=with_vector)

    monkeypatch.setattr(lineage, "collection_write_lock", mark_lock)
    monkeypatch.setattr(qdrant, "retrieve", racing_retrieve)
    result = indexer.index([path], dry_run=False)

    assert result["partial_failure"] is True
    assert "lineage head conflict before apply" in result["errors"][0]["error"]
    assert result["lineage_repair_ids"] == []
    assert source["payload"]["current_version_id"] == "11111111-1111-1111-1111-111111111111"


def test_exact_read_back_rejects_subset_response(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")

    class SubsetReadBack(FakeQdrant):
        def retrieve(self, name, ids, with_payload=True, with_vector=False):
            records = super().retrieve(name, ids, with_payload=with_payload, with_vector=with_vector)
            if self.upserts and len(ids) > 1 and records:
                return records[:-1]
            return records

    qdrant, embeddings = SubsetReadBack(), FakeEmbedding()
    result = _indexer(qdrant, embeddings).index([path], dry_run=False)

    assert result["partial_failure"] is True
    assert "lineage read-back missing exact IDs" in result["errors"][0]["error"]
    assert result["lineage_coverage"]["captured_files"] == 0


def test_empty_inventory_rejects_conflicting_source_head(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_bytes(b"")
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    indexer = _indexer(qdrant, embeddings)
    assert indexer.index([path], dry_run=False)["partial_failure"] is False
    source = next(point for point in qdrant.points if (point.get("payload") or {}).get("lineage_role") == "file_source")
    source["payload"]["current_version_id"] = "11111111-1111-1111-1111-111111111111"
    qdrant.call_log.clear()

    result = indexer.index([path], dry_run=False)

    assert result["refused"] is True
    assert result["partial_failure"] is True
    assert result["lineage_blocked_files"][0]["lineage_blocked_reason"] == "retirement_requires_reconcile"
    assert qdrant.call_log == []


def test_structural_records_never_copy_file_body_or_secret_shaped_metadata(tmp_path):
    path = tmp_path / "note.md"
    sentinel = "".join(("sk", "-proj-", "SENTINEL-DO-NOT-COPY-1234567890"))
    path.write_text(f"# Note\n{sentinel}", encoding="utf-8")
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()

    result = _indexer(qdrant, embeddings).index([path], dry_run=False)

    assert result["partial_failure"] is False
    structural = [point for point in qdrant.points if (point.get("payload") or {}).get("lineage_record") is True]
    assert structural
    rendered = json.dumps(structural, sort_keys=True)
    assert sentinel not in rendered
    for point in structural:
        payload = point["payload"]
        assert "lesson" not in payload
        assert sentinel not in str(payload.get("text") or "")
        assert len(str(payload.get("text") or "")) <= 256
        assert all(not str(key).startswith("secret") for key in payload)
        locator = payload.get("locator")
        assert locator is None or set(locator) <= {"line_start", "line_end", "heading"}


def test_unscoped_capture_neither_adopts_nor_deletes_foreign_scope_chunks(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# Note\nalpha", encoding="utf-8")
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()
    scoped = _indexer(qdrant, embeddings)
    scoped.user_id_hash = "u1"
    scoped.chat_id_hash = "c1"
    assert scoped.index([path], dry_run=False)["partial_failure"] is False
    foreign_before = {
        str(point["id"]): copy.deepcopy(point)
        for point in _points(qdrant, kind="source_chunk")
    }
    qdrant.call_log.clear()
    embeddings.documents.clear()

    unscoped = _indexer(qdrant, embeddings)
    result = unscoped.index([path], dry_run=False)

    assert result["partial_failure"] is False
    assert len(_points(qdrant, kind="source_chunk")) == 2
    assert all(next(point for point in qdrant.points if str(point["id"]) == point_id) == before
               for point_id, before in foreign_before.items())
    assert qdrant.deleted == [] and qdrant.delete_filters == []

    path.unlink()
    qdrant.call_log.clear()
    off = _indexer(qdrant, embeddings, mode="off").index([tmp_path], dry_run=False)
    assert off["stale_ids"] == []
    assert qdrant.deleted == [] and qdrant.delete_filters == []
    assert all(next(point for point in qdrant.points if str(point["id"]) == point_id) == before
               for point_id, before in foreign_before.items())


def test_structural_read_back_invariant_rejects_payload_corruption(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("alpha", encoding="utf-8")

    class CorruptStructuralReadback(FakeQdrant):
        def retrieve(self, name, ids, with_payload=True, with_vector=False):
            result = super().retrieve(name, ids, with_payload=with_payload, with_vector=with_vector)
            if len(ids) >= 4:
                for point in result:
                    if (point.get("payload") or {}).get("lineage_role") == "file_source":
                        point["payload"]["source_deleted"] = True
                        break
            return result

    result = _indexer(CorruptStructuralReadback(), FakeEmbedding()).index([path], dry_run=False)

    assert result["partial_failure"] is True
    assert "lineage read-back invariant mismatch" in result["errors"][0]["error"]


def test_chunk_read_back_invariant_rejects_payload_corruption(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("alpha", encoding="utf-8")

    class CorruptChunkReadback(FakeQdrant):
        def retrieve(self, name, ids, with_payload=True, with_vector=False):
            result = super().retrieve(name, ids, with_payload=with_payload, with_vector=with_vector)
            if len(ids) >= 4:
                for point in result:
                    if (point.get("payload") or {}).get("memory_kind") == "source_chunk":
                        point["payload"]["file_sha256"] = "0" * 64
                        break
            return result

    result = _indexer(CorruptChunkReadback(), FakeEmbedding()).index([path], dry_run=False)

    assert result["partial_failure"] is True
    assert "chunk read-back invariant mismatch" in result["errors"][0]["error"]


def test_off_mode_fresh_store_has_zero_graph_records(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("alpha", encoding="utf-8")
    qdrant, embeddings = FakeQdrant(), FakeEmbedding()

    result = _indexer(qdrant, embeddings, mode="off").index([path], dry_run=False)

    assert result["chunks_upserted"] == 1
    assert not any((point.get("payload") or {}).get("lineage_record") for point in qdrant.points)
    assert _points(qdrant, kind="graph_entity") == []
    assert _points(qdrant, kind="graph_edge") == []


def test_index_tool_false_strings_execute_live(tmp_path):
    from __init__ import QdrantMemoryProvider

    path = tmp_path / "note.md"
    path.write_text("alpha", encoding="utf-8")
    for args, configured in [({"paths": [str(path)], "dry_run": "false"}, True), ({"paths": [str(path)]}, "false")]:
        provider = QdrantMemoryProvider()
        provider._qdrant = FakeQdrant()
        provider._embeddings = FakeEmbedding()
        provider._config.update(collection_name="c", index_dry_run_default=configured, lineage_mode="off")
        result = json.loads(provider._tool_index(args))
        assert result["dry_run"] is False
        assert result["chunks_upserted"] == 1
