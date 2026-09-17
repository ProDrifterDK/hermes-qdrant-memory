"""W0 live-service proof: graph storage identity and payload-only round-trip.

Disposable collections only (fixed prefix `hermes_qdrant_lineage_itest`).
`hermes_memory` / `hermes_learnings` are never touched.

Bite contract: the negative-control env `LINEAGE_NEGATIVE_CONTROL=use_raw_ids`
bypasses the UUID mapping inside THIS TEST ONLY (never production code), so the
designated positive persistence assertions fail against real Qdrant exactly as
they would under the pre-W0 raw logical-handle mechanism:

    LINEAGE_NEGATIVE_CONTROL=use_raw_ids pytest ... -k graph_storage   # must FAIL
    pytest ... -k graph_storage                                        # must PASS
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.parse

import pytest

from conftest import LINEAGE_TEST_PREFIX, LineageContext

from qdrant_memory import lineage
from qdrant_memory.backup import create_backup, restore_backup
from qdrant_memory.graph_schema import is_uuid_string, make_graph_point_id

TRUTHY = {"1", "true", "yes", "on"}


def _integration_enabled() -> bool:
    return os.environ.get("RUN_QDRANT_INTEGRATION", "").strip().lower() in TRUTHY


pytestmark = pytest.mark.skipif(
    not _integration_enabled(),
    reason="set RUN_QDRANT_INTEGRATION=1 to run live Qdrant integration tests",
)


def _negative_control() -> bool:
    """Fixture-only negative control: bypass the UUID storage-ID mapping."""
    return os.environ.get("LINEAGE_NEGATIVE_CONTROL", "").strip().lower() == "use_raw_ids"


def _structural_records(profile: str = "lineage-itest"):
    scope_key = lineage.make_scope_key(collection_name="itest", profile_id=profile)
    source_key = lineage.make_source_key(scope_key=scope_key, resolved_file_path="/itest/Src File.md")
    file_sha = hashlib.sha256(b"itest bytes").hexdigest()
    src = lineage.build_source_node_payload(
        source_key=source_key,
        scope_key=scope_key,
        profile_id=profile,
        file_path="/itest/Src File.md",
        source_uri="file:///itest/Src%20File.md",
    )
    ver = lineage.build_version_node_payload(
        source_key=source_key,
        scope_key=scope_key,
        profile_id=profile,
        file_path="/itest/Src File.md",
        file_sha256=file_sha,
        source_uri="file:///itest/Src%20File.md",
    )
    version_storage = lineage.storage_point_id(ver["entity_id"])
    source_storage = lineage.storage_point_id(src["entity_id"])
    edge = lineage.build_mechanical_edge_payload(
        relation_type="PART_OF",
        source_entity_id=ver["entity_id"],
        target_entity_id=src["entity_id"],
        profile_id=profile,
        source_point_id=version_storage,
        target_point_id=source_storage,
        source_entity_type="source",
        target_entity_type="source",
        lineage_operation="index_capture",
        lineage_source_key=source_key,
        lineage_scope_key=scope_key,
        provenance_content_hash=f"sha256:{file_sha}",
        file_version_id=version_storage,
        file_path="/itest/Src File.md",
        file_sha256=file_sha,
    )
    return src, ver, edge


def test_graph_storage_ids_and_payload_only_roundtrip(lineage_context: LineageContext):
    """Metadata-only upsert, exact UUID retrieve, export/restore round-trip,
    and an unchanged content-search control — against real Qdrant."""
    ctx = lineage_context
    qdrant = ctx.qdrant
    collection = ctx.primary_collection
    src, ver, edge = _structural_records()

    structural_points = [
        {"id": lineage.storage_point_id(src["entity_id"]), "vector": {}, "payload": src},
        {"id": lineage.storage_point_id(ver["entity_id"]), "vector": {}, "payload": ver},
        {"id": lineage.storage_point_id(edge["edge_id"]), "vector": {}, "payload": edge},
    ]
    stored_ids = [point["id"] for point in structural_points]
    if _negative_control():
        # Deliberately use the raw logical handles the pre-W0 mechanism sent
        # to Qdrant. Real Qdrant rejects these; the positive assertions below
        # then fail instead of passing.
        stored_ids = [src["entity_id"], ver["entity_id"], edge["edge_id"]]
        structural_points = [
            {"id": point_id, "vector": {}, "payload": payload}
            for point_id, payload in zip(stored_ids, (src, ver, edge))
        ]

    # --- metadata-only persistence (no embeddings called anywhere so far) ---
    qdrant.upsert(collection, structural_points)

    # --- exact retrieve by storage ID ---
    retrieved = {point["id"]: point for point in qdrant.retrieve(collection, stored_ids, with_payload=True)}
    assert len(retrieved) == 3
    for point in structural_points:
        got = retrieved[point["id"]]
        assert got["payload"] == point["payload"]
    if not _negative_control():
        for point_id in stored_ids:
            assert is_uuid_string(point_id)
    # Payload logical IDs unchanged and deterministic.
    assert src["entity_id"] == lineage.source_logical_id(
        source_key=src["lineage_source_key"], profile_id=src["profile_id"]
    )
    assert stored_ids[0] == make_graph_point_id(src["entity_id"])

    # --- ordinary chunk with a real embedding: content-search control ---
    chunk_id = "77777777-8888-8888-8888-000000000001"
    chunk_vector = ctx.embeddings.embed_document("unique itest pangram fountain content")
    qdrant.upsert(collection, [{
        "id": chunk_id,
        "vector": chunk_vector,
        "payload": {
            "text": "unique itest pangram fountain content",
            "source_type": "project_doc",
            "profile_id": "lineage-itest",
            "user_id_hash": "",
            "chat_id_hash": "",
            "fact_status": "active",
            "importance": 8,
        },
    }])
    hits = qdrant.search(
        collection,
        ctx.embeddings.embed_query("unique itest pangram fountain content"),
        10,
    )
    hit_ids = [hit["id"] for hit in hits]
    assert hit_ids == [chunk_id], f"structural records must not surface in content search: {hit_ids}"

    # --- backup / restore round-trip of the payload-only points ---
    config = {
        "qdrant_url": "http://127.0.0.1:6333",
        "collection_name": collection,
        "learning_collection_name": ctx.restore_collection,
        "vector_size": ctx.vector_size,
        "distance": ctx.distance,
    }
    export_target = ctx.restore_collection
    # Backup first (with all points present), then delete and restore.
    backup = create_backup(qdrant, config, hermes_home=str(_hermes_home()), scope="memory")
    backup_id = backup["backup_id"]

    qdrant.delete_ids(collection, stored_ids + [chunk_id])
    assert qdrant.count(collection) == 0
    restore_config = dict(config)
    restore_config["collection_name"] = export_target
    result = restore_backup(
        qdrant,
        restore_config,
        hermes_home=str(_hermes_home()),
        backup_id=backup_id,
        dry_run=False,
        backup_first=False,
    )
    assert result["applied"] is True
    restored = {point["id"]: point for point in qdrant.retrieve(export_target, stored_ids + [chunk_id], with_payload=True)}
    for point in structural_points:
        got = restored[point["id"]]
        # Retrieved with with_vector=False; an explicit vector fetch must show
        # the empty vector map, proving the point stayed payload-only.
        assert not got.get("vector")  # absent/empty = payload-only survives restore
        fetched_vec = qdrant.retrieve(export_target, [point["id"]], with_payload=False, with_vector=True)
        assert fetched_vec and fetched_vec[0].get("vector") == {}
        assert got["payload"] == point["payload"]

    # --- bite: raw logical handles are rejected by the real service ---
    if not _negative_control():
        with pytest.raises(Exception) as excinfo:
            qdrant.retrieve(collection, [src["entity_id"]])
        assert "400" in str(excinfo.value) or "point id" in str(excinfo.value).lower()
        with pytest.raises(Exception) as excinfo:
            qdrant.upsert(collection, [{"id": edge["edge_id"], "vector": {}, "payload": edge}])
        assert "400" in str(excinfo.value) or "point id" in str(excinfo.value).lower()


def _hermes_home(tmp_path: str | None = None) -> str:
    import tempfile
    from pathlib import Path

    base = Path(tempfile.gettempdir()) / "hermes-lineage-itest-home"
    base.mkdir(parents=True, exist_ok=True)
    return str(base)
