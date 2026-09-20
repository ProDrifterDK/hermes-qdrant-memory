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
from pathlib import Path

import pytest

from conftest import LINEAGE_TEST_PREFIX, LineageContext

from qdrant_memory import lineage
from qdrant_memory.backup import create_backup, restore_backup
from qdrant_memory.graph_schema import is_uuid_string, make_graph_point_id
from qdrant_memory.indexer import FileIndexer

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
        "qdrant_url": os.environ["QDRANT_TEST_URL"],
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

    base = Path(tempfile.gettempdir()) / "hermes-lineage-itest-home"
    base.mkdir(parents=True, exist_ok=True)
    return str(base)


def _all_points(ctx: LineageContext) -> list[dict]:
    data = ctx.qdrant._request(
        "POST",
        f"/collections/{urllib.parse.quote(ctx.primary_collection, safe='')}/points/scroll",
        {"limit": 256, "with_payload": True, "with_vector": False},
    )
    result = data.get("result", {}) or {}
    assert result.get("next_page_offset") is None, "lineage integration fixture exceeded one bounded page"
    return result.get("points", []) or []


def _live_indexer(ctx: LineageContext, tmp_path: Path, *, profile_id: str, mode: str = "capture") -> FileIndexer:
    return FileIndexer(
        qdrant=ctx.qdrant,
        embeddings=ctx.embeddings,
        collection_name=ctx.primary_collection,
        profile_id=profile_id,
        platform="pytest",
        config={
            "lineage_mode": mode,
            "qdrant_url": os.environ["QDRANT_TEST_URL"],
            "lineage_lock_dir": str(tmp_path / "locks"),
            "max_chunk_tokens": 128,
        },
    )


def test_index_capture_roundtrip(lineage_context: LineageContext, tmp_path: Path):
    path = tmp_path / "capture.md"
    path.write_text("# Capture\nalpha\n\n## Detail\nbeta", encoding="utf-8")
    negative = os.environ.get("LINEAGE_CAPTURE_NEGATIVE_CONTROL") == "off"
    indexer = _live_indexer(
        lineage_context, tmp_path, profile_id="lineage-itest",
        mode="off" if negative else "capture",
    )

    result = indexer.index([path], dry_run=False)
    points = _all_points(lineage_context)
    structural = [point for point in points if (point.get("payload") or {}).get("lineage_record") is True]
    assert structural, "missing lineage graph records after index capture"
    assert result["partial_failure"] is False
    assert result["lineage_coverage"]["captured_files"] == 1
    assert len(points) == len(result["lineage_point_ids"]) + result["chunks_prepared"]
    retrieved = lineage_context.qdrant.retrieve(
        lineage_context.primary_collection,
        result["lineage_read_back_ids"],
        with_payload=True,
        with_vector=False,
    )
    assert {str(point["id"]) for point in retrieved} == set(result["lineage_read_back_ids"])
    ids = {str(point["id"]) for point in points}
    for edge in [point for point in points if (point.get("payload") or {}).get("memory_kind") == "graph_edge"]:
        assert edge["payload"]["source_point_id"] in ids
        assert edge["payload"]["target_point_id"] in ids


def test_capture_repair_and_scope_isolation(lineage_context: LineageContext, tmp_path: Path):
    path = tmp_path / "scoped.md"
    path.write_text("# Scoped\nalpha", encoding="utf-8")
    first_indexer = _live_indexer(lineage_context, tmp_path, profile_id="lineage-itest-a")
    first = first_indexer.index([path], dry_run=False)
    assert first["partial_failure"] is False
    points = _all_points(lineage_context)
    missing_edge = next(
        point for point in points
        if (point.get("payload") or {}).get("relation_type") == "DERIVED_FROM"
        and point["payload"].get("profile_id") == "lineage-itest-a"
    )
    lineage_context.qdrant.delete_ids(lineage_context.primary_collection, [str(missing_edge["id"])])

    repaired = first_indexer.index([path], dry_run=False)
    assert repaired["partial_failure"] is False
    assert repaired["lineage_repair_ids"] == [str(missing_edge["id"])]

    second = _live_indexer(
        lineage_context, tmp_path, profile_id="lineage-itest-b"
    ).index([path], dry_run=False)
    assert second["partial_failure"] is False
    points = _all_points(lineage_context)
    sources = [
        point for point in points
        if (point.get("payload") or {}).get("lineage_role") == "file_source"
    ]
    chunks = [
        point for point in points
        if (point.get("payload") or {}).get("memory_kind") == "source_chunk"
    ]
    assert {point["payload"]["profile_id"] for point in sources} == {
        "lineage-itest-a", "lineage-itest-b",
    }
    assert len({str(point["id"]) for point in sources}) == 2
    assert {point["payload"]["profile_id"] for point in chunks} == {
        "lineage-itest-a", "lineage-itest-b",
    }
    assert any(str(point["id"]) == str(missing_edge["id"]) for point in points)


def test_off_mode_fresh_store_has_no_lineage_graph(lineage_context: LineageContext, tmp_path: Path):
    path = tmp_path / "legacy.md"
    path.write_text("legacy path", encoding="utf-8")

    result = _live_indexer(
        lineage_context, tmp_path, profile_id="lineage-itest", mode="off"
    ).index([path], dry_run=False)
    points = _all_points(lineage_context)

    assert result["chunks_upserted"] == 1
    assert len(points) == 1
    assert not any((point.get("payload") or {}).get("lineage_record") for point in points)
    assert not any((point.get("payload") or {}).get("memory_kind") in {"graph_entity", "graph_edge"} for point in points)


def test_off_mode_inventory_failure_fails_closed_live(
    lineage_context: LineageContext, tmp_path: Path, monkeypatch
):
    path = tmp_path / "inventory-failure.md"
    path.write_text("# Inventory failure\nalpha", encoding="utf-8")
    profile = "lineage-itest-fail-closed"
    owner = _live_indexer(lineage_context, tmp_path, profile_id=profile)
    assert owner.index([path], dry_run=False)["partial_failure"] is False
    before_points = _all_points(lineage_context)
    before = json.dumps(before_points, sort_keys=True, separators=(",", ":"))
    original_scroll = lineage_context.qdrant.scroll_by_filter
    raised = False

    def raise_once(name, filter, limit=256, with_payload=True, with_vector=False):
        nonlocal raised
        if not raised and str(path.resolve()) in json.dumps(filter):
            raised = True
            raise RuntimeError("simulated live inventory transport error")
        return original_scroll(
            name, filter, limit=limit, with_payload=with_payload, with_vector=with_vector
        )

    monkeypatch.setattr(lineage_context.qdrant, "scroll_by_filter", raise_once)
    result = _live_indexer(
        lineage_context, tmp_path, profile_id=profile, mode="off"
    ).index([path], dry_run=False)
    after = json.dumps(_all_points(lineage_context), sort_keys=True, separators=(",", ":"))

    assert raised is True
    assert result["refused"] is True
    assert result["partial_failure"] is True
    assert result["chunks_upserted"] == 0
    assert result["chunks_deleted"] == 0
    assert result["foreign_scope_chunks"] == []
    assert result["refusals"] == [{
        "file_path": str(path.resolve()),
        "reason": "lineage_inventory_unavailable_refused_fail_closed",
    }]
    assert result["errors"] == [{
        "file_path": str(path.resolve()),
        "error": "manifest sync failed: simulated live inventory transport error",
    }]
    assert after == before


def test_foreign_scope_dry_run_redacts_chunk_ids_live(
    lineage_context: LineageContext, tmp_path: Path
):
    path = tmp_path / "foreign-report.md"
    path.write_text("# Foreign report\nalpha", encoding="utf-8")
    owner = _live_indexer(lineage_context, tmp_path, profile_id="lineage-itest-owner")
    assert owner.index([path], dry_run=False)["partial_failure"] is False
    owner_ids = {
        str(point["id"])
        for point in _all_points(lineage_context)
        if (point.get("payload") or {}).get("memory_kind") == "source_chunk"
    }
    before = json.dumps(_all_points(lineage_context), sort_keys=True, separators=(",", ":"))

    result = _live_indexer(
        lineage_context, tmp_path, profile_id="lineage-itest-other", mode="off"
    ).index([path], dry_run=True)

    report = result["foreign_scope_chunks"]
    rendered = json.dumps(report, sort_keys=True)
    assert report == [{"file_path": str(path.resolve()), "count": len(owner_ids)}]
    assert all("chunk_ids" not in item for item in report)
    assert all(point_id not in rendered for point_id in owner_ids)
    assert json.dumps(_all_points(lineage_context), sort_keys=True, separators=(",", ":")) == before


def test_index_failure_payload_uses_stderr_live(
    lineage_context: LineageContext, tmp_path: Path
):
    import argparse
    import importlib.util
    import io

    from qdrant_memory.cli_core import execute_command

    path = tmp_path / "cli-channel.md"
    path.write_text("# CLI channel\nalpha", encoding="utf-8")
    owner = _live_indexer(lineage_context, tmp_path, profile_id="lineage-itest-cli-owner")
    assert owner.index([path], dry_run=False)["partial_failure"] is False
    summary = _live_indexer(
        lineage_context, tmp_path, profile_id="lineage-itest-cli-other", mode="off"
    ).index([path], dry_run=True)
    raw = json.dumps(summary)

    class Provider:
        def handle_tool_call(self, tool_name, args):
            assert tool_name == "qdrant_memory_index"
            return raw

    spec = importlib.util.spec_from_file_location(
        "qdrant_plugin_cli_fix4_live", Path(__file__).resolve().parents[2] / "cli.py"
    )
    plugin_cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin_cli)
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command", required=True)
    qdrant_parser = subparsers.add_parser("qdrant")
    plugin_cli.register_cli(qdrant_parser)

    stdout, stderr = io.StringIO(), io.StringIO()
    human_args = parser.parse_args(["qdrant", "index", str(path)])
    assert execute_command(
        human_args, provider_factory=Provider, stdout=stdout, stderr=stderr
    ) == 1
    assert stdout.getvalue() == ""
    assert "Index refused" in stderr.getvalue()
    assert f"foreign_scope_chunks: {path.resolve()} count=1" in stderr.getvalue()

    stdout, stderr = io.StringIO(), io.StringIO()
    json_args = parser.parse_args(["qdrant", "index", str(path), "--json"])
    assert execute_command(
        json_args, provider_factory=Provider, stdout=stdout, stderr=stderr
    ) == 1
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == raw + "\n"
