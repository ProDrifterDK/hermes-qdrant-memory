from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from conftest import LineageContext
from qdrant_memory import lineage
from qdrant_memory.indexer import FileIndexer

TRUTHY = {"1", "true", "yes", "on"}
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_QDRANT_INTEGRATION", "").strip().lower() not in TRUTHY,
    reason="set RUN_QDRANT_INTEGRATION=1 to run live Qdrant integration tests",
)
PROFILE = "lineage-w2-itest"


def _id(label: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"lineage-w2-itest:{label}"))


def _payload(**values):
    return {"profile_id": PROFILE, "user_id_hash": "", "chat_id_hash": "", **values}


def _indexer(ctx: LineageContext, tmp_path: Path, qdrant=None) -> FileIndexer:
    return FileIndexer(
        qdrant=qdrant or ctx.qdrant,
        embeddings=ctx.embeddings,
        collection_name=ctx.primary_collection,
        profile_id=PROFILE,
        platform="pytest",
        config={
            "lineage_mode": "reconcile",
            "qdrant_url": os.environ["QDRANT_TEST_URL"],
            "lineage_lock_dir": str(tmp_path / "locks"),
            "max_chunk_tokens": 128,
        },
    )


def _seed_chain(ctx: LineageContext, path: Path) -> dict[str, str]:
    ids = {name: _id(name) for name in ("old", "summary", "claim", "unrelated")}
    vector = [0.0] * ctx.vector_size
    points = [
        {
            "id": ids["old"],
            "vector": vector,
            "payload": _payload(
                text="legacy no-hash bytes", chunk_type="file_chunk",
                memory_kind="source_chunk", file_path=str(path),
                chunk_index=0, chunk_count=1, canonical=True,
                truth_confidence=0.77, usefulness_weight=0.66,
                requires_review=False,
            ),
        },
    ]
    for name in ("summary", "claim", "unrelated"):
        points.append({
            "id": ids[name],
            "vector": vector,
            "payload": _payload(
                text=f"{name} stable text", memory_kind="fact", canonical=True,
                truth_confidence=0.91, usefulness_weight=0.81,
                requires_review=False,
            ),
        })
    for label, source, target, relation in (
        ("summary-edge", ids["summary"], ids["old"], "SUMMARIZES"),
        ("claim-edge", ids["claim"], ids["summary"], "DERIVED_FROM"),
        ("cycle-edge", ids["summary"], ids["claim"], "DERIVED_FROM"),
        ("support-edge", ids["unrelated"], ids["old"], "SUPPORTS"),
    ):
        points.append({
            "id": _id(label), "vector": {},
            "payload": _payload(
                memory_kind="graph_edge", source_point_id=source,
                target_point_id=target, relation_type=relation,
            ),
        })
    ctx.qdrant.upsert(ctx.primary_collection, points)
    return ids


def test_reconcile_invalidates_transitive_dependents(
    lineage_context: LineageContext, tmp_path: Path, monkeypatch,
):
    path = tmp_path / "replace.md"
    path.write_text("verified replacement bytes", encoding="utf-8")
    ids = _seed_chain(lineage_context, path)
    before = {
        str(point["id"]): point for point in lineage_context.qdrant.retrieve(
            lineage_context.primary_collection,
            [ids["summary"], ids["claim"], ids["unrelated"]],
            with_payload=True, with_vector=True,
        )
    }
    if os.environ.get("W2_BYPASS_APPLY_INVALIDATION") == "1":
        monkeypatch.setattr(
            lineage,
            "apply_invalidation",
            lambda **kwargs: {
                "changed_ids": [],
                "verified_ids": list(kwargs["plan"].get("dependent_ids") or []),
            },
        )

    result = _indexer(lineage_context, tmp_path).index([path], dry_run=False)

    assert result["errors"] == []
    assert lineage_context.qdrant.retrieve(
        lineage_context.primary_collection, [ids["old"]], with_payload=True,
    ) == []
    after = {
        str(point["id"]): point for point in lineage_context.qdrant.retrieve(
            lineage_context.primary_collection,
            [ids["summary"], ids["claim"], ids["unrelated"]],
            with_payload=True, with_vector=True,
        )
    }
    for point_id in (ids["summary"], ids["claim"]):
        assert after[point_id]["payload"].get("fact_status") == "review_required"
        assert after[point_id]["payload"]["text"] == before[point_id]["payload"]["text"]
        assert after[point_id]["payload"]["truth_confidence"] == before[point_id]["payload"]["truth_confidence"]
        assert after[point_id]["payload"]["usefulness_weight"] == before[point_id]["payload"]["usefulness_weight"]
        assert after[point_id]["vector"] == before[point_id]["vector"]
    assert after[ids["unrelated"]] == before[ids["unrelated"]]
    records = lineage_context.qdrant.scroll_by_filter(
        lineage_context.primary_collection,
        {"must": [{"key": "lineage_role", "match": {"any": ["file_version", "change_event"]}}]},
        limit=32, with_payload=True, with_vector=False,
    )
    assert {point["payload"]["lineage_role"] for point in records} == {"file_version", "change_event"}


class _FailDeleteOnce:
    def __init__(self, delegate):
        self.delegate = delegate
        self.failed = False

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def delete_ids(self, name, ids):
        if not self.failed:
            self.failed = True
            raise RuntimeError("injected delete boundary")
        return self.delegate.delete_ids(name, ids)


def test_reconcile_failure_retry_converges(lineage_context: LineageContext, tmp_path: Path):
    path = tmp_path / "retry.md"
    path.write_text("retry replacement", encoding="utf-8")
    ids = _seed_chain(lineage_context, path)
    before = {
        str(point["id"]): point for point in lineage_context.qdrant.retrieve(
            lineage_context.primary_collection,
            [ids["summary"], ids["claim"], ids["unrelated"]],
            with_payload=True, with_vector=True,
        )
    }
    wrapped = _FailDeleteOnce(lineage_context.qdrant)
    indexer = _indexer(lineage_context, tmp_path, qdrant=wrapped)

    assert indexer.index([path], dry_run=False)["partial_failure"] is True
    retried = indexer.index([path], dry_run=False)

    assert retried["errors"] == []
    assert lineage_context.qdrant.retrieve(lineage_context.primary_collection, [ids["old"]]) == []
    after = {
        str(point["id"]): point for point in lineage_context.qdrant.retrieve(
            lineage_context.primary_collection,
            [ids["summary"], ids["claim"], ids["unrelated"]],
            with_payload=True, with_vector=True,
        )
    }
    for point_id in (ids["summary"], ids["claim"]):
        assert after[point_id]["payload"]["fact_status"] == "review_required"
        assert after[point_id]["payload"]["requires_review"] is True
        for key in ("text", "truth_confidence", "usefulness_weight"):
            assert after[point_id]["payload"][key] == before[point_id]["payload"][key]
        assert after[point_id]["vector"] == before[point_id]["vector"]
    assert after[ids["unrelated"]] == before[ids["unrelated"]]
    chunks = lineage_context.qdrant.scroll_by_filter(
        lineage_context.primary_collection,
        {"must": [{"key": "file_path", "match": {"value": str(path)}}]},
        limit=32, with_payload=True, with_vector=False,
    )
    active_chunks = [point for point in chunks if (point.get("payload") or {}).get("memory_kind") == "source_chunk"]
    assert len(active_chunks) == 1
    assert active_chunks[0]["payload"]["lineage_pending"] is False
    records = lineage_context.qdrant.scroll_by_filter(
        lineage_context.primary_collection,
        {"must": [{"key": "lineage_role", "match": {"any": ["file_source", "file_version"]}}]},
        limit=32, with_payload=True, with_vector=False,
    )
    assert sum(point["payload"]["lineage_role"] == "file_source" for point in records) == 1
    assert sum(point["payload"]["lineage_role"] == "file_version" for point in records) == 1
    non_events = lineage_context.qdrant.scroll_by_filter(
        lineage_context.primary_collection,
        {"must": [
            {"key": "lineage_record", "match": {"value": True}},
            {"key": "lineage_role", "match": {"any": ["file_source", "file_version", "mechanical_edge"]}},
        ]},
        limit=64, with_payload=True, with_vector=False,
    )
    identities = [point["payload"]["lineage_identity_digest"] for point in non_events]
    assert len(identities) == len(set(identities))
    events = lineage_context.qdrant.scroll_by_filter(
        lineage_context.primary_collection,
        {"must": [{"key": "lineage_role", "match": {"value": "change_event"}}]},
        limit=32, with_payload=True, with_vector=False,
    )
    assert len(events) == 1
    assert events[0]["payload"]["event_state"] == "committed"


def test_reconcile_delete_empty_and_reversion(lineage_context: LineageContext, tmp_path: Path):
    path = tmp_path / "history.md"
    indexer = _indexer(lineage_context, tmp_path)
    for text in ("A", "B", "A", ""):
        path.write_text(text, encoding="utf-8")
        assert indexer.index([path], dry_run=False)["errors"] == []
    path.unlink()
    assert indexer.index([tmp_path], dry_run=False)["errors"] == []

    events = lineage_context.qdrant.scroll_by_filter(
        lineage_context.primary_collection,
        {"must": [{"key": "lineage_role", "match": {"value": "change_event"}}]},
        limit=32, with_payload=True, with_vector=False,
    )
    assert {point["payload"]["event_kind"] for point in events} == {
        "observed", "modified", "restored", "deleted"
    }
    assert len(events) == 5
    by_id = {str(point["id"]): point["payload"] for point in events}
    source = lineage_context.qdrant.scroll_by_filter(
        lineage_context.primary_collection,
        {"must": [{"key": "lineage_role", "match": {"value": "file_source"}}]},
        limit=4, with_payload=True, with_vector=False,
    )[0]["payload"]
    chronology = []
    event_id = source["head_event_id"]
    while event_id is not None:
        chronology.append(by_id[event_id]["event_kind"])
        event_id = by_id[event_id].get("previous_event_id")
    assert chronology == ["deleted", "modified", "restored", "modified", "observed"]
    assert source["source_deleted"] is True
    assert source.get("current_version_id") is None


def test_consolidation_lineage_impact_gates(lineage_context: LineageContext):
    root_id = _id("impact-root")
    lineage_context.qdrant.upsert(lineage_context.primary_collection, [{
        "id": root_id,
        "vector": [0.0] * lineage_context.vector_size,
        "payload": _payload(text="ordinary root", memory_kind="fact"),
    }])

    impact = lineage.build_lineage_impact_snapshot(
        qdrant=lineage_context.qdrant,
        collection_name=lineage_context.primary_collection,
        root_point_ids=[root_id],
        profile_id=PROFILE,
    )

    assert impact["complete"] is False
    assert impact["root_ids"] == [root_id]
    assert impact["errors"] == ["ordinary_root_transition_cause_unratified"]
    validation = lineage.validate_lineage_impact_snapshot(
        qdrant=lineage_context.qdrant,
        collection_name=lineage_context.primary_collection,
        impact=impact,
        expected_root_ids=[root_id],
    )
    assert validation["valid"] is False
    assert "lineage impact is incomplete" in validation["problems"]
