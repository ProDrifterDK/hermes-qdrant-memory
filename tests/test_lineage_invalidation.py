from __future__ import annotations

import copy
import os
import uuid
from pathlib import Path

import pytest

import qdrant_memory.indexer as indexer_module
import qdrant_memory.lineage as lineage
from qdrant_memory.indexer import FileIndexer


COLLECTION = "w2_disposable"
PROFILE = "default"


def _id(label: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"w2:{label}"))


def _value(payload: dict, key: str):
    value = payload
    for part in key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _matches(payload: dict, filter_value: dict) -> bool:
    for clause in filter_value.get("must", []):
        if "is_empty" in clause:
            if _value(payload, clause["is_empty"]["key"]) not in (None, "", [], {}):
                return False
            continue
        if "nested" in clause:
            nested = clause["nested"]
            items = _value(payload, nested["key"])
            if not isinstance(items, list) or not any(
                isinstance(item, dict) and _matches(item, nested["filter"])
                for item in items
            ):
                return False
            continue
        actual = _value(payload, clause["key"])
        match = clause.get("match", {})
        if "value" in match and actual != match["value"]:
            return False
        if "any" in match and actual not in match["any"]:
            return False
    for clause in filter_value.get("must_not", []):
        actual = _value(payload, clause["key"])
        match = clause.get("match", {})
        if "value" in match and actual == match["value"]:
            return False
        if "any" in match and actual in match["any"]:
            return False
    return True


class FakeEmbedding:
    def __init__(self):
        self.documents: list[str] = []

    def embed_document(self, text: str) -> list[float]:
        self.documents.append(text)
        return [float(len(text)), 0.25]


class FakeQdrant:
    def __init__(self, points: list[dict] | None = None, *, fail_at: tuple[str, int] | None = None):
        self.points = {str(point["id"]): copy.deepcopy(point) for point in (points or [])}
        self.fail_at = fail_at
        self.counts: dict[str, int] = {}
        self.failed = False

    def _boundary(self, name: str) -> None:
        self.counts[name] = self.counts.get(name, 0) + 1
        if not self.failed and self.fail_at == (name, self.counts[name]):
            self.failed = True
            raise RuntimeError(f"injected {name} boundary {self.counts[name]}")

    def upsert(self, name: str, points: list[dict]):
        self._boundary("upsert")
        for point in points:
            self.points[str(point["id"])] = copy.deepcopy(point)
        return {"status": "ok"}

    def update_payload(self, name: str, point_id: str, payload: dict):
        self._boundary("update_payload")
        point = self.points[str(point_id)]
        point["payload"] = {**(point.get("payload") or {}), **copy.deepcopy(payload)}
        return {"status": "ok"}

    def delete_ids(self, name: str, ids: list[str]):
        self._boundary("delete_ids")
        for point_id in ids:
            self.points.pop(str(point_id), None)
        return {"status": "ok"}

    def retrieve(self, name: str, ids: list[str], *, with_payload=True, with_vector=False):
        result = []
        for point_id in ids:
            point = self.points.get(str(point_id))
            if point is None:
                continue
            item = {"id": point["id"]}
            if with_payload:
                item["payload"] = copy.deepcopy(point.get("payload") or {})
            if with_vector:
                item["vector"] = copy.deepcopy(point.get("vector"))
            result.append(item)
        return result

    def scroll_page(
        self, name: str, filter: dict, *, limit=256, offset=None,
        with_payload=True, with_vector=False,
    ):
        matched = [point for point in self.points.values() if _matches(point.get("payload") or {}, filter)]
        matched.sort(key=lambda point: str(point["id"]))
        start = int(offset or 0)
        batch = matched[start : start + int(limit)]
        next_offset = start + len(batch) if start + len(batch) < len(matched) else None
        result = []
        for point in batch:
            item = {"id": point["id"]}
            if with_payload:
                item["payload"] = copy.deepcopy(point.get("payload") or {})
            if with_vector:
                item["vector"] = copy.deepcopy(point.get("vector"))
            result.append(item)
        return result, next_offset

    def scroll_by_filter(
        self, name: str, filter: dict, *, limit=256,
        with_payload=True, with_vector=False, max_total=None,
    ):
        result: list[dict] = []
        offset = None
        while True:
            page_limit = int(limit)
            if max_total is not None:
                page_limit = min(page_limit, int(max_total) - len(result))
                if page_limit <= 0:
                    break
            batch, offset = self.scroll_page(
                name, filter, limit=page_limit, offset=offset,
                with_payload=with_payload, with_vector=with_vector,
            )
            result.extend(batch)
            if offset is None:
                break
        return result


def _payload(**extra):
    return {
        "profile_id": PROFILE,
        "user_id_hash": "",
        "chat_id_hash": "",
        **extra,
    }


def _legacy_chunk(path: Path, label: str = "legacy") -> dict:
    return {
        "id": _id(label),
        "vector": [9.0, 8.0],
        "payload": _payload(
            text=f"{label} text",
            chunk_type="file_chunk",
            memory_kind="source_chunk",
            file_path=str(path),
            chunk_index=0,
            chunk_count=1,
            canonical=True,
            truth_confidence=0.77,
            usefulness_weight=0.66,
            requires_review=False,
        ),
    }


def _dependent(label: str, *, status: str | None = None) -> dict:
    payload = _payload(
        text=f"{label} immutable text",
        memory_kind="fact",
        canonical=True,
        truth_confidence=0.91,
        usefulness_weight=0.81,
        requires_review=False,
    )
    if status is not None:
        payload["fact_status"] = status
    return {"id": _id(label), "vector": [1.25, 2.5], "payload": payload}


def _edge(label: str, source: str, target: str, relation: str) -> dict:
    return {
        "id": _id(label),
        "vector": {},
        "payload": _payload(
            memory_kind="graph_edge",
            target_point_id=target,
            source_point_id=source,
            relation_type=relation,
        ),
    }


def _mechanical_edge(
    label: str,
    source: str,
    target: str,
    *,
    relation: str = "DERIVED_FROM",
    operation: str = "index_capture",
    target_entity_type: str = "source",
    retired_by_event_id: str | None = None,
) -> dict:
    edge = _edge(label, source, target, relation)
    edge["payload"].update(
        lineage_operation=operation,
        target_entity_type=target_entity_type,
    )
    if retired_by_event_id is not None:
        edge["payload"]["lineage_retired_by_event_id"] = retired_by_event_id
    return edge


def _indexer(
    qdrant: FakeQdrant, *, lock_dir: Path, mode: str = "reconcile",
    max_chunk_tokens: int = 128,
) -> FileIndexer:
    return FileIndexer(
        qdrant=qdrant,
        embeddings=FakeEmbedding(),
        collection_name=COLLECTION,
        config={
            "lineage_mode": mode,
            "qdrant_url": "http://disposable.invalid:6333",
            "lineage_lock_dir": str(lock_dir),
            "max_chunk_tokens": max_chunk_tokens,
        },
    )


def _seed_chain(path: Path) -> tuple[FakeQdrant, dict[str, str], dict[str, dict]]:
    old = _legacy_chunk(path)
    summary = _dependent("summary")
    claim = _dependent("claim")
    unrelated = _dependent("unrelated")
    ids = {name: str(point["id"]) for name, point in (
        ("old", old), ("summary", summary), ("claim", claim), ("unrelated", unrelated)
    )}
    points = [
        old, summary, claim, unrelated,
        _edge("summary-edge", ids["summary"], ids["old"], "SUMMARIZES"),
        _edge("claim-edge", ids["claim"], ids["summary"], "DERIVED_FROM"),
        _edge("cycle-edge", ids["summary"], ids["claim"], "DERIVED_FROM"),
        _edge("support-edge", ids["unrelated"], ids["old"], "SUPPORTS"),
        _edge("opposite-edge", ids["old"], ids["unrelated"], "DERIVED_FROM"),
    ]
    return FakeQdrant(points), ids, {str(point["id"]): copy.deepcopy(point) for point in points}


@pytest.fixture
def negative_control(monkeypatch):
    if os.environ.get("W2_BYPASS_APPLY_INVALIDATION") == "1":
        monkeypatch.setattr(
            lineage,
            "apply_invalidation",
            lambda **kwargs: {
                "changed_ids": [],
                "verified_ids": list(kwargs["plan"].get("dependent_ids") or []),
            },
        )


def test_reconcile_invalidates_transitive_dependents(tmp_path: Path, negative_control):
    path = tmp_path / "source.md"
    path.write_text("# Source\n\nreplacement bytes\n", encoding="utf-8")
    qdrant, ids, before = _seed_chain(path)

    result = _indexer(qdrant, lock_dir=tmp_path / "locks").index([path], dry_run=False)

    assert result["errors"] == []
    assert ids["old"] not in qdrant.points
    for label in ("summary", "claim"):
        current = qdrant.points[ids[label]]
        prior = before[ids[label]]
        assert current["payload"].get("fact_status") == "review_required"
        assert current["payload"]["requires_review"] is True
        assert current["payload"]["text"] == prior["payload"]["text"]
        assert current["payload"]["truth_confidence"] == prior["payload"]["truth_confidence"]
        assert current["payload"]["usefulness_weight"] == prior["payload"]["usefulness_weight"]
        assert current["vector"] == prior["vector"]
    assert qdrant.points[ids["unrelated"]] == before[ids["unrelated"]]
    roles = {(point.get("payload") or {}).get("lineage_role") for point in qdrant.points.values()}
    assert {"file_version", "change_event"} <= roles


@pytest.mark.parametrize(
    "boundary",
    [
        ("upsert", 1), ("upsert", 2), ("delete_ids", 1),
        *(("update_payload", occurrence) for occurrence in range(1, 9)),
    ],
)
def test_reconcile_failure_retry_converges(tmp_path: Path, boundary):
    path = tmp_path / "retry.md"
    path.write_text("retry-safe replacement", encoding="utf-8")
    qdrant, ids, before = _seed_chain(path)
    qdrant.fail_at = boundary
    indexer = _indexer(qdrant, lock_dir=tmp_path / "locks")

    first = indexer.index([path], dry_run=False)
    assert first["partial_failure"] is True
    second = indexer.index([path], dry_run=False)

    assert second["errors"] == []
    assert ids["old"] not in qdrant.points
    for label in ("summary", "claim"):
        current = qdrant.points[ids[label]]
        assert current["payload"]["fact_status"] == "review_required"
        assert current["payload"]["requires_review"] is True
        for key in ("text", "truth_confidence", "usefulness_weight"):
            assert current["payload"][key] == before[ids[label]]["payload"][key]
        assert current["vector"] == before[ids[label]]["vector"]
    assert qdrant.points[ids["unrelated"]] == before[ids["unrelated"]]
    new_chunks = [
        point for point in qdrant.points.values()
        if (point.get("payload") or {}).get("memory_kind") == "source_chunk"
    ]
    assert len(new_chunks) == 1
    assert new_chunks[0]["payload"]["lineage_pending"] is False
    assert sum((point.get("payload") or {}).get("lineage_role") == "file_source" for point in qdrant.points.values()) == 1
    assert sum((point.get("payload") or {}).get("lineage_role") == "file_version" for point in qdrant.points.values()) == 1
    identity_digests = [
        point["payload"]["lineage_identity_digest"]
        for point in qdrant.points.values()
        if (point.get("payload") or {}).get("lineage_record") is True
    ]
    assert len(identity_digests) == len(set(identity_digests))
    events = [
        point for point in qdrant.points.values()
        if (point.get("payload") or {}).get("lineage_role") == "change_event"
    ]
    assert len(events) == 1
    assert events[0]["payload"]["event_state"] == "committed"


def test_reconcile_delete_empty_and_reversion(tmp_path: Path):
    path = tmp_path / "history.md"
    indexer_qdrant = FakeQdrant()
    indexer = _indexer(indexer_qdrant, lock_dir=tmp_path / "locks")

    path.write_text("A", encoding="utf-8")
    assert indexer.index([path], dry_run=False)["errors"] == []
    path.write_text("B", encoding="utf-8")
    assert indexer.index([path], dry_run=False)["errors"] == []
    path.write_text("A", encoding="utf-8")
    assert indexer.index([path], dry_run=False)["errors"] == []
    path.write_text("", encoding="utf-8")
    assert indexer.index([path], dry_run=False)["errors"] == []
    path.unlink()
    deleted = indexer.index([tmp_path], dry_run=False)

    assert deleted["errors"] == []
    events = {
        point_id: point["payload"] for point_id, point in indexer_qdrant.points.items()
        if (point.get("payload") or {}).get("lineage_role") == "change_event"
    }
    assert len(events) == 5
    source = next(
        point["payload"] for point in indexer_qdrant.points.values()
        if (point.get("payload") or {}).get("lineage_role") == "file_source"
    )
    chronology = []
    event_id = source["head_event_id"]
    while event_id is not None:
        chronology.append(events[event_id]["event_kind"])
        event_id = events[event_id].get("previous_event_id")
    assert chronology == ["deleted", "modified", "restored", "modified", "observed"]
    assert source["source_deleted"] is True
    assert source["current_version_id"] is None


def test_consolidation_lineage_impact_gates():
    qdrant = FakeQdrant()
    plan = lineage.plan_invalidation(
        qdrant=qdrant,
        collection_name=COLLECTION,
        root_point_ids=[_id("ordinary-root")],
        event_id=None,
        profile_id=PROFILE,
    )
    assert plan["complete"] is False
    assert plan["errors"] == ["ordinary_root_transition_cause_unratified"]


def test_no_hash_legacy_replacement_uses_existing_caps_for_1450_roots(tmp_path: Path, monkeypatch):
    path = tmp_path / "shared-ledger.md"
    path.write_text("verified current snapshot", encoding="utf-8")
    points = []
    for index in range(1450):
        point = _legacy_chunk(path, f"legacy-{index}")
        point["payload"]["chunk_index"] = index
        point["payload"]["chunk_count"] = 1450
        points.append(point)
    qdrant = FakeQdrant(points)
    before = copy.deepcopy(qdrant.points)
    captured = {}
    original = lineage.plan_reconciliation

    def capture_plan(**kwargs):
        plan = original(**kwargs)
        captured.update(copy.deepcopy(plan))
        return plan

    monkeypatch.setattr(indexer_module, "plan_reconciliation", capture_plan)
    result = _indexer(qdrant, lock_dir=tmp_path / "locks").index([path], dry_run=True)

    assert result["errors"] == []
    assert result["refusals"] == []
    assert result["lineage_files"][0]["lineage_baseline_basis"] == "missing_indexed_file_sha256"
    assert result["lineage_files"][0]["event_kind"] == "observed"
    assert len(result["lineage_event_ids"]) == 1
    assert captured["invalidation"]["bounds"] == {
        "max_depth": 8, "max_points": 4096, "max_edges": 8192,
    }
    expected_ids = sorted(before)
    assert captured["invalidation"]["root_ids"] == expected_ids
    assert captured["retired_ids"] == expected_ids
    assert qdrant.points == before


def test_inline_uri_and_raptor_reverse_lookup_is_exact_and_bounded():
    root = _id("inline-root")
    uri = _dependent("uri-child")
    uri["payload"]["derived_from"] = [{
        "source_uri": f"memory://point/{root}",
        "relation_type": "SUMMARIZES",
    }]
    raptor = _dependent("raptor-child")
    raptor["payload"]["derived_from"] = [{
        "child_node_id": root,
        "relation_type": "DERIVED_FROM",
        "source_uri": "raptor://node/root",
    }]
    qdrant = FakeQdrant([uri, raptor])

    plan = lineage.plan_invalidation(
        qdrant=qdrant, collection_name=COLLECTION,
        root_point_ids=[root], event_id=_id("inline-event"), profile_id=PROFILE,
    )

    assert plan["complete"] is True
    assert plan["dependent_ids"] == sorted([uri["id"], raptor["id"]])


def test_malformed_inline_entry_cap_plus_one_and_point_drift_fail_closed():
    root = _id("bounded-root")
    first = _dependent("first-child")
    first["payload"]["derived_from"] = [
        {"point_id": root, "relation_type": "DERIVED_FROM"},
        "malformed",
    ]
    second = _dependent("second-child")
    second["payload"]["derived_from"] = [{"point_id": root, "relation_type": "DERIVED_FROM"}]
    qdrant = FakeQdrant([first, second])

    malformed = lineage.plan_invalidation(
        qdrant=qdrant, collection_name=COLLECTION,
        root_point_ids=[root], event_id=_id("malformed-event"), profile_id=PROFILE,
    )
    assert malformed["complete"] is False
    assert any("entry is malformed" in error for error in malformed["errors"])

    first["payload"]["derived_from"] = [{"point_id": root, "relation_type": "DERIVED_FROM"}]
    capped = lineage.plan_invalidation(
        qdrant=FakeQdrant([first, second]), collection_name=COLLECTION,
        root_point_ids=[root], event_id=_id("cap-event"), profile_id=PROFILE,
        max_points=1,
    )
    assert capped["complete"] is False
    assert any("bound" in error for error in capped["errors"])

    single = FakeQdrant([first])
    planned = lineage.plan_invalidation(
        qdrant=single, collection_name=COLLECTION,
        root_point_ids=[root], event_id=_id("drift-event"), profile_id=PROFILE,
    )
    single.points[str(first["id"])]["payload"]["text"] = "drift after preview"
    with pytest.raises(RuntimeError, match="dependent payload drift"):
        lineage.apply_invalidation(qdrant=single, collection_name=COLLECTION, plan=planned)


def test_depth_boundary_and_cross_collection_fail_closed():
    root = _id("root")
    child = _dependent("child")
    child["payload"]["derived_from"] = [{
        "point_id": root,
        "relation_type": "DERIVED_FROM",
        "collection": "other_collection",
    }]
    qdrant = FakeQdrant([child])

    cross_collection = lineage.plan_invalidation(
        qdrant=qdrant, collection_name=COLLECTION,
        root_point_ids=[root], event_id=_id("event"), profile_id=PROFILE,
    )
    assert cross_collection["complete"] is False
    assert any("cross-collection" in error for error in cross_collection["errors"])

    child["payload"]["derived_from"][0].pop("collection")
    qdrant = FakeQdrant([child])
    bounded = lineage.plan_invalidation(
        qdrant=qdrant, collection_name=COLLECTION,
        root_point_ids=[root], event_id=_id("event"), profile_id=PROFILE,
        max_depth=0,
    )
    assert bounded["complete"] is False
    assert bounded["errors"] == ["dependency depth bound reached with unseen dependents"]


def _events(qdrant: FakeQdrant) -> dict[str, dict]:
    return {
        point_id: point["payload"]
        for point_id, point in qdrant.points.items()
        if (point.get("payload") or {}).get("lineage_role") == "change_event"
    }


def test_locked_apply_refuses_dependency_published_during_embedding(tmp_path: Path):
    path = tmp_path / "race.md"
    path.write_text("replacement bytes", encoding="utf-8")
    root = _legacy_chunk(path)
    late = _dependent("late-dependent")
    qdrant = FakeQdrant([root])

    class PublishingEmbedding(FakeEmbedding):
        def embed_document(self, text: str) -> list[float]:
            if late["id"] not in qdrant.points:
                qdrant.upsert(COLLECTION, [
                    late,
                    _edge("late-edge", late["id"], root["id"], "DERIVED_FROM"),
                ])
            return super().embed_document(text)

    indexer = _indexer(qdrant, lock_dir=tmp_path / "locks")
    indexer.embeddings = PublishingEmbedding()
    result = indexer.index([path], dry_run=False)

    assert result["partial_failure"] is True
    assert any("locked invalidation closure grew" in item["error"] for item in result["errors"])
    assert root["id"] in qdrant.points
    assert qdrant.points[late["id"]]["payload"].get("fact_status") is None
    assert _events(qdrant) == {}


def test_w1_unchanged_baseline_bootstraps_one_observed_event(tmp_path: Path):
    path = tmp_path / "w1.md"
    path.write_text("A", encoding="utf-8")
    qdrant = FakeQdrant()
    assert _indexer(qdrant, lock_dir=tmp_path / "locks", mode="capture").index([path], dry_run=False)["errors"] == []

    result = _indexer(qdrant, lock_dir=tmp_path / "locks").index([path], dry_run=False)

    assert result["errors"] == []
    events = list(_events(qdrant).values())
    assert len(events) == 1
    event = events[0]
    assert event["event_kind"] == "observed"
    assert event["previous_event_id"] is None
    assert event["from_version_id"] is None
    assert "from_file_sha256" not in event
    assert event["lineage_observation"] == "indexed_payload"
    assert event["lineage_history_complete"] is False
    source = next(point["payload"] for point in qdrant.points.values() if (point.get("payload") or {}).get("lineage_role") == "file_source")
    assert source["head_event_id"] == next(iter(_events(qdrant)))


def test_w1_a_then_disk_b_bootstraps_before_transition(tmp_path: Path):
    path = tmp_path / "w1-edit.md"
    path.write_text("A", encoding="utf-8")
    qdrant = FakeQdrant()
    assert _indexer(qdrant, lock_dir=tmp_path / "locks", mode="capture").index([path], dry_run=False)["errors"] == []
    path.write_text("B", encoding="utf-8")

    assert _indexer(qdrant, lock_dir=tmp_path / "locks").index([path], dry_run=False)["errors"] == []

    events = _events(qdrant)
    assert len(events) == 2
    observed_id, observed = next((point_id, payload) for point_id, payload in events.items() if payload["event_kind"] == "observed")
    modified = next(payload for payload in events.values() if payload["event_kind"] == "modified")
    assert observed["event_state"] == "committed"
    assert observed["lineage_observation"] == "indexed_payload"
    assert modified["previous_event_id"] == observed_id
    assert modified["lineage_observation"] == "read_bytes"
    assert modified["lineage_history_complete"] is False


def test_w1_empty_file_bootstrap(tmp_path: Path):
    path = tmp_path / "empty.md"
    path.write_text("", encoding="utf-8")
    qdrant = FakeQdrant()
    assert _indexer(qdrant, lock_dir=tmp_path / "locks", mode="capture").index([path], dry_run=False)["errors"] == []

    assert _indexer(qdrant, lock_dir=tmp_path / "locks").index([path], dry_run=False)["errors"] == []

    event = next(iter(_events(qdrant).values()))
    assert event["event_kind"] == "observed"
    assert event["created_point_ids"] == []
    assert event["retired_point_ids"] == []
    assert event["to_version_id"] is not None


def test_w1_bootstrap_resume_reuses_event_after_head_commit_interruption(tmp_path: Path):
    path = tmp_path / "resume-bootstrap.md"
    path.write_text("A", encoding="utf-8")
    qdrant = FakeQdrant()
    assert _indexer(qdrant, lock_dir=tmp_path / "locks", mode="capture").index([path], dry_run=False)["errors"] == []
    qdrant.counts = {}
    qdrant.failed = False
    qdrant.fail_at = ("upsert", 2)
    indexer = _indexer(qdrant, lock_dir=tmp_path / "locks")

    assert indexer.index([path], dry_run=False)["partial_failure"] is True
    first_events = _events(qdrant)
    assert len(first_events) == 1
    event_id = next(iter(first_events))
    created_at = first_events[event_id]["created_at"]
    source = next(point["payload"] for point in qdrant.points.values() if (point.get("payload") or {}).get("lineage_role") == "file_source")
    assert source.get("head_event_id") is None

    qdrant.fail_at = None
    assert indexer.index([path], dry_run=False)["errors"] == []
    assert list(_events(qdrant)) == [event_id]
    assert _events(qdrant)[event_id]["created_at"] == created_at
    source = next(point["payload"] for point in qdrant.points.values() if (point.get("payload") or {}).get("lineage_role") == "file_source")
    assert source["head_event_id"] == event_id


def test_rechunk_keeps_current_version_active(tmp_path: Path):
    path = tmp_path / "rechunk.md"
    path.write_text("one two three four five six seven eight", encoding="utf-8")
    qdrant = FakeQdrant()
    assert _indexer(qdrant, lock_dir=tmp_path / "locks", mode="capture", max_chunk_tokens=128).index([path], dry_run=False)["errors"] == []

    assert _indexer(qdrant, lock_dir=tmp_path / "locks", max_chunk_tokens=2).index([path], dry_run=False)["errors"] == []

    source = next(point["payload"] for point in qdrant.points.values() if (point.get("payload") or {}).get("lineage_role") == "file_source")
    version = qdrant.points[source["current_version_id"]]["payload"]
    assert version["fact_status"] == "active"
    assert version["stale"] is False
    assert any(event["event_kind"] == "rechunked" for event in _events(qdrant).values())


def test_bare_legacy_file_hash_does_not_invent_old_version(tmp_path: Path):
    path = tmp_path / "bare-hash.md"
    path.write_text("new bytes", encoding="utf-8")
    legacy = _legacy_chunk(path)
    legacy["payload"]["file_sha256"] = "a" * 64
    qdrant = FakeQdrant([legacy])

    assert _indexer(qdrant, lock_dir=tmp_path / "locks").index([path], dry_run=False)["errors"] == []

    event = next(iter(_events(qdrant).values()))
    assert event["event_kind"] == "observed"
    assert event["from_version_id"] is None
    assert "from_file_sha256" not in event
    assert not any((point.get("payload") or {}).get("relation_type") == "SUPERSEDES" for point in qdrant.points.values())
    assert sum((point.get("payload") or {}).get("lineage_role") == "file_version" for point in qdrant.points.values()) == 1


def test_retired_mechanical_edges_do_not_consume_dependency_bound():
    root = _dependent("retired-bound-root")
    live = _dependent("retired-bound-live")
    retired_event_id = str(uuid.uuid4())
    points = [
        root,
        live,
        _edge("retired-bound-live-edge", str(live["id"]), str(root["id"]), "DERIVED_FROM"),
    ]
    retired_edges = [
        _mechanical_edge(
            f"retired-bound-{index}", _id(f"retired-chunk-{index}"), str(root["id"]),
            retired_by_event_id=retired_event_id,
        )
        for index in range(5)
    ]
    for edge in retired_edges:
        edge["payload"]["lineage_retired"] = True
    points.extend(retired_edges)

    result = lineage.find_direct_dependents(
        qdrant=FakeQdrant(points), collection_name=COLLECTION,
        target_point_id=str(root["id"]), profile_id=PROFILE, max_results=1,
    )

    assert result["complete"] is True
    assert [point["id"] for point in result["points"]] == [live["id"]]
    assert result["errors"] == []


def test_inline_non_dependencies_are_ignored_and_cap_boundary_is_exact():
    root = _id("inline-boundary-root")
    support = _dependent("inline-support")
    support["payload"]["derived_from"] = [{"point_id": root, "relation_type": "SUPPORTS"}]
    first = _dependent("inline-cap-first")
    first["payload"]["derived_from"] = [{"point_id": root, "relation_type": "DERIVED_FROM"}]
    second = _dependent("inline-cap-second")
    second["payload"]["derived_from"] = [{"point_id": root, "relation_type": "DERIVED_FROM"}]
    third = _dependent("inline-cap-third")
    third["payload"]["derived_from"] = [{"point_id": root, "relation_type": "DERIVED_FROM"}]

    exact = lineage.plan_invalidation(
        qdrant=FakeQdrant([support, first, second]), collection_name=COLLECTION,
        root_point_ids=[root], event_id=_id("exact-cap"), profile_id=PROFILE,
        max_points=2,
    )
    overflow = lineage.plan_invalidation(
        qdrant=FakeQdrant([first, second, third]), collection_name=COLLECTION,
        root_point_ids=[root], event_id=_id("overflow-cap"), profile_id=PROFILE,
        max_points=2,
    )

    assert exact["complete"] is True
    assert exact["dependent_ids"] == sorted([first["id"], second["id"]])
    assert overflow["complete"] is False
    assert any("bound" in error for error in overflow["errors"])


@pytest.mark.parametrize("mode", ["off", "capture"])
def test_prepare_errors_do_not_change_off_or_capture_directory_retirement(mode, tmp_path: Path):
    missing = tmp_path / "gone.md"
    legacy = _legacy_chunk(missing)
    qdrant = FakeQdrant([legacy])
    indexer = _indexer(qdrant, lock_dir=tmp_path / "locks", mode=mode)
    indexer.prepare = lambda *_args, **_kwargs: {
        "files": [], "file_manifests": [], "files_seen": 0, "files_indexed": 0,
        "files_skipped": 0, "skipped": [], "chunks": [], "chunks_prepared": 0,
        "errors": [{"path": str(tmp_path / "bad.md"), "error": "injected prepare error"}],
        "max_files_truncated": False,
    }

    result = indexer.index([tmp_path], dry_run=True)

    assert result["directory_manifest_checked"] is True
    assert result["deleted_file_paths"] == [str(missing)]
    if mode == "off":
        assert result["stale_ids"] == [legacy["id"]]
    else:
        assert result["refusals"] == [{
            "file_path": str(missing), "reason": "retirement_requires_reconcile",
        }]


def test_event_matches_locked_closure_after_dependent_deleted(tmp_path: Path):
    path = tmp_path / "deleted-race.md"
    path.write_text("replacement", encoding="utf-8")
    root = _legacy_chunk(path, "deleted-race-root")
    dependent = _dependent("deleted-race-dependent")
    edge = _edge("deleted-race-edge", dependent["id"], root["id"], "DERIVED_FROM")
    qdrant = FakeQdrant([root, dependent, edge])

    class DeletingEmbedding(FakeEmbedding):
        def embed_document(self, text: str) -> list[float]:
            qdrant.points.pop(str(dependent["id"]), None)
            qdrant.points.pop(str(edge["id"]), None)
            return super().embed_document(text)

    indexer = _indexer(qdrant, lock_dir=tmp_path / "locks")
    indexer.embeddings = DeletingEmbedding()
    first = indexer.index([path], dry_run=False)
    assert first["partial_failure"] is True
    assert any("closure changed" in item["error"] for item in first["errors"])

    indexer.embeddings = FakeEmbedding()
    assert indexer.index([path], dry_run=False)["errors"] == []
    event = next(iter(_events(qdrant).values()))
    demoted = sorted(
        point_id for point_id, point in qdrant.points.items()
        if (point.get("payload") or {}).get("lineage_review_event_ids")
    )
    assert event["patched_point_ids"] == demoted == []
    assert event["review_changes"] == []


def test_event_matches_locked_closure_after_dependent_status_change(tmp_path: Path):
    path = tmp_path / "status-race.md"
    path.write_text("replacement", encoding="utf-8")
    root = _legacy_chunk(path, "status-race-root")
    dependent = _dependent("status-race-dependent")
    edge = _edge("status-race-edge", dependent["id"], root["id"], "DERIVED_FROM")
    qdrant = FakeQdrant([root, dependent, edge])

    class DisputingEmbedding(FakeEmbedding):
        def embed_document(self, text: str) -> list[float]:
            qdrant.points[str(dependent["id"])]["payload"]["fact_status"] = "disputed"
            return super().embed_document(text)

    indexer = _indexer(qdrant, lock_dir=tmp_path / "locks")
    indexer.embeddings = DisputingEmbedding()
    first = indexer.index([path], dry_run=False)
    assert first["partial_failure"] is True
    assert any("closure changed" in item["error"] for item in first["errors"])

    indexer.embeddings = FakeEmbedding()
    assert indexer.index([path], dry_run=False)["errors"] == []
    event_id, event = next(iter(_events(qdrant).items()))
    payload = qdrant.points[str(dependent["id"])]["payload"]
    assert event["patched_point_ids"] == [dependent["id"]]
    assert event["review_changes"][0]["point_id"] == dependent["id"]
    assert event["review_changes"][0]["patch"] == {
        "requires_review": True,
        "lineage_review_event_ids": [event_id],
    }
    assert payload["fact_status"] == "disputed"
    assert payload["requires_review"] is True


def test_pending_event_rebuilds_closure_after_chunks_staged(tmp_path: Path):
    path = tmp_path / "pending-late.md"
    path.write_text("replacement", encoding="utf-8")
    root = _legacy_chunk(path, "pending-late-root")
    late = _dependent("pending-late-dependent")
    late_edge = _edge("pending-late-edge", late["id"], root["id"], "DERIVED_FROM")

    class PublishAfterStaging(FakeQdrant):
        published = False

        def update_payload(self, name: str, point_id: str, payload: dict):
            result = super().update_payload(name, point_id, payload)
            if payload.get("event_state") == "chunks_staged" and not self.published:
                self.published = True
                self.points[str(late["id"])] = copy.deepcopy(late)
                self.points[str(late_edge["id"])] = copy.deepcopy(late_edge)
                raise RuntimeError("interrupted after chunks staged")
            return result

    qdrant = PublishAfterStaging([root])
    indexer = _indexer(qdrant, lock_dir=tmp_path / "locks")
    assert indexer.index([path], dry_run=False)["partial_failure"] is True
    assert next(iter(_events(qdrant).values()))["event_state"] == "chunks_staged"

    assert indexer.index([path], dry_run=False)["errors"] == []
    event = next(iter(_events(qdrant).values()))
    payload = qdrant.points[str(late["id"])]["payload"]
    assert event["patched_point_ids"] == [late["id"]]
    assert [change["point_id"] for change in event["review_changes"]] == [late["id"]]
    assert payload["fact_status"] == "review_required"
    assert payload["requires_review"] is True


def test_w1_bootstrap_refuses_incomplete_chunk_inventory(tmp_path: Path):
    path = tmp_path / "partial-w1.md"
    path.write_text("word " * 200, encoding="utf-8")
    qdrant = FakeQdrant()
    assert _indexer(
        qdrant, lock_dir=tmp_path / "locks", mode="capture", max_chunk_tokens=2,
    ).index([path], dry_run=False)["errors"] == []
    chunks = [
        point for point in qdrant.points.values()
        if (point.get("payload") or {}).get("memory_kind") == "source_chunk"
    ]
    assert len(chunks) > 1
    qdrant.points.pop(str(chunks[0]["id"]))

    result = _indexer(
        qdrant, lock_dir=tmp_path / "locks", max_chunk_tokens=2,
    ).index([path], dry_run=False)

    assert result["refused"] is True
    assert any(
        "event-free W1 chunk inventory is incomplete or ambiguous" in item["reason"]
        for item in result["refusals"]
    )
    assert _events(qdrant) == {}


def test_w1_bootstrap_refuses_when_source_node_is_lost(tmp_path: Path):
    path = tmp_path / "lost-source-w1.md"
    path.write_text("A", encoding="utf-8")
    qdrant = FakeQdrant()
    assert _indexer(
        qdrant, lock_dir=tmp_path / "locks", mode="capture",
    ).index([path], dry_run=False)["errors"] == []
    source_id = next(
        str(point["id"]) for point in qdrant.points.values()
        if (point.get("payload") or {}).get("lineage_role") == "file_source"
    )
    qdrant.points.pop(source_id)
    path.write_text("B", encoding="utf-8")
    before = copy.deepcopy(qdrant.points)

    result = _indexer(qdrant, lock_dir=tmp_path / "locks").index([path], dry_run=False)

    assert result["refused"] is True
    assert any("event-free W1 source baseline is incomplete" in item["reason"] for item in result["refusals"])
    assert qdrant.points == before
    assert _events(qdrant) == {}


def test_w1_bootstrap_rejects_permuted_chunk_indices(tmp_path: Path):
    path = tmp_path / "permuted-w1.md"
    path.write_text("word " * 200, encoding="utf-8")
    qdrant = FakeQdrant()
    assert _indexer(
        qdrant, lock_dir=tmp_path / "locks", mode="capture", max_chunk_tokens=2,
    ).index([path], dry_run=False)["errors"] == []
    chunks = sorted(
        (
            point for point in qdrant.points.values()
            if (point.get("payload") or {}).get("memory_kind") == "source_chunk"
        ),
        key=lambda point: point["payload"]["chunk_index"],
    )
    assert len(chunks) > 1
    chunks[0]["payload"]["chunk_index"], chunks[1]["payload"]["chunk_index"] = (
        chunks[1]["payload"]["chunk_index"], chunks[0]["payload"]["chunk_index"],
    )

    result = _indexer(
        qdrant, lock_dir=tmp_path / "locks", max_chunk_tokens=2,
    ).index([path], dry_run=False)

    assert result["refused"] is True
    assert any("event-free W1 chunk binding mismatch" in item["reason"] for item in result["refusals"])
    assert _events(qdrant) == {}


def test_w1_bootstrap_rejects_lost_nonempty_inventory_without_file_size(tmp_path: Path):
    path = tmp_path / "stripped-size-w1.md"
    path.write_text("not empty", encoding="utf-8")
    qdrant = FakeQdrant()
    assert _indexer(
        qdrant, lock_dir=tmp_path / "locks", mode="capture",
    ).index([path], dry_run=False)["errors"] == []
    for point_id, point in list(qdrant.points.items()):
        payload = point.get("payload") or {}
        if payload.get("memory_kind") == "source_chunk":
            qdrant.points.pop(point_id)
        elif payload.get("lineage_role") == "file_version":
            payload.pop("file_size", None)

    result = _indexer(qdrant, lock_dir=tmp_path / "locks").index([path], dry_run=False)

    assert result["refused"] is True
    assert any("event-free W1 chunk inventory is incomplete or ambiguous" in item["reason"] for item in result["refusals"])
    assert _events(qdrant) == {}


@pytest.mark.parametrize(
    "case",
    ["over-count", "under-count", "duplicate-index", "non-int-count", "non-int-index", "all-chunks-lost"],
)
def test_w1_bootstrap_pins_each_inventory_invariant(case: str, tmp_path: Path):
    path = tmp_path / f"inventory-{case}.md"
    content = ""
    if case in {"non-int-count", "non-int-index"}:
        content = "A"
    elif case != "all-chunks-lost":
        content = "word " * 200
    path.write_text(content, encoding="utf-8")
    qdrant = FakeQdrant()
    assert _indexer(
        qdrant, lock_dir=tmp_path / "locks", mode="capture", max_chunk_tokens=2,
    ).index([path], dry_run=False)["errors"] == []
    chunks = sorted(
        (
            point for point in qdrant.points.values()
            if (point.get("payload") or {}).get("memory_kind") == "source_chunk"
        ),
        key=lambda point: point["payload"]["chunk_index"],
    )
    if case in {"over-count", "under-count", "duplicate-index"}:
        assert len(chunks) > 1
    if case == "over-count":
        for chunk in chunks:
            chunk["payload"]["chunk_count"] = len(chunks) + 1
    elif case == "under-count":
        first_chunk = next(
            point for point in qdrant.points.values()
            if (point.get("payload") or {}).get("memory_kind") == "source_chunk"
        )
        changed_chunk = next(chunk for chunk in chunks if chunk is not first_chunk)
        changed_chunk["payload"]["chunk_count"] = len(chunks) + 1
    elif case == "duplicate-index":
        chunks[1]["payload"]["chunk_index"] = chunks[0]["payload"]["chunk_index"]
    elif case == "non-int-count":
        assert len(chunks) == 1
        chunks[0]["payload"]["chunk_count"] = True
    elif case == "non-int-index":
        assert len(chunks) == 1
        chunks[0]["payload"]["chunk_index"] = False
    else:
        assert chunks == []
        version = next(
            point for point in qdrant.points.values()
            if (point.get("payload") or {}).get("lineage_role") == "file_version"
        )
        version["payload"]["file_size"] = 1

    result = _indexer(
        qdrant, lock_dir=tmp_path / "locks", max_chunk_tokens=2,
    ).index([path], dry_run=False)

    assert result["refused"] is True
    assert any("event-free W1 chunk inventory is incomplete or ambiguous" in item["reason"] for item in result["refusals"])
    assert _events(qdrant) == {}


def test_w1_bootstrap_checks_every_chunk_binding(tmp_path: Path):
    path = tmp_path / "binding-w1.md"
    path.write_text("word " * 200, encoding="utf-8")
    qdrant = FakeQdrant()
    assert _indexer(
        qdrant, lock_dir=tmp_path / "locks", mode="capture", max_chunk_tokens=2,
    ).index([path], dry_run=False)["errors"] == []
    chunk = next(
        point for point in qdrant.points.values()
        if (point.get("payload") or {}).get("memory_kind") == "source_chunk"
    )
    chunk["payload"]["content_hash"] = "sha256:" + "0" * 64

    result = _indexer(
        qdrant, lock_dir=tmp_path / "locks", max_chunk_tokens=2,
    ).index([path], dry_run=False)

    assert result["refused"] is True
    assert any("event-free W1 chunk binding mismatch" in item["reason"] for item in result["refusals"])
    assert _events(qdrant) == {}


def test_rechunk_then_modify_converges(tmp_path: Path):
    path = tmp_path / "rechunk-modify.md"
    path.write_text("one two three four five six seven eight", encoding="utf-8")
    qdrant = FakeQdrant()
    assert _indexer(qdrant, lock_dir=tmp_path / "locks", mode="capture").index([path], dry_run=False)["errors"] == []
    assert _indexer(qdrant, lock_dir=tmp_path / "locks", max_chunk_tokens=2).index([path], dry_run=False)["errors"] == []
    path.write_text("one two three four five six seven changed", encoding="utf-8")

    assert _indexer(qdrant, lock_dir=tmp_path / "locks", max_chunk_tokens=2).index([path], dry_run=False)["errors"] == []
    source = next(
        point["payload"] for point in qdrant.points.values()
        if (point.get("payload") or {}).get("lineage_role") == "file_source"
    )
    kinds = []
    event_id = source["head_event_id"]
    events = _events(qdrant)
    while event_id:
        kinds.append(events[event_id]["event_kind"])
        event_id = events[event_id].get("previous_event_id")
    assert kinds == ["modified", "rechunked", "observed"]


def test_rechunk_then_delete_converges_with_tombstone(tmp_path: Path):
    path = tmp_path / "rechunk-delete.md"
    path.write_text("one two three four five six seven eight", encoding="utf-8")
    qdrant = FakeQdrant()
    assert _indexer(qdrant, lock_dir=tmp_path / "locks", mode="capture").index([path], dry_run=False)["errors"] == []
    assert _indexer(qdrant, lock_dir=tmp_path / "locks", max_chunk_tokens=2).index([path], dry_run=False)["errors"] == []
    path.unlink()

    assert _indexer(qdrant, lock_dir=tmp_path / "locks", max_chunk_tokens=2).index([tmp_path], dry_run=False)["errors"] == []
    source = next(
        point["payload"] for point in qdrant.points.values()
        if (point.get("payload") or {}).get("lineage_role") == "file_source"
    )
    events = _events(qdrant)
    assert source["source_deleted"] is True
    assert source["current_version_id"] is None
    assert events[source["head_event_id"]]["event_kind"] == "deleted"


def test_w1_capture_rechunk_delete_converges(tmp_path: Path):
    path = tmp_path / "w1-rechunk-delete.md"
    path.write_text("one two three four five six seven eight", encoding="utf-8")
    qdrant = FakeQdrant()
    assert _indexer(qdrant, lock_dir=tmp_path / "locks", mode="capture").index([path], dry_run=False)["errors"] == []
    assert _indexer(qdrant, lock_dir=tmp_path / "locks", max_chunk_tokens=2).index([path], dry_run=False)["errors"] == []
    path.unlink()
    result = _indexer(qdrant, lock_dir=tmp_path / "locks", max_chunk_tokens=2).index([tmp_path], dry_run=False)
    assert result["errors"] == []
    source = next(
        point["payload"] for point in qdrant.points.values()
        if (point.get("payload") or {}).get("lineage_role") == "file_source"
    )
    assert source["source_deleted"] is True


def test_unmarked_dangling_edge_still_refuses():
    root = _id("dangling-root")
    missing = _id("dangling-source")
    result = lineage.find_direct_dependents(
        qdrant=FakeQdrant([_edge("dangling-edge", missing, root, "DERIVED_FROM")]),
        collection_name=COLLECTION,
        target_point_id=root,
        profile_id=PROFILE,
    )
    assert result["complete"] is False
    assert result["errors"] == [f"dependency source points are missing: ['{missing}']"]


@pytest.mark.parametrize(
    "retired_by_event_id",
    [pytest.param(None, id="absent"), pytest.param("not-a-uuid", id="malformed")],
)
def test_dangling_mechanical_edge_requires_valid_retirement_marker(retired_by_event_id):
    root = _id("mechanical-marker-root")
    missing = _id("mechanical-marker-source")
    edge = _mechanical_edge(
        "mechanical-marker-edge",
        missing,
        root,
        retired_by_event_id=retired_by_event_id,
    )

    result = lineage.find_direct_dependents(
        qdrant=FakeQdrant([edge]),
        collection_name=COLLECTION,
        target_point_id=root,
        profile_id=PROFILE,
    )

    assert result["complete"] is False
    assert result["errors"] == [f"dependency source points are missing: ['{missing}']"]


def test_retired_mechanical_edge_resolves_without_missing_source_error():
    root = _id("retired-mechanical-root")
    missing = _id("retired-mechanical-source")
    edge = _mechanical_edge(
        "retired-mechanical-edge",
        missing,
        root,
        retired_by_event_id=_id("retirement-event"),
    )

    result = lineage.find_direct_dependents(
        qdrant=FakeQdrant([edge]),
        collection_name=COLLECTION,
        target_point_id=root,
        profile_id=PROFILE,
    )

    assert result == {"complete": True, "points": [], "edge_keys": [], "errors": []}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("relation_type", "SUMMARIZES", id="derived-from-relation"),
        pytest.param("lineage_operation", "manual", id="mechanical-operation"),
        pytest.param("target_entity_type", "version", id="source-target"),
    ],
)
def test_retired_edge_exemption_requires_every_mechanical_field(field, value):
    root = _id(f"retired-shape-root-{field}")
    missing = _id(f"retired-shape-source-{field}")
    edge = _mechanical_edge(
        f"retired-shape-edge-{field}",
        missing,
        root,
        retired_by_event_id=_id(f"retired-shape-event-{field}"),
    )
    edge["payload"][field] = value

    result = lineage.find_direct_dependents(
        qdrant=FakeQdrant([edge]),
        collection_name=COLLECTION,
        target_point_id=root,
        profile_id=PROFILE,
    )

    assert result["complete"] is False
    assert result["errors"] == [f"dependency source points are missing: ['{missing}']"]


def test_inline_bound_counts_dependencies_not_non_dependency_citers():
    root = _id("inline-real-bound-root")
    supports = []
    for index in range(3):
        point = _dependent(f"inline-support-{index}")
        point["payload"]["derived_from"] = [{"point_id": root, "relation_type": "SUPPORTS"}]
        supports.append(point)
    real = _dependent("inline-real")
    real["payload"]["derived_from"] = [{"point_id": root, "relation_type": "DERIVED_FROM"}]

    result = lineage.plan_invalidation(
        qdrant=FakeQdrant([*supports, real]), collection_name=COLLECTION,
        root_point_ids=[root], event_id=_id("inline-real-event"), profile_id=PROFILE,
        max_points=2,
    )
    assert result["complete"] is True
    assert result["dependent_ids"] == [real["id"]]


@pytest.mark.parametrize("mode, expected", [("off", []), ("capture", ["lineage_capture_requires_exact_scroll_support"])])
def test_off_and_capture_scroll_support_behavior_is_pinned(mode, expected, tmp_path: Path):
    path = tmp_path / "scroll.md"
    path.write_text("content", encoding="utf-8")

    class NoScrollQdrant:
        pass

    result = _indexer(NoScrollQdrant(), lock_dir=tmp_path / "locks", mode=mode).index([path], dry_run=True)
    assert [item["reason"] for item in result["refusals"]] == expected


def test_bootstrap_retry_reuses_existing_event_payload(tmp_path: Path):
    path = tmp_path / "bootstrap-payload.md"
    path.write_text("A", encoding="utf-8")
    qdrant = FakeQdrant()
    assert _indexer(qdrant, lock_dir=tmp_path / "locks", mode="capture").index([path], dry_run=False)["errors"] == []
    qdrant.counts = {}
    qdrant.fail_at = ("upsert", 2)
    indexer = _indexer(qdrant, lock_dir=tmp_path / "locks")
    assert indexer.index([path], dry_run=False)["partial_failure"] is True
    event_id = next(iter(_events(qdrant)))
    qdrant.points[event_id]["payload"]["retry_marker"] = "preserve-me"
    qdrant.fail_at = None

    assert indexer.index([path], dry_run=False)["errors"] == []
    assert _events(qdrant)[event_id]["retry_marker"] == "preserve-me"


def test_a_to_b_bootstrap_retry_reuses_original_event_timestamps(tmp_path: Path):
    path = tmp_path / "bootstrap-a-to-b.md"
    path.write_text("A", encoding="utf-8")
    qdrant = FakeQdrant()
    assert _indexer(
        qdrant, lock_dir=tmp_path / "locks", mode="capture",
    ).index([path], dry_run=False)["errors"] == []
    path.write_text("B", encoding="utf-8")
    qdrant.counts = {}
    qdrant.fail_at = ("update_payload", 1)
    indexer = _indexer(qdrant, lock_dir=tmp_path / "locks")

    assert indexer.index([path], dry_run=False)["partial_failure"] is True
    observed_id, observed = next(
        (event_id, event) for event_id, event in _events(qdrant).items()
        if event["event_kind"] == "observed"
    )
    original_times = (observed["created_at"], observed["observed_at"])
    qdrant.fail_at = None

    assert indexer.index([path], dry_run=False)["errors"] == []
    retried = _events(qdrant)[observed_id]
    assert (retried["created_at"], retried["observed_at"]) == original_times
    modified = next(event for event in _events(qdrant).values() if event["event_kind"] == "modified")
    assert modified["previous_event_id"] == observed_id


def test_locked_invalidation_complete_is_required(monkeypatch, tmp_path: Path):
    path = tmp_path / "locked-complete.md"
    path.write_text("replacement", encoding="utf-8")
    root = _legacy_chunk(path, "locked-complete-root")
    qdrant = FakeQdrant([root])
    original = lineage.plan_invalidation
    calls = 0

    def incomplete_second_call(**kwargs):
        nonlocal calls
        calls += 1
        result = original(**kwargs)
        if calls == 2:
            result = {**result, "complete": False, "errors": ["forced incomplete"]}
        return result

    monkeypatch.setattr(lineage, "plan_invalidation", incomplete_second_call)
    result = _indexer(qdrant, lock_dir=tmp_path / "locks").index([path], dry_run=False)
    assert result["partial_failure"] is True
    assert any("locked invalidation closure is incomplete" in item["error"] for item in result["errors"])
    assert root["id"] in qdrant.points


def test_apply_requires_chunk_points_to_cover_new_chunk_ids(monkeypatch, tmp_path: Path):
    path = tmp_path / "missing-staged-chunks.md"
    path.write_text("replacement", encoding="utf-8")
    root = _legacy_chunk(path, "missing-staged-root")
    qdrant = FakeQdrant([root])
    original = lineage.apply_reconciliation_plan

    def omit_chunks(**kwargs):
        return original(**{**kwargs, "chunk_points": []})

    monkeypatch.setattr(indexer_module, "apply_reconciliation_plan", omit_chunks)
    result = _indexer(qdrant, lock_dir=tmp_path / "locks").index([path], dry_run=False)
    assert result["partial_failure"] is True
    assert any("chunk_points do not exactly cover new_chunk_ids" in item["error"] for item in result["errors"])
    assert root["id"] in qdrant.points
    assert _events(qdrant) == {}


def test_apply_rejects_extra_chunk_points(monkeypatch, tmp_path: Path):
    path = tmp_path / "extra-staged-chunks.md"
    path.write_text("replacement", encoding="utf-8")
    root = _legacy_chunk(path, "extra-staged-root")
    qdrant = FakeQdrant([root])
    original = lineage.apply_reconciliation_plan

    def append_chunk(**kwargs):
        extra = copy.deepcopy(kwargs["chunk_points"][0])
        extra["id"] = _id("unexpected-extra-chunk")
        return original(**{**kwargs, "chunk_points": [*kwargs["chunk_points"], extra]})

    monkeypatch.setattr(indexer_module, "apply_reconciliation_plan", append_chunk)
    result = _indexer(qdrant, lock_dir=tmp_path / "locks").index([path], dry_run=False)

    assert result["partial_failure"] is True
    assert any("chunk_points do not exactly cover new_chunk_ids" in item["error"] for item in result["errors"])
    assert root["id"] in qdrant.points
    assert _events(qdrant) == {}


def test_retirement_marking_requires_complete_edge_lookup(monkeypatch):
    monkeypatch.setattr(lineage, "_scroll_bounded", lambda *args, **kwargs: ([], False))

    with pytest.raises(RuntimeError, match="lookup exceeded bound"):
        lineage._retire_chunk_derivation_edges(
            qdrant=FakeQdrant(), collection_name=COLLECTION,
            retired_ids=[_id("retired-bound")], old_version_id=_id("old-version"),
            profile_id=PROFILE, event_id=str(uuid.uuid4()),
        )


def test_retirement_marking_requires_old_version_target():
    retired_id = _id("retired-target")
    edge = _mechanical_edge(
        "wrong-version-edge", retired_id, _id("wrong-version"),
    )
    qdrant = FakeQdrant([edge])

    lineage._retire_chunk_derivation_edges(
        qdrant=qdrant, collection_name=COLLECTION,
        retired_ids=[retired_id], old_version_id=_id("expected-version"),
        profile_id=PROFILE, event_id=str(uuid.uuid4()),
    )

    assert "lineage_retired_by_event_id" not in qdrant.points[str(edge["id"])]["payload"]


def test_retirement_marking_requires_marker_readback():
    retired_id = _id("retired-readback")
    old_version_id = _id("old-version-readback")
    edge = _mechanical_edge("readback-edge", retired_id, old_version_id)

    class LostMarkerQdrant(FakeQdrant):
        def update_payload(self, name: str, point_id: str, payload: dict):
            self._boundary("update_payload")
            self.points[str(point_id)]["payload"]["lineage_retired_by_event_id"] = payload[
                "lineage_retired_by_event_id"
            ]
            return {"status": "ok"}

    with pytest.raises(RuntimeError, match="marking failed"):
        lineage._retire_chunk_derivation_edges(
            qdrant=LostMarkerQdrant([edge]), collection_name=COLLECTION,
            retired_ids=[retired_id], old_version_id=old_version_id,
            profile_id=PROFILE, event_id=str(uuid.uuid4()),
        )


def test_brand_new_event_payload_uses_indexed_basis_and_contract_fields(tmp_path: Path):
    path = tmp_path / "new.md"
    path.write_text("brand new", encoding="utf-8")
    qdrant = FakeQdrant()

    assert _indexer(qdrant, lock_dir=tmp_path / "locks").index([path], dry_run=False)["errors"] == []

    event = next(iter(_events(qdrant).values()))
    assert event["lineage_baseline_basis"] == "indexed_file_sha256"
    assert event["lineage_observation"] == "read_bytes"
    assert event["lineage_history_complete"] is False
