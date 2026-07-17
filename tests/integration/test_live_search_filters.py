from __future__ import annotations

import json
import os
import urllib.parse
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pytest

from qdrant_memory.client import QdrantClient
from qdrant_memory.embeddings import EmbeddingClient
from qdrant_memory.indexer import make_file_chunk_id
from qdrant_memory.learning import LearningStore, build_learning_payload
from qdrant_memory.retriever import MemoryRetriever
from qdrant_memory.schema import build_payload
from qdrant_memory.writer import ConversationWriter

TRUTHY = {"1", "true", "yes", "on"}
PROFILE_ID = "integration-profile"
PLATFORM = "pytest"
PROJECT_A = "/tmp/hermes-qdrant-live-search/project-a"
PROJECT_B = "/tmp/hermes-qdrant-live-search/project-b"
CURRENT_DATE = "2026-02-15T12:00:00+00:00"
OLD_DATE = "2024-01-15T12:00:00+00:00"
FUTURE_DATE = "2027-01-15T12:00:00+00:00"


def _integration_enabled() -> bool:
    return os.environ.get("RUN_QDRANT_INTEGRATION", "").strip().lower() in TRUTHY


pytestmark = pytest.mark.skipif(
    not _integration_enabled(),
    reason="set RUN_QDRANT_INTEGRATION=1 to run live Qdrant/embedding integration tests",
)


@dataclass(frozen=True)
class LiveContext:
    qdrant: QdrantClient
    embeddings: EmbeddingClient
    memory_collection: str
    learning_collection: str
    prefix: str
    vector_size: int
    distance: str
    scope: dict[str, str]


@pytest.fixture
def live_context() -> Iterator[LiveContext]:
    prefix = os.environ.get("QDRANT_TEST_COLLECTION_PREFIX", "hermes_qdrant_itest").strip()
    if not prefix:
        pytest.fail("QDRANT_TEST_COLLECTION_PREFIX must not be empty")

    qdrant_url = os.environ.get("QDRANT_TEST_URL", "http://127.0.0.1:6333")
    qdrant_credential = os.environ.get("QDRANT_TEST_API_KEY", "")
    embedding_url = os.environ.get("QDRANT_TEST_EMBEDDING_URL", "http://127.0.0.1:8080/v1")
    embedding_model = os.environ.get("QDRANT_TEST_EMBEDDING_MODEL", "bge-m3")
    vector_size = int(os.environ.get("QDRANT_TEST_VECTOR_SIZE", "1024"))
    distance = os.environ.get("QDRANT_TEST_DISTANCE", "Cosine")

    random_suffix = uuid.uuid4().hex[:12]
    base_name = f"{prefix}_{random_suffix}"
    memory_collection = f"{base_name}_memory"
    learning_collection = f"{base_name}_learnings"
    for collection_name in (memory_collection, learning_collection):
        if not collection_name.startswith(prefix):
            pytest.fail(f"Refusing to use non-prefixed test collection: {collection_name!r}")

    qdrant = QdrantClient(qdrant_url, timeout=10.0, **{"api_key": qdrant_credential})
    embeddings = EmbeddingClient(embedding_url, embedding_model, timeout=30.0)

    try:
        qdrant._request("GET", "/collections")
    except Exception as exc:  # pragma: no cover - depends on live service state
        pytest.fail(f"Qdrant health check failed for {qdrant_url}: {exc}")

    try:
        probe_vector = embeddings.embed_query("Hermes Qdrant live integration health probe")
    except Exception as exc:  # pragma: no cover - depends on live service state
        pytest.fail(f"Embedding health check failed for {embedding_url} model={embedding_model!r}: {exc}")
    if len(probe_vector) != vector_size:
        pytest.fail(
            f"Embedding vector size mismatch for {embedding_url} model={embedding_model!r}: "
            f"expected QDRANT_TEST_VECTOR_SIZE={vector_size}, got {len(probe_vector)}"
        )

    created: list[str] = []
    try:
        existing_collections = set(qdrant.get_collections())
        for collection_name in (memory_collection, learning_collection):
            if collection_name in existing_collections:
                pytest.fail(f"Refusing to reuse existing test collection: {collection_name!r}")
        for collection_name in (memory_collection, learning_collection):
            result = qdrant.ensure_collection(collection_name, vector_size, distance)
            if result.get("exists"):
                pytest.fail(f"Refusing to reuse concurrently created test collection: {collection_name!r}")
            created.append(collection_name)
        yield LiveContext(
            qdrant=qdrant,
            embeddings=embeddings,
            memory_collection=memory_collection,
            learning_collection=learning_collection,
            prefix=prefix,
            vector_size=vector_size,
            distance=distance,
            scope={"profile_id": PROFILE_ID, "platform": PLATFORM},
        )
    finally:
        cleanup_errors: list[str] = []
        for collection_name in reversed(created):
            if not collection_name.startswith(prefix):
                cleanup_errors.append(f"refused to delete non-prefixed collection {collection_name!r}")
                continue
            try:
                quoted = urllib.parse.quote(collection_name, safe="")
                qdrant._request("DELETE", f"/collections/{quoted}")
            except Exception as exc:  # pragma: no cover - depends on live service state
                cleanup_errors.append(f"{collection_name}: {exc}")
        if cleanup_errors:
            pytest.fail("Live Qdrant integration cleanup failed: " + "; ".join(cleanup_errors))


def _point_id(collection_name: str, label: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{collection_name}:{label}"))


def _memory_point(
    ctx: LiveContext,
    label: str,
    text: str,
    *,
    source: str,
    tags: list[str],
    file_path: str,
    project_path: str = PROJECT_A,
    created_at: str = CURRENT_DATE,
    profile_id: str = PROFILE_ID,
    platform: str = PLATFORM,
    source_type: str = "project_doc",
    chunk_type: str = "fact",
    importance: int = 8,
    confidence: float = 1.0,
) -> dict[str, Any]:
    payload = build_payload(
        text=text,
        source=source,
        source_type=source_type,
        chunk_type=chunk_type,
        importance=importance,
        confidence=confidence,
        tags=tags,
        profile_id=profile_id,
        platform=platform,
        project_path=project_path,
        model="live-integration",
        created_at=created_at,
    )
    payload["file_path"] = file_path
    return {
        "id": _point_id(ctx.memory_collection, label),
        "vector": ctx.embeddings.embed_document(payload["text"]),
        "payload": payload,
    }


def _learning_point(
    ctx: LiveContext,
    label: str,
    lesson: str,
    *,
    learning_type: str,
    tags: list[str],
    file_path: str,
    project_path: str = PROJECT_A,
    source: str = "hermes_learning",
    created_at: str = CURRENT_DATE,
    profile_id: str = PROFILE_ID,
    platform: str = PLATFORM,
) -> dict[str, Any]:
    payload = build_learning_payload(
        lesson=lesson,
        learning_type=learning_type,
        trigger=f"live integration trigger {label}",
        correction=f"live integration correction {label}",
        evidence=f"live integration evidence {label}",
        tool_name="pytest",
        project_path=project_path,
        profile_id=profile_id,
        platform=platform,
        model="live-integration",
        importance=8,
        confidence=1.0,
        tags=tags,
    )
    payload["source"] = source
    payload["created_at"] = created_at
    payload["last_accessed"] = created_at
    payload["file_path"] = file_path
    return {
        "id": _point_id(ctx.learning_collection, label),
        "vector": ctx.embeddings.embed_document(payload["text"]),
        "payload": payload,
    }


def _ids(results: list[Any]) -> list[str]:
    return [str(item.id) for item in results]


def _assert_only_result(results: list[Any], expected_id: str) -> None:
    assert _ids(results) == [expected_id]


def _provider_for_live_context(ctx: LiveContext, tmp_path: Path):
    from __init__ import QdrantMemoryProvider

    provider = QdrantMemoryProvider()
    provider._qdrant = ctx.qdrant
    provider._embeddings = ctx.embeddings
    provider._active = True
    provider._profile_id = PROFILE_ID
    provider._platform = PLATFORM
    provider._session_id = "live-integration-session"
    provider._hermes_home = str(tmp_path)
    provider._config.update(
        {
            "collection_name": ctx.memory_collection,
            "learning_collection_name": ctx.learning_collection,
            "qdrant_url": os.environ.get("QDRANT_TEST_URL", "http://127.0.0.1:6333"),
            "embedding_url": os.environ.get("QDRANT_TEST_EMBEDDING_URL", "http://127.0.0.1:8080/v1"),
            "embedding_model": os.environ.get("QDRANT_TEST_EMBEDDING_MODEL", "bge-m3"),
            "vector_size": ctx.vector_size,
            "distance": ctx.distance,
            "learning_enabled": True,
            "search_candidates": 20,
            "decay_rate": 0.001,
            "min_raw_score": -1.0,
            "min_final_score": 0.0,
            "index_dry_run_default": True,
            "index_extensions": [".md", ".txt"],
            "index_exclude_dirs": [".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", "target"],
            "max_chunk_tokens": 128,
            "consolidation_report_max_points": 200,
            "consolidation_report_max_groups": 20,
            "consolidation_duplicate_threshold": 0.92,
            "consolidation_stale_days": 90,
            "consolidation_min_importance_for_keep": 4,
            "consolidation_persist_reports": True,
            "consolidation_artifact_dir": str(tmp_path / "consolidation-artifacts"),
            "consolidation_apply_dry_run_default": True,
            "reconsolidation_enabled": True,
            "reconsolidation_report_only": True,
            "reconsolidation_min_confidence": 0.6,
            "reconsolidation_max_candidates": 10,
        }
    )
    provider._retriever = MemoryRetriever(
        qdrant=ctx.qdrant,
        embeddings=ctx.embeddings,
        collection_name=ctx.memory_collection,
        scope=ctx.scope,
        search_candidates=20,
        min_raw_score=-1.0,
    )
    provider._writer = ConversationWriter(
        qdrant=ctx.qdrant,
        embeddings=ctx.embeddings,
        collection_name=ctx.memory_collection,
        profile_id=PROFILE_ID,
        platform=PLATFORM,
        session_id=provider._session_id,
        model=provider._config.get("embedding_model", ""),
    )
    provider._learning_store = LearningStore(
        qdrant=ctx.qdrant,
        embeddings=ctx.embeddings,
        collection_name=ctx.learning_collection,
        profile_id=PROFILE_ID,
        platform=PLATFORM,
        session_id=provider._session_id,
        model=provider._config.get("embedding_model", ""),
        scope=ctx.scope,
        search_candidates=20,
        min_raw_score=-1.0,
    )
    return provider


def _retrieve_one(ctx: LiveContext, collection_name: str, point_id: str) -> dict[str, Any]:
    points = ctx.qdrant.retrieve(collection_name, [point_id], with_payload=True, with_vector=False)
    assert len(points) == 1
    return points[0]


def _proposal_by_type(report: dict[str, Any], proposal_type: str) -> dict[str, Any]:
    matches = [proposal for proposal in report.get("proposals", []) if proposal.get("proposal_type") == proposal_type]
    assert matches, f"missing proposal_type={proposal_type}; summary={report.get('summary')}"
    return matches[0]


def _assert_path_under(path: str | Path, root: Path) -> Path:
    resolved_path = Path(path).resolve()
    resolved_root = root.resolve()
    assert resolved_path == resolved_root or resolved_root in resolved_path.parents
    return resolved_path


def test_live_memory_search_filters_against_real_qdrant(live_context: LiveContext) -> None:
    ctx = live_context
    source = f"{ctx.memory_collection}:source:target"
    other_source = f"{ctx.memory_collection}:source:other"
    target = _memory_point(
        ctx,
        "target",
        "Live memory target marker for Qdrant filter integration search.",
        source=source,
        tags=["live-memory-target", "live-memory-date", "live-memory-source"],
        file_path=f"{PROJECT_A}/docs/target.md",
    )
    old = _memory_point(
        ctx,
        "old-out-of-range",
        "Old live memory marker outside the requested creation window.",
        source=source,
        tags=["live-memory-date"],
        file_path=f"{PROJECT_A}/docs/old.md",
        created_at=OLD_DATE,
    )
    future = _memory_point(
        ctx,
        "future-out-of-range",
        "Future live memory marker outside the requested creation window.",
        source=source,
        tags=["live-memory-date"],
        file_path=f"{PROJECT_A}/docs/future.md",
        created_at=FUTURE_DATE,
    )
    wrong_scope = _memory_point(
        ctx,
        "wrong-scope",
        "Wrong profile live memory marker that scope filters must hide.",
        source=source,
        tags=["live-memory-target", "live-memory-date", "live-memory-source"],
        file_path=f"{PROJECT_A}/docs/wrong-scope.md",
        profile_id="other-profile",
    )
    source_decoy = _memory_point(
        ctx,
        "source-decoy",
        "Live memory marker with a different source for source filter isolation.",
        source=other_source,
        tags=["live-memory-source"],
        file_path=f"{PROJECT_A}/docs/source-decoy.md",
    )
    file_match = _memory_point(
        ctx,
        "file-path-match",
        "Live memory marker for file path filter isolation.",
        source=f"{ctx.memory_collection}:source:file",
        tags=["live-memory-file"],
        file_path=f"{PROJECT_A}/src/file_filter.py",
    )
    project_match = _memory_point(
        ctx,
        "project-path-match",
        "Live memory marker for project path filter isolation.",
        source=f"{ctx.memory_collection}:source:project",
        tags=["live-memory-project"],
        file_path=f"{PROJECT_B}/docs/project.md",
        project_path=PROJECT_B,
    )
    ctx.qdrant.upsert(ctx.memory_collection, [target, old, future, wrong_scope, source_decoy, file_match, project_match])

    retriever = MemoryRetriever(
        qdrant=ctx.qdrant,
        embeddings=ctx.embeddings,
        collection_name=ctx.memory_collection,
        scope=ctx.scope,
        search_candidates=20,
        min_raw_score=-1.0,
    )

    _assert_only_result(
        retriever.search("live memory target marker", top_k=10, tags=["live-memory-target"]),
        str(target["id"]),
    )
    _assert_only_result(
        retriever.search("live memory source marker", top_k=10, tags=["live-memory-source"], source=source),
        str(target["id"]),
    )
    _assert_only_result(
        retriever.search("live memory file path marker", top_k=10, file_path=f"{PROJECT_A}/src/file_filter.py"),
        str(file_match["id"]),
    )
    _assert_only_result(
        retriever.search("live memory project path marker", top_k=10, project_path=PROJECT_B),
        str(project_match["id"]),
    )
    _assert_only_result(
        retriever.search(
            "live memory date marker",
            top_k=10,
            tags=["live-memory-date"],
            since="2026-01-01T00:00:00+00:00",
            until="2026-12-31T23:59:59+00:00",
        ),
        str(target["id"]),
    )

    scoped_results = retriever.search("wrong profile live memory marker", top_k=20)
    assert str(wrong_scope["id"]) not in _ids(scoped_results)
    assert scoped_results
    assert all(item.payload.get("profile_id") == PROFILE_ID for item in scoped_results)
    assert all(item.payload.get("platform") == PLATFORM for item in scoped_results)


def test_live_learning_search_filters_against_real_qdrant(live_context: LiveContext) -> None:
    ctx = live_context
    target = _learning_point(
        ctx,
        "target-learning",
        "Live learning target marker for collection filter integration search.",
        learning_type="user_correction",
        tags=["live-learning-type", "live-learning-date", "live-learning-source"],
        file_path=f"{PROJECT_A}/lessons/target.md",
    )
    old = _learning_point(
        ctx,
        "old-learning",
        "Old live learning marker outside the requested creation window.",
        learning_type="workflow_lesson",
        tags=["live-learning-date"],
        file_path=f"{PROJECT_A}/lessons/old.md",
        created_at=OLD_DATE,
    )
    future = _learning_point(
        ctx,
        "future-learning",
        "Future live learning marker outside the requested creation window.",
        learning_type="workflow_lesson",
        tags=["live-learning-date"],
        file_path=f"{PROJECT_A}/lessons/future.md",
        created_at=FUTURE_DATE,
    )
    wrong_scope = _learning_point(
        ctx,
        "wrong-scope-learning",
        "Wrong profile live learning marker that scope filters must hide.",
        learning_type="user_correction",
        tags=["live-learning-type", "live-learning-date", "live-learning-source"],
        file_path=f"{PROJECT_A}/lessons/wrong-scope.md",
        profile_id="other-profile",
    )
    source_decoy = _learning_point(
        ctx,
        "source-decoy-learning",
        "Live learning marker with a different source for source filter isolation.",
        learning_type="workflow_lesson",
        tags=["live-learning-source"],
        file_path=f"{PROJECT_A}/lessons/source-decoy.md",
        source="external_learning_import",
    )
    file_match = _learning_point(
        ctx,
        "file-path-learning",
        "Live learning marker for file path filter isolation.",
        learning_type="environment_quirk",
        tags=["live-learning-file"],
        file_path=f"{PROJECT_A}/lessons/file_filter.md",
    )
    project_match = _learning_point(
        ctx,
        "project-path-learning",
        "Live learning marker for project path filter isolation.",
        learning_type="tool_failure_lesson",
        tags=["live-learning-project"],
        file_path=f"{PROJECT_B}/lessons/project.md",
        project_path=PROJECT_B,
    )
    ctx.qdrant.upsert(ctx.learning_collection, [target, old, future, wrong_scope, source_decoy, file_match, project_match])

    store = LearningStore(
        qdrant=ctx.qdrant,
        embeddings=ctx.embeddings,
        collection_name=ctx.learning_collection,
        scope=ctx.scope,
        search_candidates=20,
        min_raw_score=-1.0,
    )

    _assert_only_result(
        store.search("live learning target marker", top_k=10, learning_type="user_correction"),
        str(target["id"]),
    )
    _assert_only_result(
        store.search("live learning tag marker", top_k=10, tags=["live-learning-type"]),
        str(target["id"]),
    )
    _assert_only_result(
        store.search("live learning source marker", top_k=10, tags=["live-learning-source"], source="hermes_learning"),
        str(target["id"]),
    )
    _assert_only_result(
        store.search("live learning file marker", top_k=10, file_path=f"{PROJECT_A}/lessons/file_filter.md"),
        str(file_match["id"]),
    )
    _assert_only_result(
        store.search("live learning project marker", top_k=10, project_path=PROJECT_B),
        str(project_match["id"]),
    )
    _assert_only_result(
        store.search(
            "live learning date marker",
            top_k=10,
            tags=["live-learning-date"],
            since="2026-01-01T00:00:00+00:00",
            until="2026-12-31T23:59:59+00:00",
        ),
        str(target["id"]),
    )

    scoped_results = store.search("wrong profile live learning marker", top_k=20)
    assert str(wrong_scope["id"]) not in _ids(scoped_results)
    assert scoped_results
    assert all(item.payload.get("profile_id") == PROFILE_ID for item in scoped_results)
    assert all(item.payload.get("platform") == PLATFORM for item in scoped_results)


def test_live_provider_memory_search_routes_learning_collection(live_context: LiveContext) -> None:
    from __init__ import QdrantMemoryProvider

    ctx = live_context
    route_target = _learning_point(
        ctx,
        "provider-route-learning",
        "Provider route live learning marker for qdrant_memory_search collection routing.",
        learning_type="workflow_lesson",
        tags=["live-provider-route"],
        file_path=f"{PROJECT_A}/lessons/provider-route.md",
    )
    ctx.qdrant.upsert(ctx.learning_collection, [route_target])

    provider = QdrantMemoryProvider()
    provider._qdrant = ctx.qdrant
    provider._embeddings = ctx.embeddings
    provider._active = True
    provider._profile_id = PROFILE_ID
    provider._platform = PLATFORM
    provider._config.update(
        {
            "collection_name": ctx.memory_collection,
            "learning_collection_name": ctx.learning_collection,
            "qdrant_url": os.environ.get("QDRANT_TEST_URL", "http://127.0.0.1:6333"),
            "embedding_url": os.environ.get("QDRANT_TEST_EMBEDDING_URL", "http://127.0.0.1:8080/v1"),
            "embedding_model": os.environ.get("QDRANT_TEST_EMBEDDING_MODEL", "bge-m3"),
            "vector_size": ctx.vector_size,
            "distance": ctx.distance,
            "learning_enabled": True,
            "search_candidates": 20,
            "decay_rate": 0.001,
            "min_raw_score": -1.0,
            "min_final_score": 0.0,
        }
    )
    provider._retriever = MemoryRetriever(
        qdrant=ctx.qdrant,
        embeddings=ctx.embeddings,
        collection_name=ctx.memory_collection,
        scope=ctx.scope,
        search_candidates=20,
        min_raw_score=-1.0,
    )
    provider._learning_store = LearningStore(
        qdrant=ctx.qdrant,
        embeddings=ctx.embeddings,
        collection_name=ctx.learning_collection,
        scope=ctx.scope,
        search_candidates=20,
        min_raw_score=-1.0,
    )

    response = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_search",
            {
                "query": "provider route live learning marker",
                "collection": "learning",
                "tags": ["live-provider-route"],
                "top_k": 10,
                "include_metadata": True,
            },
        )
    )

    assert "error" not in response
    assert response["collection_name"] == ctx.learning_collection
    assert response["count"] == 1
    assert response["results"][0]["id"] == str(route_target["id"])
    assert response["results"][0]["metadata"]["source_type"] == "learning"


def test_live_provider_store_tools_persist_to_disposable_collections(live_context: LiveContext, tmp_path: Path) -> None:
    ctx = live_context
    provider = _provider_for_live_context(ctx, tmp_path)

    memory_response = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_store",
            {
                "text": "Live provider memory store marker for real Qdrant write coverage.",
                "source_type": "manual",
                "importance": 7,
                "tags": ["live-provider-store", "memory-write"],
                "dry_run": False,
                "approve": True,
            },
        )
    )
    assert "error" not in memory_response
    assert memory_response["dry_run"] is False
    assert memory_response["saved"] is True
    memory_id = memory_response["id"]

    memory_point = _retrieve_one(ctx, ctx.memory_collection, memory_id)
    memory_payload = memory_point["payload"]
    assert memory_payload["source_type"] == "manual"
    assert memory_payload["chunk_type"] == "fact"
    assert memory_payload["profile_id"] == PROFILE_ID
    assert memory_payload["platform"] == PLATFORM
    assert memory_payload["session_id"] == "live-integration-session"
    assert "live-provider-store" in memory_payload["tags"]
    assert ctx.qdrant.retrieve(ctx.learning_collection, [memory_id], with_payload=True, with_vector=False) == []

    learning_response = json.loads(
        provider.handle_tool_call(
            "qdrant_learning_store",
            {
                "lesson": "Live provider learning store marker for real Qdrant write coverage.",
                "learning_type": "workflow_lesson",
                "trigger": "live provider learning trigger",
                "correction": "live provider learning correction",
                "evidence": "live provider learning evidence",
                "importance": 9,
                "confidence": 0.95,
                "tags": ["live-provider-store", "learning-write"],
                "promote_to_skill_candidate": True,
                "dry_run": False,
                "approve": True,
            },
        )
    )
    assert "error" not in learning_response
    assert learning_response["saved"] is True
    learning_id = learning_response["id"]

    learning_point = _retrieve_one(ctx, ctx.learning_collection, learning_id)
    learning_payload = learning_point["payload"]
    assert learning_payload["source_type"] == "learning"
    assert learning_payload["learning_type"] == "workflow_lesson"
    assert learning_payload["profile_id"] == PROFILE_ID
    assert learning_payload["platform"] == PLATFORM
    assert learning_payload["session_id"] == "live-integration-session"
    assert learning_payload["trigger"] == "live provider learning trigger"
    assert learning_payload["correction"] == "live provider learning correction"
    assert learning_payload["evidence"] == "live provider learning evidence"
    assert learning_payload["promote_to_skill_candidate"] is True
    assert "learning-write" in learning_payload["tags"]
    assert ctx.qdrant.retrieve(ctx.memory_collection, [learning_id], with_payload=True, with_vector=False) == []


def test_live_provider_index_upserts_and_deletes_stale_file_chunks(live_context: LiveContext, tmp_path: Path) -> None:
    ctx = live_context
    provider = _provider_for_live_context(ctx, tmp_path)
    docs_dir = tmp_path / "project-docs"
    docs_dir.mkdir()
    note = docs_dir / "live-index.md"
    note.write_text(
        "# Alpha\n\nLive index alpha marker for Qdrant upsert coverage.\n\n"
        "# Beta\n\nLive index beta marker for stale chunk deletion coverage.\n",
        encoding="utf-8",
    )

    first = json.loads(provider.handle_tool_call("qdrant_memory_index", {"paths": [str(note)], "dry_run": False}))
    assert "error" not in first
    assert first["dry_run"] is False
    assert first["chunks_prepared"] == 2
    assert first["chunks_upserted"] == 2
    assert first["chunks_deleted"] == 0
    assert first["manifest_checked"] is True

    first_id = make_file_chunk_id(str(note.resolve()), 0)
    stale_id = make_file_chunk_id(str(note.resolve()), 1)
    first_point = _retrieve_one(ctx, ctx.memory_collection, first_id)
    stale_point = _retrieve_one(ctx, ctx.memory_collection, stale_id)
    assert first_point["payload"]["chunk_type"] == "file_chunk"
    assert stale_point["payload"]["chunk_type"] == "file_chunk"
    assert first_point["payload"]["file_path"] == str(note.resolve())

    note.write_text("# Alpha\n\nLive index alpha marker after shrinking the indexed file.\n", encoding="utf-8")

    preview = json.loads(provider.handle_tool_call("qdrant_memory_index", {"paths": [str(note)], "dry_run": True}))
    assert "error" not in preview
    assert preview["dry_run"] is True
    assert preview["chunks_prepared"] == 1
    assert preview["stale_count"] == 1
    assert preview["stale_ids"] == [stale_id]
    assert preview["chunks_deleted"] == 0
    assert _retrieve_one(ctx, ctx.memory_collection, stale_id)

    applied = json.loads(provider.handle_tool_call("qdrant_memory_index", {"paths": [str(note)], "dry_run": False}))
    assert "error" not in applied
    assert applied["dry_run"] is False
    assert applied["chunks_prepared"] == 1
    assert applied["chunks_deleted"] == 1
    assert applied["stale_ids"] == [stale_id]
    assert applied["delete_mode"] == "ids"
    assert ctx.qdrant.retrieve(ctx.memory_collection, [stale_id], with_payload=True, with_vector=False) == []
    current_point = _retrieve_one(ctx, ctx.memory_collection, first_id)
    assert "after shrinking" in current_point["payload"]["text"]


def test_live_provider_consolidation_report_and_gated_apply_paths(live_context: LiveContext, tmp_path: Path) -> None:
    ctx = live_context
    provider = _provider_for_live_context(ctx, tmp_path)
    duplicate_text = "Live consolidation duplicate marker should merge through explicit proposal approval."
    duplicate_a = _memory_point(
        ctx,
        "duplicate-a",
        duplicate_text,
        source=f"{ctx.memory_collection}:consolidation:duplicate-a",
        tags=["live-consolidation-duplicate"],
        file_path=f"{PROJECT_A}/consolidation/duplicate-a.md",
        importance=8,
        confidence=0.95,
    )
    duplicate_b = _memory_point(
        ctx,
        "duplicate-b",
        duplicate_text,
        source=f"{ctx.memory_collection}:consolidation:duplicate-b",
        tags=["live-consolidation-duplicate"],
        file_path=f"{PROJECT_A}/consolidation/duplicate-b.md",
        importance=7,
        confidence=0.9,
    )
    stale = _memory_point(
        ctx,
        "stale-low-value",
        "Live consolidation stale marker with low value and old timestamp.",
        source=f"{ctx.memory_collection}:consolidation:stale",
        tags=["live-consolidation-stale"],
        file_path=f"{PROJECT_A}/consolidation/stale.md",
        created_at=OLD_DATE,
        importance=1,
        confidence=0.4,
    )
    recon_a = _memory_point(
        ctx,
        "recon-a",
        "Live reconsolidation fact marker says the test status is alpha.",
        source=f"{ctx.memory_collection}:consolidation:recon-a",
        tags=["live-consolidation-recon"],
        file_path=f"{PROJECT_A}/consolidation/recon-a.md",
        source_type="manual",
        importance=9,
        confidence=0.95,
    )
    recon_b = _memory_point(
        ctx,
        "recon-b",
        "Live reconsolidation fact marker says the test status is beta.",
        source=f"{ctx.memory_collection}:consolidation:recon-b",
        tags=["live-consolidation-recon"],
        file_path=f"{PROJECT_A}/consolidation/recon-b.md",
        source_type="manual",
        importance=8,
        confidence=0.9,
    )
    for point in (recon_a, recon_b):
        point["payload"]["fact_key"] = "live-integration-status"
    seeded_points = [duplicate_a, duplicate_b, stale, recon_a, recon_b]
    ctx.qdrant.upsert(ctx.memory_collection, seeded_points)

    report = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_consolidate",
            {
                "scope": "memory",
                "persist": True,
                "include_examples": True,
                "include_reconsolidation": True,
                "max_groups": 10,
            },
        )
    )
    assert "error" not in report
    assert report["dry_run"] is True
    assert report["report_only"] is True
    assert report["mutations_performed"] is False
    assert report["persisted"] is True
    report_path = _assert_path_under(report["artifact"]["path"], tmp_path)
    assert report_path.exists()
    assert report["summary"]["duplicate_cluster"] >= 1
    assert report["summary"]["stale_low_value"] >= 1
    assert report["summary"]["reconsolidation_candidate"] >= 1
    for point in seeded_points:
        assert _retrieve_one(ctx, ctx.memory_collection, str(point["id"]))

    delete_proposal = _proposal_by_type(report, "stale_low_value")
    delete_preview = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_consolidation_apply",
            {
                "report_id": report["report_id"],
                "proposal_id": delete_proposal["proposal_id"],
                "action": "delete",
            },
        )
    )
    assert "error" not in delete_preview
    assert delete_preview["dry_run"] is True
    assert delete_preview["would_apply"] is True
    assert _retrieve_one(ctx, ctx.memory_collection, str(stale["id"]))

    delete_denied = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_consolidation_apply",
            {
                "report_id": report["report_id"],
                "proposal_id": delete_proposal["proposal_id"],
                "action": "delete",
                "dry_run": False,
            },
        )
    )
    assert delete_denied["error"] == "approve=true is required when dry_run=false"
    assert _retrieve_one(ctx, ctx.memory_collection, str(stale["id"]))

    delete_applied = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_consolidation_apply",
            {
                "report_id": report["report_id"],
                "proposal_id": delete_proposal["proposal_id"],
                "action": "delete",
                "dry_run": False,
                "approve": True,
            },
        )
    )
    assert "error" not in delete_applied
    assert delete_applied["applied"] is True
    assert delete_applied["deleted_ids"] == [str(stale["id"])]
    assert _assert_path_under(delete_applied["application_artifact"], tmp_path).exists()
    assert ctx.qdrant.retrieve(ctx.memory_collection, [str(stale["id"])], with_payload=True, with_vector=False) == []

    merge_proposal = _proposal_by_type(report, "duplicate_cluster")
    merge_applied = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_consolidation_apply",
            {
                "report_id": report["report_id"],
                "proposal_id": merge_proposal["proposal_id"],
                "action": "merge",
                "dry_run": False,
                "approve": True,
            },
        )
    )
    assert "error" not in merge_applied
    assert merge_applied["applied"] is True
    canonical_id = merge_applied["canonical_id"]
    deleted_duplicate_ids = merge_applied["deleted_ids"]
    assert deleted_duplicate_ids
    assert _assert_path_under(merge_applied["application_artifact"], tmp_path).exists()
    canonical_point = _retrieve_one(ctx, ctx.memory_collection, canonical_id)
    assert canonical_point["payload"]["consolidation_proposal_id"] == merge_proposal["proposal_id"]
    assert sorted(canonical_point["payload"]["consolidated_from"]) == sorted(deleted_duplicate_ids)
    assert ctx.qdrant.retrieve(ctx.memory_collection, deleted_duplicate_ids, with_payload=True, with_vector=False) == []

    draft_proposal = _proposal_by_type(report, "reconsolidation_candidate")
    draft_applied = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_consolidation_apply",
            {
                "report_id": report["report_id"],
                "proposal_id": draft_proposal["proposal_id"],
                "action": "draft_review",
                "dry_run": False,
                "approve": True,
            },
        )
    )
    assert "error" not in draft_applied
    assert draft_applied["applied"] is True
    draft_path = _assert_path_under(draft_applied["reconsolidation_draft_path"], tmp_path)
    assert draft_path.exists()
    assert _assert_path_under(draft_applied["application_artifact"], tmp_path).exists()
    draft_text = draft_path.read_text(encoding="utf-8")
    assert "# Reconsolidation review draft" in draft_text
    assert "review artifact" in draft_text.lower()
    assert "does not mutate qdrant memory" in draft_text.lower()
    assert draft_proposal["proposal_id"] in draft_text
    recon_a_after = _retrieve_one(ctx, ctx.memory_collection, str(recon_a["id"]))
    recon_b_after = _retrieve_one(ctx, ctx.memory_collection, str(recon_b["id"]))
    assert recon_a_after["payload"] == recon_a["payload"]
    assert recon_b_after["payload"] == recon_b["payload"]


def test_live_guarded_auto_canary_applies_one_exact_duplicate_and_leaves_controls_untouched(
    live_context: LiveContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qdrant_memory.guarded_auto import GuardedAutoPolicy, apply_guarded_auto

    ctx = live_context
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    provider = _provider_for_live_context(ctx, tmp_path)
    provider._hermes_home = str(tmp_path / "hermes-home")
    provider._config["consolidation_artifact_dir"] = str(tmp_path / "artifacts")

    exact_text = "Guarded auto exact duplicate canary marker retains one implementation fact."
    exact_a = _memory_point(
        ctx,
        "guarded-auto-exact-a",
        exact_text,
        source=f"{ctx.memory_collection}:guarded-auto:exact-a",
        tags=["live-guarded-auto", "exact-duplicate"],
        file_path=f"{PROJECT_A}/guarded-auto/exact-a.md",
        importance=9,
        confidence=0.99,
    )
    exact_b = _memory_point(
        ctx,
        "guarded-auto-exact-b",
        f"  {exact_text.upper()}  ",
        source=f"{ctx.memory_collection}:guarded-auto:exact-b",
        tags=["live-guarded-auto", "exact-duplicate"],
        file_path=f"{PROJECT_A}/guarded-auto/exact-b.md",
        importance=8,
        confidence=0.99,
    )
    near_text = (
        "This controlled medium risk canary record requires an operator review before mutation of isolated memory data."
    )
    near_a = _memory_point(
        ctx,
        "guarded-auto-near-a",
        near_text,
        source=f"{ctx.memory_collection}:guarded-auto:near-a",
        tags=["live-guarded-auto", "near-duplicate-control"],
        file_path=f"{PROJECT_A}/guarded-auto/near-a.md",
        source_type="manual",
        importance=8,
        confidence=0.99,
    )
    near_b = _memory_point(
        ctx,
        "guarded-auto-near-b",
        f"{near_text[:-1]} carefully.",
        source=f"{ctx.memory_collection}:guarded-auto:near-b",
        tags=["live-guarded-auto", "near-duplicate-control"],
        file_path=f"{PROJECT_A}/guarded-auto/near-b.md",
        source_type="manual",
        importance=8,
        confidence=0.99,
    )
    for point in (near_a, near_b):
        point["payload"]["fact_key"] = "guarded-auto-medium-risk-control"

    secret_sentinel = "canary-" + "secret" + "-sentinel-" + uuid.uuid4().hex
    secret_control = _memory_point(
        ctx,
        "guarded-auto-secret-control",
        "This isolated canary control carries " + "".join(("se", "cret", "=", secret_sentinel)),
        source=f"{ctx.memory_collection}:guarded-auto:secret-control",
        tags=["live-guarded-auto", "secret-control"],
        file_path=f"{PROJECT_A}/guarded-auto/secret-control.md",
        importance=8,
        confidence=0.99,
    )
    seeded_points = [exact_a, exact_b, near_a, near_b, secret_control]
    ctx.qdrant.upsert(ctx.memory_collection, seeded_points)
    seeded_payloads = {
        str(point["id"]): _retrieve_one(ctx, ctx.memory_collection, str(point["id"]))["payload"]
        for point in seeded_points
    }
    memory_count_before = ctx.qdrant.count(ctx.memory_collection)
    learning_count_before = ctx.qdrant.count(ctx.learning_collection)

    report = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_consolidate",
            {"scope": "memory", "persist": True, "include_examples": False, "max_groups": 10},
        )
    )
    assert "error" not in report
    report_path = _assert_path_under(report["artifact"]["path"], tmp_path)
    assert report_path.exists()
    exact_proposal = next(
        proposal
        for proposal in report["proposals"]
        if proposal["proposal_type"] == "duplicate_cluster"
        and set(proposal["affected_ids"]) == {str(exact_a["id"]), str(exact_b["id"])}
    )
    assert exact_proposal["match_kind"] == "exact_normalized"
    assert exact_proposal["guarded_auto_eligible"] is True
    near_proposal = next(
        proposal
        for proposal in report["proposals"]
        if proposal["proposal_type"] == "duplicate_cluster"
        and set(proposal["affected_ids"]) == {str(near_a["id"]), str(near_b["id"])}
    )
    assert near_proposal["match_kind"] == "near_duplicate"
    assert near_proposal["risk"] == "medium"
    assert near_proposal["guarded_auto_eligible"] is False

    policy = GuardedAutoPolicy(mode="guarded-auto", max_actions=1)
    guarded_auto = apply_guarded_auto(provider, report, policy)
    assert guarded_auto["attempted"] == 1
    assert guarded_auto["errors"] == []
    assert len(guarded_auto["applied"]) == 1
    applied = guarded_auto["applied"][0]
    assert applied["proposal_id"] == exact_proposal["proposal_id"]
    assert applied["action"] == "merge"
    result = applied["result"]
    assert result["applied"] is True
    assert {result["canonical_id"], *result["deleted_ids"]} == {str(exact_a["id"]), str(exact_b["id"])}
    application_path = _assert_path_under(result["application_artifact"], tmp_path)
    assert application_path.exists()

    assert _retrieve_one(ctx, ctx.memory_collection, result["canonical_id"])
    assert ctx.qdrant.retrieve(ctx.memory_collection, result["deleted_ids"], with_payload=True, with_vector=False) == []
    for point in (near_a, near_b, secret_control):
        actual_payload = _retrieve_one(ctx, ctx.memory_collection, str(point["id"]))["payload"]
        assert actual_payload == seeded_payloads[str(point["id"])]
    assert ctx.qdrant.count(ctx.memory_collection) == memory_count_before - 1
    assert ctx.qdrant.count(ctx.learning_collection) == learning_count_before

    public_output = json.dumps({"report": report, "guarded_auto": guarded_auto}, sort_keys=True)
    assert secret_sentinel not in public_output
    assert secret_sentinel not in report_path.read_text(encoding="utf-8")
    assert secret_sentinel not in application_path.read_text(encoding="utf-8")

    replay = apply_guarded_auto(provider, report, policy)
    assert replay["attempted"] == 1
    assert replay["applied"] == []
    assert replay["errors"] == [
        {
            "proposal_id": exact_proposal["proposal_id"],
            "action": "merge",
            "reason": "exact normalized duplicate cluster is preauthorized for merge",
            "error": "affected point missing; rerun consolidation",
        }
    ]
