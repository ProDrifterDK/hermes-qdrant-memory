from __future__ import annotations

import json
import os
import urllib.parse
import uuid
from dataclasses import dataclass
from typing import Any, Iterator

import pytest

from qdrant_memory.client import QdrantClient
from qdrant_memory.embeddings import EmbeddingClient
from qdrant_memory.learning import LearningStore, build_learning_payload
from qdrant_memory.retriever import MemoryRetriever
from qdrant_memory.schema import build_payload

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
) -> dict[str, Any]:
    payload = build_payload(
        text=text,
        source=source,
        source_type="project_doc",
        chunk_type="fact",
        importance=8,
        confidence=1.0,
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
